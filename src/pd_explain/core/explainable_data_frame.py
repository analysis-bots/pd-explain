from __future__ import annotations

import warnings
from copy import copy as cpy

import numpy as np
import pandas as pd
from matplotlib.axis import Axis
from pandas import DataFrame
from pandas._libs.lib import no_default
import dill

import sys

# adding Folder_2/subfolder to the system path

sys.path.insert(0, 'C:/Users/itaye/Desktop/pdexplain/FEDEx_Generator-1/src/')
sys.path.insert(0, "C:\\Users\\Yuval\\PycharmProjects\\FEDEx_Generator\\src")
sys.path.insert(0, "C:\\Users\\Yuval\\PycharmProjects\\cluster-explorer\\src")
sys.path.insert(0, "C:\\Users\\Yuval\\PycharmProjects\\ExternalExplainers\\src")
# sys.path.insert(0, 'C:/Users/User/Desktop/pd_explain_test/FEDEx_Generator-1/src')
from fedex_generator.Operations.Filter import Filter
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Join import Join
from fedex_generator.Operations.BJoin import BJoin
from fedex_generator.commons import utils
from fedex_generator.commons.utils import get_calling_params_name
from typing import (
    Hashable,
    Sequence,
    Union,
    List, Callable, Literal, Tuple, )
from pd_explain.explainers import ExplainerFactory
from pd_explain.utils.global_values import get_use_sampling_value
from pandas._typing import Level, Renamer, IndexLabel, Axes, Dtype, DropKeep

sys.path.insert(0, 'C:/Users/itaye/Desktop/pdexplain/pd-explain/src/')
sys.path.insert(0, "C:\\Users\\Yuval\\PycharmProjects\\pd-explain\\src")
# sys.path.insert(0, 'C:/Users/User/Desktop/pd_explain_test/pd-explain/src')
from pd_explain.core.explainable_series import ExpSeries
from pd_explain.experimental.query_recommenders.llm_based_query_recommender import LLMBasedQueryRecommender
from pd_explain.llm_integrations.automated_data_exploration import AutomatedDataExploration
from pd_explain.explainers.outlier_explainer import OutlierExplainer
from pd_explain.explainers.explainer_interface import ExplainerInterface


class ExpDataFrame(pd.DataFrame):
    """
    Explainable dataframe, extends Pandas DataFrame adding state management and explain() function.
    """

    def __init__(
            self,
            data=None,
            index: Axes | None = None,
            columns: Axes | None = None,
            dtype: Dtype | None = None,
            copy: bool | None = None,
    ):
        """
        Initialize explain dataframe
        Two-dimensional, size-mutable, potentially heterogeneous tabular data.

        Data structure also contains labeled axes (rows and columns).
        Arithmetic operations align on both row and column labels.
        Can be thought of as a dict-like container for Series objects. The primary pandas data structure.

        :param data: Dict can contain Series, arrays, constants, dataclass or list-like objects.
                     If data is a dict, column order follows insertion-order.
                     If a dict contains Series which have an index defined, it is aligned by its index.
        :param index: Index to use for resulting frame.
                      Will default to RangeIndex if no indexing information part of input data and no index provided.
        :param columns: Column labels to use for resulting frame when data does not have them,
                        defaulting to RangeIndex(0, 1, 2, …, n).
                        If data contains column labels, will perform column selection instead.
        :param dtype: Data type to force. Only a single dtype is allowed. If None, infer.
        :param copy: Copy data from inputs. For dict data, the default of None behaves like copy=True.
                     For DataFrame or 2d ndarray input, the default of None behaves like copy=False.
        """
        super().__init__(data, index, columns, dtype, copy)
        self.operation = None
        self.explanation = None
        self.filter_items = None
        self.last_used_explainer: ExplainerInterface | None = None
        self.data_explorer = None

    # We overwrite the constructor to ensure that an ExpDataFrame is returned when a new DataFrame is created.
    # This is necessary so that methods not overridden in this class, like iloc, return an ExpDataFrame.
    @property
    def _constructor(self) -> Callable[..., DataFrame]:

        # We define a new constructor that returns an ExpDataFrame, with the same properties as the original dataframe.
        def _c(*args, **kwargs):
            df = ExpDataFrame(*args, **kwargs)
            df.operation = self.operation
            df.explanation = self.explanation
            df.filter_items = self.filter_items
            return df

        return _c

    def llm_recommend_experimental(self, custom_requests=None, num_recommendations=5, num_iterations=3,
                                   return_all_options: bool = True):
        """
        Generate queries for the DataFrame using the LLM.
        Please note that this feature is experimental and may not work as expected or produce meaningful results.

        :param custom_requests: Custom requests to be sent to the LLM. Optional.
        :param num_recommendations: Number of recommendations to generate. Default is 4.
        :param num_iterations: Number of iterations to run the query refinement process. Default is 2. Note that every
        iteration will call the LLM twice, so this will result in 2 * num_iterations calls to the LLM.
        :param return_all_options: If True, returns all options generated throughout the iterations, instead of just the
        top k options. Default is False.

        :return: A Series of generated queries or None if no queries are generated.
        """
        source_name = ""
        # If we can get the source name from the operation, we will use it, since it is more likely to be the name of the
        # original dataframe.
        if self.operation is not None:
            if hasattr("self.operation", "source_name"):
                source_name = self.operation.source_name
            else:
                source_name = self.operation.left_name
        # Otherwise, if we had no operation, we will use the name of the dataframe.
        else:
            source_name = get_calling_params_name(self)
        recommender = LLMBasedQueryRecommender(
            df=self,
            df_name=source_name,
            user_requests=custom_requests,
            k=num_recommendations,
            n=num_iterations,
            return_all_options=return_all_options,
        )
        return recommender.recommend()

    def automated_data_exploration(self, user_query: str, num_iterations: int = 10, fedex_top_k: int = 3, metainsight_top_k: int = 2,
                                   metainsight_max_filter_cols: int = 3, metainsight_max_agg_cols: int = 3,
                                   verbose: bool = False,
                                   input_df: 'ExpDataFrame' = None,
                                   max_iterations_to_add: int = 3,
                                   beautify_fedex_visualizations: bool = False,
                                   beautify_metainsight_visualizations: bool = False,
                                   ):
        """
        Use LLMs to perform automated exploration and analysis on the DataFrame based on the user's request.
        Automated exploration iteratively generates a query tree on the DataFrame, where at each iteration the
        queries are ran, analyzed, and the next queries are generated based on the results of the previous queries.
        The end result includes a report summarizing the analysis, as well as visualizations of the most important
        queries and of the query tree in a widget.
        This may take a while to run, depending on the number of iterations and queries per iteration.

        :param user_query: What the user wants to analyze in the DataFrame. Example: "Explore the relationship between
        column A and column B".
        :param num_iterations: Number of iterations to run the automated exploration for. Default is 10. Note that each iteration
        will call the LLM once.
        :param fedex_top_k: Number of top findings to return from the FEDEx explainer. Default is 3.
        :param metainsight_top_k: Number of top findings to return from the MetaInsight explainer. Default is 2.
        :param metainsight_max_filter_cols: Maximum number of columns to analyze distribution of in the MetaInsight
        explainer. Default is 3.
        :param metainsight_max_agg_cols: Maximum number of columns to aggregate by in the MetaInsight explainer. Default is 3.
        :param verbose: If True, will print additional information about the process. Default is False.
        :param input_df: Optional parameter to pass an input DataFrame to replace the self DataFrame.
        :param max_iterations_to_add: The maximum number of iterations to add in case the LLM fails during some iterations.
        Default is 3. This can help mitigate cases where the LLM fails too many iterations and thus does not get enough
        information.
        :param beautify_fedex_visualizations: If True, will use the LLM beautify feature to try and beautify the FEDEx visualizations.
        Default is False.
        :param beautify_metainsight_visualizations: If True, will use the LLM beautify feature to try and beautify the MetaInsight visualizations.
        Default is False.

        :return: A widget containing the automated exploration's analysis results, including a report and visualizations.

        :raises ValueError: If the user_query is not a string, or if any of the parameters are not positive integers.
        """
        if not isinstance(user_query, str):
            raise ValueError("user_query must be a string describing what you want to analyze in the DataFrame.")
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        if not isinstance(fedex_top_k, int) or fedex_top_k <= 0:
            raise ValueError("fedex_top_k must be a positive integer.")
        if not isinstance(metainsight_top_k, int) or metainsight_top_k <= 0:
            raise ValueError("metainsight_top_k must be a positive integer.")
        if not isinstance(metainsight_max_filter_cols, int) or metainsight_max_filter_cols <= 0:
            raise ValueError("metainsight_max_filter_cols must be a positive integer.")
        if not isinstance(metainsight_max_agg_cols, int) or metainsight_max_agg_cols <= 0:
            raise ValueError("metainsight_max_agg_cols must be a positive integer.")

        # Create the AutomatedDataExploration object with the provided parameters.
        self.data_explorer = AutomatedDataExploration(
            dataframe=self if input_df is None else input_df,
            source_name=get_calling_params_name(self),
            beautify_fedex=beautify_fedex_visualizations,
            beautify_metainsight=beautify_metainsight_visualizations,
        )
        # Run the automated exploration with the user query and the parameters.
        self.data_explorer.do_llm_action(
            user_query=user_query,
            num_iterations=num_iterations,
            fedex_top_k=fedex_top_k,
            metainsight_top_k=metainsight_top_k,
            metainsight_max_filter_cols=metainsight_max_filter_cols,
            metainsight_max_agg_cols=metainsight_max_agg_cols,
            verbose=verbose,
            max_iterations_to_add=max_iterations_to_add
        )
        # Visualize and save the results in the exploration_visualization property.
        exploration_visualization = self.data_explorer.do_follow_up_action()
        return exploration_visualization

    def save_data_exploration(self, file_path: str):
        """
        Save the data exploration results to a file.
        Uses dill to serialize the data_explorer object.
        :param file_path: The path to save the data exploration results to.
        """
        if self.data_explorer is None:
            raise ValueError(
                "No data exploration has been performed yet. Please run automated_data_exploration() first.")

        attributes = {
            'history': self.data_explorer.history,
            'query_and_results': self.data_explorer.query_and_results,
            'query_tree': self.data_explorer.query_tree,
            'final_report': self.data_explorer.final_report,
            'source_name': self.data_explorer.source_name,
            'beautify_fedex': self.data_explorer.beautify_fedex,
            'beautify_metainsight': self.data_explorer.beautify_metainsight,
            'fedex_beautify_code': self.data_explorer.visualizer.fedex_beautify_code,
            'metainsight_beautify_code': self.data_explorer.visualizer.metainsight_beautify_code,
            'log': self.data_explorer.log,
        }

        with open(file_path, 'wb') as file:
            dill.dump(attributes, file)

    @staticmethod
    def visualize_from_saved_data_exploration(file_path: str,
                                              visualization_type: Literal['graph', 'simple'] = 'graph'):
        """
        Visualize the data exploration results from a saved file.
        Uses dill to deserialize the data_explorer object.
        :param file_path: The path to load the data exploration results from.
        :param visualization_type: The type of visualization for the query tree. Can be 'graph' for an interactive graph
        visualization, or 'simple' for a simpler, static HTML visualization. Default is 'graph'.
        """
        with open(file_path, 'rb') as file:
            data_explorer_attributes = dill.load(file)
        # Don't actually need the dataframe here, as we are only visualizing the results.
        data_explorer = AutomatedDataExploration(pd.DataFrame())
        return data_explorer.do_follow_up_action(history=data_explorer_attributes['history'],
                                                 query_and_results=data_explorer_attributes['query_and_results'],
                                                 query_tree=data_explorer_attributes['query_tree'],
                                                 final_report=data_explorer_attributes['final_report'],
                                                 source_name=data_explorer_attributes['source_name'],
                                                 beautify_fedex=data_explorer_attributes['beautify_fedex'],
                                                 log=data_explorer_attributes['log'],
                                                 beautify_metainsight=data_explorer_attributes['beautify_metainsight'],
                                                 fedex_beautify_code=data_explorer_attributes['fedex_beautify_code'],
                                                 metainsight_beautify_code=data_explorer_attributes['metainsight_beautify_code'])

    def follow_up_with_automated_data_exploration(self, explanation_index: int = None,
                                                  num_iterations: int = 10, fedex_top_k: int = 3,
                                                  metainsight_top_k: int = 2,
                                                  metainsight_max_filter_cols: int = 3,
                                                  metainsight_max_agg_cols: int = 3,
                                                  verbose=False,
                                                  max_iterations_to_add: int = 3,
                                                  beautify_fedex_visualizations: bool = False,
                                                  beautify_metainsight_visualizations: bool = False,
                                                  ):
        """
        Use the automated data exploration feature to follow up on specific explanations received from the last called explain() method.
        This method will automatically try to follow up on explanations, to try and contextualize them in the data, and
        if a previous explanation was generated by a LLM using the reasoning feature, it will see if it can corroborate
        or reject that explanaton based on the data.
        Refer to the documentation of the automated_data_exploration() method for more details on the parameters aside from
        the explanation_index parameter.

        :param explanation_index: The index of the explanation to follow up on. The indexes for each explainer go as follows:
                                    - FEDEx: Top left to bottom right, row by row. Top left plot is index 0, to the right on the
                                    same row is index 1 (and so on the same row), then the next row starts with index 2, and so on.
                                    - MetaInsight: Top to bottom. Index 0 is the top plot, index 1 is the plot below it, and so on.
                                    - Many to One explainer: by row order. First row is index 0, second row is index 1, and so on.
                                    - Outlier explainer: Irrelevant, as it only has one explanation.
                                    All explainers but the Outlier explainer require at least one explanation index to be provided.
        :param num_iterations: Number of iterations to run the automated exploration for. Default is 10.
        :param fedex_top_k: Number of top findings to return from the FEDEx explainer. Default is 3.
        :param metainsight_top_k: Number of top findings to return from the MetaInsight explainer. Default is 2.
        :param metainsight_max_filter_cols: Maximum number of columns to analyze distribution of in the MetaInsight
                                            explainer. Default is 3.
        :param metainsight_max_agg_cols: Maximum number of columns to aggregate by in the MetaInsight explainer. Default is 3.
        :param verbose: If True, will print additional information about the process. Default is False.
        :param max_iterations_to_add: The maximum number of iterations to add in case the LLM fails during some iterations.
        Default is 3.
        :param beautify_fedex_visualizations: If True, will use the LLM beautify feature to try and beautify the FEDEx visualizations.
        Default is False.
        :param beautify_metainsight_visualizations: If True, will use the LLM beautify feature to try and beautify the MetaInsight visualizations.
        Default is False.

        :return: A widget containing the automated exploration's analysis results, including a report and visualizations.
        """
        if self.last_used_explainer is None:
            raise ValueError("No explainer has been used yet. Please run explain() first.")
        indexes_valid = isinstance(explanation_index, int) and explanation_index >= 0
        if not isinstance(self.last_used_explainer, OutlierExplainer) and not indexes_valid:
            raise ValueError("If the explainer last used on this DataFrame is not the outlier explainer, you must "
                             "provide a non-negative integer index to follow up on the explanation. ")

        description = self.last_used_explainer.get_explanation_in_textual_description(index=explanation_index)
        explorer_query = f"{description}\n"
        explorer_query += (
            "Your goal is to use the data to follow up on the findings, to try and provide context to them "
            "and further explain them and draw more information from them using the data, by querying the original "
            "dataframe that had no queries applied to it yet.\n"
            "If there is context guessed by a LLM included, in your report you should state whether your findings "
            "corroborate, reject, or are inconclusive about the context provided. You may try to use this "
            "context as a starting point for what to look for, but do not base your entire analysis on it.\n")
        input_df = None
        if self.operation is not None:
            if hasattr(self.operation, 'source_df'):
                input_df = self.operation.source_df
            elif hasattr(self.operation, 'left_df'):
                input_df = self.operation.left_df
        return self.automated_data_exploration(
            user_query=explorer_query,
            num_iterations=num_iterations,
            fedex_top_k=fedex_top_k,
            metainsight_top_k=metainsight_top_k,
            metainsight_max_filter_cols=metainsight_max_filter_cols,
            metainsight_max_agg_cols=metainsight_max_agg_cols,
            verbose=verbose,
            input_df=input_df,
            max_iterations_to_add=max_iterations_to_add,
            beautify_fedex_visualizations=beautify_fedex_visualizations,
            beautify_metainsight_visualizations=beautify_metainsight_visualizations,
        )

    @property
    def _constructor_sliced(self) -> Callable[..., ExpSeries]:
        return ExpSeries

    def __getitem__(self, key):
        """
        Get item from dataframe, save the item key
        :param key: key to the item in dataframe

        :return: item from dataframe
        """

        if isinstance(key, str):
            if self.filter_items is None:
                self.filter_items = []
            self.filter_items.append(key)
        if isinstance(key, list):
            if self.filter_items is None:
                self.filter_items = []
            self.filter_items.extend(key)
        to_return = super().__getitem__(key)

        # Convert the result to an explainable dataframe or series if it is not already.
        if isinstance(to_return, pd.DataFrame) and not isinstance(to_return, ExpDataFrame):
            to_return = ExpDataFrame(to_return)
        elif isinstance(to_return, pd.Series) and not isinstance(to_return, ExpSeries):
            to_return = ExpSeries(to_return)

        # If the item is an explainable dataframe or series, we want to update its operation.
        if isinstance(to_return, ExpDataFrame) or isinstance(to_return, ExpSeries):
            # We only want to make the updates if the operation is not None, and if the get_item is a column
            # selection, not a row selection (a filter operation).
            if self.operation is not None and (
                    isinstance(key, str) or (isinstance(key, list)) and all([x in self.columns for x in key])):

                # Copy the operation, to avoid changing the original operation of the dataframe.
                to_return.operation = cpy(self.operation)

                # Filter and GroupBy operations: perform the same selection on the source dataframe
                if hasattr(to_return.operation, 'source_df') and to_return.operation.source_df is not None:
                    to_return.operation.source_df = to_return.operation.source_df.__getitem__(key)

                # Join operations: perform the same selection on the left and right dataframes.
                elif hasattr(to_return.operation, 'left_df'):
                    try:
                        to_return.operation.left_df = to_return.operation.left_df.__getitem__(key)
                    except KeyError:
                        pass
                    try:
                        to_return.operation.right_df = to_return.operation.right_df.__getitem__(key)
                    except KeyError:
                        pass

                # Finally, update the result dataframe of the operation to have the same selection applied.
                to_return.operation.result_df = to_return if isinstance(to_return,
                                                                        ExpDataFrame) else to_return.to_frame()

        return to_return

    def copy(self, deep=True):
        """
        Make a copy of this object’s indices and data.
        :param deep: Make a deep copy, including a copy of the data and the indices.
                     With deep=False neither the indices nor the data are copied.
        :return: explain dataframe copy
        """
        return super().copy(deep)

    def drop(
            self,
            labels=None,
            axis: Union[str, int] = 0,
            index=None,
            columns=None,
            level: Level | None = None,
            inplace: bool = False,
            errors: str = "raise",
            update_operation: bool = True,
    ) -> ExpDataFrame | None:
        """
        Drop specified labels from rows or columns.
        Remove rows or columns by specifying label names and corresponding axis,
        or by specifying directly index or column names.
        When using a multi-index, labels on different levels can be removed by specifying the level.

        :param labels: Index or column labels to drop.
                       A tuple will be used as a single label and not treated as a list-like.
        :param axis: Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
        :param index: Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).
        :param columns: Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
        :param level: For MultiIndex, level from which the labels will be removed.
        :param inplace: If False, return a copy. Otherwise, do operation inplace and return None.
        :param errors: If ‘ignore’, suppress error and only existing labels are dropped.
        :param update_operation: Whether to update operations with the rename applied. This is for internal usage,
        to prevent recursive calls. Users should not set this to False when calling rename.

        :return: Explain DataFrame without the removed index or column labels or None if inplace=True.
        """
        if inplace:
            super(ExpDataFrame, self).drop(labels=labels, axis=axis, index=index, columns=columns, level=level,
                                           inplace=inplace, errors=errors)
            res = self
        else:
            res = super(ExpDataFrame, self).drop(labels=labels, axis=axis, index=index, columns=columns, level=level,
                                                 inplace=inplace, errors=errors)

        # Set the result dataframe of the operation to the new dataframe.
        if res.operation is not None:
            res.operation.result_df = res

        # If needed, update the operation, so it will use the new dataframe.
        if res.operation is not None and update_operation:
            # We copy the operation, as we don't want to change the original operation of the dataframe when not
            # doing an inplace operation.
            res.operation = cpy(self.operation)
            # Filter and GroupBy operations have a source_df field that needs to be updated.
            if hasattr(res.operation, 'source_df') and res.operation.source_df is not None:
                # update_operation is our own addition, so we can't pass it to the original rename method, if the
                # df is not an ExpDataFrame.
                if isinstance(res.operation.source_df, ExpDataFrame):
                    # We always set update_operation to False, as we don't want to update other dataframes' operations.
                    # We also always set inplace to False, as we don't want to change the original dataframe.
                    res.operation.source_df = res.operation.source_df.drop(labels=labels, axis=axis, index=index,
                                                                           columns=columns, level=level,
                                                                           inplace=False,
                                                                           errors=errors, update_operation=False)
                else:
                    res.operation.source_df = res.operation.source_df.drop(labels=labels, axis=axis, index=index,
                                                                           columns=columns, level=level,
                                                                           inplace=False,
                                                                           errors=errors)
            # Join operations have a left_df and right_df field that needs to be updated.
            elif hasattr(res.operation, 'left_df') and res.operation.left_df is not None:
                # The drop operation may cause a KeyError if the labels are not found in the dataframe.
                # In joins, we don't know if the labels are in the left or right dataframe, so we need to try both.
                try:
                    if isinstance(res.operation.left_df, ExpDataFrame):
                        res.operation.left_df = res.operation.left_df.drop(labels=labels, axis=axis, index=index,
                                                                           columns=columns, level=level,
                                                                           inplace=False,
                                                                           errors=errors, update_operation=False)
                    else:
                        res.operation.left_df = res.operation.left_df.drop(labels=labels, axis=axis, index=index,
                                                                           columns=columns, level=level,
                                                                           inplace=False,
                                                                           errors=errors)
                except KeyError:
                    pass
                try:
                    if isinstance(res.operation.right_df, ExpDataFrame):
                        res.operation.right_df = res.operation.right_df.drop(labels=labels, axis=axis, index=index,
                                                                             columns=columns, level=level,
                                                                             inplace=False,
                                                                             errors=errors, update_operation=False)
                    else:
                        res.operation.right_df = res.operation.right_df.drop(labels=labels, axis=axis, index=index,
                                                                             columns=columns, level=level,
                                                                             inplace=False,
                                                                             errors=errors)
                except KeyError:
                    pass

        return res if not inplace else None

    def rename(self,
               mapper: Renamer | None = None,
               *,
               index: Renamer | None = None,
               columns: Renamer | None = None,
               axis: Axis | None = None,
               copy: bool = True,
               inplace: bool = False,
               level: Level | None = None,
               errors: str = "ignore",
               update_operation: bool = True,
               ) -> ExpDataFrame | None:
        """
        Alter axes labels.
        Function / dict values must be unique (1-to-1). Labels not contained in a dict / Series will be left as-is.
        Extra labels listed don’t throw an error.

        :param mapper: Dict-like or function transformations to apply to that axis’ values.
                       Use either mapper and axis to specify the axis to target with mapper, or index and columns.
        :param index: Alternative to specifying axis (mapper, axis=0 is equivalent to index=mapper).
        :param columns: Alternative to specifying axis (mapper, axis=1 is equivalent to columns=mapper).
        :param axis: Axis to target with mapper. Can be either the axis name (‘index’, ‘columns’) or number (0, 1).
                        The default is ‘index’.
        :param copy: Also copy underlying data.
        :param inplace: Whether to return a new DataFrame. If True then value of copy is ignored.
        :param level: In case of a MultiIndex, only rename labels in the specified level.
        :param errors: If ‘raise’, raise a KeyError when a dict-like mapper, index,
                       or columns contains labels that are not present in the Index being transformed.
                       If ‘ignore’, existing keys will be renamed and extra keys will be ignored.
        :param update_operation: Whether to update operations with the rename applied. This is for internal usage,
        to prevent recursive calls. Users should not set this to False when calling rename.

        :return: Explain DataFrame with the renamed axis labels or None if inplace=True.
        """
        if inplace:
            super(ExpDataFrame, self).rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                             copy=copy, inplace=inplace, level=level, errors=errors)
            res = self
        else:
            res = super(ExpDataFrame, self).rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                                   copy=copy, inplace=inplace, level=level, errors=errors)

        if res.operation is not None:
            res.operation.result_df = res

        # Update the operation, if needed.
        if res.operation is not None and update_operation:

            # Update the columns of the operation, if the columns were renamed.
            res.operation = cpy(self.operation)
            # Filter and GroupBy operations have a source_df field that needs to be updated.
            if hasattr(res.operation, 'source_df') and res.operation.source_df is not None:
                # update_operation is our own addition, so we can't pass it to the original rename method, if the
                # df is not an ExpDataFrame.
                if isinstance(res.operation.source_df, ExpDataFrame):
                    # We always set update_operation to False, as we don't want to update other dataframes' operations.
                    # We also always set inplace to False, as we don't want to change the original dataframe.
                    res.operation.source_df = res.operation.source_df.rename(mapper=mapper, index=index,
                                                                             columns=columns, axis=axis,
                                                                             copy=copy, inplace=False, level=level,
                                                                             errors=errors, update_operation=False)
                else:
                    res.operation.source_df = res.operation.source_df.rename(mapper=mapper, index=index,
                                                                             columns=columns, axis=axis,
                                                                             copy=copy, inplace=False, level=level,
                                                                             errors=errors)
            # Join operations have a left_df and right_df field that needs to be updated.
            elif hasattr(res.operation, 'left_df') and res.operation.left_df is not None:
                if isinstance(res.operation.left_df, ExpDataFrame):
                    res.operation.left_df = res.operation.left_df.rename(mapper=mapper, index=index, columns=columns,
                                                                         axis=axis, copy=copy, inplace=False,
                                                                         level=level,
                                                                         errors=errors, update_operation=False)
                else:
                    res.operation.left_df = res.operation.left_df.rename(mapper=mapper, index=index, columns=columns,
                                                                         axis=axis, copy=copy, inplace=False,
                                                                         level=level,
                                                                         errors=errors)
                if isinstance(res.operation.right_df, ExpDataFrame):
                    res.operation.right_df = res.operation.right_df.rename(mapper=mapper, index=index, columns=columns,
                                                                           axis=axis, copy=copy, inplace=False,
                                                                           level=level,
                                                                           errors=errors, update_operation=False)
                else:
                    res.operation.right_df = res.operation.right_df.rename(mapper=mapper, index=index, columns=columns,
                                                                           axis=axis, copy=copy, inplace=False,
                                                                           level=level,
                                                                           errors=errors)

            # Filter and join operations have an attribute field that needs to be updated.
            if hasattr(res.operation, 'attribute'):

                if columns is not None and res.operation.attribute in columns:
                    res.operation.attribute = columns[res.operation.attribute]

                # In the case of a mapper, we only care about making the update to the operation if it affects the columns.
                elif mapper is not None and axis == 'columns':

                    # If the mapper is of the form {old_name: new_name}, we need to update the attribute name if it was
                    # renamed.
                    if hasattr(mapper, '__getitem__') and res.operation.attribute in mapper:
                        res.operation.attribute = mapper[res.operation.attribute]
                    # Otherwise, if the mapper is a function, we need to call the function on the attribute name.
                    elif callable(mapper):
                        res.operation.attribute = mapper(res.operation.attribute)

            # GroupBy operations have a group_attributes field that needs to be updated.
            elif hasattr(res.operation, 'group_attributes'):

                # Extract the group attributes, and figure out if any of them were renamed.
                group_attributes = res.operation.group_attributes if len(res.operation.group_attributes) > 1 else [
                    res.operation.group_attributes]
                shared_attributes = set(res.operation.group_attributes) & set(columns)

                # If the group attributes were renamed, we need to update the group_attributes field.
                if columns is not None and len(shared_attributes) > 0:
                    res.operation.group_attributes = [columns[attr] if attr in shared_attributes else attr for attr in
                                                      group_attributes]

                # If there is a mapper, we need to update attributes that were renamed.
                elif mapper is not None and axis == 'columns':
                    # If the mapper is of the form {old_name: new_name}, we need to update the attribute name if it was
                    # renamed.
                    if hasattr(mapper, '__getitem__'):
                        res.operation.group_attributes = [mapper[attr] if attr in mapper else attr for attr in
                                                          group_attributes]
                    # Otherwise, if the mapper is a function, we need to call the function on the attribute name.
                    elif callable(mapper):
                        res.operation.group_attributes = [mapper(attr) for attr in group_attributes]

        # Return the result. None if inplace=True, otherwise the new dataframe.
        return res if not inplace else None

    def sample(
            self,
            n=None,
            frac=None,
            replace=False,
            weights=None,
            random_state=None,
            axis=None,
            ignore_index=False,
    ):
        """
        Return a random sample of items from an axis of object.
        You can use random_state for reproducibility.

        :param n: Number of items from axis to return. Cannot be used with frac. Default = 1 if frac = None.
        :param frac: Fraction of axis items to return. Cannot be used with n.
        :param replace: Allow or disallow sampling of the same row more than once.
        :param weights: Default ‘None’ results in equal probability weighting.
                        If passed a Series, will align with target object on index.
                        Index values in weights not found in sampled object will be ignored and index values in sampled
                        object not in weights will be assigned weights of zero. If called on a DataFrame,
                        will accept the name of a column when axis = 0. Unless weights are a Series,
                        weights must be same length as axis being sampled.
                        If weights do not sum to 1, they will be normalized to sum to 1.
                        Missing values in the weights column will be treated as zero. Infinite values not allowed.
        :param random_state: If int, array-like, or BitGenerator, seed for random number generator.
                             If np.random.RandomState or np.random.Generator, use as given.
        :param axis: Axis to sample. Accepts axis number or name.
                     Default is stat axis for given data type (0 for Series and DataFrames).
        :param ignore_index: If True, the resulting index will be labeled 0, 1, …, n - 1.

        :return: A new object of same type as caller containing n items randomly sampled from the caller object.
        """
        return super().sample(n, frac, replace, weights, random_state, axis, ignore_index)

    def where(
            self,
            cond,
            other=np.nan,
            inplace=False,
            axis=None,
            level=None,
            errors="raise",
            try_cast=False,
    ):
        """
        Override Dataframe where, save filter operation on the result.
        :param cond: Where cond is True, keep the original value.
                     Where False, replace with corresponding value from other.
                     If cond is callable, it is computed on the Series/DataFrame and
                     should return boolean Series/DataFrame or array.
                     The callable must not change input Series/DataFrame (though pandas doesn’t check it).
        :param other: Entries where cond is False are replaced with corresponding value from other.
                      If other is callable, it is computed on the
                      Series/DataFrame and should return scalar or Series/DataFrame.
                      The callable must not change input Series/DataFrame (though pandas doesn’t check it).
        :param inplace: Alignment axis if needed.
        :param axis: Alignment axis if needed.
        :param level: Alignment level if needed.
        :param errors: Note that currently this parameter won’t affect the results and
                       will always coerce to a suitable dtype.
                       * ‘raise’ : allow exceptions to be raised.
                       * ‘ignore’ : suppress exceptions. On error return original object.
        :param try_cast: Try to cast the result back to the input type (if possible).

        :return: Same type as caller or None if inplace=True.
        """
        result_df = ExpDataFrame(super().where(cond))  # , other, inplace, axis, level, errors, try_cast)
        try:
            if self.filter_items:
                result_df.operation = Filter(source_df=self,
                                             source_scheme={},
                                             attribute=self.filter_items.pop(),
                                             result_df=result_df)

        except Exception as error:
            print(f'Error {error} with operation filter explanation')

        return result_df

    def groupby(
            self,
            by=None,
            level=None,
            as_index: bool = True,
            sort: bool = True,
            group_keys: bool = True,
            squeeze: bool = no_default,
            observed: bool = False,
            dropna: bool = True,
    ):
        """
        Override Dataframe groupby, use many to one columns as groupby filed, return explain groupby dataframe.

        :param by: Used to determine the groups for the groupby.
                   If by is a function, it’s called on each value of the object’s index.
                   If a dict or Series is passed, the Series or dict VALUES will be used to determine the groups
                   (the Series’ values are first aligned; see .align() method).
                   If a list or ndarray of length equal to the selected axis is passed (see the groupby user guide),
                   the values are used as-is to determine the groups.
                   A label or list of labels may be passed to group by the columns in self.
                   Notice that a tuple is interpreted as a (single) key.
        :param axis: Split along rows (0) or columns (1).
        :param level: If the axis is a MultiIndex (hierarchical), group by a particular level or levels.
        :param as_index: For aggregated output, return object with group labels as the index.
                         Only relevant for DataFrame input. as_index=False is effectively “SQL-style” grouped output.
        :param sort: Sort group keys. Get better performance by turning this off.
                     Note this does not influence the order of observations within each group.
                     Groupby preserves the order of rows within each group.
        :param group_keys: When calling apply, add group keys to index to identify pieces.
        :param squeeze: Reduce the dimensionality of the return type if possible, otherwise return a consistent type.
        :param observed: This only applies if any of the groupers are Categoricals.
                         If True: only show observed values for categorical groupers.
                         If False: show all values for categorical groupers.
        :param dropna: If True, and if group keys contain NA values, NA values together with row/column will be dropped.
                       If False, NA values will also be treated as the key in groups.

        :return:Explain DataFrameGroupBy object that contains information about the groups.
        """
        try:
            from pd_explain.core.explainable_group_by_dataframe import ExpDataFrameGroupBy
            # group_attributes = GroupBy.get_one_to_many_attributes(self, [by] if isinstance(by, str) else by)
            group_attributes = by
            tmp = pd.core.groupby.generic.DataFrameGroupBy
            pd.core.groupby.generic.DataFrameGroupBy = ExpDataFrameGroupBy
            g = super().groupby(by=group_attributes, level=level, as_index=as_index, sort=sort,
                                group_keys=group_keys
                                , observed=observed, dropna=dropna)
            g.group_attributes = by
            g.source_name = utils.get_calling_params_name(self)
            g.operation = GroupBy(source_df=self, group_attributes=by, result_df=g, source_scheme=None, agg_dict=None)
            g.original = super().groupby(by=by, level=level, as_index=as_index, sort=sort,
                                         group_keys=group_keys
                                         , observed=observed, dropna=dropna)

            pd.core.groupby.generic.DataFrameGroupBy = tmp
            return g

        except Exception as error:
            print(f'Error {error} with operation group by explanation')
            g = super().groupby(by=by, level=level, as_index=as_index, sort=sort, group_keys=group_keys
                                , observed=observed, dropna=dropna)
            # g.group_attributes = by
            # g.operation = GroupBy(source_df=self, group_attributes=by, result_df=g)
            return super().groupby(by=by, level=level, as_index=as_index, sort=sort, group_keys=group_keys
                                   , observed=observed, dropna=dropna)

    def _getitem_bool_array(self, key):
        """
        Get filtered dataframe
        :param key: item key
        :return: Explain dataframe with filter operation
        """
        result_df = ExpDataFrame(super()._getitem_bool_array(key))
        try:
            if self.filter_items:
                if isinstance(key, ExpSeries) and key.filter_query is not None:
                    op = key.filter_query['op']
                    other = key.filter_query['other']
                else:
                    op = None
                    other = None
                result_df.operation = Filter(source_df=self,
                                             source_scheme={},
                                             attribute=self.filter_items.pop(),
                                             result_df=result_df,
                                             operation_str=op,
                                             value=other)
        except Exception as error:
            print(f'Error {error} with operation filter explanation')

        return result_df

    def merge(
            self,
            right,
            how="inner",
            on=None,
            left_on=None,
            right_on=None,
            left_index=False,
            right_index=False,
            sort=False,
            suffixes=("_x", "_y"),
            copy=True,
            indicator=False,
            validate=None
    ):
        """

        :param right: Object to merge with.
        :param how: Type of merge to be performed.
                    * left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
                    * right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
                    * outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
                    * inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
                    * cross: creates the cartesian product from both frames, preserves the order of the left keys.
        :param on: Column or index level names to join on. These must be found in both DataFrames.
                   If on is None and not merging on indexes then this defaults to the intersection
                   of the columns in both DataFrames.
        :param left_on: Column or index level names to join on in the left DataFrame.
                        Can also be an array or list of arrays of the length of the left DataFrame.
                        These arrays are treated as if they are columns.
        :param right_on: Column or index level names to join on in the left DataFrame.
                        Can also be an array or list of arrays of the length of the right DataFrame.
                        These arrays are treated as if they are columns.
        :param left_index: Use the index from the left DataFrame as the join key(s).
                           If it is a MultiIndex, the number of keys in the other DataFrame
                           (either the index or a number of columns) must match the number of levels.
        :param right_index: Use the index from the right DataFrame as the join key. Same caveats as left_index.
        :param sort: Sort the join keys lexicographically in the result DataFrame.
                     If False, the order of the join keys depends on the join type (how keyword).
        :param suffixes: A length-2 sequence where each element is optionally a string indicating the suffix
                         to add to overlapping column names in left and right respectively.
                         Pass a value of None instead of a string to indicate that the column name from left or right
                          should be left as-is, with no suffix. At least one of the values must not be None.


        :param copy: If False, avoid copy if possible.
        :param indicator: If True, adds a column to the output DataFrame called “_merge” with information on the source
                          of each row. The column can be given a different name by providing a string argument.
                          The column will have a Categorical type with the value of “left_only” for observations whose
                          merge key only appears in the left DataFrame, “right_only” for observations whose merge key
                          only appears in the right DataFrame, and “both” if the observation’s merge key is found in
                          both DataFrames.
        :param validate: If specified, checks if merge is of specified type.
                         * “one_to_one” or “1:1”: check if merge keys are unique in both left and right datasets.
                         * “one_to_many” or “1:m”: check if merge keys are unique in left dataset.
                         * “many_to_one” or “m:1”: check if merge keys are unique in right dataset.
                         * “many_to_many” or “m:m”: allowed, but does not result in checks.
        :return: A Explain DataFrame of the two merged objects with join operation filed.
        """
        try:
            left_name = utils.get_calling_params_name(self)
            right_name = utils.get_calling_params_name(right)
            self = self.reset_index()
            right_df = right.copy()
            ignore_columns = [attribute for attribute in on] if on is not None else []
            ignore_columns.append('index')
            self.columns = [col if col in ignore_columns else left_name + "_" + col
                            for col in self]
            right_df.columns = [col if col in ignore_columns else right_name + "_" + col
                                for col in right_df]
            result_df = ExpDataFrame(super().merge(right_df, how, on, left_on, right_on, left_index, right_index,
                                                   sort, suffixes, copy, indicator, validate))
            result_df.operation = Join(self, right_df, None, on, result_df, left_name, right_name)
            return result_df

        except Exception as error:
            print(f'Error {error} with operation merge explanation')
            return super().merge(right, how, on, left_on, right_on, left_index,
                                 right_index, sort, suffixes, copy, indicator, validate)

    def join(
            self,
            other: ExpDataFrame | ExpSeries,
            on: IndexLabel | None = None,
            left_on=None,
            right_on=None,
            how: str = "inner",
            lsuffix: str = "",
            rsuffix: str = "",
            sort: bool = False,
    ) -> ExpDataFrame:
        """

        :param other: Index should be similar to one of the columns in this one. If a Series is passed,
                      its name attribute must be set, and that will be used as the column name in the
                      resulting joined DataFrame.
        :param on: Column or index level name(s) in the caller to join on the index in other,
                   otherwise joins index-on-index. If multiple values given, the other DataFrame must have a MultiIndex.
                    Can pass an array as the join key if it is not already contained in the calling DataFrame.
                    Like an Excel VLOOKUP operation.
        :param how: How to handle the operation of the two objects.
                    * left: use calling frame’s index (or column if on is specified)
                    * right: use other’s index.
                    * outer: form union of calling frame’s index (or column if on is specified) with other’s index,
                             and sort it. lexicographically.
                    * inner: form intersection of calling frame’s index (or column if on is specified) with other’s
                             index, preserving the order of the calling’s one.
                    * cross: creates the cartesian product from both frames, preserves the order of the left keys.
        :param lsuffix: Suffix to use from left frame’s overlapping columns.
        :param rsuffix: Suffix to use from right frame’s overlapping columns.
        :param sort: Order result DataFrame lexicographically by the join key.
                     If False, the order of the join key depends on the join type (how keyword).

        :return: A Explain DataFrame of the two merged objects with join operation filed.
        """
        try:
            # If no on is specified, we raise a warning to let the user know that the operation and explanation may not
            # work as expected.
            if on is None:
                warnings.warn(
                    "No 'on' parameter specified in join operation. The operation and explanation may not work as expected.")

            left_name = utils.get_calling_params_name(self)
            right_name = utils.get_calling_params_name(other)
            self = self.reset_index()
            self.df_name = left_name
            other.df_name = right_name
            right_df = ExpDataFrame(other.copy())
            right_df.df_name = right_name

            result_df = ExpDataFrame(pd.merge(self, right_df, on=on, left_on=left_on,
                                              right_on=right_on, how=how, suffixes=(lsuffix, rsuffix), sort=sort))

            # Check if the resulting df is empty. If it is, we raise a warning to let the user know.
            if result_df.empty:
                warnings.warn("The resulting dataframe is empty. Check the join operation to ensure it is correct.")

            # This is a complete hack to fix the issue: applying suffixes to the columns of the resulting dataframe
            # causes the explanation to fail, since it can no longer match up the columns of the resulting dataframe
            # with the columns of the original dataframes. This is a temporary fix until a better solution is found.
            # We simply apply the suffixes to the original dataframes, before passing them to the Join operation.
            left_cols = [col for col in self.columns if col not in ['index', on]]
            right_cols = [col for col in right_df.columns if col not in ['index', on]]
            coinciding_cols = set(left_cols) & set(right_cols)
            left_df = self.copy()
            right_df = right_df.copy()

            left_df = left_df.rename(columns={col: col + lsuffix for col in coinciding_cols}, inplace=False,
                                     update_operation=False)
            right_df = right_df.rename(columns={col: col + rsuffix for col in coinciding_cols}, inplace=False,
                                       update_operation=False)

            result_df.operation = Join(left_df, right_df, None, on, result_df, left_name, right_name)

            return result_df

        except Exception as error:
            print(f'Error {error} with operation merge explanation')
            return ExpDataFrame(pd.merge(self, right_df, on=on, how=how))

            # return ExpDataFrame(super().join(other, on, how, lsuffix, rsuffix, sort))

    def b_join(
            self,
            other: ExpDataFrame | ExpSeries,
            on: IndexLabel | None = None,
            how: str = "left",
            lsuffix: str = "",
            rsuffix: str = "",
            sort: bool = False,
            explain=False,
            consider='left',
            top_k=1
    ):
        """

        :param other: Index should be similar to one of the columns in this one. If a Series is passed,
                      its name attribute must be set, and that will be used as the column name in the
                      resulting joined DataFrame.
        :param on: Column or index level name(s) in the caller to join on the index in other,
                   otherwise joins index-on-index. If multiple values given, the other DataFrame must have a MultiIndex.
                    Can pass an array as the join key if it is not already contained in the calling DataFrame.
                    Like an Excel VLOOKUP operation.
        :param how: How to handle the operation of the two objects.
                    * left: use calling frame’s index (or column if on is specified)
                    * right: use other’s index.
                    * outer: form union of calling frame’s index (or column if on is specified) with other’s index,
                             and sort it. lexicographically.
                    * inner: form intersection of calling frame’s index (or column if on is specified) with other’s
                             index, preserving the order of the calling’s one.
                    * cross: creates the cartesian product from both frames, preserves the order of the left keys.
        :param lsuffix: Suffix to use from left frame’s overlapping columns.
        :param rsuffix: Suffix to use from right frame’s overlapping columns.
        :param sort: Order result DataFrame lexicographically by the join key.
                     If False, the order of the join key depends on the join type (how keyword).

        :return: A Explain DataFrame of the two merged objects with join operation filed.
        """
        try:
            left_name = utils.get_calling_params_name(self)
            right_name = utils.get_calling_params_name(other)
            self = self.reset_index()
            self.df_name = left_name
            other.df_name = right_name
            right_df = ExpDataFrame(other.copy())
            right_df.df_name = right_name
            # ignore_columns = [attribute for attribute in on] if on is not None else []
            # ignore_columns.append('index')

            # self.columns = [col if col in ignore_columns else left_name + "_" + col
            #                 for col in self]
            # right_df.columns = [col if col in ignore_columns else right_name + "_" + col
            #                     for col in right_df]
            operation = BJoin(self, right_df, None, on, None, left_name, right_name)
            # result_df.operation = BJoin(self, right_df, None, on, result_df, left_name, right_name)
            if (explain):
                operation.explain(consider=consider, top_k=top_k)
            return operation.result

        except Exception as error:
            print(f'Error {error} with operation merge explanation')
            return ExpDataFrame(super().join(other, on, how, lsuffix, rsuffix, sort))

    def reset_index(
            self,
            level: Hashable | Sequence[Hashable] | None = None,
            drop: bool = False,
            inplace: bool = False,
            col_level: Hashable = 0,
            col_fill: Hashable = "",
    ) -> ExpDataFrame | None:
        """
        Reset the index, or a level of it.
        Reset the index of the DataFrame, and use the default one instead.
        If the DataFrame has a MultiIndex, this method can remove one or more levels.

        :param level: Only remove the given levels from the index. Removes all levels by default.
        :param drop: Do not try to insert index into dataframe columns.
                     This resets the index to the default integer index.
        :param inplace: Modify the DataFrame in place (do not create a new object).
        :param col_level: If the columns have multiple levels, determines which level the labels are inserted into.
                          By default it is inserted into the first level.
        :param col_fill: If the columns have multiple levels, determines how the other levels are named.
                        If None then the index name is repeated.
        :return: Explain DataFrame with the new index or None if inplace=True.
        """
        return super().reset_index(drop=drop, inplace=inplace, level=level, col_level=col_level, col_fill=col_fill)

    def drop_duplicates(
            self,
            subset: Hashable | Sequence[Hashable] | None = None,
            *,
            keep: DropKeep = "first",
            inplace: bool = False,
            ignore_index: bool = False,
    ) -> ExpDataFrame | None:
        # Drop duplicates does not interact well with __get_item__ in ExpDataFrame. So, we cast it back to a normal
        # DataFrame, drop the duplicates, and then cast it back to an ExpDataFrame.
        res_df = DataFrame(self)
        res_df = res_df.drop_duplicates(subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index)
        res_df = ExpDataFrame(res_df)
        res_df.operation = self.operation
        res_df.explanation = self.explanation
        res_df.filter_items = self.filter_items
        return res_df

    def __repr__(self):
        """
        repr object
        :return: super repr
        """
        return super().__repr__()

    def present_deleted_correlated(self, figs_in_row: int = 2):  #####
        return self.operation.present_deleted_correlated(figs_in_row=figs_in_row)

    def explain(self, schema: dict = None, attributes: List = None, use_sampling: bool | None = None,
                sample_size: int | float = 5000, top_k: int = None,
                explainer: Literal['fedex', 'outlier', 'many_to_one', 'shapley', 'metainsight'] = 'fedex',
                target=None, dir=None,
                figs_in_row: int = 2, show_scores: bool = False, title: str = None, corr_TH: float = 0.7,
                consider='right', value=None, attr=None, ignore=None,
                labels=None, coverage_threshold: float = 0.7, max_explanation_length: int = 3,
                separation_threshold: float = 0.3, p_value: int = 1,
                explanation_form: Literal['conj', 'disj', 'conjunction', 'disjunction'] = 'conj',
                prune_if_too_many_labels: bool = True, max_labels: int = 10, pruning_method='largest',
                bin_numeric: bool = False, num_bins: int = 10, binning_method: str = 'quantile',
                label_name: str = 'label', explain_errors=True,
                error_explanation_threshold: float = 0.05, debug_mode: bool = False,
                add_llm_explanation_reasoning: bool = False,
                min_commonness: float = 0.5, no_exception_penalty_weight=0.1,
                balance_factor: float = 1, filter_columns: List[str] | str = None,
                aggregations: List[Tuple[str, str]] = None, groupby_columns: List[List[str]] | List[str] = None,
                correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                max_filter_columns: int = 3, max_aggregation_columns: int = 3,
                allow_multiple_aggregations: bool = False, allow_multiple_groupbys: bool = False,
                use_all_groupby_combinations: bool = False,
                do_not_visualize: bool = False,
                log_query: bool = False,
                display_mode: Literal['grid', 'carousel'] = 'grid',
                beautify: bool = False, beautify_max_fix_attempts: int = 10, silent_beautify: bool = True,
                max_labels_per_plot: int = 8,
                max_common_categories_per_plot: int = 3,
                ):
        """
        Generate an explanation for the dataframe, using the selected explainer and based on the last operation performed.

        :param explainer: The explainer to use. Currently supported: 'fedex', 'many to one', 'shapley', 'outlier', 'metainsight'. Note
        that 'outlier' is only supported for series, not for dataframes. Please note that the metainsight explainer is still
        in beta, and may not work as expected. Defaults to 'fedex'.
        :param attributes: All explainers. Which columns to consider in the explanation.
        :param use_sampling: All explainers. Whether or not to use sampling when generating an explanation. This can massively speed up
        the explanation generation process, but may result in less accurate explanations. We use sampling methods that
        we have empirically tested to only minimally affect the accuracy of the explanations. Defaults to None, in which
        case the value set in the global configuration is used (which defaults to True).
        :param sample_size: All explainers. The number of samples to use when use_sampling is True. Can be either an integer or a float.
        If it is an integer, that number of samples will be used. If it is a float, it will be interpreted as a percentage
        of the total number of samples. Defaults to 5000, which is also the minimum value.
        :param schema: Fedex explainer. Result columns, can change columns name and ignored columns.
        :param top_k: Fedex explainer. Number of explanations.
        :param figs_in_row: Fedex and MetaInsight explainers. Number of explanations figs in one row.
        :param show_scores: Fedex explainer. show scores on explanation.
        :param title: Fedex / outlier / shapley explainers. explanation title.
        :param corr_TH: Fedex explainer. Correlation threshold, above this threshold the columns are considered correlated.
        :param log_query: Fedex explainer. If true, the query that produced the explanation will be logged.
        :param target: Outlier explainer. Target value for the outlier explainer
        :param dir: Outlier explainer. Direction for the outlier explainer. Can be either 'high' or 'low'.
        :param consider: Fedex explainer. Which side of a join to consider for the explanation. Can be either 'left' or 'right'.
        :param labels: Many to one explainer. Cluster / group labels. Can either be a series or a column name.
        If a column name is provided, the column must be present in the dataframe.
        If you wish to explain the groups of a groupby operation, leave this parameter as None while calling explain on the
        groupy result. The labels will be automatically extracted from the groupby operation.
        :param coverage_threshold: Many to one explainer. Minimum coverage threshold. Coverage is
        defined as the % of the data in the group that is explained by the explanation. Defaults to 0.7.
        :param max_explanation_length: Many to one explainer. Maximum explanation length permitted. Defaults to 3.
        :param separation_threshold: Many to one explainer. Maximum separation threshold. Separation error is defined as the
        % of the data in groups other than the one being explained that is explained by the explanation. Defaults to 0.3.
        :param p_value: Many to one explainer. A scaling factor for the maximum number of attributes that will be considered
        as candidates for the explanation. n_attr = max_explanation_length * p_value. Setting this to a higher value may
        result in a more accurate explanation, but will also increase the computation time. Defaults to 1.
        :param explanation_form: Many to one explainer. The form of the explanation. Can be either 'conj' or 'disj', for
        conjunction and disjunction respectively. Defaults to 'conj'.
        :param prune_if_too_many_labels: Many to one explainer. If True, the labels will be pruned if there are too many
        labels to consider. Defaults to True.
        :param max_labels: Many to one explainer. The maximum number of labels permitted. Above this number, the labels
        will be pruned if prune_if_too_many_labels is True. Defaults to 10.
        :param pruning_method: Many to one explainer. The method to use when selecting which labels to prune. Cab be
        'largest' - where the k labels with the most values are kept, 'smallest', 'random', 'max_dist' - where the k labels
        with the highest distance between their means * group size are kept, 'min_dist', 'max_silhouette' - where the k groups with the
        highest silhouette score * group size are kept, or 'min_silhouette'. Defaults to 'largest'.
        :param bin_numeric: Many to one explainer. Whether or not to bin numeric labels, if there are more labels than
        the specified number of bins. Defaults to False.
        :param num_bins: Many to one explainer. The number of bins to use when binning numeric labels. Defaults to 10.
        :param binning_method: The method to use when binning numeric labels. Can be either 'quantile' or 'uniform'.
        :param label_name: Many to one explainer. How to call the labels column in the explanation, if binning was used
        and the labels column did not have a name. Defaults to 'label'.
        :param explain_errors: Many to one explainer. Whether or not to explain where the separation error originates from
        for each explanation. Defaults to True.
        :param error_explanation_threshold: Many to one explainer. The threshold for how much a group needs to contribute
        to the separation error to be included in the explanation. Groups that contribute less than this threshold will
        be aggregated into a single group. Defaults to 0.05.
        :param add_llm_explanation_reasoning: All explainers. Enables using a LLM to generate additional context explanations, explaining why
        the explanations found occur. Defaults to False. Requires setting an API key. See the documentation for more information.
        Note that setting this to True will increase the computation time by a potentially large amount, entirely dependent on the LLM API response time.
        Also note that the output of the LLM is not guaranteed to be accurate, and may contain errors, so use with caution.
        :param debug_mode: Developer option. Disables multiprocessing and enables debug prints. Defaults to False.
        :param min_commonness: MetaInsight explainer. Patterns must encompass at-least this percentage of the values to
        be considered a common pattern. Defaults to 0.5.
        :param no_exception_penalty_weight: MetaInsight explainer. The weight given to the penalty in the case no exceptions
        are found to a common pattern. Defaults to 0.1. Higher values will give more priority to common patterns with exceptions.
        :param balance_factor: MetaInsight explainer. The weight given to exceptions when computing the score of a common pattern.
        Defaults to 1 - same ratio for both common patterns and exceptions. Higher values will give more priority to common patterns without exceptions.
        :param filter_columns: MetaInsight explainer. The columns to filter on when mining for common patterns.
        :param max_filter_columns: MetaInsight explainer. The maximum number of filter columns to use when automatically
        selecting the filter columns. Defaults to 3.
        :param aggregations: MetaInsight explainer. The aggregations to use when mining for common patterns.
        :param max_aggregation_columns: MetaInsight explainer. The maximum number of aggregation columns to use when automatically
        selecting the aggregation columns. Defaults to 3.
        :param groupby_columns: MetaInsight explainer. The columns to group by when mining for common patterns. If not provided,
        will be inferred automatically from the filter / aggregation columns.
        :correlation_aggregation_method: MetaInsight explainer. When auto-selecting groupby / filter / aggregation columns,
        a correlation based method is used to determine the best method to use. This parameter determines which aggregation
        function is used to aggregate the computed correlation in the case of multiple columns. Can be either 'avg', 'max' or 'sum'.
        :param allow_multiple_aggregations: MetaInsight explainer. Whether or not to allow multiple aggregations to be used
        in the same pattern. Defaults to False. May result in more complex and less interpretable patterns if set to True.
        :param allow_multiple_groupbys: MetaInsight explainer. Whether or not to allow multiple groupbys to be used in the same pattern.
        Defaults to False. May result in more complex and less interpretable patterns, possibly with multiple
        (almost or completely) disjoint indexes if set to True.
        :param use_all_groupby_combinations:MetaInsight explainer. When automatically inferring on a result of a groupby operation, whether to
        use all combinations of the groupby columns or just the provided ones. For example, if set to True and the groupby columns are ['A', 'B'],
        the groupby columns will be [['A'], ['B'], ['A', 'B']]. If set to False, only the provided groupby columns will be used.
        :param do_not_visualize: If True, the explanation will not be visualized (if the explainer supports disabling visualization).
        :param display_mode: Fedex explainer and MetaInsight explainer. How to visualize the multiple figures returned by
        the explainer. Can be either 'grid' for a regular plot in a grid format displaying all plots at once,
         or 'carousel' for a carousel that shows one figure at a time with navigation buttons.
        :param beautify: MetaInsight and Fedex explainers. If True, we will attempt to beautify the explanation by having a LLM generate code for producing
        a more visually appealing explanation for this specific case. Defaults to False. Please note that:
        1. This will increase the computation time by a potentially large amount, entirely dependent on the LLM API response time.
        2. The output of the LLM is not guaranteed to be accurate, and may contain errors, so use with caution.
        Please also note that this feature is still in beta, and may not work as expected.
        :param beautify_max_fix_attempts: MetaInsight and Fedex explainers. The maximum number of attempts to fix the
        returned code from the LLM to make it work or improve the visualization,, if the beautify parameter is set to True. Defaults to 10.
        :param silent_beautify: MetaInsight and Fedex explainers. If True, the beautify process will not print any information
        about its progress, and will only return the final result. Defaults to False.
        :param max_labels_per_plot: MetaInsight explainer. The maximum number of labels to display per plot. If there are more labels, they will be truncated.
        There may be more labels than this number in the final plot if there are more than this number of indexes with
        highlights in them.
        :param max_common_categories_per_plot: MetaInsight explainer. The maximum number of common categories to display per plot. If there are more categories,
        they will be grouped together and their average value will be displayed.

        :return: A visualization of the explanation, if possible. Otherwise, the raw explanation.
        """

        # Ensure that the user does not get a non-informative error message if they try to use the outlier explainer.
        # Without this line, the user gets an AttributeError 'str' object has no attribute 'items',
        # which provides no information to the user.
        if str.lower(explainer) == 'outlier':
            raise ValueError("Outlier explainer is not supported for multi-attribute dataframes, only for series.")

        use_sampling = use_sampling if use_sampling is not None else get_use_sampling_value()

        if str.lower(explainer) == "metainsight":
            warnings.warn("The MetaInsight explainer is still in beta, and may not work as expected. "
                          "Please use with caution. ")
        if beautify:
            warnings.warn("The beautify feature is still in beta, and may not work as expected. "
                          "It may also take a long time to run, depending on the LLM API response time. "
                          "Please use with caution.")

        factory = ExplainerFactory()
        explainer = factory.create_explainer(explainer=explainer, operation=self.operation,
                                             schema=schema, attributes=attributes, top_k=top_k, figs_in_row=figs_in_row,
                                             show_scores=show_scores, title=title, corr_TH=corr_TH,
                                             consider=consider, cont=value, attr=attr, ignore=ignore,
                                             labels=labels, coverage_threshold=coverage_threshold,
                                             max_explanation_length=max_explanation_length,
                                             separation_threshold=separation_threshold, p_value=p_value,
                                             target=target, dir=dir,
                                             source_df=self, explanation_form=explanation_form,
                                             prune_if_too_many_labels=prune_if_too_many_labels, max_labels=max_labels,
                                             pruning_method=pruning_method,
                                             bin_numeric=bin_numeric, num_bins=num_bins, binning_method=binning_method,
                                             label_name=label_name,
                                             use_sampling=use_sampling, sample_size=sample_size,
                                             explain_errors=explain_errors,
                                             error_explanation_threshold=error_explanation_threshold,
                                             debug_mode=debug_mode,
                                             add_llm_context_explanations=add_llm_explanation_reasoning,
                                             min_commonness=min_commonness,
                                             no_exception_penalty_weight=no_exception_penalty_weight,
                                             balance_factor=balance_factor, filter_columns=filter_columns,
                                             aggregations=aggregations, groupby_columns=groupby_columns,
                                             correlation_aggregation_method=correlation_aggregation_method,
                                             max_filter_columns=max_filter_columns,
                                             max_aggregation_columns=max_aggregation_columns,
                                             allow_multiple_aggregations=allow_multiple_aggregations,
                                             allow_multiple_groupbys=allow_multiple_groupbys,
                                             use_all_groupby_combinations=use_all_groupby_combinations,
                                             do_not_visualize=do_not_visualize,
                                             log_query=log_query,
                                             display_mode=display_mode, beautify=beautify,
                                             beautify_max_fix_attempts=beautify_max_fix_attempts,
                                             silent_beautify=silent_beautify,
                                             max_labels_per_plot=max_labels_per_plot,
                                             max_common_categories_per_plot=max_common_categories_per_plot
                                             )
        self.last_used_explainer = explainer
        explanation = explainer.generate_explanation()

        if explainer.can_visualize():
            return explainer.visualize()

        return explanation
