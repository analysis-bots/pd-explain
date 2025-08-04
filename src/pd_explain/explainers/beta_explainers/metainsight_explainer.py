"""
Note: this feature is currently in beta.
To whoever takes up the project next, please note that this code may still contain bugs and is not fully optimized.
There are also still some issues with the visualizations, which are not yet satisfactory.
"""

import math
from collections import defaultdict
import numpy as np
from scipy import stats
import itertools
import warnings

import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display

from pd_explain.explainers.explainer_interface import ExplainerInterface
from external_explainers.metainsight_explainer.metainsight_mining import MetaInsightMiner, MetaInsight
from fedex_generator.Operations.Operation import Operation
from fedex_generator.Operations.Filter import Filter
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Join import Join
from fedex_generator.commons.utils import get_calling_params_name
from pd_explain.visualizer_adaptations.carousel_adapter import CarouselAdapter
from pd_explain.llm_integrations.beta_integrations.visualization_beautifier import VisualizationBeautifier
from pd_explain.llm_integrations.explanation_reasoning import ExplanationReasoning

from typing import List, Tuple, Literal

RANDOM_SEED = 42


class MetaInsightExplainer(ExplainerInterface):
    """
    An implementation of MetaInsight from the paper MetaInsight: Automatic Discovery of Structured Knowledge for
    Exploratory Data Analysis by Ma et al. (2021).
    """

    def __len__(self):
        if self.metainsights is None:
            return 0
        return len(self.metainsights)

    def __init__(self, source_df, top_k=4, min_commonness: float = 0.5,
                 no_exception_penalty_weight=0.1, balance_factor: float = 1,
                 filter_columns: List[str] | str = None, aggregations: List[Tuple[str, str]] = None,
                 groupby_columns: List[List[str]] | List[str] = None,
                 operation: Operation = None, correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                 use_sampling: bool = True, sample_size: int | float = 5000,
                 max_filter_columns: int = 3, max_aggregation_columns: int = 3,
                 allow_multiple_aggregations: bool = False,
                 allow_multiple_groupbys: bool = False, num_bins: int = 10,
                 use_all_groupby_combinations: bool = False,
                 do_not_visualize: bool = False,
                 display_mode: Literal['carousel', 'grid'] = 'grid',
                 beautify: bool = False,
                 beautify_max_fix_attempts: int = 3,
                 add_llm_context_explanations: bool = False,
                 silent_beautify: bool = False,
                 return_beautify_code: bool = False,
                 generalize_beautify_code: bool = False,
                 figs_in_row: int = 2,
                 max_labels_per_plot: int = 8,
                 max_common_categories_per_plot: int = 3,
                 *args, **kwargs):
        """
        Initialize the MetaInsightExplainer with the provided arguments.
        :param source_df: The source dataframe to explain.
        :param top_k: The number of top MetaInsights to return.
        :param min_commonness: The minimum proportion of values that a pattern must cover to be considered common.
        :param no_exception_penalty_weight: The weight to apply to the actionability regularizer. This penalty lowers the
        score of patterns that have no exceptions, making them less likely to be selected.
        :param balance_factor: The weight to apply to exceptions, compared to the common patterns, when computing the score
        of a MetaInsight. A higher value means causes MetaInsights with many exceptions to have a lower score.
        :param filter_columns: The columns to filter on. If None, the filter columns will be inferred from the operation.
        :param aggregations: The aggregations to use. If None, the aggregations will be inferred from the operation.
        :param groupby_columns: The columns to group by. If None, the groupby columns will be inferred from the operation.
        :param operation: The operation to use to infer the filter and groupby columns. If None, the filter_columns and
        groupby_columns must be provided.
        :param correlation_aggregation_method: The method to use for aggregating the correlation values. Can be 'avg', 'max' or 'sum'.
        These values are used to determine the most correlated columns to the filter and groupby columns.
        :param use_sampling: Whether to use sampling to speed up the explanation generation. If True, the source_df will be sampled.
        :param sample_size: The size of the sample to take. Can be an integer for an absolute amount or a float between 0 and 1 for a relative amount.
        :param max_filter_columns: The maximum number of filter columns to use when automatically inferring the filter columns.
        :param max_aggregation_columns: The maximum number of aggregation columns to use when automatically inferring the aggregations.
        :param allow_multiple_aggregations: Whether to allow multiple aggregations in the same MetaInsight. Can cause the MetaInsight to be more complex
        and harder to interpret, but can also lead to more interesting insights.
        :param allow_multiple_groupbys: Whether to allow multiple groupbys in the same MetaInsight. Can cause the MetaInsight to be more complex
        and harder to interpret, but can also lead to more interesting insights.
        :param num_bins: The number of bins to use when a filter column is numeric. This is used to discretize the numeric columns.
        :param use_all_groupby_combinations: When automatically inferring on a result of a groupby operation, whether to
        use all combinations of the groupby columns or just the provided ones. For example, if set to True and the groupby columns are ['A', 'B'],
        the groupby columns will be [['A'], ['B'], ['A', 'B']]. If set to False, only the provided groupby columns will be used.
        :param do_not_visualize: If True, the visualizations will not be generated. This is useful for when the explainer
        is used in a pipeline and the visualizations are not needed.
        :param beautify: If True, use a LLM to create new visualizations for the explanations, which should look (maybe) better
        and be easier to understand compared to the templates used. Defaults to False.
        :param beautify_max_fix_attempts: The maximum number of attempts to fix the code returned by the LLM beautifier.
        :param add_llm_context_explanations: Whether to add LLM context explanations to the explanation. Defaults to False.
        :param figs_in_row: How many figures to display in a row when visualizing the explanations.
        :param max_labels_per_plot: The maximum number of labels to display per plot. If there are more labels, they will be truncated.
        There may be more labels than this number in the final plot if there are more than this number of indexes with
        highlights in them.
        :param max_common_categories_per_plot: The maximum number of common categories to display per plot. If there are more categories,
        they will be grouped together and their average value will be displayed.
        """
        self.metainsights = None
        self.source_df = pd.DataFrame(source_df)
        if top_k is None:
            top_k = 4
        self.top_k = top_k
        self.min_commonness = min_commonness
        self.actionability_regulizer = no_exception_penalty_weight
        self.balance_factor = balance_factor
        self.use_sampling = use_sampling
        self.sample_size = sample_size
        self.max_filter_columns = max_filter_columns
        self.max_aggregation_columns = max_aggregation_columns
        self.aggregations = None
        self.filter_columns = None
        self.groupby_columns = None
        self.allow_multiple_aggregations = allow_multiple_aggregations
        self.allow_multiple_groupbys = allow_multiple_groupbys
        self.n_bins = num_bins
        self.use_all_groupby_combinations = use_all_groupby_combinations
        self._do_not_visualize = do_not_visualize
        self.beautify = beautify
        self.beautify_max_fix_attempts = beautify_max_fix_attempts
        self.silent_beautify = silent_beautify
        self.return_beautify_code = return_beautify_code
        self.generalize_beautify_code = generalize_beautify_code
        self._do_not_visualize_beautify = False
        self.add_llm_context_explanations = add_llm_context_explanations
        self._source_name = get_calling_params_name(source_df)
        self.figs_per_row = figs_in_row
        if display_mode not in ['carousel', 'grid']:
            warnings.warn(f"Display mode {display_mode} is not supported. Defaulting to 'grid'.")
            display_mode = 'grid'
        self._display_mode = display_mode
        self.max_labels_per_plot = max_labels_per_plot
        self.max_common_categories_per_plot = max_common_categories_per_plot

        if self.source_df is None:
            raise ValueError("Source dataframe cannot be None")

        if self.source_df.empty:
            raise ValueError("Source dataframe cannot be empty")

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("k must be a positive integer")

        if 1 < self.min_commonness <= 0:
            raise ValueError("min_commonness must be in the range (0, 1]")

        self.can_run_visualize = False

        handle_filter_or_join_or_provided_filters = False
        handle_groupby = False

        # If filter_columns is None, we try to draw them from the operation, if possible.
        if not filter_columns and operation is not None:
            if isinstance(operation, Filter) or isinstance(operation, Join):
                self.filter_columns = operation.attribute
                self.groupby_columns = None
                handle_filter_or_join_or_provided_filters = True
            elif isinstance(operation, GroupBy) and not groupby_columns:
                self.groupby_columns = operation.group_attributes
                # We want to convert the groupby columns to a list of lists, so that we can handle multiple groupby columns.
                if isinstance(self.groupby_columns, str):
                    self.groupby_columns = [self.groupby_columns]
                if isinstance(self.groupby_columns, list) and any(
                        isinstance(group, str) for group in self.groupby_columns):
                    self.groupby_columns = [self.groupby_columns]
                self.source_df = pd.DataFrame(operation.source_df)
                self.filter_columns = []
                handle_groupby = True
            else:
                raise NotImplementedError(
                    "Unknown operation. The developers added an operation to Fedex_Generator and forgot to update here.")
            if isinstance(self.filter_columns, str):
                self.filter_columns = [self.filter_columns]

        # Otherwise, use the user provided filter_columns
        elif filter_columns is not None:
            handle_filter_or_join_or_provided_filters = True
            if isinstance(filter_columns, str):
                self.filter_columns = [filter_columns]
            else:
                self.filter_columns = filter_columns

        else:
            if groupby_columns and aggregations:
                handle_groupby = True
            else:
                raise ValueError(
                    "If no filter_columns / groupby_columns + aggregations are provided, an operation must have been performed prior to using the MetaInsight explainer.")

        if handle_filter_or_join_or_provided_filters:
            self._prepare_case_filter_join(
                groupby_columns=groupby_columns,
                correlation_aggregation_method=correlation_aggregation_method,
                aggregations=aggregations,
                operation=operation
            )
        elif handle_groupby:
            self._prepare_case_groupby(
                correlation_aggregation_method=correlation_aggregation_method,
                operation=operation
            )

        # If the operation is available, store a string representing it and its type for later use.
        if operation is not None:
            self._query, self._query_type = self._create_query_string(operation)
        else:
            self._query, self._query_type = None, None

        if self.aggregations is None and not aggregations:
            raise ValueError("No aggregations provided, and no viable aggregation options were found automatically.")
        if self.groupby_columns is None:
            self.groupby_columns = self.filter_columns
        if isinstance(self.groupby_columns, list):
            if any(isinstance(group, str) for group in self.groupby_columns):
                self.groupby_columns = [[group] if isinstance(group, str) else group for group in self.groupby_columns]

    def _prepare_case_groupby(self,
                              operation: GroupBy,
                              correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                              aggregations: List[Tuple[str, str]] = None
                              ) -> None:
        """
        Prepare the filter columsn, aggregations and groupby columns in the case we are automatically
        inferring from a groupby operation.

        :param correlation_aggregation_method: The method to use for aggregating the correlation values.
        :param operation: The GroupBy object to use.
        :param aggregations: The aggregations to use. If None, we will use the most correlated columns to the groupby columns.
        """
        # Check if groupby_columns are in the source_df
        groupby_columns_set = set()
        for group in self.groupby_columns:
            if isinstance(group, str):
                groupby_columns_set.add(group)
            else:
                groupby_columns_set.update(group)
        if not all(col in self.source_df.columns for col in groupby_columns_set):
            raise ValueError("All groupby_columns must be present in the source dataframe")

        correlated_cols, numerical_cols = self._find_correlated_columns_multi(groupby_columns_set,
                                                                              method=correlation_aggregation_method)

        len_filters = len(self.filter_columns)
        # If there are less than k target columns, we add the most correlated columns to the target_columns
        if len_filters < self.max_filter_columns:
            difference = self.max_filter_columns - len_filters
            self.filter_columns += [col for col in correlated_cols[:difference]]

        # If the user provided a list of aggregations, we use them. Otherwise, we automatically infer them
        # from the operation.
        if aggregations:
            self.aggregations = aggregations
        else:
            agg_dict = operation.agg_dict
            self.aggregations = []
            if "All" in agg_dict:
                aggregations_on_all = [agg_method for agg_method in agg_dict["All"]]
            else:
                aggregations_on_all = []
            # Remove any "All" columns from the agg_dict, because they are not valid aggregations.
            agg_dict = {k: v for k, v in agg_dict.items() if k not in ["All", "all"]}
            # If there are aggregations on all columns, we will use them as the aggregations functions on the most highly
            # correlated columns.
            if len(aggregations_on_all) > 0:
                best_numerical_cols = [col for col in numerical_cols if col not in self.groupby_columns][
                                      :self.max_aggregation_columns]
                for agg_func in aggregations_on_all:
                    self.aggregations += [(col, agg_func) for col in best_numerical_cols]
            # If the operation has no aggregations at all, we will use the most correlated columns to the groupby columns,
            # with mean and std as the aggregation functions.
            if len(agg_dict) == 0 and len(aggregations_on_all) == 0:
                best_numerical_cols = [col for col in numerical_cols if col not in self.groupby_columns][
                                      :self.max_aggregation_columns]
                self.aggregations += [(col, 'mean') for col in best_numerical_cols] + [(col, 'std') for col in
                                                                                       best_numerical_cols]
            # Else, we will use the aggregations from the operation.
            else:
                col_agg_tuples = []
                for col, agg_func_list in agg_dict.items():
                    # If the column is not in the source_df, we skip it.
                    if col not in self.source_df.columns:
                        continue
                    # If the column is in the source_df, we will use the aggregations from the operation.
                    if isinstance(agg_func_list, str):
                        agg_func_list = [agg_func_list]
                    for agg_func in agg_func_list:
                        col_agg_tuples.append((col, agg_func))
                self.aggregations += [col_agg_tuple for col_agg_tuple in col_agg_tuples if
                                      col_agg_tuple[0] not in self.groupby_columns]

        # If the user wants to use all combinations of the groupby columns, we will do that.
        if self.use_all_groupby_combinations:
            # Set the groupby columns to be every combination of the groupby columns
            groupby_columns = []
            for i in range(1, len(groupby_columns_set) + 1):
                groupby_columns += list(itertools.combinations(groupby_columns_set, i))

            # Turn each tuple in that list into a list
            self.groupby_columns = [list(group) for group in groupby_columns]
        # Otherwise, only make sure that the groupby_columns are lists
        else:
            if isinstance(self.groupby_columns, str):
                self.groupby_columns = [self.groupby_columns]

    def _prepare_case_filter_join(self,
                                  groupby_columns: List[List[str]] = None,
                                  correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                                  aggregations: List[Tuple[str, str]] = None,
                                  operation: Operation = None) -> None:
        """
        Prepare the filter columns, aggregations and groupby columns in the case we are automatically
        inferring from a filter or join operation.

        :param groupby_columns: The groupby columns to use. If None, we will use the filter_columns.
        :param correlation_aggregation_method: The method to use for aggregating the correlation values.
        :param aggregations: The aggregations to use. If None, we will use the most correlated columns.
        """

        # Check if target_columns are in the source_df
        if not all(col in self.source_df.columns for col in self.filter_columns):
            raise ValueError("All target_columns must be present in the source dataframe")

        # Handle the case where there is only 1 value to the filter column after a filter operation
        need_restore_source_df = False
        if len(self.source_df[self.filter_columns].nunique()) == 1:
            source_df = self.source_df
            # This case should really only happen with filters. If it happened with a join, it may just be
            # that there was only one value to the column to begin with
            if isinstance(operation, Filter):
                self.source_df = operation.source_df
            need_restore_source_df = True

        correlated_cols, numerical_cols = self._find_correlated_columns_multi(self.filter_columns,
                                                                              method=correlation_aggregation_method)

        if need_restore_source_df:
            # noinspection PyUnboundLocalVariable
            self.source_df = source_df

        # It is possible that filtering only left us with 1 value in the filtered columns. If that happened,
        # we want to remove those columns because we won't be able to get anything from them.
        # We do this after the correlation computing because we still need some columns to compute the correlation.
        self.filter_columns = [col for col in self.filter_columns if self.source_df[col].nunique() > 1]

        # If there are less than k target columns, we add the most correlated columns to the target_columns
        if len(self.filter_columns) < self.max_filter_columns:
            difference = self.max_filter_columns - len(self.filter_columns)
            self.filter_columns += [col for col in correlated_cols[:difference]]

        if groupby_columns is not None:
            if isinstance(groupby_columns, str):
                groupby_columns = [groupby_columns]
            self.groupby_columns = groupby_columns
        else:
            # If no groupby_columns are provided, we just use the filter_columns
            self.groupby_columns = self.filter_columns

        # Aggregations can be None, in which case we will figure out the measures automatically via the correlation.
        if aggregations is not None:
            self.aggregations = aggregations
        else:
            # If no aggregations are provided, we will use a mean on the k most correlated numerical columns
            best_numerical_cols = [col for col in numerical_cols if col not in self.groupby_columns][
                                  :self.max_aggregation_columns]
            self.aggregations = [(col, 'mean') for col in best_numerical_cols] + [(col, 'std') for col in
                                                                                  best_numerical_cols]

    def _add_llm_reasoning_to_plot(self, llm_reasoning_series: pd.Series,
                                   explanation_nums: list[int] | None = None) -> None:
        """
        Adds LLM reasoning text to the bottom left of the plot.
        :param llm_reasoning_series: A series the LLM reasoning text for each explanation.
        :param explanation_nums: A list of integers representing the explanation numbers. If None, we will not add the
        explanation numbers to the text. If provided, they will be used to format the text.
        """
        added_text = ""
        for i, explanation_num in enumerate(explanation_nums):
            explanation_added_text = llm_reasoning_series.get(i, None)
            if not explanation_added_text:
                continue
            if not isinstance(explanation_added_text, str):
                explanation_added_text = str(explanation_added_text)
            # Replace any wrapping done previously - we want our own custom wrapping here, since we know
            # what length of text we want to display.
            explanation_added_text = explanation_added_text.replace("\n", " ")
            added_text += f"[{explanation_num}] {explanation_added_text}\n\n"

        # If there is text to add, add it to the bottom left of the plot.
        if added_text:
            plt.figtext(0, 0, added_text, horizontalalignment='left', verticalalignment='top',
                        fontsize=22, wrap=True, )

    def visualize(self, metainsights: List[MetaInsight] = None, beautify_code: str = None) -> None | str:
        if metainsights is None:
            metainsights = self.metainsights
        if len(metainsights) == 0:
            return (f"""No common patterns detected for the given parameters:
                    Filter columns: {self.filter_columns}
                    Groupby columns: {self.groupby_columns}
                    Aggregations: {self.aggregations}""")
        else:
            explanations_str = [
                metainsight.__str__() for metainsight in metainsights
            ]
            if self.add_llm_context_explanations:
                reasoner = ExplanationReasoning(
                    data=self.source_df,
                    explanations_found=explanations_str,
                    query_type='metainsight',
                    source_name=self._source_name,
                    query=self._query,
                )
                added_explanations = reasoner.do_llm_action()
            else:
                added_explanations = pd.Series([None] * self.top_k, index=range(self.top_k))
            # Grid display mode - use normal matplotlib visualization
            if self._display_mode == 'grid':
                num_rows = math.ceil(len(metainsights) / self.figs_per_row)
                fig, axs = plt.subplots(
                    nrows=num_rows, ncols=self.figs_per_row, figsize=(11 * self.figs_per_row, 13 * num_rows),
                )
                plt.subplots_adjust(hspace=0.5)
                if len(axs.shape) != 1:
                    axs = axs.flatten()
                for i, mi in enumerate(metainsights[:self.top_k]):
                    mi.visualize(plt_ax=axs[i],
                                 plot_num=i + 1,
                                 max_labels=self.max_labels_per_plot,
                                 max_common_categories=self.max_common_categories_per_plot)

                # Hide any unused subplots
                for j in range(len(metainsights[:self.top_k]), len(axs)):
                    axs[j].set_visible(False)

                if added_explanations is not None:
                    self._add_llm_reasoning_to_plot(
                        llm_reasoning_series=added_explanations,
                        explanation_nums=list(range(1, len(metainsights) + 1))
                    )

                if self.beautify:
                    beautifier = VisualizationBeautifier(
                        visualization_object=fig,
                        max_fix_attempts=self.beautify_max_fix_attempts,
                        data=self.source_df,
                        requester_name='MetaInsight',
                        visualization_params={
                            'top_k': self.top_k,
                            'metainsights': metainsights,
                        },
                        silent=self.silent_beautify,
                        must_generalize=self.generalize_beautify_code
                    )
                    if not beautify_code:
                        try:
                            fig_tab, code = beautifier.do_llm_action()
                        except Exception as e:
                            warnings.warn(f"Beautification failed: {e}. Returning the original figure.")
                            fig_tab = None
                    # If we do have code already (e.g. from a previous beautification attempt),
                    # we will use it instead of generating new code.
                    else:
                        fig_tab = beautifier.beautify_from_code(beautify_code)
                        code = None
                    if fig_tab is not None:
                        if not self._do_not_visualize_beautify:
                            display(fig_tab)
                        if self.return_beautify_code:
                            return code
                    else:
                        warnings.warn("Beautification failed. Returning the original figure.")
                        if not self._do_not_visualize_beautify:
                            display(fig)
                return None
            elif self._display_mode == 'carousel':
                if self.beautify:
                    print("Beautification is not supported in carousel mode. ")
                with CarouselAdapter() as adapter:
                    for i, mi in enumerate(metainsights[:self.top_k]):
                        fig, ax = plt.subplots(figsize=(9, 11))
                        mi.visualize(plt_ax=ax, plot_num=i + 1)
                        if added_explanations is not None:
                            self._add_llm_reasoning_to_plot(
                                llm_reasoning_series=added_explanations,
                                explanation_nums=[i + 1]
                            )
                        plt.close(fig)
                        adapter.capture_output(fig)
                return None
            return None

    def can_visualize(self) -> bool:
        return self.can_run_visualize and not self._do_not_visualize

    def _find_correlated_columns(self, column_name: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Finds the correlated columns for a given column name with optimized performance.
        """
        # Determine if target column is numeric
        target_is_numeric = self.source_df[column_name].dtype.kind in 'ifmM' and self.source_df[
            column_name].nunique() > 6

        all_correlations = []
        numerical_columns = []

        # Pre-compute target data for categorical column
        if not target_is_numeric:
            target_categories = self.source_df[column_name].unique()
            category_indices = {cat: self.source_df[column_name] == cat for cat in target_categories}
            category_counts = {cat: indices.sum() for cat, indices in category_indices.items()}

        for col in self.source_df.columns:
            if col == column_name:
                continue

            # Check if column is numeric
            is_numeric = self.source_df[col].dtype.kind in 'ifmMcf' and self.source_df[col].nunique() > 6

            # Case 1: Both columns are numeric - Pearson correlation
            if target_is_numeric and is_numeric:
                correlation = abs(self.source_df[[column_name, col]].corr().iloc[0, 1])
                numerical_columns.append(col)

            # Case 2: Target is numeric, other column is categorical
            elif target_is_numeric and not is_numeric:
                target_mean = self.source_df[column_name].mean()
                target_sst = ((self.source_df[column_name] - target_mean) ** 2).sum()

                if target_sst <= 0:
                    correlation = 0
                else:
                    group_stats = self.source_df.groupby(col)[column_name].agg(['mean', 'count'])
                    ssb = (group_stats['count'] * ((group_stats['mean'] - target_mean) ** 2)).sum()
                    correlation = np.sqrt(ssb / target_sst)

            # Case 3: Target is categorical, other column is numeric
            elif not target_is_numeric and is_numeric:
                col_values = self.source_df[col].values
                col_mean = col_values.mean()
                col_sst = ((col_values - col_mean) ** 2).sum()

                if col_sst <= 0:
                    correlation = 0
                else:
                    ssb = 0
                    for cat in target_categories:
                        indices = category_indices[cat]
                        count = category_counts[cat]
                        if count > 0:
                            cat_mean = col_values[indices].mean()
                            ssb += count * ((cat_mean - col_mean) ** 2)
                    correlation = np.sqrt(ssb / col_sst)

                numerical_columns.append(col)

            # Case 4: Both columns are categorical - Cramer's V
            else:
                try:
                    # Handle GroupBy case - use Series objects directly to avoid index issues
                    series1 = self.source_df[column_name]
                    series2 = self.source_df[col]

                    # Ensure we have pandas Series with correct lengths
                    if len(series1) == len(series2):
                        contingency = pd.crosstab(series1, series2)

                        # Safety check for crosstab result
                        if contingency.size > 0:
                            chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
                            n = contingency.values.sum()
                            phi2 = chi2 / n
                            r, k = contingency.shape
                            correlation = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0
                        else:
                            correlation = 0
                    else:
                        correlation = 0
                except (TypeError, ValueError):
                    # Fallback if crosstab fails
                    correlation = 0

            all_correlations.append((col, correlation))

        # Sort both lists by correlation strength (descending)
        all_correlations.sort(key=lambda x: x[1], reverse=True)
        numerical_corr = [(col, corr) for col, corr in all_correlations if col in numerical_columns]
        numerical_corr.sort(key=lambda x: x[1], reverse=True)

        return all_correlations, numerical_corr

    def _find_correlated_columns_multi(self, column_names: List[str] | List[List[str]] | set[str], method='avg') -> \
            Tuple[
                List[str], List[str]]:
        """
        Finds correlated columns for multiple target columns.

        :param column_names: List of column names to find correlations for
        :param method: Method to combine correlations - 'avg' for average, 'max' for maximum
        :return: A tuple containing two lists:
            - All correlated columns (in order of correlation)
            - Only the numerical correlated columns (in order of correlation)
        """

        # Dictionary to store correlations for each column
        all_column_names = set()
        for col_list in column_names:
            if isinstance(col_list, str):
                all_column_names.add(col_list)
            else:
                all_column_names.update(col_list)
        all_column_correlations = defaultdict(list)
        all_numerical_columns = set()

        # Calculate correlations for each target column
        for target_column in column_names:
            all_cols, num_cols = self._find_correlated_columns(target_column)

            # Add to numerical columns set
            all_numerical_columns.update([col[0] for col in num_cols])

            # Get correlations for all columns
            for col, corr in all_cols:
                all_column_correlations[col].append(corr)

        # Combine correlations based on method
        combined_correlations = []
        for col, corr_values in all_column_correlations.items():
            if method == 'avg':
                combined_corr = np.mean(corr_values)
            elif method == 'max':
                combined_corr = max(corr_values)
            elif method == 'sum':
                combined_corr = sum(corr_values)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'avg', 'max' or 'sum'")

            combined_correlations.append((col, combined_corr))

        # Sort by correlation strength
        combined_correlations.sort(key=lambda x: x[1], reverse=True)
        all_cols = [col for col, _ in combined_correlations]

        # Filter and sort numerical columns
        numerical_corrs = [(col, corr) for col, corr in combined_correlations if col in all_numerical_columns]
        numerical_corrs.sort(key=lambda x: x[1], reverse=True)
        numerical_cols = [col for col, _ in numerical_corrs]

        return all_cols, numerical_cols

    @staticmethod
    def _sample(df: pd.DataFrame, sample_size: int | float):
        """
        Samples a dataframe.
        :param df: The dataframe to sample.
        :param sample_size: The number of samples to take. Can be an integer for an absolute amount or
        a float between 0 and 1 for a relative amount.
        """
        if isinstance(sample_size, float):
            if 0 < sample_size < 1:
                sample_size = df.shape[0] * sample_size
            else:
                raise ValueError("Sample size be an integer or a float in (0, 1)")
        if sample_size <= 0:
            raise ValueError("Sample size must be a positive number.")

        if df.shape[0] <= sample_size:
            return df
        else:
            # Convert the sample size to an integer, in case it was passed as a float above 1.
            sample_size = int(sample_size)
            # We use a set seed so that the user will always get the same explanation when using sampling.
            generator = np.random.default_rng(RANDOM_SEED)
            uniform_indexes = generator.choice(df.index, sample_size, replace=False)
            return df.loc[uniform_indexes]

    def generate_explanation(self) -> List[MetaInsight] | None:
        """
        Generate explanations using the MetaInsight algorithm.
        """
        if self.metainsights is None:
            miner = MetaInsightMiner(
                k=self.top_k,
                min_commonness=self.min_commonness,
                actionability_regularizer=self.actionability_regulizer,
                balance_factor=self.balance_factor
            )
            if self.use_sampling:
                source_df = self._sample(self.source_df, self.sample_size)
            else:
                source_df = self.source_df
            self.metainsights = miner.mine_metainsights(source_df=source_df,
                                                        filter_dimensions=self.filter_columns,
                                                        measures=self.aggregations,
                                                        breakdown_dimensions=self.groupby_columns,
                                                        extend_by_measure=self.allow_multiple_aggregations,
                                                        extend_by_breakdown=self.allow_multiple_groupbys,
                                                        n_bins=self.n_bins
                                                        )
        self.can_run_visualize = True
        return self.metainsights

    def get_explanation_in_textual_description(self, index: int) -> str:
        """
        Get explanations after they have already been generated.
        If the explanations have not been generated yet, this method will generate them.

        :param index: The index of the explanation to get.
        :return: A human-readable string that explains what was found, and the explanation itself.
        """
        if self.metainsights is None:
            raise ValueError("Explanations have not been generated yet. Please call generate_explanation() first.")

        metainsight = self.metainsights[index]

        textual_description = (f"Using automated analysis on the dataframe {self._source_name}, we have found "
                               f"the common pattern in the data: {metainsight.__str__().replace("\n", "")}\n")
        if len(metainsight.exceptions) > 0:
            textual_description += f"Exceptions to this pattern were found:\n"
        textual_description += metainsight.get_exceptions_string()
        return textual_description
