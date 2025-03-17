from __future__ import annotations
from fedex_generator.Operations.GroupBy import GroupBy
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
from pandas._libs import lib
from typing import Literal

from pd_explain import ExpDataFrame
from pd_explain.core.explainable_group_by_series import ExpSeriesGroupBy

# These used to be in the __getitem__ method, but doing it that way made it so the classes could end up
# remaining wrong in multi-threaded environments.
tmp_dataframe = pd.core.groupby.generic.DataFrameGroupBy
tmp_series = pd.core.groupby.generic.SeriesGroupBy


class ExpDataFrameGroupBy(DataFrameGroupBy):
    """
    Explain group_by dataframe
    """

    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        """
        Aggregate using one or more operations over the specified axis.

        :param func: Function to use for aggregating the data.
                     If a function, must either work when passed a DataFrame or when passed to DataFrame.apply.

                     Accepted combinations are:
                     * function
                     * string function name
                     * list of functions and/or function names, e.g. [np.sum, 'mean']
                     * dict of axis labels -> functions, function names or list of such.
                     Can also accept a Numba JIT function with engine='numba' specified.
                     Only passing a single function is supported with this engine.
                     If the 'numba' engine is chosen, the function must be a user defined function with values and index
                      as the first and second arguments respectively in the function signature. Each group’s index will
                      be passed to the user defined function and optionally available for use.
        :param args: Positional arguments to pass to func.
        :param engine: * 'cython' : Runs the function through C-extensions from cython.
                       * 'numba' : Runs the function through JIT compiled code from numba.
                       * None : Defaults to 'cython' or globally setting compute.use_numba
        :param engine_kwargs:
        :param kwargs:
        :return: Explain Groupby Dataframe
        """
        from pd_explain.core.explainable_data_frame import ExpDataFrame
        result = ExpDataFrame(super().aggregate(func, engine, engine_kwargs, kwargs))

        if hasattr(self, 'original'):
            original_result = self.original.aggregate(func, engine, engine_kwargs, kwargs)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict=func,
                                                result_df=result,
                                                source_name=self.source_name)

            return original_result

        return result

    agg = aggregate

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        """
        Get item from dataframe
        :param key: item key
        :return: item as ExplainDataFrameGroupBy or ExplainSeriesGroupBy
        """
        pd.core.groupby.generic.DataFrameGroupBy = ExpDataFrameGroupBy
        pd.core.groupby.generic.SeriesGroupBy = ExpSeriesGroupBy

        try:
            item = super().__getitem__(key)
            if hasattr(self, 'group_attributes'):
                item.group_attributes = self.group_attributes

            if hasattr(self, 'source_name'):
                item.source_name = self.source_name
            if hasattr(self, 'operation'):
                item.operation = self.operation

            if hasattr(self, 'original'):
                item.original = self.original.__getitem__(key)
        # If ANY exception occurs, without this, we will be stuck with the wrong classes.
        finally:
            pd.core.groupby.generic.DataFrameGroupBy = tmp_dataframe
            pd.core.groupby.generic.SeriesGroupBy = tmp_series

        return item

    def nunique(self, dropna: bool = True):
        """
        Return DataFrame with counts of unique elements in each position.

        :param dropna: Don’t include NaN in the counts.

        :return: DataFrame
        """
        try:
            result = ExpDataFrame(super().nunique(dropna))
            agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
            result.name = '_'.join([agg_attr, 'nunique'])

            if hasattr(self, 'original'):
                original_result = self.original.nunique(dropna)
                original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                    source_scheme={},
                                                    group_attributes=self.group_attributes,
                                                    agg_dict={agg_attr: ['nunique']},
                                                    result_df=result,
                                                    source_name=self.source_name)
                return original_result

        except Exception as error:
            print(error)
            result = super().nunique(dropna)

        return result

    def count(self):
        """
        Compute count of group, excluding missing values.
        Add operation groupby to the result object.

        :return: count for each group
        """

        try:
            result = ExpDataFrame(super().count())
            agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
            result.name = '_'.join([agg_attr, 'count'])

            if hasattr(self, 'original'):
                original_result = self.original.count()
                original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                    source_scheme={},
                                                    group_attributes=self.group_attributes,
                                                    agg_dict={agg_attr: ['count']},
                                                    result_df=result,
                                                    source_name=self.source_name)
                return original_result

        except Exception as error:
            print(error)
            result = super().count()

        return result

    def mean(self,
             numeric_only: bool | lib.NoDefault = lib.no_default,
             engine: str = "cython",
             engine_kwargs: dict[str, bool] | None = None):
        """
        Compute mean of groups, excluding missing values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param engine: * 'cython' : Runs the operation through C-extensions from cython.
                       * numba' : Runs the operation through JIT compiled code from numba.
                       * None : Defaults to 'cython' or globally setting compute.use_numba
        :param engine_kwargs: * For 'cython' engine, there are no accepted engine_kwargs
                              * For 'numba' engine, the engine can accept nopython, nogil and parallel dictionary keys.
                                The values must either be True or False.
                                The default engine_kwargs for the 'numba' engine is
                                {{'nopython': True, 'nogil': False, 'parallel': False}}
        :return: Mean value for each group
        """
        try:
            result = ExpDataFrame(super().mean(numeric_only, engine, engine_kwargs))
            agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
            result.name = '_'.join([agg_attr, 'mean'])
            # result.operation = self.operation

            if hasattr(self, 'original'):
                original_result = self.original.mean(numeric_only, engine, engine_kwargs)
                original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                    source_scheme={},
                                                    group_attributes=self.group_attributes,
                                                    agg_dict={agg_attr: ['mean']},
                                                    result_df=result,
                                                    source_name=self.source_name)
                return original_result

        except Exception as error:
            print(error)
            result = super().mean()
        # result.source_df = self.source_df
        return result

    def median(self, numeric_only: bool | lib.NoDefault = lib.no_default):
        """
        Compute median of groups, excluding missing values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :return: Median of values within each group.
        """
        try:
            result = ExpDataFrame(super().median(numeric_only))
            agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
            result.name = '_'.join([agg_attr, 'median'])

            if hasattr(self, 'original'):
                original_result = self.original.median(numeric_only)
                original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                    source_scheme={},
                                                    group_attributes=self.group_attributes,
                                                    agg_dict={agg_attr: ['median']},
                                                    result_df=result,
                                                    source_name=self.source_name)
                return original_result

        except Exception as error:
            print(error)
            result = super().median()
        return result

    def std(
            self,
            ddof: int = 1,
            engine: str | None = None,
            engine_kwargs: dict[str, bool] | None = None,
            numeric_only: bool | lib.NoDefault = lib.no_default,
    ):
        """
        Compute standard deviation of groups, excluding missing values.
        Add operation groupby to the result object.

        :param ddof: Degrees of freedom.
        :param engine:
                        * 'cython' : Runs the operation through C-extensions from cython.
                        * 'numba' : Runs the operation through JIT compiled code from numba.
                        * None : Defaults to 'cython' or globally setting compute.use_numba
        :param engine_kwargs:
                              * For 'cython' engine, there are no accepted engine_kwargs
                              * For 'numba' engine, the engine can accept nopython, nogil and parallel dictionary keys.
                                The values must either be True or False.
                                The default engine_kwargs for the 'numba' engine is
                                {{'nopython': True, 'nogil': False, 'parallel': False}}
        :return: Standard deviation of values within each group.
        """
        result = ExpDataFrame(super().std(ddof, engine, engine_kwargs, numeric_only=numeric_only))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'std'])

        if hasattr(self, 'original'):
            original_result = self.original.std(ddof, engine, engine_kwargs)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['std']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def var(
            self,
            ddof: int = 1,
            engine: str | None = None,
            engine_kwargs: dict[str, bool] | None = None,
            numeric_only: bool | lib.NoDefault = lib.no_default,
    ):
        """
        Compute variance of groups, excluding missing values.
        For multiple groupings, the result index will be a MultiIndex.
        Add operation groupby to the result object.

        :param ddof: Degrees of freedom.
        :param engine:
                        * 'cython' : Runs the operation through C-extensions from cython.
                        * 'numba' : Runs the operation through JIT compiled code from numba.
                        * None : Defaults to 'cython' or globally setting compute.use_numba
        :param engine_kwargs:
                              * For 'cython' engine, there are no accepted engine_kwargs
                              * For 'numba' engine, the engine can accept nopython, nogil and parallel dictionary keys.
                                The values must either be True or False.
                                The default engine_kwargs for the 'numba' engine is
                                {{'nopython': True, 'nogil': False, 'parallel': False}}
        :return: Variance of values within each group.
        """
        result = ExpDataFrame(super().var(ddof, engine, engine_kwargs, numeric_only=numeric_only))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'var'])

        if hasattr(self, 'original'):
            original_result = self.original.var(ddof, engine, engine_kwargs, numeric_only=numeric_only)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['var']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def sem(self,
            ddof: int = 1,
            numeric_only: bool | lib.NoDefault = lib.no_default
            ):
        """
        Compute standard error of the mean of groups, excluding missing values.
        For multiple groupings, the result index will be a MultiIndex.
        Add operation groupby to the result object.

        :param ddof: Degrees of freedom.
        :return:Standard error of the mean of values within each group.
        """
        result = ExpDataFrame(super().sem(ddof, numeric_only=numeric_only))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'sem'])

        if hasattr(self, 'original'):
            original_result = self.original.sem(ddof)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['sem']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def size(self):
        """
        Compute group sizes.
        Add operation groupby to the result object.

        :return: group sizes
        """
        result = ExpDataFrame(super().size())
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'size'])

        if hasattr(self, 'original'):
            original_result = self.original.size()
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['size']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def sum(
            self,
            numeric_only: bool | lib.NoDefault = lib.no_default,
            min_count: int = 0,
            engine: str | None = None,
            engine_kwargs: dict[str, bool] | None = None,
    ):
        """
        Compute sum of group values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param min_count: The required number of valid values to perform the operation.
                          If fewer than min_count non-NA values are present the result will be NA.
        :param engine:
                        * 'cython' : Runs the operation through C-extensions from cython.
                        * 'numba' : Runs the operation through JIT compiled code from numba.
                        * None : Defaults to 'cython' or globally setting compute.use_numba
        :param engine_kwargs:
                              * For 'cython' engine, there are no accepted engine_kwargs
                              * For 'numba' engine, the engine can accept nopython, nogil and parallel dictionary keys.
                                The values must either be True or False.
                                The default engine_kwargs for the 'numba' engine is
                                {{'nopython': True, 'nogil': False, 'parallel': False}}
        :return: Computed sum of values within each group.
        """
        result = ExpDataFrame(super().sum(numeric_only, min_count, engine, engine_kwargs))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'sum'])

        if hasattr(self, 'original'):
            original_result = self.original.sum(numeric_only, min_count, engine, engine_kwargs)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['sum']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def prod(self, numeric_only: bool | lib.NoDefault = lib.no_default, min_count: int = 0):
        """
        Compute prod of group values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param min_count: The required number of valid values to perform the operation.
                          If fewer than min_count non-NA values are present the result will be NA.
        :return: Computed prod of values within each group.
        """
        result = ExpDataFrame(super().prod(numeric_only, min_count))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'prod'])

        if hasattr(self, 'original'):
            original_result = self.original.prod(numeric_only, min_count)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['prod']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def min(self, numeric_only: bool = False, min_count: int = -1,
            engine: Literal["cython", "numba"] | None = None,
            engine_kwargs: dict[str, bool] | None = None, ):
        """
        Compute min of group values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param min_count: The required number of valid values to perform the operation.
                          If fewer than min_count non-NA values are present the result will be NA.
        :return: Computed min of values within each group.
        """
        result = ExpDataFrame(super().min(numeric_only, min_count))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'min'])

        if hasattr(self, 'original'):
            original_result = self.original.min(numeric_only, min_count)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['min']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def drop_duplicates(
            self,
    ) -> ExpDataFrame | None:
        return ExpDataFrame(super().drop_duplicates())

    def max(self, numeric_only: bool = False, min_count: int = -1,
            engine: Literal["cython", "numba"] | None = None,
            engine_kwargs: dict[str, bool] | None = None, ):
        """
        Compute max of group values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param min_count: The required number of valid values to perform the operation.
                          If fewer than min_count non-NA values are present the result will be NA.
        :return: Computed max of values within each group.
        """
        result = ExpDataFrame(super().max(numeric_only=numeric_only))
        agg_attr = result.name if hasattr(result, 'name') and not 'name' in result.columns else 'All'
        result.name = '_'.join([agg_attr, 'max'])

        if hasattr(self, 'original'):
            original_result = self.original.max(numeric_only, min_count)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['max']},
                                                result_df=result,
                                                source_name=self.source_name)
            return original_result

        return result

    def explain(self, schema: dict = None, attributes: list = None, top_k: int = 1, figs_in_row: int = 2,
                target=None, dir=None, control=None, hold_out=[],
                show_scores: bool = False, title: str = None, ignore=[]):

        if attributes is None:
            attributes = []

        if schema is None:
            schema = {}
        if self.operation is None:
            print('no operation was found.')
            return
        mean_agg = self.mean()
        mean_agg.explain(schema=schema, attributes=attributes, top_k=top_k, explainer='fedex', target=target, dir=dir,
                         figs_in_row=figs_in_row, show_scores=show_scores, title=title, ignore=ignore)
