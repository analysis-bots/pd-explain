from __future__ import annotations

from fedex_generator.Operations.GroupBy import GroupBy
from pandas import Series, DataFrame
from pandas._libs import lib
from pandas.core.groupby.generic import SeriesGroupBy
from pd_explain.core.explainable_series import ExpSeries


class ExpSeriesGroupBy(SeriesGroupBy):
    """
    Explain group_by series
    """

    def __init__(self,
                 obj,
                 keys=None,
                 axis=0,
                 level=None,
                 grouper=None,
                 exclusions=None,
                 selection=None,
                 as_index=True,
                 sort=True,
                 group_keys=True,
                 squeeze: bool = False,
                 observed: bool = False,
                 mutated: bool = False,
                 dropna: bool = True,
                 ):
        super().__init__(obj=obj, keys=keys, axis=axis, level=level, grouper=grouper, exclusions=exclusions,
                         selection=selection, as_index=as_index
                         , sort=sort,
                         group_keys=group_keys, observed=observed)


    def nunique(self, dropna: bool = True) -> Series | DataFrame:
        """
        Compute count of distinct observations.
        Add operation groupby to the result object.

        :param dropna: Don’t include NaN in the counts.
        :return: Count of distinct observations for each group.
        """
        result = ExpSeries(super().nunique(dropna))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'nunique'])

        if hasattr(self, 'original'):
            original_result = self.original.nunique(dropna)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['nunique']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def count(self):
        """
       Compute count of group, excluding missing values.
       Add operation groupby to the result object.

       :return: count for each group
       """
        result = ExpSeries(super().count())
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'count'])

        if hasattr(self, 'original'):
            original_result = self.original.count()
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['count']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

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
        result = ExpSeries(super().mean(numeric_only, engine, engine_kwargs))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'mean'])

        if hasattr(self, 'original'):
            original_result = self.original.mean(numeric_only, engine, engine_kwargs)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['mean']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def median(self, numeric_only: bool | lib.NoDefault = lib.no_default):
        """
       Compute median of groups, excluding missing values.
       Add operation groupby to the result object.

       :param numeric_only: Include only float, int, boolean columns.
                            If None, will attempt to use everything, then use only numeric data.
       :return: Median of values within each group.
       """
        result = ExpSeries(super().median(numeric_only))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'median'])

        if hasattr(self, 'original'):
            original_result = self.original.median(numeric_only)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['median']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def std(self,
            ddof: int = 1,
            engine: str | None = None,
            engine_kwargs: dict[str, bool] | None = None,
            numeric_only: bool | lib.NoDefault = lib.no_default
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
        result = ExpSeries(super().std(ddof, engine, engine_kwargs, numeric_only=numeric_only))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'std'])

        if hasattr(self, 'original'):
            original_result = self.original.std(ddof, engine, engine_kwargs, numeric_only=numeric_only)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['std']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
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
        result = ExpSeries(super().var(ddof, engine, engine_kwargs, numeric_only=numeric_only))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'var'])

        if hasattr(self, 'original'):
            original_result = self.original.var(ddof, engine, engine_kwargs, numeric_only=numeric_only)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['var']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def sem(self, ddof: int = 1, numeric_only: bool | lib.NoDefault = lib.no_default):
        """
       Compute standard error of the mean of groups, excluding missing values.
       For multiple groupings, the result index will be a MultiIndex.
       Add operation groupby to the result object.

       :param ddof: Degrees of freedom.
       :return:Standard error of the mean of values within each group.
       """
        result = ExpSeries(super().sem(ddof, numeric_only=numeric_only))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'sem'])

        if hasattr(self, 'original'):
            original_result = self.original.sem(ddof, numeric_only=numeric_only)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['sem']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def size(self):
        """
       Compute group sizes.
       Add operation groupby to the result object.

       :return: group sizes
       """
        result = ExpSeries(super().size())
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'size'])

        if hasattr(self, 'original'):
            original_result = self.original.size()
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['size']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
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
        result = ExpSeries(super().sum(numeric_only, min_count, engine, engine_kwargs))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'sum'])

        if hasattr(self, 'original'):
            original_result = self.original.sum(numeric_only, min_count, engine, engine_kwargs)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['sum']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
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
        result = ExpSeries(super().prod(numeric_only, min_count))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'prod'])

        if hasattr(self, 'original'):
            original_result = self.original.prod(numeric_only, min_count)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['prod']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def min(self, numeric_only: bool = False, min_count: int = -1):
        """
        Compute min of group values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param min_count: The required number of valid values to perform the operation.
                          If fewer than min_count non-NA values are present the result will be NA.
        :return: Computed min of values within each group.
        """
        result = ExpSeries(super().min(numeric_only, min_count))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'min'])

        if hasattr(self, 'original'):
            original_result = self.original.min(numeric_only, min_count)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['min']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def max(self, numeric_only: bool = False, min_count: int = -1):
        """
        Compute max of group values.
        Add operation groupby to the result object.

        :param numeric_only: Include only float, int, boolean columns.
                             If None, will attempt to use everything, then use only numeric data.
        :param min_count: The required number of valid values to perform the operation.
                          If fewer than min_count non-NA values are present the result will be NA.
        :return: Computed max of values within each group.
        """
        result = ExpSeries(super().max(numeric_only, min_count))
        agg_attr = result.name
        result.name = '_'.join([agg_attr, 'max'])

        if hasattr(self, 'original'):
            original_result = self.original.max(numeric_only, min_count)
            original_result.operation = GroupBy(source_df=self.operation.source_df,
                                                source_scheme={},
                                                group_attributes=self.group_attributes,
                                                agg_dict={agg_attr: ['max']},
                                                result_df=result.to_frame(),
                                                source_name=self.source_name,
                                                column_mapping={
                                                    result.name: agg_attr
                                                })
            return original_result

        return result

    def drop_duplicates(
            self,
    ) -> ExpDataFrame | None:
        return ExpDataFrame(super().drop_duplicates())
