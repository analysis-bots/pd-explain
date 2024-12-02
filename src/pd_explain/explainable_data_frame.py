from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.axis import Axis
from pandas._libs.lib import no_default



# importing sys
import sys
 
# adding Folder_2/subfolder to the system path

sys.path.insert(0, 'C:/Users/itaye/Desktop/pdexplain/FEDEx_Generator-1/src/')
sys.path.insert(0, "C:/Users/Yuval/PycharmProjects/FEDEx_Generator/src")
# sys.path.insert(0, 'C:/Users/User/Desktop/pd_explain_test/FEDEx_Generator-1/src')
from fedex_generator.Operations.Filter import Filter
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Join import Join
from fedex_generator.Operations.BJoin import BJoin
from fedex_generator.commons import utils
from typing import (
    Hashable,
    Sequence,
    Union,
    List,
)
from pandas._typing import Level, Renamer, IndexLabel, Axes, Dtype
sys.path.insert(0, 'C:/Users/itaye/Desktop/pdexplain/pd-explain/src/')
sys.path.insert(0, "C:/Users/Yuval/PycharmProjects/pd-explain/src")
# sys.path.insert(0, 'C:/Users/User/Desktop/pd_explain_test/pd-explain/src')
from pd_explain.explainable_series import ExpSeries


class ExpDataFrame(pd.DataFrame):
    """
    Explainable dataframe, extents Pandas DataFrame adding state management and explain() function.
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



    @property
    def _constructor(self):
        return ExpDataFrame

    @property
    def _constructor_sliced(self):
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
        to_return = super().__getitem__(key)
        t = str(type(to_return))
        if str(type(to_return)) == "<class 'pandas.core.frame.DataFrame'>":
            return ExpDataFrame(to_return)
        if t == "<class 'pd_explain.explainable_data_frame.ExpDataFrame'>":
            return to_return
        # to_return.source_df = self.operation.source_df
        return ExpSeries(to_return)

    def copy(self, deep=True):
        """
        Make a copy of this object’s indices and data.
        :param deep: Make a deep copy, including a copy of the data and the indices.
                     With deep=False neither the indices nor the data are copied.
        :return: explain dataframe copy
        """
        return ExpDataFrame(super().copy(deep))

    def drop(
            self,
            labels=None,
            axis: Union[str, int] = 0,
            index=None,
            columns=None,
            level: Level | None = None,
            inplace: bool = False,
            errors: str = "raise",
    ):
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

        :return: Explain DataFrame without the removed index or column labels or None if inplace=True.
        """
        return ExpDataFrame(super().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors))

    def rename(self,
               mapper: Renamer | None = None,
               *,
               index: Renamer | None = None,
               columns: Renamer | None = None,
               axis: Axis | None = None,
               copy: bool = True,
               inplace: bool = False,
               level: Level | None = None,
               errors: str = "ignore", ):
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

        :return: Explain DataFrame with the renamed axis labels or None if inplace=True.
        """
        if inplace:
            # When doing an inplace rename, we need to update the source dataframe of the operation.
            # Otherwise, the operation may use the old dataframes or cause an error.
            # Please note that this causes a recursive call, as the source_df should also be an ExpDataFrame.
            if self.operation is not None:
                self.operation.source_df.rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                                copy=copy, inplace=inplace, level=level, errors=errors)
                # If we change the name of the attribute the operation is performed on, we need to update the operation.
                if columns is not None and self.operation.attribute in columns:
                    self.operation.attribute = columns[self.operation.attribute]
                elif mapper is not None and self.operation.attribute in mapper:
                    self.operation.attribute = mapper[self.operation.attribute]
                elif index is not None and self.operation.attribute in index:
                    self.operation.attribute = index[self.operation.attribute]
            # Perform the renaming, and save the result to a variable so we can return it later.
            super(ExpDataFrame, self).rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                             copy=copy, inplace=inplace, level=level, errors=errors)
            res = self
        else:
            # If the operation is not inplace, we can just return the new dataframe.
            # However, we need to make sure to update the operation of the new dataframe, as otherwise we may get a
            # no operation found error.
            res = super(ExpDataFrame, self).rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                                    copy=copy, inplace=inplace, level=level, errors=errors)
            if self.operation is not None:
                res.operation = self.operation
                res.operation.source_df = res.operation.source_df.rename(mapper=mapper, index=index, columns=columns, axis=axis,
                                                    copy=copy, inplace=inplace, level=level, errors=errors)
                res.operation.result_df = res

        # Finally, update the attribute name in the operation if it was renamed.
        if self.operation is not None:
            if columns is not None and self.operation.attribute in columns:
                self.operation.attribute = columns[self.operation.attribute]
            # In the case of a mapper, we only care about making the update to the operation if it affects the columns.
            elif mapper is not None and axis == 'columns':
                # If the mapper is of the form {old_name: new_name}, we need to update the attribute name if it was
                # renamed.
                if (isinstance(mapper, dict) or isinstance(mapper, pd.Series)) and self.operation.attribute in mapper:
                    self.operation.attribute = mapper[self.operation.attribute]
                # Otherwise, if the mapper is a function, we need to call the function on the attribute name.
                elif callable(mapper):
                    self.operation.attribute = mapper(self.operation.attribute)


        # Then return the result.
        return res


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
        result_df = ExpDataFrame(super().where(cond))#, other, inplace, axis, level, errors, try_cast)
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
            axis=0,
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
            from pd_explain.explainable_group_by_dataframe import ExpDataFrameGroupBy
            # group_attributes = GroupBy.get_one_to_many_attributes(self, [by] if isinstance(by, str) else by)
            group_attributes = by
            tmp = pd.core.groupby.generic.DataFrameGroupBy
            pd.core.groupby.generic.DataFrameGroupBy = ExpDataFrameGroupBy
            g = super().groupby(by=group_attributes, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys
                                   , observed=observed, dropna=dropna)
            g.group_attributes = by
            g.source_name = utils.get_calling_params_name(self)
            g.operation = GroupBy(source_df=self, group_attributes=by, result_df=g, source_scheme=None, agg_dict=None)
            g.original = super().groupby(by=by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys
                                   , observed=observed, dropna=dropna)

            pd.core.groupby.generic.DataFrameGroupBy = tmp
            return g

        except Exception as error:
            print(f'Error {error} with operation group by explanation')
            g = super().groupby(by=by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys
                                   , observed=observed, dropna=dropna)
            # g.group_attributes = by
            # g.operation = GroupBy(source_df=self, group_attributes=by, result_df=g)
            return super().groupby(by=by, axis=axis, level=level, as_index=as_index, sort=sort, group_keys=group_keys
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
                result_df.operation = Filter(source_df=self,
                                             source_scheme={},
                                             attribute=self.filter_items.pop(),
                                             result_df=result_df)
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
            left_on = None,
            right_on = None,
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
                            # for col in self]
            # right_df.columns = [col if col in ignore_columns else right_name + "_" + col
                                # for col in right_df]
            result_df = ExpDataFrame(pd.merge(self, right_df, on=on, left_on=left_on, right_on=right_on, how=how))
            # result_df = ExpDataFrame(super().join(right_df, on, how, lsuffix, rsuffix, sort))
            result_df.operation = Join(self, right_df, None, on, result_df, left_name, right_name)

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
            explain = False,
            consider = 'left',
            top_k = 1
    ) :
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
            if(explain):
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
        return ExpDataFrame(super().reset_index(drop=drop))


    def drop_duplicates(
            self,
    ) -> ExpDataFrame | None:
        return ExpDataFrame(super().drop_duplicates())
    def __repr__(self):
        """
        repr object
        :return: super repr
        """
        return super().__repr__()

    def present_deleted_correlated(self, figs_in_row: int = 2): #####
        return self.operation.present_deleted_correlated(figs_in_row = figs_in_row)
        
        
        
    def explain(self, schema: dict = None, attributes: List = None, top_k: int = None, explainer='fedex', target=None, dir=None,
                figs_in_row: int = 2, show_scores: bool = False, title: str = None, corr_TH: float = 0.7, consider='right', value=None, attr=None, ignore=[]):
        """
        Generate explanation to series base on the operation lead to this series result
        :param schema: result columns, can change columns name and ignored columns
        :param attributes: list of specific columns to consider in the explanation
        :param top_k: number of explanations
        :param figs_in_row: number of explanations figs in one row
        :param show_scores: show scores on explanation
        :param title: explanation title


        :return: explanation figures
        """
        if attributes is None:
            attributes = []
            if top_k is None:
                top_k=1
        else:
            if top_k is None:
                top_k=len(attributes)

        if schema is None:
            schema = {}
        if self.operation is None:
            print('no operation was found.')
            return

        return self.operation.explain(schema=schema, attributes=attributes, top_k=top_k,figs_in_row=figs_in_row, show_scores=show_scores, title=title, corr_TH=corr_TH, explainer=explainer, consider=consider, cont=value, attr=attr, ignore=ignore)

