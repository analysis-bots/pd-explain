from __future__ import annotations

from typing import List, Literal, Callable

import pandas as pd
from pandas import Series
from pandas._typing import Dtype, DropKeep
import matplotlib.pyplot as plt
from fedex_generator.Operations.BJoin import BJoin
from fedex_generator.commons import utils
from pd_explain.explainers import ExplainerFactory
from pd_explain.utils.global_values import get_use_sampling_value

import numpy as np

df_loc = 'C:/Users/itaye/Desktop/pdexplain/pd-explain/Examples/Datasets/spotify_all.csv'

op_table = {
    "eq": "==",
    "ne": "!=",
    "le": "<=",
    "lt": "<",
    "ge": ">=",
    "gt": ">",
    "and": "&",
    "or": "|",
}


class ExpSeries(pd.Series):
    """
    Explainable series, Inherit from pandas Series.
    """

    def __init__(
            self,
            data=None,
            index=None,
            dtype: Dtype | None = None,
            name=None,
            copy: bool = False,
    ):
        """
        Initialize new explain series

        :param data: Contains data stored in Series. If data is a dict, argument order is maintained.
        :param index: Values must be hashable and have the same length as data. Non-unique index values are allowed.
                      Will default to RangeIndex (0, 1, 2, …, n) if not provided.
                      If data is dict-like and index is None, then the keys in the data are used as the index.
                      If the index is not None, the resulting Series is reindexed with the index values.
        :param dtype: Data type for the output Series. If not specified, this will be inferred from data.
        :param name: The name to give to the Series.
        :param copy: Copy input data. Only affects Series or 1d ndarray input. See examples.
        """
        super().__init__(data, index, dtype, name, copy)
        self.explanation = None
        self.operation = None
        self.filter_items = []
        self.filter_query = None
        self.last_used_explainer = None


    # We overwrite the constructor to ensure that an ExpSeries is returned when a new Series is created.
    # This is necessary so that methods not overridden in this class, like iloc, return a Series.
    @property
    def _constructor(self) -> Callable[..., pd.Series]:

        # We define a new constructor that returns a Series, with the same properties as the original dataframe.
        def _c(*args, **kwargs):
            s = ExpSeries(*args, **kwargs)
            s.operation = self.operation
            s.explanation = self.explanation
            s.filter_items = self.filter_items
            s.filter_query = self.filter_query
            return s

        return _c


    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:
        result = super().value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
        )
        ret_val = ExpSeries(result)
        ret_val.operation = self.operation
        ret_val.explanation = self.explanation
        ret_val.filter_items = self.filter_items
        ret_val.filter_query = self.filter_query
        return ret_val


    def std_int(self, df, target):
        return abs(df[target].mean() - df.mean()) / df.std()

    def calc_influence_std(self, df_agg, df_ex, g_att, g_agg, target):
        try:
            return abs(self.std_int(df_ex.groupby(g_att)[g_agg].mean(), target) - self.std_int(df_agg, target))
        except:
            return 0

    def b_join(
            self,
            other,
            on=None,
            how: str = "left",
            lsuffix: str = "",
            rsuffix: str = "",
            sort: bool = False,
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
        from pd_explain.core.explainable_data_frame import ExpDataFrame
        try:
            left_name = utils.get_calling_params_name(self)
            right_name = utils.get_calling_params_name(other)
            right_df = other.copy()
            ignore_columns = [attribute for attribute in on] if on is not None else []
            ignore_columns.append('index')
            self = self.reset_index()
            self.columns = [col if col in ignore_columns else left_name + "_" + col
                            for col in self]
            right_df.columns = [col if col in ignore_columns else right_name + "_" + col
                                for col in right_df]
            result_df = ExpDataFrame(super().join(right_df, on, how, lsuffix, rsuffix, sort))
            result_df.operation = BJoin(ExpDataFrame(self), right_df, None, on, result_df, left_name, right_name)
            return result_df

        except Exception as error:
            print(f'Error {error} with operation merge explanation')
            return ExpDataFrame(super().join(other, on, how, lsuffix, rsuffix, sort))

    def explain_outlier(self, df_agg, df_in, g_att, g_agg, target):
        attrs = df_in.select_dtypes(include='number').columns.tolist()[:10]
        attrs = [a for a in attrs if a not in [g_att, g_agg]]
        top_bin_all = None
        top_inf_all = 0
        top_attr = None
        for attr in attrs:
            if attr != 'explicit':
                continue
            _, bins = pd.cut(df_in[attr], 30, retbins=True, duplicates='drop')
            df_bins_in = pd.cut(df_in[attr], bins=bins).value_counts(normalize=True).sort_index().rename(df_in.id)

            top_inf = 0
            top_bin = None
            for bin in df_bins_in.keys():
                # print(bin.left, bin.right)
                df_in_exc = df_in[(df_in[attr] < bin.left) | (df_in[attr] > bin.right)]
                # print(df_in_exc.head())
                inf = self.calc_influence_std(df_agg, df_in_exc, g_att, g_agg, target) / (
                        df_in.id.count() / df_in_exc.id.count())
                if inf > top_inf:
                    top_inf = inf
                    top_bin = bin
            # print(f'bin of {attr}: {bin}, influence: {inf}')
            # print(f'most influencing bin of {attr}: {top_bin}, influence: {top_inf}')
            if top_inf > top_inf_all:
                top_inf_all = top_inf
                top_bin_all = top_bin
                top_attr = attr

        # print(f'overall, the top contributing bin is {top_bin_all} of {top_attr}. influence {top_inf_all}')
        df = df_in[(df_in[top_attr] < top_bin_all.left) | (df_in[top_attr] > top_bin_all.right)].groupby(g_att)[
            g_agg].mean()
        fig, ax = plt.subplots(layout='constrained', figsize=(7, 7))
        x1 = list(df_agg.index)
        ind1 = np.arange(len(x1))
        y1 = df_agg.values

        x2 = list(df.index)
        ind2 = np.arange(len(x2))
        y2 = df.values

        bar1 = ax.bar(ind1 - 0.2, y1, 0.4, alpha=1., label='All')
        bar2 = ax.bar(ind2 + 0.2, y2, 0.4, alpha=1., label='Without (\'Explicit\' = 0)')
        ax.set_ylabel(f'{g_agg} (mean)')
        ax.set_xlabel(f'{g_att}')
        ax.set_xticks(ind1)
        ax.set_xticklabels(tuple([str(i) for i in x1]), rotation=45)
        ax.legend(loc='best')
        ax.set_title('The predicate (\'Explicit\' = 0) has high influence on this outlier')
        # items_to_bold=[target]
        bar1[x1.index(target)].set_edgecolor('tab:green')
        bar1[x1.index(target)].set_linewidth(2)
        bar2[x2.index(target)].set_edgecolor('tab:green')
        bar2[x2.index(target)].set_linewidth(2)
        ax.get_xticklabels()[-1].set_color('tab:green')

    def drop_duplicates(
            self,
            *,
            keep: DropKeep = "first",
            inplace: bool = False,
            ignore_index: bool = False,
    ):
        return super().drop_duplicates(keep=keep, inplace=inplace, ignore_index=ignore_index)

    # We override the comparison methods to store the comparison operation in the filter_query attribute.
    # There are no other changes to the behavior of the comparison methods.
    def _cmp_method(self, other, op):
        self.filter_query = {
            'op': op_table[op.__name__],
            'other': other
        }
        result = super()._cmp_method(other, op)
        return result

    def __and__(self, other):
        if self.filter_query is not None:
            current_filter_query = self.filter_query
        else:
            current_filter_query = {
                'op': '',
                'other': None
            }
        if isinstance(other, ExpSeries) and other.filter_query is not None:
            self.filter_query = {
                'op': f'{current_filter_query["op"]} {current_filter_query["other"]} & {other.filter_query["op"]}',
                'other': other.filter_query['other']
            }
        else:
            # We have no way to interpret the other since it's a boolean array, therefore we can only rely
            # on the filter query of the current object.
            self.filter_query = {
                'op': f'{current_filter_query["op"]} {current_filter_query["other"]} &',
                'other': None
            }
        result = super().__and__(other)
        return result


    def __or__(self, other):
        if self.filter_query is not None:
            current_filter_query = self.filter_query
        else:
            current_filter_query = {
                'op': '',
                'other': None
            }
        if isinstance(other, ExpSeries) and other.filter_query is not None:
            self.filter_query = {
                'op': f'{current_filter_query["op"]} {current_filter_query["other"]} | {other.filter_query["op"]}',
                'other': other.filter_query['other']
            }
        else:
            # We have no way to interpret the other since it's a boolean array, therefore we can only rely
            # on the filter query of the current object.
            self.filter_query = {
                'op': f'{current_filter_query["op"]} {current_filter_query["other"]} |',
                'other': None
            }
        result = super().__or__(other)
        return result


    def explain(self, schema: dict = None, attributes: List = None, use_sampling: None | bool = None,
                sample_size: int | float = 5000, top_k: int = 1, figs_in_row: int = 2,
                explainer: Literal['fedex', 'outlier', 'many_to_one', 'shapley']='fedex',
                target=None, dir: Literal["high", "low", 1, -1] = None, control=None, hold_out=None,
                show_scores: bool = False, title: str = None,
                labels=None, coverage_threshold: float = 0.7, max_explanation_length: int = 3,
                separation_threshold: float = 0.3, p_value: int = 1,
                explanation_form: Literal['conj', 'disj', 'conjunction', 'disjunction'] = 'conj',
                prune_if_too_many_labels: bool = True, max_labels: int = 10, pruning_method='largest',
                bin_numeric: bool = False, num_bins: int = 10, binning_method: str = 'quantile',
                label_name: str = 'label', explain_errors=True,
                error_explanation_threshold: float = 0.05,
                debug_mode: bool = False,
                add_llm_explanation_reasoning=False,
                display_mode: Literal['grid', 'carousel'] = 'grid',
                beautify: bool = False,
                beautify_max_fix_attempts: int = 10,
                silent_beautify: bool = False,
                do_not_visualize: bool = False,
                log_query: bool = False,
                ):
        """
        Generate an explanation for the dataframe.

        :param explainer: The explainer to use. Currently supported: 'fedex', 'many to one', 'outlier'.
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
        :param figs_in_row: Fedex explainer. Number of explanations figs in one row.
        :param show_scores: Fedex explainer. show scores on explanation.
        :param title: Fedex / outlier / shapley explainers. explanation title.
        :param target: Outlier explainer. Target value for the outlier explainer
        :param dir: Outlier explainer. Direction for the outlier explainer. Can be either 'high' or 'low'.
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
        with the highest mean distance * group size are kept, 'min_dist', 'max_silhouette' - where the k groups with the
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
        :param debug_mode: Developer option. Disables multiprocessing and enables debug prints. Defaults to False.
        :param add_llm_explanation_reasoning: All explainers. Enables using a LLM to generate additional context explanations, explaining why
        the explanations found occur. Defaults to False. Requires setting an API key. See the documentation for more information.
        Note that setting this to True will increase the computation time by a potentially large amount, entirely dependent on the LLM API response time.
        Also note that the output of the LLM is not guaranteed to be accurate, and may contain errors, so use with caution.
        :param display_mode: Fedex explainer. Chooses how to display multiple explanations. Can be either 'grid' or 'carousel'.
        'grid' will display all explanations together in a grid layout, while 'carousel' will display them one by one, with navigation buttons.
        :param beautify: MetaInsight and Fedex explainers. If True, we will attempt to beautify the explanation by having a LLM generate code for producing
        a more visually appealing explanation for this specific case. Defaults to False. Please note that:
        1. This will increase the computation time by a potentially large amount, entirely dependent on the LLM API response time.
        2. The output of the LLM is not guaranteed to be accurate, and may contain errors, so use with caution.
        :param beautify_max_fix_attempts: MetaInsight and Fedex explainers. The maximum number of attempts to fix the
        returned code from the LLM to make it work, if the beautify parameter is set to True. Defaults to 10.
        :param silent_beautify: MetaInsight and Fedex explainers. If True, the beautify process will not print any information
        about its progress, and will only return the final result. Defaults to False.
        :param do_not_visualize: Prevents the explainer from visualizing the explanation, even if it is possible to do so.
        :param log_query: Fedex explainer. If True, the query used to generate the explanation as well as its score will be logged.

        :return: explanation figures
        """

        if explainer.lower().startswith('metainsight'):
            raise ValueError("MetaInsight explainer is not supported for Series.")

        use_sampling = use_sampling if use_sampling is not None else get_use_sampling_value()

        factory = ExplainerFactory()
        explainer = factory.create_explainer(
            schema=schema, attributes=attributes, top_k=top_k, explainer=explainer,
            target=target, dir=dir, control=control, hold_out=hold_out,
            figs_in_row=figs_in_row, show_scores=show_scores, title=title,
            operation=self.operation, source_df=self,
            labels=labels, coverage_threshold=coverage_threshold,
            max_explanation_length=max_explanation_length,
            separation_threshold=separation_threshold, p_value=p_value,
            prune_if_too_many_labels=prune_if_too_many_labels, max_labels=max_labels,
            pruning_method=pruning_method, bin_numeric=bin_numeric, num_bins=num_bins,
            binning_method=binning_method, label_name=label_name,
            explanation_form=explanation_form, use_sampling=use_sampling,
            sample_size=sample_size,
            explain_errors=explain_errors, error_explanation_threshold=error_explanation_threshold,
            debug_mode=debug_mode,
            add_llm_context_explanations=add_llm_explanation_reasoning,
            display_mode=display_mode,
            beautify=beautify,
            beautify_max_fix_attempts=beautify_max_fix_attempts,
            silent_beautify=silent_beautify,
            do_not_visualize=do_not_visualize,
            log_query=log_query
        )
        self.last_used_explainer = explainer

        explanation = explainer.generate_explanation()

        if explainer.can_visualize():
            return explainer.visualize()

        return explanation

    def to_html(self, *args, **kwargs):
        """
        Render the Series to a HTML table.
        We do it this way because for an unknown reason, the default for series does not always work.
        This way, we instead get the usual table that we get for a DataFrame.
        """
        return self.to_frame().to_html(*args, **kwargs)
