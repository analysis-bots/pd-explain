from __future__ import annotations

from typing import List

import pandas as pd
from pandas._typing import Dtype
import matplotlib.pyplot as plt
import numpy as np

df_loc = 'C:/Users/itaye/Desktop/pdexplain/pd-explain/Examples/Datasets/spotify_all.csv'
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
            fastpath: bool = False
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
        :param fastpath:
        """
        super().__init__(data, index, dtype, name, copy, fastpath)
        self.explanation = None
        self.operation = None
        self.filter_items = []

    def std_int(self, df, target):
        return abs(df[target].mean()-df.mean())/df.std()
    def calc_influence_std(self, df_agg, df_ex, g_att, g_agg, target):
        try:
            return abs(self.std_int(df_ex.groupby(g_att)[g_agg].mean(), target) - self.std_int(df_agg, target))
        except:
            return 0
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
                df_in_exc = df_in[(df_in[attr] < bin.left)|(df_in[attr] > bin.right)]
            # print(df_in_exc.head())
                inf = self.calc_influence_std(df_agg, df_in_exc, g_att, g_agg, target)/(df_in.id.count()/df_in_exc.id.count())
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
        df = df_in[(df_in[top_attr] < top_bin_all.left)|(df_in[top_attr] > top_bin_all.right)].groupby(g_att)[g_agg].mean()
        fig, ax = plt.subplots(layout='constrained', figsize=(7, 7))
        x1 = list(df_agg.index)
        ind1 = np.arange(len(x1))
        y1 = df_agg.values
    
        x2 = list(df.index)
        ind2 = np.arange(len(x2))
        y2 = df.values
    
        bar1 = ax.bar(ind1-0.2, y1, 0.4, alpha=1., label='All')
        bar2 = ax.bar(ind2+0.2, y2, 0.4,alpha=1., label='Without (\'Explicit\' = 0)')
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

    def explain(self, schema: dict = None, attributes: List = None, top_k: int = 1, figs_in_row: int = 2, explainer='fedex',
                target=None,
                show_scores: bool = False, title: str = None):
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
        # if explainer == 'outlier':

        #     df = pd.read_csv(df_loc)
        #     new_songs = df[df.year > 1990]
        #     m_p_by_decade = new_songs.groupby('decade').mean()['popularity']
        #     self.explain_outlier(m_p_by_decade, new_songs, 'decade', 'popularity', 2020)
        #     return
        if attributes is None:
            attributes = []

        if schema is None:
            schema = {}

        return self.operation.explain(schema=schema, attributes=attributes, top_k=top_k, explainer=explainer, target=target,
                                      figs_in_row=figs_in_row, show_scores=show_scores, title=title)
