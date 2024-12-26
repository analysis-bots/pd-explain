from typing import List, Dict, Any

import numpy as np
from ipywidgets import Tab
from pandas import DataFrame, Series
import pandas as pd
from ipywidgets import Output
import matplotlib.pyplot as plt

from pd_explain.recommenders.utils import consts
from pd_explain.recommenders.utils.recommendation_data_classes import FilterQuery
from pd_explain.recommenders.recommender_base import RecommenderBase
from pd_explain.recommenders.utils.util_funcs import is_numeric
from pd_explain.recommenders.configurations.filter_recommender_configuration import FilterRecommenderConfiguration
from pd_explain.recommenders.measures.correlation_based_attribute_interestingness_measure import \
    CorrelationBasedAttributeInterestingnessMeasure
from fedex_generator.Measures.ExceptionalityMeasure import ExceptionalityMeasure
from fedex_generator.Operations.Filter import Filter


class FilterRecommender(RecommenderBase):
    """
    A recommender that recommends filter queries to the user.

    This class is a subclass of RecommenderBase, and implements the filter specific logic.
    """

    def __init__(self):
        measures = [CorrelationBasedAttributeInterestingnessMeasure()]
        super().__init__(configuration=FilterRecommenderConfiguration(), measures=measures)

    __name__ = 'FilterRecommender'

    @property
    def name(self) -> str:
        return self.__name__


    def _compute_recommendation_scores_internal(self, data: DataFrame, attributes: List[str], queries: Dict[str, List], top_k: int) -> \
            dict[str, Series]:
        query_scores = {}
        for attribute in attributes:
            query_scores[attribute] = Series(np.zeros(len(queries[attribute])), index=queries[attribute])
            for i, query in enumerate(queries[attribute]):
                # Apply the filter to the data
                filtered_data = Filter(source_df=data, source_scheme={}, attribute=attribute,
                                       operation_str=query.operation, value=query.value)
                measure = ExceptionalityMeasure()
                # Get the score dict (key = attribute, value = score)
                scores = measure.calc_measure(operation_object=filtered_data,scheme={},use_only_columns=[])
                # Sort the scores in descending order, and only keep the values
                scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                scores = [score for _, score in scores]
                # The score is the sum of the top-k scores
                score = sum(scores[:top_k])
                query_scores[attribute][i] = score
            # Sort the queries based on their score
            query_scores[attribute] = query_scores[attribute].sort_values(ascending=False)
        return query_scores

    def _create_tab_internal(self, data: DataFrame, attributes: List[str],
                             queries: Dict[str, List[FilterQuery]], top_k_explanations: int) -> Tab:
        # Create the external tab that will contain all the query sub-tabs
        tab = Tab()
        children = []
        for attribute in attributes:
            # Create a sub-tab for each attribute
            queries_tab = Tab()
            queries_tab_children = []
            queries_tab_titles = []
            # Create the explanation plot for each query
            for query in queries[attribute]:
                filtered_data = Filter(source_df=data, source_scheme={}, attribute=attribute,
                                       operation_str=query.operation, value=query.value)
                out = Output()


                with out:
                    fig = filtered_data.explain(top_k=top_k_explanations)
                    plt.show(fig)

                queries_tab_children.append(out)
                queries_tab_titles.append(str(query))

            # Add the children to the tab, and set the titles
            queries_tab.children = queries_tab_children
            for i, title in enumerate(queries_tab_titles):
                queries_tab.set_title(i, title)
            children.append(queries_tab)

        # Add the children to the external tab
        tab.children = children
        for i, attribute in enumerate(attributes):
            tab.set_title(i, attribute)
        return tab

    def _create_recommendation_candidates_internal(self, data: DataFrame, attributes: List[str]) -> dict[str, list[FilterQuery]]:
        queries = {}
        for attribute in attributes:
            col = data[attribute]
            attribute_queries = []
            # If the attribute is numeric, we use the bin edges as filter values for less than and greater than filters
            if is_numeric(data, attribute):
                # Bin the attribute if it is numeric
                bins = pd.qcut(col, self._configuration.num_bins, retbins=True, duplicates='drop')[1]
                bins.sort()
                # Create the filter queries. For each bin, we create a less than and greater than query using
                # the bin edges as filter values.
                for i in range(len(bins) - 1):
                    # A greater than query on the first bin is meaningless because it would include all values
                    if i != 0:
                        attribute_queries.append(FilterQuery(attribute, ">=", bins[i]))
                    # Likewise, a less than query on the last bin is meaningless
                    if i != len(bins) - 2:
                        attribute_queries.append(FilterQuery(attribute, "<=", bins[i + 1]))

            # If the attribute is categorical, we use the bin values as filter values for equality filters
            else:
                # Keep the 10 most frequent values if the attribute is categorical
                value_counts = col.value_counts()
                bins = value_counts.index[:self._configuration.num_bins]
                for b in bins:
                    attribute_queries.append(FilterQuery(attribute, "==", b))
                    # If there are only 2 values, adding a not equal query is meaningless and would be the same as the equal query
                    if len(bins) > 2:
                        attribute_queries.append(FilterQuery(attribute, "!=", b))
            # Add the queries to the dictionary
            queries[attribute] = attribute_queries
        return queries
