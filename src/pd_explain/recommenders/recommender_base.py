from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from ipywidgets import Tab, HTML
from IPython.display import display
from pandas import DataFrame, Series
from pd_explain.recommenders.utils.data_classes import Query


class RecommenderBase(ABC):

    def __init__(self):
        self._analyzers = []

    def recommend(self, data: DataFrame, attributes: List[str] = None, top_k_attributes: int = 3,
                  top_k_recommendations: int = 1, top_k_explanations: int = 4) -> Tab:
        """
        Recommends a list of queries to the user.

        :param data: The data to recommend queries for.
        :param attributes: The attributes to recommend queries for. If None, the recommender will automatically
                            select the attributes.
        :param top_k_attributes: The maximum number of attributes to recommend queries for.
        :param top_k_recommendations: The maximum number of recommendations to return for each attribute.
        :param top_k_explanations: The maximum number of explanations to provide for each recommendation.

        :return: A tab with the recommendations.
        """
        # If the user did not specify the attributes, we will automatically select the best attributes, based
        # on an interestingness score.
        if attributes is None or len(attributes) == 0:
            # Compute the interestingness score for each attribute.
            scores = {}
            for analyzer in self._analyzers:
                scores[analyzer.__repr__()] = analyzer.analyze(data)
            # Condense the scores to a single array per attribute
            scores = self._condense_scores(scores)
            # Get the top-k skyline attributes
            attributes = self.top_k_skyline(scores=scores, top_k=top_k_attributes)

        # Get the queries for each attribute
        queries = self._create_queries_internal(data=data, attributes=attributes)

        # If the data is too large, we want to avoid computing scores on possibly many queries on the entire data.
        # Therefore, we sample the data instead.
        sampled_data = self.sample_data(data)

        # Score each query
        query_scores = self._compute_query_scores_internal(data=sampled_data, attributes=attributes,
                                                           queries=queries, top_k=top_k_explanations)

        # Take the top-k recommendations for each attribute
        for attribute in attributes:
            query_scores[attribute] = query_scores[attribute][:top_k_recommendations]
            queries[attribute] = [query for query in queries[attribute] if query in query_scores[attribute].index.tolist()]

        display(
            HTML(
                """
        <style>
        .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
            flex: 0 1 auto
        }
        </style>
        """
            )
        )

        return self._create_tab_internal(data=data, attributes=attributes, queries=queries, top_k_explanations=top_k_explanations)



    def top_k_skyline(self, scores: Dict[str, np.ndarray], top_k: int) -> List[str]:
        """
        Compute the top-k skyline attributes based on the scores.

        :param scores: The scores for each attribute.
        :param top_k: The number of attributes to return.

        :return: The top-k skyline attributes.
        """
        skyline = []
        for attribute, score in scores.items():
            if len(skyline) < top_k:
                skyline.append(attribute)
            else:
                for i in range(len(skyline)):
                    if self._is_dominated(scores[skyline[i]], score):
                        skyline[i] = attribute
                        break
        return skyline



    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the recommender.

        :return: The name of the recommender.
        """
        raise NotImplementedError

    def _condense_scores(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """
        Condense the scores to a single array per attribute.

        :param scores: The scores to condense.

        :return: The condensed scores.
        """
        condensed_scores = {}
        for score in scores.values():
            for attribute, value in score.items():
                if attribute not in condensed_scores:
                    condensed_scores[attribute] = []
                condensed_scores[attribute].append(value)

        for attribute in condensed_scores:
            condensed_scores[attribute] = np.array(condensed_scores[attribute])

        return condensed_scores

    def _is_dominated(self, score: np.ndarray, param: np.ndarray) -> bool:
        """
        Check if the score is dominated by the parameter.

        :param score: The score to check.
        :param param: The parameter to check against.

        :return: True if the score is dominated, False otherwise.
        """
        return all(score <= param) and any(score < param)

    @abstractmethod
    def _create_queries_internal(self, data: DataFrame, attributes: List[str]) -> Dict[str, List]:
        """
        Creates filter queries for each attribute.

        :param data: The data to recommend queries for.
        :param attributes: The attributes to recommend queries for.

        :return: The queries for each attribute.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_query_scores_internal(self, data: DataFrame, attributes: List[str], queries: Dict[str, List[str]], top_k: int) -> Dict[str, Series]:
        """
        Compute the scores for the queries.

        :param data: The data to recommend queries for.
        :param attributes: The attributes to recommend queries for.
        :param queries: The queries to recommend.
        :param top_k: The number of top-k scores to consider. Corresponds to the max number of explanations to provide
        for each query.

        :return: The scores for the queries.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_tab_internal(self, data: DataFrame, attributes: List[str],
                             queries: Dict[str, List], top_k_explanations: int) -> Tab:
        """
        Create the tab with the recommendations.

        :param data: The data to recommend queries for.
        :param attributes: The attributes to recommend queries for.
        :param queries: The queries to recommend.
        :param top_k_explanations: The number of top-k explanations to consider.

        :return: The tab with the recommendations.
        """
        raise NotImplementedError

    def sample_data(self, data):
        """
        Sample the data to a smaller size.
        Size is determined in proportion to the original data size.

        #TODO Implement this method if the proof of concept is successful. Right now it just returns the data.

        :param data: The data to sample.

        :return: A sampled subset of the data.
        """
        return data

