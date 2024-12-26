from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
from ipywidgets import Tab, HTML
from IPython.display import display
from pandas import DataFrame, Series
from pd_explain.recommenders.configurations.configuration_base import ConfigurationBase
from pd_explain.recommenders.measures.attribute_interestingness_measure_base import AttributeInterestingnessMeasureBase
from pd_explain.recommenders.configurations.global_configuration import GlobalConfiguration
from pd_explain.recommenders.utils.listener_interface import ListenerInterface


class RecommenderBase(ABC, ListenerInterface):
    """
    Base class for all recommenders.
    Implements all of the logic that is common to all recommenders.
    """

    def __init__(self, configuration: ConfigurationBase, measures: List[AttributeInterestingnessMeasureBase]):
        self._measures = measures
        self._configuration = configuration
        self._global_configuration = GlobalConfiguration()
        self._global_configuration.add_listener(self)
        # If the global configuration has a configuration for this recommender, we will use that configuration.
        # Otherwise, the default configuration will be used.
        self._configuration.config = self._global_configuration.get_recommender_configuration(self.name)

    @property
    def config(self):
        """
        Gets the configuration of the recommender.
        """
        return self._configuration.config

    @config.setter
    def config(self, value: dict):
        """
        Sets the configuration of the recommender.

        :param value: The new configuration. Only the keys that are present in the configuration will be updated.
        """
        self._configuration.config = value

    @property
    def config_info(self):
        """
        Gets the configuration info of the recommender.
        This info is metadata, explaining the configuration options.
        """
        return self._configuration.config_info

    @property
    def global_config_values(self):
        """
        Gets the values of the global configuration.
        """
        return self._global_configuration.recommender_configurations

    @property
    def global_config(self):
        """
        Gets the global configuration.
        """
        return self._global_configuration

    def recommend(self, data: DataFrame) -> Tab:
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
        if self._configuration.attributes is None or len(self._configuration.attributes) == 0:
            # Compute the interestingness score for each attribute.
            scores = {}
            for measure in self._measures:
                scores[measure.__repr__()] = measure.compute_measure(data)
            # Condense the scores to a single array per attribute
            scores = self._condense_scores(scores)
            # Get the top-k skyline attributes
            attributes = self.top_k_skyline(scores=scores, top_k=self._configuration.top_k_attributes)

        else:
            attributes = self._configuration.attributes

        # Get the queries for each attribute
        queries = self._create_recommendation_candidates_internal(data=data, attributes=attributes)

        # If the data is too large, we want to avoid computing scores on possibly many queries on the entire data.
        # Therefore, we sample the data instead.
        sampled_data = self.sample_data(data)

        # Score each query
        query_scores = self._compute_recommendation_scores_internal(data=sampled_data, attributes=attributes,
                                                                    queries=queries,
                                                                    top_k=self._configuration.top_k_explanations)

        # Take the top-k recommendations for each attribute
        for attribute in attributes:
            query_scores[attribute] = query_scores[attribute][:self._configuration.top_k_recommendations]
            queries[attribute] = [query for query in queries[attribute] if
                                  query in query_scores[attribute].index.tolist()]

        # A hack, to make the tabs scrollable and not squish the content if the horizontal space is too small.
        display(
            HTML(
                """
                <style>
                .jupyter-widgets.widget-tab > .p-TabBar {
                    overflow-x: auto;
                    white-space: nowrap;
                }
                .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
                    flex: 0 0 auto;
                }
                </style>
                """
            )
        )

        return self._create_tab_internal(data=data, attributes=attributes, queries=queries,
                                         top_k_explanations=self._configuration.top_k_explanations)

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
    def _create_recommendation_candidates_internal(self, data: DataFrame, attributes: List[str]) -> Dict[str, List]:
        """
        Creates the recommendation candidates for each attribute.

        :param data: The data to recommend queries for.
        :param attributes: The attributes to recommend queries for.

        :return: The recommendation candidates for each attribute, in a dict with the attribute as key, and a list of
        recommendation candidates as values.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_recommendation_scores_internal(self, data: DataFrame, attributes: List[str],
                                                queries: Dict[str, List[str]], top_k: int) -> Dict[str, Series]:
        """
        Compute the scores for each recommendation candidate.

        :param data: The data to recommend queries for.
        :param attributes: The attributes to recommend queries for.
        :param queries: The queries to recommend.
        :param top_k: The number of top-k scores to consider. Corresponds to the max number of explanations to provide
        for each query.

        :return: The scores for each recommendation candidate, in a dict with the attribute as key, and a
        Series with the scores as values and the recommendation candidates as index.
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

    def on_event(self, config_changes: Dict[str, Dict[str, Any]]):
        """
        Called whenever the global configuration changes.
        Modifies the local configuration to match the global configuration.
        """
        # Outside dict is the recommender name, inside dict is the configuration.
        for key, value in config_changes.items():
            if key in self._global_configuration.registered_recommenders and key == self.name:
                self._configuration.config = value


    def __del__(self):
        self._global_configuration.remove_listener(self)
