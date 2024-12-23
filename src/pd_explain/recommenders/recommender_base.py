from abc import ABC, abstractmethod
from typing import List


class RecommenderBase(ABC):

    @abstractmethod
    def recommend(self) -> List:
        """
        Recommends a list of items based on the query.

        :return: A list of recommendations.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the recommender.

        :return: The name of the recommender.
        """
        raise NotImplementedError