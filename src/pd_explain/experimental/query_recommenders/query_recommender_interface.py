from abc import ABC, abstractmethod
from pandas import DataFrame


class QueryRecommenderInterface(ABC):
    """
    Abstract base class for query recommenders.
    """

    @abstractmethod
    def recommend(self) -> DataFrame:
        """
        Recommend a query based on the input dataframe and user requests.
        """
        raise NotImplementedError()