from typing import List

from pd_explain.recommenders.recommender_base import RecommenderBase

class FilterRecommender(RecommenderBase):

    __name__ = 'FilterRecommender'

    @property
    def name(self) -> str:
        return self.__name__

    def recommend(self) -> List:
        raise NotImplementedError
