import numpy as np
from pandas import DataFrame, Series

from pd_explain.experimental.query_recommenders.query_recommender_interface import QueryRecommenderInterface
from pd_explain.experimental.experimental_llm_integrations.llm_query_recommender import LLMQueryRecommender
from pd_explain.experimental.experimental_llm_integrations.query_refiner import QueryRefiner
from pd_explain.experimental.query_recommenders.query_logger import QueryLogger
from pd_explain.experimental.query_recommenders.query_score_functions import score_queries

from fedex_generator.Operations.Filter import Filter

class LLMBasedQueryRecommender(QueryRecommenderInterface):
    """
    LLMQueryRecommender is a class that generates queries for a given DataFrame using a large language model (LLM).
    It uses the LLM to generate queries based on the provided DataFrame and the history of queries.
    It then refines the queries using the LLM.
    """

    def __init__(self, df, df_name, user_requests=None, k=4, n=2, return_all_options: bool = False):
        """
        Initialize the LLMQueryRecommender with a DataFrame and an optional history of queries.

        :param df: The DataFrame to generate queries for.
        :param df_name: The name of the DataFrame.
        :param user_requests: An optional list of user requests.
        :param k: The number of queries to generate.
        :param n: The number of refining iterations to perform.
        :param return_all_options: If True, returns all options instead of just the top k.
        """
        self.df = df
        self.df_name = df_name
        self.user_requests = user_requests if user_requests is not None else []
        self.k = k
        self.n = n
        self.return_all_options = return_all_options


    def _score_query(self, query: str, query_result: DataFrame) -> tuple[dict, float]:
        """
        Score the query based on its result.
        """
        # If the query result is not a DataFrame or Series, or if it is empty, return an empty score.
        # The scenario where it is not a DF or series may be when it filters down to a single value, and the result is a scalar.
        if not (isinstance(query_result, DataFrame) or isinstance(query_result, Series)) or query_result.empty:
            return {}, 0
        # If a query is not a groupby or a join, its a filter
        if "groupby" not in query and "join" not in query:
            operation = Filter(source_df=self.df, result_df=query_result, source_scheme={})
        elif "groupby" in query:
            # For groupbys, we know for sure that we can just extract it since it will for sure by a ExpDataFrame
            # or ExpSeries.
            # With filters, we can't be 100% sure since we didn't override every single method in DataFrame, and some
            # of the Pandas internals might convert to a DataFrame or Series. I encountered one such case with value_counts,
            # but there may be more.
            operation = query_result.operation
        elif "join" in query:
            operation = query_result.operation

        try:
            scores = operation.explain(top_k=4, measure_only=True)
            score = score_queries(scores)
        except Exception as e:
            scores = {"Scoring Error": str(np.nan)}
            score = 0

        return scores, score



    def recommend(self) -> DataFrame:
        """
        Generate queries for the DataFrame using the LLM.
        """
        logger = QueryLogger()
        # Get the last queries done on the dataframe
        history = logger.get_log(dataframe_name=self.df_name, k=40)
        recommender = LLMQueryRecommender(self.df, self.df_name, history=history, user_requests=self.user_requests, k=self.k)
        # Apply the generated queries to the dataframe then score them
        recommendations = recommender.do_llm_action()
        print("Finished generating initial recommendations.")
        if self.n == 0:
            applied_recommendations = recommender.do_follow_up_action(recommendations)
            scores = []
            for query, query_result in applied_recommendations.items():
                if query_result["error"]:
                    scores.append(0)
                    continue
                _, score = self._score_query(query, query_result["result"])
                scores.append(score)
            recommendations = DataFrame({"query": recommendations, "score": scores})
            recommendations = recommendations.sort_values(by="score", ascending=False).reset_index(drop=True)
            return recommendations
        refiner = QueryRefiner(self.df, self.df_name, recommendations, score_function=self._score_query, k=self.k, n=self.n,
                               user_requests=self.user_requests, return_all_options=self.return_all_options)
        refined_recommendations = refiner.do_llm_action()
        return refined_recommendations

