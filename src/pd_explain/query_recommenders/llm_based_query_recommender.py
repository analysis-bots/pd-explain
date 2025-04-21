import numpy as np
from pandas import DataFrame, Series

from pd_explain.query_recommenders.query_recommender_interface import QueryRecommenderInterface
from pd_explain.llm_integrations.llm_query_recommender import LLMQueryRecommender
from pd_explain.llm_integrations.query_refiner import QueryRefiner
from pd_explain.query_recommenders.query_logger import QueryLogger
from pd_explain.query_recommenders.query_score_functions import score_queries

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
        This is a placeholder method and should be implemented based on the specific scoring criteria.
        """
        if isinstance(query_result, dict):
            error = query_result.get("error", None)
            if error:
                return {"error": error}, 0
            query_result = query_result.get("result", None)
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
            print(f"An error occurred while scoring query {query}: {e}. \n"
                  f"This is not intentional, so we would appreciate it if you could report this issue at "
                  f"https://github.com/analysis-bots/pd-explain")

        return scores, score



    def recommend(self) -> DataFrame:
        """
        Generate queries for the DataFrame using the LLM.
        """
        logger = QueryLogger()
        # Get the last queries done on the dataframe
        history = logger.get_log(dataframe_name=self.df_name)
        recommender = LLMQueryRecommender(self.df, self.df_name, history=history, user_requests=self.user_requests, k=self.k)
        # Apply the generated queries to the dataframe then score them
        recommendations = recommender.do_llm_action()
        applied_recommendations = recommender.do_follow_up_action(recommendations)
        recommendations = {}
        for query, query_result in applied_recommendations.items():
            # Score the query
            scores, score = self._score_query(query, query_result)
            recommendations[query] = {
                "query_result": query_result["result"],
                "score_dict": scores,
                "score": score
            }
        # Sort the recommendations by score
        recommendations = {k: v for k, v in sorted(recommendations.items(), key=lambda item: -item[1]["score"])}
        print("Finished generating initial recommendations.")
        # Refine the recommendations
        if self.n == 0:
            return_df = DataFrame.from_dict(recommendations, orient='index')
            return_df = return_df[["score"]]
            return_df = return_df.sort_values(by="score", ascending=False)
            return_df.index.name = "query"
            return_df.reset_index(inplace=True)
            return_df.rename(columns={"query": "query"}, inplace=True)
            return return_df
        refiner = QueryRefiner(self.df, self.df_name, recommendations, score_function=self._score_query, k=self.k, n=self.n,
                               user_requests=self.user_requests, return_all_options=self.return_all_options)
        refined_recommendations = refiner.do_llm_action()
        return refined_recommendations

