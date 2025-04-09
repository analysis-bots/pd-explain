from typing import List

import pandas as pd
import re

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client

class LLMQueryRecommender(LLMIntegrationInterface):
    """
    LLMQueryRecommender is a class that generates queries for a given DataFrame using a large language model (LLM).
    It uses the LLM to generate queries based on the provided DataFrame and the history of queries.
    """

    def __init__(self, df, df_name, history=None, user_requests=None, k=3):
        """
        Initialize the LLMQueryRecommender with a DataFrame and an optional history of queries.

        :param df: The DataFrame to generate queries for.
        :param history: An optional history of queries.
        """
        self.df = df
        self.df_name = df_name
        self.history = history if history is not None else []
        self.user_requests = user_requests if user_requests is not None else []
        self.k = k


    def _create_task_explanation(self) -> str:
        """
        Create an explanation for the LLM, explaining the task it needs to perform.
        This is a general explanation that can be used for any DataFrame.
        """
        task_explanation = (f"You are a query recommender for a Pandas DataFrame. "
                            f"Your task is to generate interesting queries for the DataFrame. "
                            f"You will be provided with some context about the DataFrame, as well as a history of queries"
                            f"performed by the user with their interestingness scores. "
                            f"The user may also provide additional requests. "
                            f"Your goal is to generate queries that are interesting and relevant to the user.")
        return task_explanation


    def _create_context_explanation(self) -> str:
        """
        Create an explanation for the LLM, explaining the context of the DataFrame.
        This includes the columns, their types, and any other relevant information.
        """
        context_explanation = f"The DataFrame is named {self.df_name}. "
        context_explanation += f"The DataFrame has the following columns: {self.df.columns.tolist()}. "
        context_explanation += f"The column types are: {self.df.dtypes.to_dict()}. "
        context_explanation += f"Using df.describe() we get the following statistics: {self.df.describe(include='all').to_dict()}."
        context_explanation += f"The history of the most recent queries is: {self.history}. "
        if self.user_requests:
            context_explanation += f"The user requests are: {self.user_requests}. "
        return context_explanation


    def _create_format_instructions(self) -> str:
        """
        Explain the expected format of the output to the LLM.
        """
        format_instructions = (f"The output should be a list of {self.k} queries. Denote each query with a * and a newline. "
                               f"The list of queries should be surrounded by @@@@@@@@ both at the beginning and the end, so they can be easily extracted. "
                               f"The queries should be valid Pandas queries that can be run as-is. "
                               f"If you use aggregation functions, do not use the .aggregate() or .agg() methods, but instead use the aggregation functions directly, i.e. call mean(), sum(), etc. ")
        return format_instructions


    def do_llm_action(self) -> pd.Series | None:
        """
        Generate queries for the DataFrame using the LLM.

        :return: A Series of generated queries or None if no queries are generated.
        """
        client = Client()
        task_explanation = self._create_task_explanation()
        context_explanation = self._create_context_explanation()
        format_instructions = self._create_format_instructions()
        user_messages = [context_explanation + format_instructions]
        response = client(
            system_messages=[task_explanation],
            user_messages=user_messages,
        )
        recommendations = self._extract_response(response, "@")
        if recommendations is None or len(recommendations) <= 1:
            return None
        recommendations = recommendations.split("\n")
        recommendations = [x.strip() for x in recommendations if x.strip()]
        recommendations = [x[1:] for x in recommendations if x.startswith("*")]
        # We do a strip again to remove any leading or trailing whitespace after removing the "*".
        recommendations = [x.strip() for x in recommendations if x.strip()]
        # And finally, we remove any empty strings from the list.
        recommendations = [x for x in recommendations if len(x) > 0]
        # We return the recommendations as a pandas Series.
        recommendations = pd.Series(recommendations)
        recommendations.name = "recommendations"
        return recommendations


    def do_follow_up_action(self, response: pd.Series) -> List[pd.DataFrame] | None:
        """
        Use the generated queries to query the DataFrame, and return the results.
        """
        results = []
        for query in response:
            if query.startswith(self.df_name):
                # We remove the dataframe name from the query, because we want to run the query as-is on the DataFrame.
                query = query.replace(self.df_name, "")
            try:
                # We use eval to run the query on the DataFrame.
                result = eval(f"self.df{query}")
                results.append(result)
            except Exception as e:
                print(f"Error running query {query}: {e}")
                continue
        if len(results) == 0:
            return None
        return results



