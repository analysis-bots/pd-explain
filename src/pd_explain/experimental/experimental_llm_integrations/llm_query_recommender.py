import os
from typing import Dict

import pandas as pd

from pandas import DataFrame

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client
import pd_explain.llm_integrations.consts as consts

class LLMQueryRecommender(LLMIntegrationInterface):
    """
    LLMQueryRecommender is a class that generates queries for a given DataFrame using a large language model (LLM).
    It uses the LLM to generate queries based on the provided DataFrame and the history of queries.
    """

    def __init__(self, df: pd.DataFrame, df_name: str, history=None, user_requests=None, k=4):
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
                            f"performed by the user with their interestingness scores between 0 and 1. Higher is better. "
                            f"The user may also provide additional requests. "
                            f"Your goal is to generate queries that are interesting and relevant to the user. "
                            f"Try to make the queries varied, if possible. "
                            f"Try to avoid generating queries that are too similar to the ones already in the history, or only creating of queries one type (only groupbys, only filters, etc.). "
                            f"Note that even if one type of queries seems to dominate another (i.e. groupbys typically having higher scores than filters), you should still try to create a variety of queries of all types, and not "
                            f"just one type. "
                            f"Also note that the queries in the history may not be syntactically correct, so it is your responsibility to generate queries that are syntactically correct. ")
        return task_explanation


    def _create_context_explanation(self) -> str:
        """
        Create an explanation for the LLM, explaining the context of the DataFrame.
        This includes the columns, their types, and any other relevant information.
        """
        context_explanation = f"The DataFrame is named {self.df_name}. "
        context_explanation += f"The DataFrame has the following columns: {self.df.columns.tolist()}. "
        context_explanation += f"The column types are: {self.df.dtypes.to_dict()}. "
        context_explanation += f"The DataFrame has {self.df.shape[0]} rows and {self.df.shape[1]} columns. "
        # This line is commented out because it can take up too much of the context window, depending on the size of the DataFrame,
        # leading to the LLM not being able to generate any queries.
        # context_explanation += f"Using df.describe() we get the following statistics: {self.df.describe(include='all').to_dict()}."
        context_explanation += f"The history of the most recent queries is: {self.history}. "
        if self.user_requests:
            context_explanation += f"The user requests are: {self.user_requests}. These should be given the highest priority. "
        return context_explanation


    def _create_format_instructions(self) -> str:
        """
        Explain the expected format of the output to the LLM.
        """
        format_instructions = (f"The output should be a list of {self.k} queries. Denote each query with a * and a newline. "
                               f"The list of queries should be surrounded by <recs> and </recs> at the start and end respectively, so they can be easily extracted. "
                               f"Not abiding by this format may lead to errors and program crashes. "
                               f"The queries should be valid Pandas queries that can be run as-is. "
                               "If you use aggregation functions, use the function call format, e.g. df['column'].agg('mean'). On groupbys, you can also use the format df.groupby('column').agg({'column1': 'mean', 'column2': 'sum'}). "
                               "Other formats are not allowed and may lead to errors. "
                               "Try to avoid only using a small subset of the columns (for example, selecting 3 or less columns or aggregating a small number of columns) unless the user explicitly requests it. "
                               "Always opt for having at-least 4 columns in the query's output if possible. "
                               "You may use filter, groupby, and join operations, plus any valid aggregation function. "
                               "The following function and their like must NEVER be used: head, tail, sample, iloc, loc, at, iat, reset_index, set_index, merge, concat, "
                               "and any other function that is not expressly part of a query that may make sense in a SQL context.  ")
        return format_instructions


    def do_llm_action(self, system_messages: list[str] = None,
                      user_messages: list[str] = None, assistant_messages: list[str] = None) -> pd.Series | None:
        """
        Generate queries for the DataFrame using the LLM.
        If the optional parameters system_messages, user_messages, and assistant_messages are provided,
        they will be used instead of the default ones.

        :param system_messages: System messages to be sent to the LLM. Optional.
        :param user_messages: User messages to be sent to the LLM. Optional.
        :param assistant_messages: Assistant messages which were sent by the LLM. Optional.

        :return: A Series of generated queries or None if no queries are generated.
        """
        client = Client(
            api_key=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONiNG_LLM_KEY),
            model=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_MODEL, consts.DEFAULT_REASONING_LLM_MODEL),
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER, consts.DEFAULT_REASONING_LLM_PROVIDER),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL, consts.DEFAULT_REASONING_LLM_PROVIDER_URL)
        )
        if system_messages is None:
            system_messages = [self._create_task_explanation()]
        if user_messages is None:
            context_explanation = self._create_context_explanation()
            format_instructions = self._create_format_instructions()
            user_messages = [context_explanation + format_instructions]
        response = client(
            system_messages=system_messages,
            user_messages=user_messages,
            assistant_messages=assistant_messages
        )
        recommendations = self._extract_response(response, "<recs>", "</recs>")
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


    def do_follow_up_action(self, response: pd.Series) -> Dict[str, DataFrame]:
        """
        Use the generated queries to query the DataFrame, and return the results.
        """
        results = {}
        for query in response:
            query_fixed = query.replace(self.df_name, "self.df")
            results[query] = {}
            try:
                # We use eval to run the query on the DataFrame.
                result = eval(f"{query_fixed}")
                results[query]["result"] = result
                results[query]["error"] = False
            except Exception as e:
                # Put the exception in the results dictionary.
                results[query]["result"] = e
                results[query]["error"] = True
                continue
        return results



