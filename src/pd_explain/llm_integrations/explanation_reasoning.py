import pandas as pd

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client

class ExplanationReasoning(LLMIntegrationInterface):


    def __init__(self, data: pd.DataFrame | pd.Series, source_name: str, query: str, explanations_found: list[str],
                 right_df: pd.DataFrame | pd.Series):
        self._data = data
        self._data_description = data.describe()
        self._data_columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]
        self._data_type = "dataframe" if isinstance(data, pd.DataFrame) else "series"
        self._source_name = source_name
        self._query = query
        self._explanations_found = explanations_found
        if right_df is not None:
            self._right_df = right_df
            self._right_df_description = right_df.describe()
            self._right_df_columns = right_df.columns if isinstance(right_df, pd.DataFrame) else [right_df.name]
            self._right_df_type = "dataframe" if isinstance(right_df, pd.DataFrame) else "series"


    def _explanations_to_numbered_list(self):
        return "\n".join([f"{i+1}. {explanation}" for i, explanation in enumerate(self._explanations_found)])

    def explain(self) -> str:

        client = Client()
        system_messages = [
            f"Our system automatically analyzes a user's query over their data and finds some interesting statistical insights. Your task is to reason about the insights and explain why they occur, so they can be presented to the user alongside the insight."
        ]
        user_messages = [
            f"The user has performed the query {self._query} on dataset {self._source_name}. "
            f"The dataset is a {self._data_type} with the following columns: {', '.join(self._data_columns)}. "
            f"Using pd.describe() on the dataset, we find the following statistics: {self._data_description}. "
            f"The insights found are as follows:\n"
            f"{self._explanations_to_numbered_list()}\n"
            f"Provide a short, 1-sentence explanation, reasoning about why each of these insights occur. "
            f"The explanations should be in a numbered list format, with each explanation corresponding to the insight number. "
            f"Surround the list with @@@@@@@@@ to separate it from the rest of the message, and so it can be easily identified by the program. "
            f"Do not repeat the insight in the explanation, as it will already be included in the final output."
        ]
        response = client(
            system_messages=system_messages,
            user_messages=user_messages
        )
        return response