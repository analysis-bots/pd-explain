import pandas as pd

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client

class ExplanationReasoning(LLMIntegrationInterface):


    def __init__(self, data: pd.DataFrame | pd.Series, source_name: str):
        self._data = data
        self._data_description = data.describe()
        self._data_columns = data.columns if isinstance(data, pd.DataFrame) else data.name
        self._data_type = "dataframe" if isinstance(data, pd.DataFrame) else "series"
        self._source_name = source_name

    def explain(self) -> str:

        client = Client()
        response = client(
            system_messages=["Our system has automatically analyzed a user's query over their data and found some interesting statistical insights. Your task is to reason about the insights and explain why they occur, so they can be presented to the user alongside the insight."],
            user_messages=[
                "The user has performed the query SELECT * FROM adults WHERE class = ' >50k'. The insights found are as follows:"
                "1. marital-status value 'Never-married' appears 5 times less than before"
                "2. relationship value 'Own-child' appears 16 times less than before"
                "3. education-num values above 13 appear 3 times more than before"
                "4. age values between 16 and 22 appear 52 times less than before"
                "Provide a short explanation for why each of these insights occur. The explanations should be in a numbered list format, with each explanation corresponding to the insight number."
                "Surround the list with @@@@@@@@@ to separate it from the rest of the message, and so it can be easily identified by the program."
                "Do not repeat the insight in the explanation, as it will already be included in the final output."
            ]
        )
        return response