import pandas as pd
import re

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

    def explain(self) -> pd.Series | None:

        client = Client()
        # Create the system and user messages
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
            f"Do not repeat the insight in the explanation, as it will already be included in the final output. "
            f"Do not add anything after the closing @@@@@@@@@, as it will hurt the program's ability to extract the explanations. "
            f"Do not start the explanation with sayings like 'this insight' or 'this occurs', as that is redundant. "
            f"If you include numbers in the explanation, if they are integers, make sure they are not followed by a period. If they are decimals, only use them if they will be followed by a digit, for example 19.9 or 15.0"
        ]
        response = client(
            system_messages=system_messages,
            user_messages=user_messages
        )
        # Extract the explanations from the response.
        # We use a regex to find the text between the @ symbols.
        # Regex means: Find any text that starts with one or more @ symbols, followed by any string of characters that is not an @ symbol, and ends with one or more @ symbols.
        pattern = r"@[^@]+@"
        explanations = re.findall(pattern, response)
        # If we have no explanations, we return an empty string.
        if len(explanations) == 0:
            return None
        # We only expect one explanation result from the regex, so we take the first one.
        explanation = explanations[0]
        explanation = explanation.replace("@", "").strip()
        # Split the explanation into an array of explanations, with the indexes matching the explanations found and provided
        # as a numbered list to the model and by the model.
        # Regex means: Find any string of one or more digits, followed by a period, and not followed by more digits (to avoid matching decimal points).
        pattern = r"\d+\.[^\d]"
        explanations = re.split(pattern, explanation)
        explanations = [explanation.strip() for explanation in explanations]
        # General cleanup of the explanations by assuming that any explanation with less than 5 characters is noise mistakenly extracted.
        explanations = [explanation for explanation in explanations if len(explanation) > 5]

        if len(explanations) < len(self._explanations_found):
            # If we got less explanations than we should, add empty strings to match the expected number of explanations.
            explanations += [""] * (len(self._explanations_found) - len(explanations))

        explanations = pd.Series(explanations, index=self._explanations_found.index)

        return explanations