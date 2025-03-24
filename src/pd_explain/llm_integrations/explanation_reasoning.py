import pandas as pd
import re
import textwrap

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client

class ExplanationReasoning(LLMIntegrationInterface):


    def __init__(self, data: pd.DataFrame | pd.Series, source_name: str, query: str, explanations_found: list[str],
                 right_df: pd.DataFrame | pd.Series, query_type: str):
        self._data = data
        self._data_description = data.describe()
        self._data_columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]
        self._data_type = "dataframe" if isinstance(data, pd.DataFrame) else "series"
        self._source_name = source_name
        self._query = query
        self._explanations_found = explanations_found
        self._query_type = query_type
        if right_df is not None:
            self._right_df = right_df
            self._right_df_description = right_df.describe()
            self._right_df_columns = right_df.columns if isinstance(right_df, pd.DataFrame) else [right_df.name]
            self._right_df_type = "dataframe" if isinstance(right_df, pd.DataFrame) else "series"


    def _explanations_to_numbered_list(self):
        # clean up the explanations of bold and italic formatting
        explanations_found = [re.sub(r"[${}]|(\\bf)", "", explanation) for explanation in self._explanations_found]
        return "\n".join([f"{i+1}. {explanation}" for i, explanation in enumerate(explanations_found)])

    def explain(self) -> pd.Series | None:

        client = Client()
        # Create the system and user messages
        system_messages = [
            f"Our system automatically analyzes a user's query over their data and finds some interesting statistical insights. "
            f"You have been provided with the insights found, as well as the specific query and some information about the dataset. "
            f"Your task is to reason and explain, using your domain knowledge, why these insights occur in the context of the query. "
        ]
        explanation_format_explanation = ""
        if self._query_type == "groupby":
            explanation_format_explanation = (f"The insights on groupby operations are of the form 'groups with property = x has property y z standard deviations from the mean'. "
                                              f"Your explanations should reason about why these groups with this specific property x display this large deviation from the mean in property y (and not other groups). "
                                              f"Your explanation must always revolve around why property x causes this deviation in property y, and not the other way around. "
                                              f"It should also always explicitly refer to the group with property x.")
        elif self._query_type == "filter":
            explanation_format_explanation = (f"The insights on filter operations are of the form 'property x value y appears z times more / less than before'. "
                                              f"Your explanations should reason about why this this specific value y of property x appears z times more / less than before. "
                                              f"Your explanation must always revolve around why this specific value y appears z times more / less than before, and not the other way around. ")

        user_messages = [
            f"The user has performed the query {self._query} on dataset {self._source_name}. "
            f"The dataset is a {self._data_type} with the following columns: {', '.join(self._data_columns)}. "
            f"{explanation_format_explanation}"
            f"The insights found are as follows:\n"
            f"{self._explanations_to_numbered_list()}\n"
            f"Provide a short, 1-sentence explanation, reasoning about why each of these insights occur. "
            f"The explanation should always be in the context of the query, and not a general explanation. "
            f"It should also always refer to the insight. For example, if the insight on a groupby includes 'group = some_group', it should explicitly refer to 'some_group', or if the insight on a filter includes 'x value y', it should always refer to 'y'."
            f"The explanations should be in a numbered list format, with each explanation corresponding to the insight number. "
            f"Surround the list with @@@@@@@@@ to separate it from the rest of the message, and so it can be easily identified by the program. "
            f"While you need to explicitly refer to the insight's details, do not quote the insight verbatim. "
            f"Do not start the explanation with sayings like 'this insight' or 'this occurs', as that is redundant. "
        ]
        response = client(
            system_messages=system_messages,
            user_messages=user_messages
        )
        # Client can return None if no API key is set.
        if response is None:
            return None
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
        explanations = ["LLM: " + explanation for explanation in explanations if len(explanation) > 5]
        explanations = ["\n".join(textwrap.wrap(explanation, width=50)) for explanation in explanations]

        if len(explanations) < len(self._explanations_found):
            # If we got less explanations than we should, add empty strings to match the expected number of explanations.
            explanations += [""] * (len(self._explanations_found) - len(explanations))

        explanations = pd.Series(explanations, index=self._explanations_found.index)

        return explanations