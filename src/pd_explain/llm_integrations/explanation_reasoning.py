import pandas as pd
import re
import textwrap
import os

from numba.scripts.generate_lower_listing import description

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client
from pd_explain.llm_integrations import consts


class ExplanationReasoning(LLMIntegrationInterface):
    """
    This class is responsible for creating reasoning based explanations for the typically statistical insights found by the
    other explainers.
    It uses a LLM to generate explanations that utilize the LLM's reasoning capabilities and domain knowledge.
    """

    def __init__(self, data: pd.DataFrame | pd.Series, source_name: str,  explanations_found: pd.DataFrame | pd.Series,
                 query_type: str, query: str = None, right_df: pd.DataFrame | pd.Series = None, right_name: str = None,
                 labels: pd.Series = None, target: str = None, dir: str = None, after_op_data: pd.DataFrame = None):
        self._data = data
        self._data_columns = data.columns if isinstance(data, pd.DataFrame) else [data.name]
        self._data_type = "dataframe" if isinstance(data, pd.DataFrame) else "series"
        self._source_name = source_name
        self._query = query
        self._explanations_found = explanations_found
        self._query_type = query_type
        # These are all parameters that are only required for specific query types, so they are optional.
        # If they are not provided, they should cause an exception if they are accessed.
        if right_df is not None:
            self._right_df = right_df
            self._right_df_columns = right_df.columns if isinstance(right_df, pd.DataFrame) else [right_df.name]
            self._right_df_type = "dataframe" if isinstance(right_df, pd.DataFrame) else "series"
            self._right_source_name = right_name
        if labels is not None:
            self._labels = labels
        if target is not None:
            self._target = target
        if dir is not None:
            self._dir = dir
        if after_op_data is not None:
            self._after_op_data = after_op_data


    def _explanations_to_list_string(self) -> str:
        """
        Transform the explanations found a string in a list format.
        :return: The explanations in a numbered list format.
        """
        if self._query_type in ["join", "filter", "groupby"]:
            # clean up the explanations of bold and italic LaTeX formatting
            explanations_found = [re.sub(r"[${}]|(\\bf)", "", explanation) for explanation in self._explanations_found]
            return "\n".join([f"{i + 1}. {explanation}" for i, explanation in enumerate(explanations_found)])
        elif self._query_type == "many_to_one":
            idx = 1
            out_string = ""
            for row in self._explanations_found.iterrows():
                index = row[0]
                row = row[1]
                group = index[0]
                explanation = index[1]

                out_string += (f"{idx}. "
                               f"Group / Cluster: {group}, Explanation: {explanation}, "
                               f"Coverage: {row['Coverage']}, Separation Error: {row['Separation Error']}")
                if 'Separation Error Origins' in self._explanations_found.columns:
                    out_string += f", Separation Error Origins: {row['Separation Error Origins']}"
                out_string += "\n"
                idx += 1
            return out_string
        elif self._query_type == "outlier":
            out = re.sub(r"[${}]|(\\bf)", "", self._explanations_found)
            return out.replace("\n", ' AND ')
        elif self._query_type == "metainsight":
            # MetaInsight explanations are already in a list format, so we just join them.
            return "\n\n".join(self._explanations_found)


    def _create_explanation_format_explanation(self) -> str:
        """
        Create the string that explains the format of the explanations to the LLM, and what is expected from it in
        terms of reasoning.
        """
        explanation_format_explanation = ""
        if self._query_type == "groupby":
            explanation_format_explanation = (
                f"The insights on groupby operations are of the form 'groups with property = x has property y z standard deviations from the mean'. "
                f"Your explanations should reason about why these groups with this specific property x display this large deviation from the mean in property y (and not other groups). "
                f"Your explanation must always revolve around why property x causes this deviation in property y, and not the other way around. "
                f"It should also always explicitly refer to the group with property x."
                f"Your explanations should not be statistical, but should be based on domain knowledge and reasoning. ")
        elif self._query_type == "filter" or self._query_type == "join":
            explanation_format_explanation = (
                f"The insights on {self._query_type} operations are of the form 'property x value y appears z times more / less than before'. "
                f"Your explanations should reason about why this this specific value y of property x appears z times more / less than before. "
                f"Your explanation must always revolve around why this specific value y appears z times more / less than before, and not the other way around. "
                f"Your explanations should not be statistical, but should be based on domain knowledge and reasoning. ")
        elif self._query_type == "outlier":
            explanation_format_explanation = (f"The outlier explainer creates an explanation of the form 'the outlier is not as significant when excluding rows with property x = y' on series that are the result of a groupby operation."
                                              f"Your explanation should reason about why the outlier is not as significant when excluding rows with property x = y. "
                                              f"Your explanation must always revolve around why the outlier is not as significant when excluding rows with property x = y, and not the other way around. "
                                              f"It should never be some general explanation, the sorts of 'x is not as significant when excluding y because x is more / less prevalent in the data when excluding y'"
                                              f" - actually reason and think about why the outlier is not as significant when excluding rows with property x = y. "
                                              f"Your explanation should not be statistical, but should be based on domain knowledge and reasoning. ")
        elif self._query_type == "many_to_one":
            explanation_format_explanation = (
                f"The many-to-one explanainer creates a table with the columns: 'Group / Cluster', 'Explanation', 'Coverage', 'Separation Error', and possibly also 'Separation Error Origins'. "
                f"The 'Group / Cluster' column contains the group or cluster that the explanation is about. "
                f"The 'Explanation' column contains the predicate that was found to give a good explanation for the group. "
                f"The 'Coverage' column contains the percentage of the group that is covered by the explanation. "
                f"The 'Separation Error' column contains the separation error - how much of data not in the group is covered by the explanation. "
                f"The 'Separation Error Origins' column contains the columns that contributed to the separation error. "
                f"Each group or cluster may have multiple explanations, or may have no explanation at all. If there is no explanation, try and explain why that is the case, but make sure to use words like 'may', 'could', 'might', etc. when doing so. "
                f"Your explanations should reason about why the predicate in the 'Explanation' column gives a good explanation for the group or cluster in the 'Group / Cluster' column. "
                f"If the separation error is high, you should also try to explain that, using the columns in the 'Separation Error Origins' column (if it exists). "
                f"Your explanation must always revolve around why the predicate in the 'Explanation' column gives a good explanation for the group or cluster, and not the other way around. "
                f"Your explanations should not be statistical, but should be based on domain knowledge and reasoning. ")
        elif self._query_type == "metainsight":
            explanation_format_explanation = (
                "The MetaInsight explainer finds interesting patterns in the data, as well as exceptions to these patterns.\n"
                "The explanations are of the form 'Common pattern {x} detected for over {y}% of values of {z}, when grouping by "
                "{a} and aggregating by {b}. Exceptions in category {c}: {d} with values {e} = {f}, {g} = {h}'\n"
                "There are three types of exceptions: 1) Highlight change - this means that the pattern is common, but the value is different from the expected value. "
                "2) Type change - the value is part of a different pattern than the common pattern. 3) No pattern - no pattern was found for this value.\n"
                "Your explanation should reason about why the common pattern {x} is detected for over {y}% of values of {z}, "
                "and why the various exceptions of each category occur. "
            )
        else:
            raise ValueError(
                "Unrecognized query type. This may have happened if you added a new operation to Fedex, or a new explainer to pd_explain, without updating this method.")
        return explanation_format_explanation

    def _create_data_and_query_description(self):
        if self._query_type in ['filter', 'groupby']:
            description = (f"The user has performed the query {self._query} on dataset {self._source_name}. "
                           f"The dataset is a {self._data_type} with the following columns: {', '.join(self._data_columns)}.")
        elif self._query_type == "join":
            description = (
                f"The user has performed the query {self._query} on datasets {self._source_name} and {self._right_source_name}. "
                f"The dataset {self._source_name} is a {self._data_type} with the following columns: {', '.join(self._data_columns)}. "
                f"The dataset {self._right_source_name} is a {self._right_df_type} with the following columns: {', '.join(self._right_df_columns)}. "
                f"Using pd.describe() on the dataset, we find the following statistics:\n"
                )
        elif self._query_type == 'many_to_one':
            description = (f"The user has requested a many to one explanation on dataset {self._source_name}. "
                           f"This explainer creates logical explanations for the provided labels which define a many-to-one relationship with the rest of the data, using the provided data. "
                           f"The dataset is a {self._data_type} with the following columns: {', '.join(self._data_columns)}. "
                           f"The labels + their counts are as follows:\n"
                           f"{self._labels.value_counts()}\n"
                           f"Using pd.describe() on the dataset, we find the following statistics:\n"
                           )
        elif self._query_type == 'outlier':
            description = (f"The user has performed the query {self._query} on dataset {self._source_name}. "
                           f"The dataset is a {self._data_type} with the following columns: {', '.join(self._data_columns)}. "
                           f"The dataset after the groupby and aggregation is:\n"
                           f"{self._after_op_data}:\n"
                           f"Using pd.describe() on the source dataset, we find the following statistics:\n"
                           f"They requested analysis of the outlier value {self._target}, suspecting it to be an outlier in the direction {self._dir}. ")
        elif self._query_type == 'metainsight':
            description = (f"The user has requested a MetaInsight explanation on dataset {self._source_name}. "
                           f"This explainers finds interesting patterns in the data, as well as exceptions to to these patterns. \n"
                           f"The dataset is a {self._data_type} with the following columns: {', '.join(self._data_columns)}. "
                           )
            if self._query is not None:
                description += (f"The user has requested the MetaInsight explainer after performing the query {self._query} on the dataset. "
                                f"If the query is a groupby operation, the explainer will find patterns that emerge in the original dataset specifically from the groupby operation. "
                                f"If it is a filter or a join operation, the explainer will find patterns in the dataset after the operation. ")
            description += (f"Using pd.describe() on the dataset, we find the following statistics:\n"
                            f"{self._data.describe()}\n"
                            )
        else:
            raise ValueError(
                "Unrecognized query type. This may have happened if you added a new operation to Fedex, or a new explainer to pd_explain, without updating this method.")

        return description

    def _create_output_format_explanation(self):
        if self._query_type in ["join", "filter", "groupby", "metainsight"]:
            output_format_explanation = (
                f"The explanations should be in a numbered list format, with each explanation corresponding to the insight number. "
                f"Surround the list with <reasoning> and </reasoning> at the start and end respectively to separate it from the rest of the message, and so it can be easily identified by the program. "
                f"Make sure there is <reasoning> and </reasoning> at the start and end respectively, or you may crash the program and cause the poor developers to cry and lose their sanity. ")
        elif self._query_type == "many_to_one":
            output_format_explanation = (f"The explanations should be in a numbered list format, with the numbers matching what you were provided with. "
                                         f"Surround the list with <reasoning> and </reasoning> at the start and end respectively to separate it from the rest of the message, and so it can be easily identified by the program. "
                                         f"Make sure there is <reasoning> and </reasoning> at the start and end respectively, or you may crash the program and cause the poor developers to cry and lose their sanity. ")
        elif self._query_type == "outlier":
            output_format_explanation = (f"The explanation should be a single sentence,"
                                         f" surrounded by <reasoning> and </reasoning> at the start and end respectively to separate it from the rest of the message, and so it can be easily identified by the program. "
                                         f"Make sure there is <reasoning> and </reasoning> at the start and end respectively, or you may crash the program and cause the poor developers to cry and lose their sanity. ")
        else:
            raise ValueError("Unrecognized query type. This may have happened if you added a new operation to Fedex, or a new explainer to pd_explain, without updating this method.")

        return output_format_explanation

    def do_llm_action(self) -> pd.Series | None:
        client = Client(
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER, consts.DEFAULT_REASONING_LLM_PROVIDER),
            model=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_MODEL, consts.DEFAULT_REASONING_LLM_MODEL),
            api_key=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONiNG_LLM_KEY, None),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL, consts.DEFAULT_REASONING_LLM_PROVIDER_URL),
        )
        # Create the system and user messages
        system_messages = [
            f"Our system automatically analyzes a user's query over their data and finds some interesting statistical insights. "
            f"You have been provided with the insights found, as well as the specific query and some information about the dataset. "
            f"Your task is to reason and explain, using your domain knowledge, why these insights occur in the context of the query. "
        ]
        data_desciption_query_description = self._create_data_and_query_description()
        explanation_format_explanation = self._create_explanation_format_explanation()
        output_format_explanation = self._create_output_format_explanation()

        user_messages = [
            (f"{data_desciption_query_description}\n"
             f"{explanation_format_explanation}\n"
             f"The insights found are as follows:\n"
             f"{self._explanations_to_list_string()}\n"
             f"Provide a short, 1-sentence explanation, reasoning about why each of these insights occur. \n"
             f"The explanation should always be in the context of the query, and not a general explanation. \n"
             f"It should also always refer to the insight. For example, if the insight on a groupby includes 'group = some_group', it should explicitly refer to 'some_group', or if the insight on a filter includes 'x value y', it should always refer to 'y'.\n"
             f"{output_format_explanation}\n"
             f"While you need to explicitly refer to the insight's details, do not quote the insight verbatim. \n"
             f"Do not start the explanation with sayings like 'this insight', 'the insight' or 'this occurs', as that is redundant. \n"
             f"If you include decimals in the explanation, make sure they always have a number after the decimal point even if they could be an integer otherwise. I.e. do not write 10., but instead write 10.0.")
        ]
        response = client(
            system_messages=system_messages,
            user_messages=user_messages
        )
        # Client can return None if no API key is set.
        if response is None:
            return None
        # Extract the explanations from the response.
        explanation = self._extract_response(response, "<reasoning>", "</reasoning>")
        # Split the explanation into an array of explanations, with the indexes matching the explanations found and provided
        # as a numbered list to the model and by the model.
        # Regex means: Find any string of one or more digits, followed by a period, and not followed by more digits (to avoid matching decimal points).
        pattern = r"\d+\.[^\d]"
        explanations = re.split(pattern, explanation)
        explanations = [explanation.strip() for explanation in explanations]
        # General cleanup of the explanations by assuming that any explanation with less than 5 characters is noise mistakenly extracted.
        explanations = [explanation for explanation in explanations if len(explanation) > 5]
        # Explanations that are visualized in a plot are wrapped to fit the plot width. Otherwise, they are not wrapped.
        if self._query_type != "many_to_one":
            explanations = ["\n".join(textwrap.wrap(explanation, width=50)) for explanation in explanations]

        if (isinstance(self._explanations_found, pd.Series)
                or isinstance(self._explanations_found, pd.DataFrame)
                or isinstance(self._explanations_found, list)):
            if len(explanations) < len(self._explanations_found):
                # If we got less explanations than we should, add empty strings to match the expected number of explanations.
                explanations += [""] * (len(self._explanations_found) - len(explanations))
            if isinstance(self._explanations_found, pd.Series) or isinstance(self._explanations_found, pd.DataFrame):
                index = self._explanations_found.index
            else:
                index = range(len(self._explanations_found))
            explanations = pd.Series(explanations, index=index)

        else:
            explanations = explanations[0]

        return explanations
