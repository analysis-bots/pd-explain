import itertools
from typing import List, Tuple, Any
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import re
import sys
import os

import pandas as pd
from pandas import DataFrame

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client

apply_result = namedtuple("apply_result", ["index",
                                           "result",
                                           "generating_query",
                                           "error_occurred"])

@dataclass(frozen=False)
class QueryResultObjects:
    """
    A class to hold the results of a query applied to a DataFrame.
    It contains the findings from the FedEx and MetaInsight explainers, as well as any error that occurred.
    """
    fedex: object = None
    metainsight: object = None
    error: str | None = None


class DeepDive(LLMIntegrationInterface):
    """
    A class to facilitate automated "deep dive" analysis of a DataFrame using a large language model (LLM).
    Deep dive analysis is designed such that given a user query of what they want to explore, the LLM will
    generate queries to explore the DataFrame, which will then be analyzed by our explainers to provide insights,
    following which the LLM will generate more queries to further explore the DataFrame.
    At the end, the LLM will generate a final report summarizing the findings.
    """

    def __init__(self, dataframe: pd.DataFrame, num_iterations: int = 10,
                 num_queries_per_iteration: int = 5, source_name: str = None):
        self.dataframe = dataframe.copy()
        # Change all column names to lowercase, to avoid issues with case sensitivity
        self.dataframe.columns = [col.lower() for col in self.dataframe.columns]
        self.num_iterations = num_iterations
        self.num_queries_per_iteration = num_queries_per_iteration
        self.source_name = source_name if source_name else "df"

    def _define_task(self) -> str:
        """
        Define the task for the LLM to perform.
        This is a short description of the task, passed as the system prompt to the LLM.
        """
        return ("You are part of an automated data exploration system on Pandas DataFrames. "
                "Your task is to generate a query tree using Pandas syntax to explore the DataFrame provided using varied queries. "
                "You will be given a user query that describes what the user wants to explore. "
                "You will also be given the history of queries that have been generated so far in this iterative process,"
                "as well as the findings derived from those queries. You will not be given the actual DataFrame or query results. "
                "If there was an error executing a query, it will be provided in the history as well. "
                "Your available operators are: filtering using boolean conditions, groupby with aggregation functions [mean, sum, count, nunique, min, max, median, std, sem, var, size, prod], and join operations. "
                "All other operators, such as describe(), apply(), etc. are illegal and will result in an error. "
                "Do not ever perform column selection after a groupby operation, i.e. do not use .groupby('column_name')['another_column'].[some_agg_func](). "
                "You may use column selection after filtering or join operations, but it must never be a single column. "
                "Your queries must never create a Series object, they must always return a DataFrame. "
                "This is an iterative process, and it is expected that you will generate follow-up queries based on the results of previous queries. "
                "Plan your queries such that they can potentially be followed up on in future iterations. You will be told which iteration you are in, and how many iterations are left. ")

    def _describe_data(self, user_query: str) -> str:
        """
        Describe the data in the DataFrame.
        This is a short description of the data, passed as the user prompt to the LLM.
        :param user_query: The query the user gave. We use this to find the columns they (hopefully) specified,
        to give the LLM more information about those columns.
        :return: A string description of the data in the DataFrame.
        """
        data_description = (f"The DataFrame contains {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns. "
                f"The columns are: " + ", ".join(
            [f"{col}: {str(dtype)}" for col, dtype in self.dataframe.dtypes.items()])
                )
        # Create a set of words from the user query to find relevant columns
        user_query_words = re.split(r"\s+|,|;|:|-|_|\.|\(|\)|\{|}|\[|]|\"|'|`", user_query)
        user_query_words = set([word.strip().lower() for word in user_query_words])
        # Create every possible combination of words from the user query, with each combination appearing as
        # a string with no spaces, a string a space between each word, a string with hyphens between each word,
        # and a string with underscores between each word.
        user_query_combinations = set()
        # Assuming, hopefully reasonably, that column names are not longer than 5 words,
        for i in range(1, 6):
            for combination in itertools.combinations(user_query_words, i):
                user_query_combinations.add(''.join(combination))
                user_query_combinations.add(' '.join(combination))
                user_query_combinations.add('-'.join(combination))
                user_query_combinations.add('_'.join(combination))
        # Find columns that match any of the combinations
        matching_columns = user_query_combinations.intersection(self.dataframe.columns)
        if len(matching_columns) > 0:
            data_description += (f"\nThe user query potentially mentioned the following columns: {', '.join(matching_columns)}. "
                                 f"Potentially relevant data about these columns is as follows (there may be missing or irrelevant columns in this list):\n")
            for col in matching_columns:
                col_data = self.dataframe[col]
                is_numeric = col_data.dtype.kind in 'biufcmM' and col_data.nunique() > 6
                if is_numeric:
                    data_description += (f"{col} - numeric: mean={col_data.mean():.2f}, std={col_data.std():.2f}, median={col_data.median():.2f}, "
                                         f"min={col_data.min():.2f}, max={col_data.max():.2f}\n")
                else:
                    data_description += f"{col} - categorical: unique values={col_data.nunique()}, value_counts={col_data.value_counts()[:15].to_dict()}\n"
        return data_description

    def _describe_input_format(self) -> str:
        return ("The history is formatted as a DataFrame with the following columns: "
                "query, fedex_explainer_findings, metainsight_explainer_findings. "
                "The query column is listed as index: query, where index is the index of the row in the history "
                "that the query was applied to. For example, 5: query means this result was created by applying the "
                "query to the result of the query at index 5 in the history. "
                "The fedex_explainer_findings and metainsight_explainer_findings columns are lists of findings done "
                "by the FedEx and MetaInsight explainers, respectively. "
                "FedEx findings are the most important statistical changes as a result of the query, and MetaInsight "
                "findings are the most significant patterns detected in the data after the query was applied.\n")

    def _describe_output_format(self) -> str:
        return (f"You are expected to generate {self.num_queries_per_iteration} queries. "
                f"Each query must be in the format index: query, where index is the index of the row in the history "
                f"that you want to apply the query to. Use index 0 for the original DataFrame. "
                "Example for a query on index i -  i: [x > 5], i: .groupby('column_name').mean(), etc. "
                "Make sure you use the correct index for the query you want to apply, and never select an index above the current max index in the history. "
                "Do not perform column selection after a groupby operation, i.e. do not use .groupby('column_name')['another_column'].[some_agg_func](). "
                "You may however use column selection after filtering or join operations. "
                "The query must be a valid Pandas query applicable using eval() with the syntax [df]{query}, where [df] "
                "is a placeholder for the name of the actual DataFrame (which is not provided to you). "
                "If you need to use the DataFrame's name (for example to filter it), use the placeholder [df] in your query, and the system will replace it with the actual DataFrame name. "
                "For example, if you want to filter by a column, write [[df]['column_name'] > 5]. "
                f"The output must be a list of queries, where each row in the list starts with a * symbol and ends with a new line. "
                f"The list should be surrounded by <queries> and </queries> tags. so it can be easily extracted programmatically. "
                f"Do not provide conclusions or summaries - this will be done in the final report, and is not your task. "
                f"If the history is not empty, always perform follow-up queries to queries that diverged from the original DataFrame, "
                f"and only do not do so if it is completely impossible to do so. Make sure to design your queries "
                f"such that they can be followed up on in the next iteration. "
                f"Avoid repeating queries that have already been applied in the history. "
                f"If you use the std function, make sure to also specify the ddof parameters, otherwise std throws an error.")

    def _format_history(self, history) -> str:
        """
        Format the history of queries and findings for the LLM.
        This is a string representation of the history DataFrame.
        """
        if history.empty:
            return "The history is empty."
        return history.to_string(index=True, header=True)

    def _apply(self, response: pd.Series, result_mapping: dict) -> List[apply_result]:
        if result_mapping is None:
            raise ValueError("Result mapping must be provided to apply the queries to the DataFrame.")
        if response is None or len(response) == 0:
            raise ValueError("Response from LLM is empty or None. Cannot proceed with follow-up action.")
        new_results = []
        # Apply the queries from the response to the DataFrame and update the history.
        for query in response:
            # Extract the index from the query, which is in the format index[query]
            if not isinstance(query, str) or ':' not in query:
                continue
            index, query_string = query.split(':', 1)
            index = int(index.strip())
            query_string = query_string.strip()
            query_string_fixed = query_string.replace("[df]", "df_to_query")  # Replace [df] with df_to_query
            # Apply the query to the DataFrame
            df_to_query = result_mapping.get(index, None)
            if df_to_query is None:
                index = 0
                df_to_query = self.dataframe  # Default to the original DataFrame if index is not found
            try:
                # Use eval to apply the query to the DataFrame
                result = eval(f"df_to_query{query_string_fixed}")
                # Store the result along with the index in the new results list
                new_results.append(
                    apply_result(index=index, result=result, generating_query=query_string, error_occurred=False)
                )
            except Exception as e:
                new_results.append(
                    apply_result(index=index, result=e, generating_query=query_string, error_occurred=True)
                )

        return new_results

    def do_llm_action(self, user_query: str = None) -> tuple[
        DataFrame, str | None, defaultdict[Any, dict], list[Any] | list[str]]:
        if user_query is None or len(user_query) == 0:
            raise ValueError("User query must be provided for deep dive analysis.")
        history = pd.DataFrame(data=[["Original DataFrame", None, None, None]],
                               columns=["query", "fedex_explainer_findings", "metainsight_explainer_findings", "error"])
        result_history_mapping = {
            0: self.dataframe  # Start with the original DataFrame
        }
        display_max_rows = pd.get_option('display.max_rows')
        display_max_columns = pd.get_option('display.max_columns')
        display_width = pd.get_option('display.width')
        display_max_colwidth = pd.get_option('display.max_colwidth')
        # Disable the pandas options that limit the display of DataFrames, because that will truncate what the LLM sees
        # if we don't.
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 0)  # No limit on the width of the display
        pd.set_option('display.max_colwidth', None)  # No limit on column width
        client = Client()
        system_message = self._define_task()
        data_description = self._describe_data(user_query)
        format_description = self._describe_input_format() + self._describe_output_format()
        query_and_results = defaultdict(dict)
        # Temporarily disable console output, because some explainers print to the console, and we don't want that
        current_output = sys.stdout
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null device to suppress output
        try:
            for iteration in range(self.num_iterations):
                # Format the history for the LLM
                formatted_history = self._format_history(history)
                # Create the user message for the LLM
                user_message = (f"This is iteration {iteration + 1} out of {self.num_iterations} of the analysis process.\n"
                                f"User query: {user_query}\n"
                                f"History of queries and findings:\n{formatted_history}\n"
                                f"Data description:\n{data_description}\n"
                                f"Format description:\n{format_description}")
                response = client(
                    system_messages=[system_message],
                    user_messages=[user_message]
                )
                # Extract the queries from the response
                queries = self._extract_response(response, "<queries>", "</queries>")
                if queries is None or len(queries) == 0:
                    continue
                queries = queries.split("\n")
                queries = [query.replace("*", "").strip() for query in queries if query.strip() and query.startswith('*')]
                # Split the queries into a Series
                queries_series = pd.Series(queries)
                # Apply the queries to the DataFrame and update the history
                new_results = self._apply(queries_series, result_history_mapping)
                if not new_results:
                    continue
                # Update the history DataFrame with new results
                for result in new_results:
                    query_and_results[result.index][result.generating_query] = {}
                    if result.error_occurred:
                        history = history._append({
                            "query": f"{result.index}: {result.generating_query}",
                            "fedex_explainer_findings": None,
                            "metainsight_explainer_findings": None,
                            "error": str(result.result)
                        }, ignore_index=True)
                        # Store the error in the query and results mapping, so it can be used later
                        query_and_results[result.index][result.generating_query] = QueryResultObjects(
                            fedex=None,
                            metainsight=None,
                            error=str(result.result)
                        )
                    else:
                        # Assuming FedEx and MetaInsight explainers are applied here
                        result_df = result.result
                        query_and_results[result.index][result.generating_query] = QueryResultObjects(
                            fedex=None,
                            metainsight=None,
                            error=None
                        )
                        try:
                            fedex_findings = result_df.explain(
                                explainer="fedex",
                                top_k=4,
                                do_not_visualize=True
                            )
                            # Store the raw FedEx findings in the query and results mapping
                            query_and_results[result.index][result.generating_query].fedex = result_df.last_used_explainer
                            title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = fedex_findings
                            fedex_findings = explanations
                            # Remove the LaTeX formatting from the FedEx findings
                            fedex_findings = fedex_findings.values.tolist()
                            fedex_findings = [finding.replace("bf", "").replace("$", "").replace("\n", "") for finding in
                                              fedex_findings]
                        except Exception as e:
                            fedex_findings = e
                            query_and_results[result.index][result.generating_query].fedex = None
                        try:
                            metainsight_findings = result_df.explain(
                                explainer="metainsight",
                                top_k=2,
                                do_not_visualize=True,
                                max_filter_columns=3,
                                max_aggregation_columns=3,
                            )
                            # Store the MetaInsight objects in the query and results mapping
                            query_and_results[result.index][result.generating_query].metainsight = result_df.last_used_explainer
                        except Exception as e:
                            metainsight_findings = e
                            query_and_results[result.index][result.generating_query].metainsight = None
                        history = history._append({
                            "query": f"{result.index}: {result.generating_query}",
                            "fedex_explainer_findings": fedex_findings,
                            "metainsight_explainer_findings": metainsight_findings,
                            "error": None
                        }, ignore_index=True)
                    # Update the result history mapping with the new results
                    result_history_mapping[len(history) - 1] = result.result
            # At the end of the iterations, generate a final report
            final_report_system_message = ("You are part of an automated data exploration system on Pandas DataFrames. "
                                           "You have two tasks."
                                           "The first is to generate a final report summarizing the findings from an automated analysis "
                                           "done according to a user query. "
                                           "The second is to point out which queries were most important in the analysis, so they "
                                           "can be visualized to the user as part of the final report. "
                                           "You will be given the history of the queries that have been used for the analysis, along "
                                           "with the findings derived from those queries."
                                           )
            final_report_user_message = (f"This is the final report generation step. "
                                         f"The user query was: {user_query}\n"
                                         f"The history has the following format:\n"
                                         f"{self._describe_input_format()}\n"
                                         f"The history of queries and findings is as follows:\n"
                                         f"{self._format_history(history)}\n"
                                         f"Ignore the error column, it is not relevant for the final report.\n"
                                         f"Extract the findings from the history, and generate a final report summarizing the findings, "
                                         f"according to the user query. \n"
                                         f"Provide the report surrounded by <report> and </report> tags, so it can be easily extracted programmatically. "
                                         f"Also, provide a list of the most important queries that were used in the analysis. This list "
                                         f"should be a list of indexes from the history DataFrame, with a * symbol before each index and a new line after each index. "
                                         f"This list should be surrounded by <vis> and </vis> tags, so it can be easily extracted programmatically. ")
            final_report_response = client(
                system_messages=[final_report_system_message],
                user_messages=[final_report_user_message]
            )
            # Extract the final report from the response
            final_report = self._extract_response(final_report_response, "<report>", "</report>")
            visualization_queries = self._extract_response(final_report_response, "<vis>", "</vis>")
            if visualization_queries is None or len(visualization_queries) == 0:
                visualization_queries = []
            else:
                visualization_queries = visualization_queries.split("\n")
                visualization_queries = [query.replace("*", "").strip() for query in visualization_queries if
                                         query.strip() and query.startswith('*')]
                # Restore the pandas options that were changed
                # pd.set_option('display.max_rows', display_max_rows)
                # pd.set_option('display.max_columns', display_max_columns)
                # pd.set_option('display.width', display_width)
                # pd.set_option('display.max_colwidth', display_max_colwidth)
                # Re-enable console output if it was disabled
                # Re-enable console output if it was disabled
                sys.stdout = current_output if current_output else sys.__stdout__
                return history, final_report, query_and_results, visualization_queries
        finally:
            # Backup inside the finally block to ensure we restore all options even if an error occurs
            pd.set_option('display.max_rows', display_max_rows)
            pd.set_option('display.max_columns', display_max_columns)
            pd.set_option('display.width', display_width)
            pd.set_option('display.max_colwidth', display_max_colwidth)
            # Re-enable console output if it was disabled
            sys.stdout = current_output if current_output else sys.__stdout__


    def visualize_deep_dive(self, history, final_report, query_and_results, visualization_queries):
        pass
