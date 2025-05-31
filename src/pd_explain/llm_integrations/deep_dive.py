from typing import List
from collections import namedtuple

import pandas as pd

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client

apply_result = namedtuple("apply_result", ["index",
                                           "result",
                                           "generating_query",
                                           "error_occurred"])

class DeepDive(LLMIntegrationInterface):
    """
    A class to facilitate automated "deep dive" analysis of a DataFrame using a large language model (LLM).
    Deep dive analysis is designed such that given a user query of what they want to explore, the LLM will
    generate queries to explore the DataFrame, which will then be analyzed by our explainers to provide insights,
    following which the LLM will generate more queries to further explore the DataFrame.
    At the end, the LLM will generate a final report summarizing the findings.
    """

    def __init__(self, dataframe: pd.DataFrame, num_iterations: int = 5,
                 num_queries_per_iteration: int = 4):
        self.dataframe = dataframe
        self.num_iterations = num_iterations
        self.num_queries_per_iteration = num_queries_per_iteration

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
                "Your queries must never create a Series object, they must always return a DataFrame. ")

    def _describe_data(self) -> str:
        """
        Describe the data in the DataFrame.
        This is a short description of the data, passed as the user prompt to the LLM.
        """
        return (f"The DataFrame contains {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns. "
                f"The columns are: " + ", ".join([f"{col}: {str(dtype)}" for col, dtype in self.dataframe.dtypes.items()])
                )

    def _describe_format(self) -> str:
        """
        Describe the format of the expected input and output for the LLM.
        """
        input_description = ("The history is formatted as a DataFrame with the following columns: "
                             "query, fedex_explainer_findings, metainsight_explainer_findings. "
                             "The query column is listed as index: query, where index is the index of the row in the history "
                             "that the query was applied to. For example, 5: query means this result was created by applying the "
                             "query to the result of the query at index 5 in the history. "
                             "The fedex_explainer_findings and metainsight_explainer_findings columns are lists of findings done "
                             "by the FedEx and MetaInsight explainers, respectively. "
                             "FedEx findings are the most important statistical changes as a result of the query, and MetaInsight "
                             "findings are the most significant patterns detected in the data after the query was applied.\n")
        output_description = (f"You are expected to generate {self.num_queries_per_iteration} queries. "
                              f"Each query must be in the format index: query, where index is the index of the row in the history "
                              f"that you want to apply the query to. Use index 0 for the original DataFrame. "
                              "Example for a queries on index i -  i: [x > 5], i: .groupby('column_name').mean(), etc. "
                              "Do not perform column selection after a groupby operation, i.e. do not use .groupby('column_name')['another_column'].[some_agg_func](). "
                              "You may however use column selection after filtering or join operations. "
                              "The query must be a valid Pandas query applicable using eval() with the syntax [df]{query}, where [df] "
                              "is a placeholder for the name of the actual DataFrame (which is not provided to you). "
                              f"This must be a list of queries, where each row in the list starts with a * symbol and ends with a new line. "
                              f"The list should be surrounded by <queries> and </queries> tags. so it can be easily extracted programmatically. "
                              f"Do not provide conclusions or summaries - this will be done in the final report, and is not your task. "
                              f"Avoid only querying the original DataFrame, focus on developing a query tree that explores the data using previous queries as well. "
                              f"Avoid repeating queries that have already been applied in the history. ")
        return input_description + output_description

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
            # Apply the query to the DataFrame
            df_to_query = result_mapping.get(index, None)
            if df_to_query is None:
                index = 0
                df_to_query = self.dataframe  # Default to the original DataFrame if index is not found
            try:
                # Use eval to apply the query to the DataFrame
                result = eval(f"df_to_query{query_string}")
                # Store the result along with the index in the new results list
                new_results.append(
                    apply_result(index=index, result=result, generating_query=query_string, error_occurred=False)
                )
            except Exception as e:
                print(f"Error applying query '{query_string}' to DataFrame: {e}")
                new_results.append(
                    apply_result(index=index, result=e, generating_query=query_string, error_occurred=True)
                )

        return new_results


    def do_llm_action(self, user_query: str = None) -> pd.Series | None | str:
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
        data_description = self._describe_data()
        format_description = self._describe_format()
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
                print("No queries generated by the LLM.")
                return None
            queries = queries.split("\n")
            queries = [query.replace("*", "").strip() for query in queries if query.strip() and query.startswith('*')]
            # Split the queries into a Series
            queries_series = pd.Series(queries)
            # Apply the queries to the DataFrame and update the history
            new_results = self._apply(queries_series, result_history_mapping)
            if not new_results:
                print("No valid queries were applied to the DataFrame.")
                return None
            # Update the history DataFrame with new results
            for result in new_results:
                if result.error_occurred:
                    history = history._append({
                        "query": f"{result.index}: {result.generating_query}",
                        "fedex_explainer_findings": None,
                        "metainsight_explainer_findings": None,
                        "error": str(result.result)
                    }, ignore_index=True)
                else:
                    # Assuming FedEx and MetaInsight explainers are applied here
                    result_df = result.result
                    try:
                        fedex_findings = result_df.explain(
                            explainer="fedex",
                            top_k=2,
                            do_not_visualize=True
                        )
                        title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = fedex_findings
                        fedex_findings = explanations
                    except Exception as e:
                        fedex_findings = e
                    try:
                        metainsight_findings = result_df.explain(
                            explainer="metainsight",
                            top_k=2,
                            do_not_visualize=True,
                            max_filter_columns=2,
                            max_aggregation_columns=2,
                        )
                    except Exception as e:
                        metainsight_findings = e
                    history = history._append({
                        "query": f"{result.index}: {result.generating_query}",
                        "fedex_explainer_findings": fedex_findings,
                        "metainsight_explainer_findings": metainsight_findings,
                        "error": None
                    }, ignore_index=True)
                # Update the result history mapping with the new results
                result_history_mapping[len(result_history_mapping) - 1] = result.result
        pd.set_option('display.max_rows', display_max_rows)
        pd.set_option('display.max_columns', display_max_columns)
        pd.set_option('display.width', display_width)
        pd.set_option('display.max_colwidth', display_max_colwidth)
        return None


