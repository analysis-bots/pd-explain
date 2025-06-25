import itertools
import math
from typing import List, Any, Literal
from collections import defaultdict
import re
import os

import pandas as pd
from pandas import DataFrame
from together.error import InvalidRequestError

from pd_explain.llm_integrations import Client
from pd_explain.llm_integrations import consts
from pd_explain.llm_integrations.automated_data_exploration.simple_visualizer import \
    SimpleAutomatedExplorationVisualizer
from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.automated_data_exploration.data_structures import apply_result, QueryResultObject, \
    QueryTree
from pd_explain.llm_integrations.automated_data_exploration.graph_visualizer import GraphAutomatedExplorationVisualizer


class AutomatedDataExploration(LLMIntegrationInterface):
    """
    A class to facilitate automated "deep dive" exploration and analysis of a DataFrame using a large language model (LLM).
    Deep dive analysis is designed such that given a user query of what they want to explore, the LLM will
    generate queries to explore the DataFrame, which will then be analyzed by our explainers to provide insights,
    following which the LLM will generate more queries to further explore the DataFrame.
    At the end, the LLM will generate a final report summarizing the findings.
    """

    def __init__(self, dataframe: pd.DataFrame, source_name: str = None, beautify: bool = False):
        self.dataframe = dataframe.copy()
        # Change all column names to lowercase, to avoid issues with case sensitivity
        self.dataframe.columns = [col.lower() for col in self.dataframe.columns]
        self.source_name = source_name if source_name else "df"
        self.history = None
        self.final_report = None
        self.query_and_results = None
        self.visualization_queries = None
        self.query_tree = None
        self.beautify = beautify

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
                "If there was an error executing a query, it will be provided in the history as well. Make absolutely sure you do not repeat errors. "
                "Your available operators are: filtering using boolean conditions and groupby with aggregation functions [mean, sum, count, nunique, min, max, median, std, sem, var, size, prod]."
                "All other operators and functions, such as describe, apply, join, quantile, query, reset_index, etc. are illegal and will result in an error. "
                "You can freely combine these operators or use lambda functions, but you must always return a DataFrame. "
                "Do not ever select only a single column after performing an operation. If you perform column selection, it must always be multiple columns."
                "Your queries must never create a Series object, they must always return a DataFrame. "
                "This is an iterative process, and it is expected that you will generate follow-up queries based on the results of previous queries. "
                "Plan your queries such that they can potentially be followed up on in future iterations. You will be told which iteration you are in, and how many iterations are left. "
                )

    def _describe_data(self, user_query: str) -> str:
        """
        Describe the data in the DataFrame.
        This is a short description of the data, passed as the user prompt to the LLM.
        :param user_query: The query the user gave. We use this to find the columns they (hopefully) specified,
        to give the LLM more information about those columns.
        :return: A string description of the data in the DataFrame.
        """
        data_description = (
                f"The DataFrame contains {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns. "
                f"The columns are: " + ", ".join(
            [f"{col}: {str(dtype)}" for col, dtype in self.dataframe.dtypes.items()])
        )
        # We want to find combinations of words in the user query that may match column names.
        # First, we identify candidate words from the user query by:
        # 1. Splitting the user query and the names of the columns into words
        # 2. Checking if any of the words in the user query are equal to any of the words in the column names.
        # 3. Finally, we generate combinations of the candidate words to find potential full column names.
        # Note: I tried an approach using Hamming distance to find similar words, but even with a threshold of 2
        # it produced too many false positives and made the output have too many columns that were not relevant.
        # Also note: trying to use the words as is without this filtering, in any case where the user query is long,
        # will result in this taking too long (I tried with an 88 word query, and it didn't end even after 20 minutes).
        pattern = re.compile(r"\s+|,|;|:|-|_|\.|\(|\)|\{|}|\[|]|\"|'|`|\n+")
        user_query_words = pattern.split(user_query)
        user_query_words = set([word.strip().lower() for word in user_query_words])
        user_query_words = {word for word in user_query_words if len(word) > 1}  # Remove empty strings
        column_words = [word for col in self.dataframe.columns for word in pattern.split(col) if word.strip()]
        candidate_set = set()
        for word in user_query_words:
            # Check if the word is a substring of any column name
            for col in column_words:
                if word == col:
                    candidate_set.add(col)
        user_query_combinations = set()
        # Assuming, hopefully reasonably, that column names are not longer than 5 words,
        for i in range(1, 6):
            for combination in itertools.combinations(candidate_set, i):
                user_query_combinations.add(''.join(combination))
                user_query_combinations.add(' '.join(combination))
                user_query_combinations.add('-'.join(combination))
                user_query_combinations.add('_'.join(combination))
        # Find columns that match any of the combinations
        matching_columns = user_query_combinations.intersection(self.dataframe.columns)
        if len(matching_columns) > 0:
            data_description += (
                f"\nThe user query potentially mentioned the following columns: {', '.join(matching_columns)}. "
                f"Potentially relevant data about these columns is as follows (there may be missing or irrelevant columns in this list):\n")
            for col in matching_columns:
                col_data = self.dataframe[col]
                is_numeric = col_data.dtype.kind in 'biufcmM' and col_data.nunique() > 6
                if is_numeric:
                    data_description += (
                        f"{col} - numeric: mean={col_data.mean():.2f}, std={col_data.std():.2f}, median={col_data.median():.2f}, "
                        f"min={col_data.min():.2f}, max={col_data.max():.2f}\n")
                else:
                    data_description += f"{col} - categorical: unique values={col_data.nunique()}\n"
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

    def _describe_output_format(self, queries_per_iteration: int, history: pd.DataFrame) -> str:
        num_errors = history[history['error'].notnull()].shape[0]
        recent_errors = history.tail(10)[history['error'].notnull()].shape[0]
        # If there are too many errors in the history, we want to increase the number of queries per iteration
        if num_errors > history.shape[0] / 3 or recent_errors > 3:
            queries_per_iteration *= 2
        return (f"You are expected to generate at-least {queries_per_iteration} queries. "
                f"Increase this number as needed if there is a significant number of failed queries (errors in the history) or if there are a lot recent errors. "
                f"Be aggressive about increasing the number of queries, especially if you see that we are not getting enough insights to produce anything meaningful. "
                f"Each query must be in the format index: query, where index is the index of the row in the history "
                f"that you want to apply the query to. Use index 0 for the original DataFrame. "
                "Example for a query on index i -  i: [x > 5], i: .groupby('column_name').mean(), etc. Never use a groupby within or as a filter query."
                "Make sure you use the correct index for the query you want to apply, and never select an index above the current max index in the history. "
                "The query must be a valid Pandas query applicable using eval(f'df_to_query{query}'), where df_to_query is an the name of some "
                "arbitrary dataframe (not given to you), and query is your own output. You should be aware of this syntax, and not add, for example,"
                "round brackets to the beginning, or [df] to the beginning of the query, as that will result in an error. "
                "Do not add round brackets anywhere unless you intend to call a function. Using round brackets for groupby(...), mean(), etc. is fine,"
                "using round brackets for filtering is not fine, as it will result in an error. "
                "If you need to use the DataFrame's name (for example to filter it), use the placeholder [df] in your query, and the system will replace it with the actual DataFrame name. "
                "For example, if you want to filter by a column, write [[df]['column_name'] > 5]. The usage of square brackets must "
                "never be done outside of a filter query or column selection, and the inside of a filter query must never be any operation but simple comparisons. "
                "For example, [[df].groupby('column_name').mean() > 5] is not a valid query, but [[df]['column_name'] > 5] is. "
                f"The output must be a list of queries, where each row in the list starts with a * symbol and ends with a new line. "
                f"The list should be surrounded by <queries> and </queries> tags. so it can be easily extracted programmatically. "
                f"Avoid repeating queries that have already been applied in the history. "
                f"If you use the std function, make sure to also specify the ddof parameters, otherwise std throws an error."
                f"If you use the agg function, make sure to provide the aggregations as a dictionary, i.e. {{column_name: 'agg_func'}}, "
                f"and not as any other type of format. Any other format except a valid Python dictionary will cause an error.\n"
                f"Make sure you always apply brackets correctly - every bracket of any kind must have an opening and closing bracket. "
                f"The program will not be fix your bracket mistakes, and will throw an error if you do not apply brackets correctly. "
                f"If you get unmatched bracket errors in the history, take a great amount of care to not repeat those mistakes.\n"
                f"Also make doubly sure to use valid pandas syntax, as the program will not fix your mistakes. "
                f"It is of paramount importance that you do not repeat errors in the history. If you see an error, "
                f"think carefully about why it happened to make sure you do not repeat it.\n")

    def _describe_additional_output_format(self):
        return ("The output should also include two more lists.\n"
                "The first, is a list giving a textual description of the queries you produced, surrounded by <desc> and </desc> tags. "
                "These descriptions should be concise, and should explain what the query does, and what it is expected to find.\n"
                "The second, is a list explaining the findings of queries from the previous iteration, surrounded by <findings> and </findings> tags. "
                "These explanations should be concise, explaining in simple words what the query found and what it means.\n"
                "Look at the 'need_explanation' column in the history DataFrame to see which queries are those from the previous iteration that need explanations.\n"
                "These two lists should not include query numbers, as those are not relevant and are already present in the history DataFrame.\n")

    def _format_history(self, history, truncate_early_by: int = 0, part: int = None,
                        total_parts: int = None,
                        remove_errors: bool = False) -> str:
        """
        Format the history of queries and findings for the LLM.
        :param history: The history DataFrame containing the queries and findings.
        :param truncate_early_by: The number of rows to truncate from the beginning of the history.
        :param part: The part of the history to return, if specified.
        :param total_parts: The total number of parts to split the history into, if specified.
        :param remove_errors: If True, remove rows with errors from the history.

        :return: A string representation of the history DataFrame, formatted for the LLM.
        """
        if history.empty:
            return "The history is empty."
        # Do not take the 'query_description' and 'query_findings' columns into account when formatting the history,
        # since they contain information meant for the user, not the LLM.
        if 'query_description' in history.columns:
            history = history.drop(columns=['query_description'])
        if 'query_findings' in history.columns:
            history = history.drop(columns=['query_findings'])
        if remove_errors:
            # Remove rows with errors from the history
            history = history[history['error'].isnull()]
        # If part or total parts is not specified, we return the whole history - possible truncation
        if part is None or total_parts is None:
            return history.tail(history.shape[0] - truncate_early_by).to_string(index=True, header=True)
        # If part and total parts are specified, we return only the specified part of the history
        else:
            if part < 1 or part > total_parts:
                raise ValueError(f"Part must be between 1 and {total_parts}, got {part}.")
            history_length = history.shape[0]
            part_length = math.ceil(history_length / total_parts)
            start_index = (part - 1) * part_length
            end_index = min(start_index + part_length, history_length)
            # Return only the specified part of the history
            return history.iloc[start_index:end_index].to_string(index=True, header=True)

    def _apply(self, response: pd.Series, result_mapping: dict) -> List[apply_result]:
        if result_mapping is None:
            raise ValueError("Result mapping must be provided to apply the queries to the DataFrame.")
        # If the LLM responded with something that ended up being None or empty and was not caught earlier,
        # we return an empty list, as there is nothing to apply.
        # Alternatively, we could raise an error, but that may crash the whole process, and possibly
        # throw away the other perfectly valid results that were generated before this point, or may
        # be generated later.
        if response is None or len(response) == 0:
            return []
        new_results = []
        # Apply the queries from the response to the DataFrame and update the history.
        for query in response:
            # Extract the index from the query, which is in the format index[query]
            if not isinstance(query, str) or ':' not in query:
                continue
            index, query_string = query.split(':', 1)
            try:
                index = int(index.strip())
            except ValueError:
                # If the index is not an integer, we just default to 0
                index = 0
            query_string = query_string.strip()
            # Sometimes, the LLM puts decides that writing [[df].groupby...] is a good idea. It is not.
            if query_string.startswith("[[df].groupby("):
                query_string = query_string[len("[[df]"):]
                query_string = query_string[:-1]  # Remove the last character, which is the closing bracket
            # Likewise, the LLM can sometimes just decide to put [df] at the start of the query, which is not valid.
            # What is valid is [[df]... which is a filter query. [df] at the start just makes it end up getting replaced
            # with df_to_query, resulting in an error.
            if query_string.startswith("[df]"):
                query_string = query_string[len("[df]"):]
            # Replace [df] placeholder with df_to_query
            query_string_fixed = query_string.replace("[df]", "df_to_query")
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

    def _format_summary_query(self, history: pd.DataFrame, user_query: str,
                              total_parts: int = None, part: int = None, existing_summary: str = None) -> tuple[
        str, str]:
        """
        Format a summary query for the LLM to generate a final report.
        This is a string representation of the history DataFrame, along with the user query.
        :param history: The history DataFrame containing the queries and findings.
        :param user_query: The user query that initiated the analysis.
        :param total_parts: The total number of parts to split the history into for the LLM.
        :param part: The current part of the history being processed.
        :param existing_summary: An existing summary of the findings, if any.
        :return: A tuple containing the system message and user message for the LLM.
        """
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
                                     f"{self._describe_input_format()}\n")
        if (total_parts is None and part is None) or total_parts <= 1:
            final_report_user_message += (
                f"The history of queries and findings is as follows:\n"
                f"{self._format_history(history, remove_errors=True)}\n"
            )
        else:
            final_report_user_message += (
                f"Due to the length of the history, you are only seeing part {part} of {total_parts} of it, and will be provided "
                f"the rest of the history in subsequent calls.\n"
                f"The current part of the history of queries and findings is as follows:\n"
                f"{self._format_history(history, part=part, total_parts=total_parts, remove_errors=True)}\n"
                f"If this is not the final part, you will see the report you write in the next call, so direct that report "
                f"so it will be useful to yourself, to draw the final conclusions from, and not the user, as the user won't see "
                f"this in-between step.\n"
            )
        if existing_summary is not None:
            final_report_user_message += (
                f"The existing summary of the findings is as follows:\n"
                f"{existing_summary}\n"
            )
        final_report_user_message += (f"Ignore the error column, it is not relevant for the final report.\n"
                                      f"Extract the findings from the history, and generate a final report summarizing the findings, "
                                      f"according to the user query. This report, while it should be concise, should be detailed enough for the user to "
                                      f"understand and draw conclusions from.\n"
                                      f"Provide the report surrounded by <report> and </report> tags, so it can be easily extracted programmatically. "
                                      f"Also, provide a list of the most important queries that were used in the analysis. This list "
                                      f"should be a list of indexes from the history DataFrame, with a * symbol before each index and a new line after each index. "
                                      f"This list should be surrounded by <vis> and </vis> tags, so it can be easily extracted programmatically. "
                                      f"This list should be short, and only contain the most important queries that were used in the analysis. ")
        return final_report_system_message, final_report_user_message

    def do_llm_action(self, user_query: str = None, num_iterations: int = 10,
                      queries_per_iteration: int = 5, fedex_top_k: int = 4, metainsight_top_k: int = 2,
                      metainsight_max_filter_cols: int = 3, metainsight_max_agg_cols: int = 3,
                      verbose=False, max_iterations_to_add: int = 3) \
            -> tuple[DataFrame, str | None, defaultdict[Any, QueryResultObject], list[str], QueryTree]:
        """
        Perform the deep dive analysis on the DataFrame using the LLM.

        :param user_query: A textual description of what the user wants to explore in the DataFrame.
        :param num_iterations: The number of iterations to perform in the analysis.
        :param queries_per_iteration: The number of queries to generate per iteration.
        :param fedex_top_k: The number of top findings to return from the FedEx explainer.
        :param metainsight_top_k: The number of top findings to return from the MetaInsight explainer.
        :param metainsight_max_filter_cols: The maximum number of filter columns to use in the MetaInsight explainer.
        :param metainsight_max_agg_cols: The maximum number of columns to use for aggregation in the MetaInsight explainer.
        :param verbose: If True, print additional information during the analysis.
        :param max_iterations_to_add: The maximum number of additional iterations to add if the LLM fails to generate queries. Defaults to 3.

        :return: A tuple containing:
            - history: A DataFrame containing the history of queries and findings.
            - final_report: A string containing the final report generated by the LLM.
            - query_and_results: A dictionary mapping query indices to their results, including FedEx and MetaInsight findings.
            - visualization_queries: A list of query indices that are deemed important for visualization.
            - query_tree: A QueryTree object containing the structure of the queries and their ancestry.
        :raises ValueError: If the user_query is None or empty.
        """
        if user_query is None or len(user_query) == 0:
            raise ValueError("User query must be provided for deep dive analysis.")
        history = pd.DataFrame(data=[["Original DataFrame", None, None, None, False, None, None]],
                               columns=["query", "fedex_explainer_findings", "metainsight_explainer_findings",
                                        "error", "need_explanation", "query_description", "query_findings"])
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
        client = Client(
            api_key=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_KEY, None),
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER, "together"),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER_URL, "https://api.together.xyz"),
            model=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_MODEL, "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")
        )
        system_message = self._define_task()
        data_description = self._describe_data(user_query)
        format_description = self._describe_input_format() + self._describe_output_format(queries_per_iteration,
                                                                                          history)
        format_description += self._describe_additional_output_format()
        query_and_results = defaultdict(QueryResultObject)
        query_tree = QueryTree(source_name=self.source_name)
        print_error = False
        truncate_by = 0
        iteration_num = 0
        max_iterations = num_iterations
        iterations_added = 0
        try:
            while iteration_num < max_iterations:
                if verbose:
                    print(f"Starting iteration {iteration_num + 1}/{max_iterations}")
                # Format the history for the LLM
                formatted_history = self._format_history(history, truncate_early_by=truncate_by)
                # Create the user message for the LLM
                user_message = (
                    f"This is iteration {iteration_num + 1} out of {max_iterations} of the analysis process.\n"
                    f"User query: {user_query}\n"
                    f"History of queries and findings:\n{formatted_history}\n"
                    f"Data description:\n{data_description}\n"
                    f"Format description:\n{format_description}")
                # If anything goes wrong with the LLM, we break out of the loop.
                # Examples of errors that can occur are:
                # - Rate limit exceeded
                # - LLM not responding and timing out
                if verbose:
                    print(f"\t - Sending request to LLM")
                try:
                    response = client(
                        system_messages=[system_message],
                        user_messages=[user_message]
                    )
                except InvalidRequestError as e:
                    truncate_by += 10
                    if iterations_added < max_iterations_to_add:
                        iterations_added += 1
                        max_iterations += 1
                    if verbose:
                        print(f"\t - LLM request failed with error: {e}")
                        print(f"\t - Truncating history seen by the LLM by {truncate_by} rows and retrying...")
                        if iterations_added <= max_iterations_to_add:
                            print(
                                f"\t - Adding one additional iteration to compensate, now {max_iterations} iterations in total.")
                        else:
                            print(
                                f"\t - Not adding any more iterations as the maximum number of additional iterations ({max_iterations_to_add}) has been reached.")
                    iteration_num += 1
                    continue
                # Extract the queries from the response
                queries = self._extract_response(response, "<queries>", "</queries>")
                if queries is None or len(queries) == 0:
                    if verbose:
                        print(
                            f"\t - LLM did not generate queries or got the format wrong and queries could not be extracted during iteration {iteration_num + 1}. ")
                        if iterations_added < max_iterations_to_add:
                            iterations_added += 1
                            max_iterations += 1
                        if iterations_added <= max_iterations_to_add:
                            print(
                                f"\t - Adding one additional iteration to compensate, now {max_iterations} iterations in total.")
                        else:
                            print(
                                f"\t - Not adding any more iterations as the maximum number of additional iterations ({max_iterations_to_add}) has been reached.")
                    iteration_num += 1
                    continue
                queries = queries.split("\n")
                queries = [query.replace("*", "").strip() for query in queries if
                           query.strip() and query.startswith('*')]
                # Split the queries into a Series
                queries_series = pd.Series(queries)
                descriptions = self._extract_response(response, "<desc>", "</desc>")
                if descriptions is not None and len(descriptions) > 0:
                    descriptions = descriptions.split("\n")
                    descriptions = [desc.replace("*", "").strip() for desc in descriptions if
                                    desc.strip() and desc.startswith('*')]
                findings = self._extract_response(response, "<findings>", "</findings>")
                if findings is not None and len(findings) > 0:
                    findings = findings.split("\n")
                    findings = [finding.replace("*", "").strip() for finding in findings if
                                finding.strip() and finding.startswith('*')]
                    # Add the findings to the history DataFrame, by adding them in order to the queries in the
                    # history where need_explanation is True.
                    history_index = 0
                    if len(findings) > 0:
                        for i, finding in enumerate(findings):
                            # Find the first row in the history where need_explanation is True
                            while history_index < history.shape[0] and not history.iloc[history_index]['need_explanation']:
                                history_index += 1
                            if history_index >= history.shape[0]:
                                break
                            # Add the finding to the history DataFrame
                            history.at[history_index, 'query_findings'] = finding
                            history.at[history_index, 'need_explanation'] = False
                if verbose:
                    print(f"\t - Generated {len(queries_series)} queries for iteration {iteration_num + 1}")
                # Apply the queries to the DataFrame and update the history
                new_results = self._apply(queries_series, result_history_mapping)
                if not new_results:
                    iteration_num += 1
                    continue
                # Update the history DataFrame with new results
                for idx, result in enumerate(new_results):
                    # print(f"\t - Checking query: {result.generating_query} (index: {result.index})")
                    curr_index = len(history)
                    if result.error_occurred:
                        if verbose:
                            print(
                                f"\t - Error occurred while applying query: {result.generating_query} - {result.result}")
                        history = history._append({
                            "query": f"{result.index}: {result.generating_query}",
                            "fedex_explainer_findings": None,
                            "metainsight_explainer_findings": None,
                            "error": str(result.result),
                            "need_explanation": False,
                            "query_description": None,
                            "query_findings": None
                        }, ignore_index=True)
                        # Store the error in the query and results mapping, so it can be used later
                        query_and_results[curr_index] = QueryResultObject(
                            fedex=None,
                            metainsight=None,
                            error=str(result.result)
                        )
                    else:
                        result_df = result.result
                        fedex_finding_str = ""
                        metainsight_finding_str = ""
                        try:
                            fedex_findings = result_df.explain(
                                explainer="fedex",
                                top_k=fedex_top_k,
                                do_not_visualize=True,
                                log_query=False,
                                display_mode='carousel' if not self.beautify else 'grid'
                            )
                            # Store the raw FedEx findings in the query and results mapping
                            query_and_results[curr_index].fedex = result_df.last_used_explainer
                            title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = fedex_findings
                            fedex_findings = explanations
                            # Remove the LaTeX formatting from the FedEx findings
                            fedex_findings = fedex_findings.values.tolist()
                            pattern = re.compile(r'\$\\+bf{(.*?)}\$')
                            fedex_findings = [pattern.sub(r'\1', finding) for finding in fedex_findings]
                            fedex_findings = [finding.replace("(in green)", "").replace("\n", " ").replace("\\", "") for
                                              finding in fedex_findings]
                            if verbose:
                                fedex_finding_str = f"{len(fedex_findings)} FEDEx findings"
                        except Exception as e:
                            # Commented out line can potentially give more information to the LLM on how to avoid the error,
                            # but in most cases, it just takes up more tokens (which we don't have a lot of),
                            # fedex_findings = f"Error: {str(e)}"
                            fedex_findings = f"Error"
                            query_and_results[curr_index].fedex = None
                        try:
                            metainsight_findings = result_df.explain(
                                explainer="metainsight",
                                top_k=2,
                                do_not_visualize=True,
                                max_filter_columns=metainsight_max_filter_cols,
                                max_aggregation_columns=metainsight_max_agg_cols,
                                display_mode='carousel' if not self.beautify else 'grid'
                            )
                            metainsight_findings = [finding.__str__() for finding in metainsight_findings]
                            # Store the MetaInsight objects in the query and results mapping
                            query_and_results[curr_index].metainsight = result_df.last_used_explainer
                            if verbose:
                                metainsight_finding_str = f"{len(metainsight_findings)} MetaInsight findings"
                        except Exception as e:
                            # metainsight_findings = f"Error: {str(e)}"
                            metainsight_findings = f"Error"
                            query_and_results[curr_index].metainsight = None
                        if verbose:
                            if metainsight_finding_str or fedex_finding_str:
                                print(
                                    f"\t - Query {result.generating_query} produced {fedex_finding_str} {'and ' if fedex_finding_str and metainsight_finding_str else ''}{metainsight_finding_str}")
                            else:
                                print(f"\t - Query {result.generating_query} produced no findings.")
                        history = history._append({
                            "query": f"{result.index}: {result.generating_query}",
                            "fedex_explainer_findings": fedex_findings,
                            "metainsight_explainer_findings": metainsight_findings,
                            "error": None,
                            "need_explanation": True,  # We need explanations for the new queries
                            "query_description": descriptions[idx] if descriptions is not None and idx < len(descriptions) else None,
                            "query_findings": None
                        }, ignore_index=True)
                    # Update the query tree with the new query
                    query_tree.add_node(result.index, result.generating_query, curr_index)
                    # Update the result history mapping with the new results
                    result_history_mapping[len(history) - 1] = result.result
                iteration_num += 1
            if history.empty:
                print_error = True
                return history, None, query_and_results, [], query_tree
            # At the end of the iterations, generate a final report
            total_parts = 1
            final_report_response = None
            report_generated = False
            if verbose:
                print("Generating final report...")
            while not report_generated:
                try:
                    # Report generation can fail if the history is too long. We first try with the full history,
                    # but if that fails, we begin splitting the history into parts and trying again.
                    for part in range(1, total_parts + 1):
                        final_report_system_message, final_report_user_message = self._format_summary_query(
                            history=history,
                            user_query=user_query,
                            total_parts=total_parts,
                            part=part,
                            existing_summary=final_report_response if final_report_response else None
                        )
                        final_report_response = client(
                            system_messages=[final_report_system_message],
                            user_messages=[final_report_user_message]
                        )
                    report_generated = True
                except InvalidRequestError as e:
                    total_parts *= 2  # Double the number of parts to try and fit the history
                    if verbose:
                        print(f"\t - LLM request for generating final report failed with error: {e}")
                        print(
                            f"\t - Trying to generate final report again by splitting history into {total_parts} parts.")

            # Extract the final report from the response
            final_report = self._extract_response(final_report_response, "<report>", "</report>")
            visualization_queries = self._extract_response(final_report_response, "<vis>", "</vis>")
            if visualization_queries is None or len(visualization_queries) == 0:
                visualization_queries = []
            else:
                visualization_queries = visualization_queries.split("\n")
                visualization_queries = [query.replace("*", "").strip() for query in visualization_queries if
                                         query.strip() and query.startswith('*')]
            self.history = history
            self.final_report = final_report
            self.query_and_results = query_and_results
            self.visualization_queries = visualization_queries
            self.query_tree = query_tree
            return history, final_report, query_and_results, visualization_queries, query_tree
        finally:
            # Restore the pandas display options and console output to their original state
            # This is inside the finally block to ensure it always runs, even if an error occurs
            pd.set_option('display.max_rows', display_max_rows)
            pd.set_option('display.max_columns', display_max_columns)
            pd.set_option('display.width', display_width)
            pd.set_option('display.max_colwidth', display_max_colwidth)
            if print_error:
                print("LLM failed to generate any queries. ")

    def do_follow_up_action(self, history: pd.DataFrame = None, final_report=None,
                            query_and_results: dict[int, QueryResultObject] = None,
                            visualization_queries: list[int | str] = None, query_tree: QueryTree = None,
                            source_name: str = None, visualization_type: Literal['graph', 'simple'] = "graph",
                            beautify: bool = False
                            ):
        """
        Visualize the results of the deep dive analysis.

        :param history: The history of queries and findings from the deep dive analysis. Optional.
        :param final_report: The final report generated by the LLM. Optional.
        :param query_and_results: A dictionary mapping query indices to their results, including FedEx and MetaInsight findings. Optional.
        :param visualization_queries: A list of query indices that are deemed important for visualization. Optional.
        :param query_tree: A QueryTree object containing the structure of the queries and their ancestry. Optional.
        :param source_name: The name of the source DataFrame, if different from the default. Optional.
        :param visualization_type: The type of visualization to use, either "rich" or "simple". Default is "rich".
        Rich visualizations use interactive graphs for the query tree, while the simple visualization uses a much more
        basic visualization.
        If all optional parameters (except for source_name) are provided, the method will visualize the deep dive
        using the provided parameters, enabling visualization without running the LLM again.
        If the parameters are not provided, it will use the results from the last run of do_llm_action() to visualize the deep dive.

        :return: The visualized deep dive results as a ipywidgets tab.
        """
        all_params_provided = history is not None and final_report is not None and query_and_results is not None \
                              and visualization_queries is not None and query_tree is not None
        visualizer_class = GraphAutomatedExplorationVisualizer if visualization_type == "graph" else SimpleAutomatedExplorationVisualizer
        if all_params_provided:
            visualizer = visualizer_class(
                history=history,
                query_and_results=query_and_results,
                visualization_queries=visualization_queries,
                query_tree=query_tree,
                final_report=final_report,
                source_name=source_name if source_name else self.source_name,
                beautify=beautify
            )
            return visualizer.visualize_data_exploration()
        all_self_params_exist = self.history is not None and self.final_report is not None \
                                and self.query_and_results is not None and self.visualization_queries is not None \
                                and self.query_tree is not None
        if not all_self_params_exist:
            raise ValueError("No deep dive analysis has been performed yet. Please run do_llm_action() first.")
        visualizer = visualizer_class(
            history=self.history,
            query_and_results=self.query_and_results,
            visualization_queries=self.visualization_queries,
            query_tree=self.query_tree,
            final_report=self.final_report,
            source_name=source_name if source_name else self.source_name,
            beautify=self.beautify
        )
        return visualizer.visualize_data_exploration()
