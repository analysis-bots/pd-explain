import itertools
import math
from typing import List, Any
from collections import defaultdict
import re
import os
import warnings

import pandas as pd
from pandas import DataFrame
from together.error import InvalidRequestError

from pd_explain.llm_integrations import Client
from pd_explain.llm_integrations import consts
from pd_explain.llm_integrations.automated_data_exploration.visualizer import \
    AutomatedExplorationVisualizer
from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.automated_data_exploration.data_structures import apply_result, QueryResultObject, \
    QueryTree


class AutomatedDataExploration(LLMIntegrationInterface):
    """
    A class to facilitate automated "deep dive" exploration and analysis of a DataFrame using a large language model (LLM).
    Deep dive analysis is designed such that given a user query of what they want to explore, the LLM will
    generate queries to explore the DataFrame, which will then be analyzed by our explainers to provide insights,
    following which the LLM will generate more queries to further explore the DataFrame.
    At the end, the LLM will generate a final report summarizing the findings.
    """

    def __init__(self, dataframe: pd.DataFrame, source_name: str = None,
                 beautify_fedex: bool = False, beautify_metainsight: bool = False):
        """
        Initialize the AutomatedDataExploration class with a DataFrame and optional parameters.

        :param dataframe: The DataFrame to be explored.
        :param source_name: An optional name for the source of the DataFrame, used for identification in reports.
        :param beautify_fedex: If True, beautify the FedEx explainer findings.
        :param beautify_metainsight: If True, beautify the MetaInsight explainer findings.
        :param beautify_query_tree: If True, beautify the query tree visualization.
        """
        self.dataframe = dataframe.copy()
        # Change all column names to lowercase, to avoid issues with case sensitivity
        self.dataframe.columns = [col.lower() for col in self.dataframe.columns]
        self.source_name = source_name if source_name else "df"
        self.history = None
        self.final_report = None
        self.query_and_results = None
        self.query_tree = None
        self.beautify_fedex = beautify_fedex
        self.beautify_metainsight = beautify_metainsight
        self.visualizer = None
        self.verbose = False
        self.log = []  # A list to store logs of the process, for debugging and analysis purposes.
        self.current_query_num = 1  # A counter to keep track of the current query number.

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
                "All other operators and functions, such as describe, apply, join, quantile, query, reset_index, sort_values, crosstab, etc. are illegal and will result in an error. "
                "You can freely combine these operators or use lambda functions, but you must always return a DataFrame. "
                "Do not ever select only a single column after performing an operation. If you perform column selection, it must always be multiple columns."
                "Your queries must never create a Series object, they must always return a DataFrame. "
                "This is an iterative process, and it is expected that you will generate follow-up queries based on the results of previous queries. "
                "You will be given an overall exploration plan. Your queries should aim to address the goals outlined in the plan."
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
        return ("You will receive a summary of the history of queries and findings.\n"
                "If LLM generated descriptions of the findings are available, you will be provided those. If not, you will "
                "receive the findings from the FedEx and MetaInsight explainers.\n"
                "FedEx findings are the most important statistical changes as a result of the query, and MetaInsight "
                "findings are the most significant patterns detected in the data after the query was applied.\n")

    def _describe_output_format(self) -> str:
        return (f"You are expected to generate a variable number of queries in each iteration. "
                f"Start with one or two queries at most. You can generate more queries in a single iteration if you "
                f"believe it is necessary to explore multiple paths based on the previous findings.\n "
                "The query must be a valid Pandas query applicable using eval(f'df{query}'), where df is an the name of some "
                "arbitrary dataframe (not given to you), and query is your own output. You should be aware of this syntax, and not add, for example,"
                "round brackets to the beginning, or df to the beginning of the query, as that will result in an error. "
                "If you need to use the DataFrame's name (for example to filter it), use the placeholder df in your query, and the system will replace it with the actual DataFrame name. "
                "For example, if you want to filter by a column, write [df['column_name'] > 5]. The usage of square brackets must "
                "never be done outside of a filter query or column selection, and the inside of a filter query must never be any operation but simple comparisons. "
                "For example, [df.groupby('column_name').mean() > 5] is not a valid query, but [df['column_name'] > 5] is. "
                "The query name will be added by the system, so you should not add it yourself. For example, if you supply the query "
                "[df['column_name'] > 5], it will be transformed to df[df['column_name'] > 5]. Trying to add this "
                "yourself will result in an error. For example, [df[df['column_name'] == 5]] is not a valid query, as it will result in an error, "
                "while [df['column_name'] == 5] is a valid query. "
                "This is for the case of filter queries. In the case of groupby queries, you need to add a . before the groupby, "
                "i.e. .groupby('column_name') is a valid query, while groupby('column_name') and df.groupby('column_name') are not. "
                "Additional examples of valid filter queries: [df['column_name'] < 2], [df['column_name'] == 'value'].groupby('column_name').mean(). "
                "Additional examples of invalid filter queries: [df.groupby('column_name')], df[df['column_name'] > 5]. "
                "Additional examples of valid groupby queries: df.groupby('column_name').mean(), df.groupby(['column_name1', 'column_name2']).agg({'column_name3': 'mean', 'column_name4': 'sum'}). "
                "Additional examples of invalid groupby queries: [df.groupby('column_name')], df.groupby('column_name').filter(lambda x: x['column_name'] > 5), df.groupby('column_name').agg(['mean', 'sum']). "
                "Do not add round brackets anywhere unless you intend to call a function. Using round brackets for groupby(...), mean(), etc. is fine,"
                "using round brackets for filtering is not fine, as it will result in an error. "
                f"The output must be a list of queries, where each row in the list starts with a * symbol and ends with a new line (provide it in list format even if you only have one query). "
                f"There must never be multiple queries in a single row, even if separated by commas. "
                f"The list should be surrounded by <queries> and </queries> tags. so it can be easily extracted programmatically. "
                f"Avoid repeating queries that have already been applied in the history. "
                f"If you use the std function, make sure to also specify the ddof parameters, otherwise std throws an error."
                f"If you use the agg function, make sure to provide the aggregations as a dictionary, i.e. {{column_name: 'agg_func'}}, "
                f"and not as any other type of format. Any other format except a valid Python dictionary (such as a list) will cause an error.\n"
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
                "These explanations should be concise, explaining in simple words what the query found and what it means. "
                "These explanations should be stand-alone, and should not reference the history or previous queries.\n"
                "Queries that require these explanations will have the words 'This query requires a summary of its findings, which will also be provided in the next iteration.' "
                "explicitly stated. Only those queries need an explanation, and the rest of the queries do not need an explanation.\n"
                "The format of each entry in this explanations list should be query number: explanation, where the query number is the number of the query in the history. "
                "Some example entries in the list may be: 4: ... \n 5: ... \n 6: ...\n "
                "It is very important the the beginning of the explanations list is the query number, and nothing but the query number followed by "
                "a colon and then the explanation, as the system will use this number to match the explanation to the query in the history.\n"
                "Writing query 4: ..., Query 4: ..., etc. will result in an error, as the system will not be able to match the explanation to the query.\n"
                "These two lists should not include query numbers, as those are not relevant and are already present in the history DataFrame.\n"
                "Both of these lists must likewise be un-numbered lists, starting with a * character and separated by new-line characters.\n"
                "The description list should be in order of the generated queries.\n"
                "The findings list should be in order of the previous queries that need an explanation to their findings.\n")

    def _format_history(self, history, truncate_early_by: int = 0, part: int = None,
                        total_parts: int = None,
                        remove_errors: bool = False) -> str:
        """
        Prepare the history DataFrame for the LLM, removing unnecessary columns and trimming it if necessary,
        then summarizing it into a string.
        :param history: The history DataFrame containing the queries and findings.
        :param truncate_early_by: The number of rows to truncate from the beginning of the history.
        :param part: The part of the history to return, if specified.
        :param total_parts: The total number of parts to split the history into, if specified.
        :param remove_errors: If True, remove rows with errors from the history.

        :return: A string summarizing the history DataFrame, formatted for the LLM.
        """
        if history.empty:
            return history
        if remove_errors:
            # Remove rows with errors from the history
            history = history[history['error'].isnull()]
        # If part or total parts is not specified, we return the whole history - possible truncation
        if part is None or total_parts is None:
            return self._summarize_history(history.tail(history.shape[0] - truncate_early_by))
        # If part and total parts are specified, we return only the specified part of the history
        else:
            if part < 1 or part > total_parts:
                raise ValueError(f"Part must be between 1 and {total_parts}, got {part}.")
            history_length = history.shape[0]
            part_length = math.ceil(history_length / total_parts)
            start_index = (part - 1) * part_length
            end_index = min(start_index + part_length, history_length)
            # Return only the specified part of the history
            return self._summarize_history(history.iloc[start_index:end_index], truncated_history = part > 1)


    def _summarize_history(self, history: pd.DataFrame, truncated_history: bool = False) -> str:
        """
        Summarize the history of queries and findings for the LLM.
        :param history: The history DataFrame containing the queries and findings.
        :param truncated_history: A boolean value indicating whether the history is truncated. If it is, the first row is not ignored.
        If it is not, then the first row is assumed to be the original DataFrame and is ignored.
        :return: A string summarizing the history DataFrame, formatted for the LLM.
        """
        if history.empty:
            return "The history is empty."
        summary_string = ""
        # First, include how many queries were generated in total, and how many of them were successful.
        total_queries = history.shape[0]
        successful_queries = history[history['error'].isnull()].shape[0]
        summary_string += (f"So far, {total_queries - 1} queries were generated, "
                           f"of which {successful_queries - 1} were successful and {total_queries - successful_queries} failed with errors.\n")
        # Loop over the history df, and summarize each query (skip the first row, which is the original DataFrame)
        for idx, row in history.iterrows():
            if idx == 0 and not truncated_history:
                continue
            summary_string += "\n\n"
            query = row['query']
            query_description = row['query_description']
            query_findings = row['query_findings']
            error = row['error']
            # Extract the query origin from the query string, which is in the format index: query
            query_split = query.split(':', 1)
            query_origin = query_split[0].strip() if len(query_split) > 1 else "0"
            if query_origin == '0':
                query_origin = "Original DataFrame"
            else:
                query_origin = f"Result of query {query_origin}"
            query = query_split[1].strip() if len(query_split) > 1 else query.strip()
            summary_string += f"Query {idx}: {query} applied to {query_origin}.\n"
            if query_description:
                summary_string += f"Description: {query_description}\n"
            # If no error occurred, we include the findings from the query.
            if error is None:
                if query_findings:
                    summary_string += f"Findings: {query_findings}\n"
                else:
                    fedex_findings = str(row['fedex_explainer_findings'])
                    metainsight_findings = str(row['metainsight_explainer_findings'])
                    if fedex_findings != "[]":
                        summary_string += f"FedEx findings: {fedex_findings}\n"
                    if metainsight_findings != "[]":
                        summary_string += f"MetaInsight findings: {metainsight_findings}\n"
                    # If the query's findings have not been explained yet, we note that.
                    if row['need_explanation']:
                        summary_string += "This query requires a summary of its findings, which will also be provided in the next iteration.\n"
            else:
                summary_string += (f"This query failed with the following syntax error, which you must take great care "
                                   f"to understand and not repeat: {error}\n")

        return summary_string


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
            query_string = query.strip()
            # Sometimes, the LLM just decides that writing [df.groupby...] is a good idea. It is not.
            if query_string.startswith("[df.groupby("):
                query_string = query_string[len("[df"):]
                query_string = query_string[:-1]  # Remove the last character, which is the closing bracket
            # Yet another mistake the LLM can make is to put df at the start of the query, which is not valid.
            # It just ends up as dfdf, which is not recognized as a variable.
            if query_string.startswith("df"):
                # If this df is followed by a dot, it is a groupby or similar operation, so we just remove it.
                if len(query_string) > 2 and query_string[2] == '.':
                    query_string = query_string[2:]
                # If this df is followed by a square bracket, it is a filter query, so we actually want to add the
                # missing square brackets.
                elif len(query_string) > 2 and query_string[2] == '[':
                    query_string = query_string[2:]
                    query_string = f"[df{query_string}"
                    # In this case, there is also a non-zero chance the bracket is not closed, so we check
                    num_opening_brackets = query_string.count('[')
                    num_closing_brackets = query_string.count(']')
                    if num_opening_brackets > num_closing_brackets:
                        # If there are more opening brackets than closing brackets, we find the first . in the query,
                        # and add the closing bracket before it. If there is no . in the query, we just add the closing bracket at the end.
                        dot_index = query_string.find('.')
                        if dot_index != -1:
                            query_string = query_string[:dot_index] + ']' + query_string[dot_index:]
                        else:
                            query_string += ']'
            # Another fun and common error  - [df[df[... is a common mistake the LLM makes. Despite
            # explicit instructions to not do that, it still does it.
            if query_string.startswith("[df[df["):
                # The first [df] is the one we want to keep, so we remove the second one.
                query_string = query_string[len("[df[df["):]
                query_string = "[df[" + query_string  # Add the first [df] back
            # Yet another common error - too many closing square brackets.
            num_closing_brackets = query_string.count('[')
            num_opening_brackets = query_string.count(']')
            if num_closing_brackets > num_opening_brackets:
                # If there are more closing brackets than opening brackets, this usually happens where the LLM
                # puts double square brackets somewhere in the query when it should not, for example: [df['column_name'] > 5]].
                # We need to remove the extra closing brackets until the number of closing brackets matches the number
                # of opening brackets.
                # This rarely happens more than once, so we just remove the last double closing brackets.
                all_double_closing_brackets = re.finditer(r']]', query_string)
                last_double_closing_brackets = None
                for match in all_double_closing_brackets:
                    last_double_closing_brackets = match.start()
                if last_double_closing_brackets is not None:
                    # Remove the last double closing brackets
                    query_string = query_string[:last_double_closing_brackets] + query_string[last_double_closing_brackets + 1:]

            # Apply the query to the DataFrame. We always query the original DataFrame, which is stored in result_mapping with index 0.
            df = result_mapping.get(0, None)
            try:
                # Use eval to apply the query to the DataFrame
                result = eval(f"df{query_string}")
                # Store the result along with the index in the new results list
                # The index parameter used to be for the originating query to allow for follow ups on the same query,
                # but we have since decided to not allow that, so we always use index 0.
                new_results.append(
                    apply_result(index=0, result=result, generating_query=query_string, error_occurred=False)
                )
            except Exception as e:
                new_results.append(
                    apply_result(index=0, result=e, generating_query=query_string, error_occurred=True)
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
                                       "Your task generate a final report summarizing the findings from an automated analysis "
                                       "done according to a user query. "
                                       "You will be given a textual summary of the queries that have been used for the analysis, along "
                                       "with the findings derived from those queries."
                                       )
        final_report_user_message = (f"This is the final report generation step. "
                                     f"The user query was: {user_query}\n")
        if (total_parts is None and part is None) or total_parts <= 1:
            final_report_user_message += (
                f"The summary history of queries and findings is as follows:\n"
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
                                      f"Every conclusion should list exactly which queries it was drawn from. "
                                      f"This referencing of queries should always be done in the format (Query x, Query y, ...) - i.e., referencing within "
                                      f"brackets and with the word query before the exact query number. This format will be used to "
                                      f"programmatically find these specified queries and create links for the user to click to get more "
                                      f"info about the query, so it is very important you use this exact format.\n"
                                      f"Any attempt to use other formats, such as (Queries x, y, z) or Queries (x, y, z) will result in an error, "
                                      f"as the system will not be able to extract the queries from the report. It is very important you use the exact format. \n"
                                      f"Provide the report surrounded by <report> and </report> tags, so it can be easily extracted programmatically.")
        return final_report_system_message, final_report_user_message

    def _generate_initial_plan(self, client: Client, user_query: str, num_iterations: int, data_description_str: str):
        """
        Have the LLM generate an initial plan for the analysis.
        :param client: The LLM client to use for the request.
        :param user_query: The user query that initiated the analysis.
        :param num_iterations: The number of iterations to perform in the analysis.
        :param data_description_str: A string description of the data in the DataFrame.
        """
        system_message = (
            "You are a data analysis planner for an iterative data exploration system, that automatically "
            "queries a DataFrame based on a user query and generates findings from those queries.\n"
            "It can not do anything except generate and run those queries, and then generate a final report at the end (after the iterative process is done. You do not need to plan for this step, it will always happen.).\n"
            "The system can not visualize the data, can not generate plots, can not perform any statistical tests, "
            "and can not perform any other operations except for generating queries and running them on the DataFrame.\n"
            "Based on a user query and a description of the data, "
            "your task is to generate an initial plan for how to query the data and explore it.\n"
            "create a step-by-step plan for exploration. The plan should consist of high-level goals."
        )
        user_message = (
            f"User query: {user_query}\n"
            f"Data description: {data_description_str}\n"
            f"You will have {num_iterations} iterations to perform the analysis.\n"
            f"Place the plan in <plan> and </plan> tags, so it can be easily extracted programmatically. "
        )

        try:
            response = client(
                system_messages=[system_message],
                user_messages=[user_message]
            )
            plan = self._extract_response(response, "<plan>", "</plan>")
            if self.verbose:
                if plan is not None and len(plan) > 0:
                    print(f"Initial plan generated by the LLM:\n{plan}")
                else:
                    print("No initial plan generated by the LLM.")
        except InvalidRequestError as e:
            if self.verbose:
                print(f"Failed to generate initial plan with error: {e}")
            plan = "No initial plan generated due to an error."
        return plan

    def _analyze_queries(self, query_results: List[apply_result],
                         fedex_top_k, metainsight_max_filter_cols,
                         metainsight_max_agg_cols,
                         verbose: bool = False) -> tuple[List[QueryResultObject], List[int]]:
        """
        Use fedex and metainsight to analyze the queries and their results.
        :param query_results: A list of DataFrames containing the results of the queries.
        :param fedex_top_k: The number of top findings to return from the FedEx explainer.
        :param metainsight_max_filter_cols: The maximum number of filter columns to use in the MetaInsight explainer.
        :param metainsight_max_agg_cols: The maximum number of columns to use for aggregation in the MetaInsight explainer.
        :param verbose: If True, print additional information during the analysis.
        """
        results: List[QueryResultObject] = []
        total_findings = []
        for idx, result in enumerate(query_results):
            if result.error_occurred:
                error_str = str(result.result)
                # Edge case - error string is too long, truncate it to 100 characters.
                # I have only seen it happen once, when it produced an error string "could not convert key '<=50k''>50k'
                # repeated some 30000 times, which bricked the process.
                if len(error_str) > 100:
                    error_str = error_str[:100] + "..."
                log_str = f"\t - Error occurred while applying query {self.current_query_num}: {result.generating_query} - {error_str}"
                if verbose:
                    print(log_str)
                self.log.append(log_str.replace("\t", "&emsp;"))
                results.append(
                    QueryResultObject(
                        fedex=None,
                        metainsight=None,
                        error=error_str,
                    )
                )
                total_findings.append(0)
            else:
                res = QueryResultObject()
                result_df = result.result
                fedex_finding_str = ""
                metainsight_finding_str = ""
                try:
                    fedex_findings = result_df.explain(
                        explainer="fedex",
                        top_k=fedex_top_k,
                        do_not_visualize=True,
                        log_query=False,
                        display_mode='grid'
                    )
                    res.fedex = result_df.last_used_explainer
                    # Store the raw FedEx findings in the query and results mapping
                    title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = fedex_findings
                    fedex_findings = explanations
                    # Remove the LaTeX formatting from the FedEx findings
                    fedex_findings = fedex_findings.values.tolist()
                    pattern = re.compile(r'\$\\+bf{(.*?)}\$')
                    fedex_findings = [pattern.sub(r'\1', finding) for finding in fedex_findings]
                    fedex_findings = [finding.replace("(in green)", "").replace("\n", " ").replace("\\", "") for
                                      finding in fedex_findings]
                    res.fedex_findings = fedex_findings
                    n_fedex_findings = len(fedex_findings)
                    fedex_finding_str = f"{len(fedex_findings)} FEDEx findings"
                except Exception as e:
                    res.fedex_findings = f"Error"
                    res.fedex = None
                    n_fedex_findings = 0
                try:
                    metainsight_findings = result_df.explain(
                        explainer="metainsight",
                        top_k=2,
                        do_not_visualize=True,
                        max_filter_columns=metainsight_max_filter_cols,
                        max_aggregation_columns=metainsight_max_agg_cols,
                        display_mode='grid'
                    )
                    metainsight_findings = [finding.__str__() for finding in metainsight_findings]
                    # Store the MetaInsight objects in the query and results mapping
                    res.metainsight = result_df.last_used_explainer
                    res.metainsight_findings = metainsight_findings
                    n_metainsight_findings = len(metainsight_findings)
                    metainsight_finding_str = f"{len(metainsight_findings)} MetaInsight findings"
                except Exception as e:
                    res.metainsight_findings = "Error"
                    res.metainsight = None
                    n_metainsight_findings = 0
                results.append(res)
                if n_fedex_findings > 0 or n_metainsight_findings > 0:
                    log_str =  f"\t - Query {self.current_query_num}: {result.generating_query} produced {fedex_finding_str if fedex_finding_str else ''} {'and ' if fedex_finding_str and metainsight_finding_str else ''}{metainsight_finding_str}"
                else:
                    log_str = f"\t - Query {self.current_query_num}: {result.generating_query} produced no findings."
                if verbose:
                    print(log_str)
                self.log.append(log_str.replace("\t", "&emsp;"))
                total_findings.append(n_fedex_findings + n_metainsight_findings)
            self.current_query_num += 1

        return results, total_findings

    def do_llm_action(self, user_query: str = None, num_iterations: int = 10, fedex_top_k: int = 4, metainsight_top_k: int = 2,
                      metainsight_max_filter_cols: int = 3, metainsight_max_agg_cols: int = 3,
                      verbose=False, max_iterations_to_add: int = 3) \
            -> tuple[DataFrame, str | None, defaultdict[Any, QueryResultObject], list[str], QueryTree]:
        """
        Perform the deep dive analysis on the DataFrame using the LLM.

        :param user_query: A textual description of what the user wants to explore in the DataFrame.
        :param num_iterations: The number of iterations to perform in the analysis.
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
        self.verbose = verbose
        self.log = []
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
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER,
                               consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_PROVIDER),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER_URL,
                                   consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_PROVIDER_URL),
            model=os.getenv(consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_MODEL,
                            consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_MODEL)
        )
        system_message = self._define_task()
        data_description = self._describe_data(user_query)
        format_description = self._describe_input_format() + self._describe_output_format()
        format_description += self._describe_additional_output_format()
        query_and_results = defaultdict(QueryResultObject)
        query_tree = QueryTree(source_name=self.source_name)
        print_error = False
        truncate_by = 0
        iteration_num = 0
        max_iterations = num_iterations
        iterations_added = 0
        # Suppress warnings while we are running the automated data exploration, since LLMs are pros at
        # generating queries that will result in warnings, and we don't want to clutter the output with them.
        warnings.filterwarnings("ignore")
        plan = self._generate_initial_plan(
            client=client, user_query=user_query, num_iterations=max_iterations,
            data_description_str=data_description
        )
        self.log.append(f"Initial plan generated: {plan.replace('\t', '&emsp;')}")
        try:
            while iteration_num < max_iterations:
                if verbose:
                    print(f"Starting iteration {iteration_num + 1}/{max_iterations}")
                self.log.append(f"Iteration {iteration_num + 1}/{max_iterations} started:")
                # Format the history for the LLM
                formatted_history = self._format_history(history, truncate_early_by=truncate_by)
                # Create the user message for the LLM
                user_message = (
                    f"This is iteration {iteration_num + 1} out of {max_iterations} of the analysis process.\n"
                    f"User query: {user_query}\n"
                    f"The exploration plan is:\n{plan}\n"
                    f"Summary of the history of queries and findings:\n{formatted_history}\n"
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
                    iteration_added = False
                    if iterations_added < max_iterations_to_add:
                        iterations_added += 1
                        max_iterations += 1
                        iteration_added = True
                    if verbose:
                        print(f"\t - LLM request failed with error: {e}")
                        print(f"\t - Truncating history seen by the LLM by {truncate_by} rows and retrying...")
                        if iteration_added:
                            print(
                                f"\t - Adding one additional iteration to compensate, now {max_iterations} iterations in total.")
                        else:
                            print(
                                f"\t - Not adding any more iterations as the maximum number of additional iterations ({max_iterations_to_add}) has been reached.")
                    self.log.append(f"&emsp; - Iteration failed - Invalid request error {e}. {'Adding additional iteration' if iteration_added else 'Not adding more iterations since maximum number of additional iterations reached.'}")
                    iteration_num += 1
                    continue

                # Extract the queries from the response
                queries = self._extract_response(response, "<queries>", "</queries>")
                if queries is None or len(queries) == 0:
                    iteration_added = False
                    if iterations_added < max_iterations_to_add:
                        iterations_added += 1
                        max_iterations += 1
                        iteration_added = True
                    if verbose:
                        print(
                            f"\t - LLM did not generate queries or got the format wrong and queries could not be extracted during iteration {iteration_num + 1}. ")
                        if iteration_added:
                            print(
                                f"\t - Adding one additional iteration to compensate, now {max_iterations} iterations in total.")
                        else:
                            print(
                                f"\t - Not adding any more iterations as the maximum number of additional iterations ({max_iterations_to_add}) has been reached.")
                    iteration_num += 1
                    self.log.append(f"&emsp; - Iteration failed to generate queries. {'Adding additional iteration' if iteration_added else 'Not adding more iterations since maximum number of additional iterations reached.'}")
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
                    # Add the findings to the history DataFrame
                    for finding in findings:
                        split = finding.split(":", 1)
                        if len(split) < 2 or not split[0].strip().isdigit():
                            # The LLM gets specific instructions to use the format "query_num: finding".
                            # However, the LLM can be stupid and not follow the instructions, instead writing Query 1
                            # or query 1 or similar. In that case, use this backup regex.
                            pattern = re.compile(r"^(query|Query)\s*(\d+)\s*(.*)$")
                            match = pattern.match(finding)
                            if match:
                                query_num = match.group(2).strip()
                                finding = match.group(3).strip()
                            # If even this fails, we just skip the finding because there is no way to extract the query number.
                            else:
                                continue
                        else:
                            query_num = split[0].strip()
                            finding = split[1].strip()
                        try:
                            query_num = int(query_num.strip())
                        except ValueError:
                            continue
                        # The LLM can sometimes hallucinate and generate query numbers of future queries,
                        # which ends up being out of bounds for the history DataFrame.
                        try:
                            query_num_row = history.iloc[query_num]
                        except IndexError:
                            continue
                        if query_num_row['need_explanation']:
                            history.at[query_num, 'query_findings'] = finding
                            history.at[query_num, 'need_explanation'] = False
                if verbose:
                    print(f"\t - Generated {len(queries_series)} queries for iteration {iteration_num + 1}")
                self.log.append(f"&emsp; - Generated {len(queries_series)} queries")
                # Apply the queries to the DataFrame and update the history
                new_results = self._apply(queries_series, result_history_mapping)
                if not new_results:
                    iteration_num += 1
                    continue

                # Update the history DataFrame with new results
                analysis_results, findings_count = self._analyze_queries(
                    query_results=new_results,
                    fedex_top_k=fedex_top_k,
                    metainsight_max_filter_cols=metainsight_max_filter_cols,
                    metainsight_max_agg_cols=metainsight_max_agg_cols,
                    verbose=verbose,
                )
                all_errors = True
                for idx, (apply_res, analysis_res) in enumerate(zip(new_results, analysis_results)):
                    if apply_res.error_occurred:
                        history = history._append({
                            "query": f"{apply_res.index}: {apply_res.generating_query}",
                            "fedex_explainer_findings": None,
                            "metainsight_explainer_findings": None,
                            "error": str(analysis_res.error),
                            "need_explanation": False,
                            "query_description": descriptions[idx] if descriptions is not None and idx < len(
                                descriptions) else None,
                            "query_findings": None
                        }, ignore_index=True)
                    else:
                        if findings_count[idx] > 0:
                            all_errors = False
                        history = history._append({
                            "query": f"{apply_res.index}: {apply_res.generating_query}",
                            "fedex_explainer_findings": analysis_res.fedex_findings,
                            "metainsight_explainer_findings": analysis_res.metainsight_findings,
                            "error": None,
                            "need_explanation": True,  # We need explanations for the new queries
                            "query_description": descriptions[idx] if descriptions is not None and idx < len(
                                descriptions) else None,
                            "query_findings": None
                        }, ignore_index=True)
                    curr_index = len(history) - 1
                    # Update the query tree with the new query
                    query_tree.add_node(apply_res.index, apply_res.generating_query, curr_index)
                    query_and_results[curr_index] = analysis_res
                    # Update the result history mapping with the new results
                    result_history_mapping[len(history) - 1] = apply_res.result

                if all_errors:
                    iteration_added = False
                    if iterations_added < max_iterations_to_add:
                        iterations_added += 1
                        max_iterations += 1
                        iteration_added = True
                    if verbose:
                        print(f"\t - All queries in iteration {iteration_num + 1} failed with errors. ")
                        if iteration_added:
                            print(f"\t - Adding one additional iteration to compensate, now {max_iterations} iterations in total.")
                        else:
                            print(f"\t - Not adding any more iterations as the maximum number of additional iterations ({max_iterations_to_add}) has been reached.")
                    self.log.append(f"&emsp; - Iteration failed, all queries produced no findings. {'Adding additional iteration' if iteration_added else 'Not adding more iterations since maximum number of additional iterations reached.'}")
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
            if final_report is None or len(final_report) == 0:
                final_report = "No final report generated."
            visualization_queries = self._extract_response(final_report_response, "<vis>", "</vis>")
            if visualization_queries is None or len(visualization_queries) == 0:
                visualization_queries = []
            else:
                visualization_queries = visualization_queries.split("\n")
                visualization_queries = [query.replace("*", "").strip() for query in visualization_queries if
                                         query.strip() and query.startswith('*')]
                # Fallback for visualization queries, if the LLM provided them in comma separated format
                if len(visualization_queries) == 1 and ',' in visualization_queries[0]:
                    visualization_queries = visualization_queries[0].split(',')
                    visualization_queries = [query.strip() for query in visualization_queries if query.strip()]
            self.history = history
            self.final_report = final_report
            self.query_and_results = query_and_results
            self.query_tree = query_tree
            return history, final_report, query_and_results, visualization_queries, query_tree
        finally:
            # Restore the pandas display options and console output to their original state
            # This is inside the finally block to ensure it always runs, even if an error occurs
            pd.set_option('display.max_rows', display_max_rows)
            pd.set_option('display.max_columns', display_max_columns)
            pd.set_option('display.width', display_width)
            pd.set_option('display.max_colwidth', display_max_colwidth)
            # Restore the warnings filter to its default state
            warnings.filterwarnings("default")
            if print_error:
                print("LLM failed to generate any queries. ")

    def do_follow_up_action(self, history: pd.DataFrame = None, final_report=None,
                            query_and_results: dict[int, QueryResultObject] = None,
                            query_tree: QueryTree = None,
                            source_name: str = None,
                            log: list[str] = None,
                            beautify_fedex: bool = False, beautify_metainsight: bool = False,
                            fedex_beautify_code: str = None, metainsight_beautify_code: str = None,
                            ):
        """
        Visualize the results of the deep dive analysis.

        :param history: The history of queries and findings from the deep dive analysis. Optional.
        :param final_report: The final report generated by the LLM. Optional.
        :param query_and_results: A dictionary mapping query indices to their results, including FedEx and MetaInsight findings. Optional.
        :param visualization_queries: A list of query indices that are deemed important for visualization. Optional.
        :param query_tree: A QueryTree object containing the structure of the queries and their ancestry. Optional.
        :param source_name: The name of the source DataFrame, if different from the default. Optional.
        If all optional parameters (except for source_name) are provided, the method will visualize the deep dive
        using the provided parameters, enabling visualization without running the LLM again.
        If the parameters are not provided, it will use the results from the last run of do_llm_action() to visualize the deep dive.
        :param beautify_fedex: If True, use the LLM beautifier to format the FedEx findings for better readability. Default is False.
        :param beautify_metainsight: If True, use the LLM beautifier to format the MetaInsight findings for better readability. Default is False.
        :param beautify_query_tree: If True, use the LLM beautifier to format the query tree for better readability. Default is False.

        :return: The visualized deep dive results as a ipywidgets tab.
        """
        all_params_provided = history is not None and final_report is not None and query_and_results is not None \
                              and query_tree is not None and log is not None
        if all_params_provided:
            visualizer = AutomatedExplorationVisualizer(
                history=history,
                query_and_results=query_and_results,
                query_tree=query_tree,
                final_report=final_report,
                source_name=source_name if source_name else self.source_name,
                beautify_fedex=beautify_fedex,
                beautify_metainsight=beautify_metainsight,
                log=log
            )
            visualizer.fedex_beautify_code = fedex_beautify_code
            visualizer.metainsight_beautify_code = metainsight_beautify_code
            return visualizer.visualize_data_exploration()
        all_self_params_exist = self.history is not None and self.final_report is not None \
                                and self.query_and_results is not None \
                                and self.query_tree is not None
        if not all_self_params_exist:
            raise ValueError("No deep dive analysis has been performed yet. Please run do_llm_action() first.")
        if self.visualizer is None:
            visualizer = AutomatedExplorationVisualizer(
                history=self.history,
                query_and_results=self.query_and_results,
                query_tree=self.query_tree,
                final_report=self.final_report,
                source_name=source_name if source_name else self.source_name,
                beautify_fedex=self.beautify_fedex,
                beautify_metainsight=self.beautify_metainsight,
                verbose=self.verbose,
                log=self.log
            )
            self.visualizer = visualizer
        return self.visualizer.visualize_data_exploration()
