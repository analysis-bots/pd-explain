import itertools
from typing import List, Any
from collections import namedtuple, defaultdict
from dataclasses import dataclass
import re
import sys
import os
import ipywidgets as widgets
import matplotlib.pyplot as plt

import pandas as pd
from pandas import DataFrame

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client
from pd_explain.explainers.fedex_explainer import FedexExplainer
from pd_explain.explainers.metainsight_explainer import MetaInsightExplainer

apply_result = namedtuple("apply_result", ["index",
                                           "result",
                                           "generating_query",
                                           "error_occurred"])


@dataclass(frozen=False)
class QueryResultObject:
    """
    A class to hold the results of a query applied to a DataFrame.
    It contains the findings from the FedEx and MetaInsight explainers, as well as any error that occurred.
    """
    fedex: FedexExplainer | None = None
    metainsight: MetaInsightExplainer | None = None
    error: str | None = None


tree_node = namedtuple("tree_node", ["source", "query", "children"])


class QueryTree:
    """
    A class to built and maintain the query tree for the deep dive analysis.
    """

    def __init__(self, source_name: str = "Original DataFrame"):
        self.tree = defaultdict(tree_node)
        self.source_name = source_name
        self.tree[0] = tree_node(source=None, query=source_name, children=[])

    def add_node(self, source_idx: int, query: str, new_index: int) -> None:
        """
        :param source_idx: The index of the query that this query is based on.
        :param query: The query to add to the tree.
        :param new_index: The index of the new query in the tree.
        """
        if source_idx not in self.tree:
            raise ValueError(f"Source index {source_idx} not found in the tree.")
        self.tree[new_index] = tree_node(source=source_idx, query=query, children=[])
        self.tree[source_idx].children.append(new_index)

    def get_node(self, idx: int) -> tree_node | None:
        return self.tree.get(idx, None)


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
                "If there was an error executing a query, it will be provided in the history as well. Make absolutely sure you do not repeat errors. "
                "Your available operators are: filtering using boolean conditions and groupby with aggregation functions [mean, sum, count, nunique, min, max, median, std, sem, var, size, prod]."
                "All other operators and functions, such as describe, apply, join, quantile, query, etc. are illegal and will result in an error. "
                "You can freely combine these operators or use lambda functions, but you must always return a DataFrame. "
                "Do not ever select only a single column after performing an operation. If you perform column selection, it must always be multiple columns."
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
        data_description = (
                    f"The DataFrame contains {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns. "
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

    def _describe_output_format(self) -> str:
        return (f"You are expected to generate {self.num_queries_per_iteration} queries. "
                f"Each query must be in the format index: query, where index is the index of the row in the history "
                f"that you want to apply the query to. Use index 0 for the original DataFrame. "
                "Example for a query on index i -  i: [x > 5], i: .groupby('column_name').mean(), etc. "
                "Make sure you use the correct index for the query you want to apply, and never select an index above the current max index in the history. "
                "The query must be a valid Pandas query applicable using eval() with the syntax [df]{query}, where [df] "
                "is a placeholder for the name of the actual DataFrame (which is not provided to you). "
                "If you need to use the DataFrame's name (for example to filter it), use the placeholder [df] in your query, and the system will replace it with the actual DataFrame name. "
                "For example, if you want to filter by a column, write [[df]['column_name'] > 5]. "
                f"The output must be a list of queries, where each row in the list starts with a * symbol and ends with a new line. "
                f"The list should be surrounded by <queries> and </queries> tags. so it can be easily extracted programmatically. "
                f"Avoid repeating queries that have already been applied in the history. "
                f"If you use the std function, make sure to also specify the ddof parameters, otherwise std throws an error."
                f"If you use the agg function, make sure to provide the aggregations as a dictionary, i.e. {{column_name: 'agg_func'}}, "
                f"and not as a list of any kind, i.e. ['agg_func_1', 'agg_func_2'] as the list format will throw a NoneType error. ")

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
        DataFrame, str | None, defaultdict[Any, QueryResultObject], list[str], QueryTree]:
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
        query_and_results = defaultdict(QueryResultObject)
        query_tree = QueryTree(source_name=self.source_name)
        # Temporarily disable console output, because some explainers print to the console, and we don't want that
        current_output = sys.stdout
        sys.stdout = open(os.devnull, 'w')  # Redirect stdout to null device to suppress output
        try:
            for iteration in range(self.num_iterations):
                # Format the history for the LLM
                formatted_history = self._format_history(history)
                # Create the user message for the LLM
                user_message = (
                    f"This is iteration {iteration + 1} out of {self.num_iterations} of the analysis process.\n"
                    f"User query: {user_query}\n"
                    f"History of queries and findings:\n{formatted_history}\n"
                    f"Data description:\n{data_description}\n"
                    f"Format description:\n{format_description}")
                try:
                    response = client(
                        system_messages=[system_message],
                        user_messages=[user_message]
                    )
                # If anything goes wrong with the LLM, we break out of the loop
                except Exception as e:
                    break
                # Extract the queries from the response
                queries = self._extract_response(response, "<queries>", "</queries>")
                if queries is None or len(queries) == 0:
                    continue
                queries = queries.split("\n")
                queries = [query.replace("*", "").strip() for query in queries if
                           query.strip() and query.startswith('*')]
                # Split the queries into a Series
                queries_series = pd.Series(queries)
                # Apply the queries to the DataFrame and update the history
                new_results = self._apply(queries_series, result_history_mapping)
                if not new_results:
                    continue
                # Update the history DataFrame with new results
                for result in new_results:
                    curr_index = len(history)
                    if result.error_occurred:
                        history = history._append({
                            "query": f"{result.index}: {result.generating_query}",
                            "fedex_explainer_findings": None,
                            "metainsight_explainer_findings": None,
                            "error": str(result.result)
                        }, ignore_index=True)
                        # Store the error in the query and results mapping, so it can be used later
                        query_and_results[curr_index] = QueryResultObject(
                            fedex=None,
                            metainsight=None,
                            error=str(result.result)
                        )
                    else:
                        # Assuming FedEx and MetaInsight explainers are applied here
                        result_df = result.result
                        try:
                            fedex_findings = result_df.explain(
                                explainer="fedex",
                                top_k=4,
                                do_not_visualize=True
                            )
                            # Store the raw FedEx findings in the query and results mapping
                            query_and_results[curr_index].fedex = result_df.last_used_explainer
                            title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = fedex_findings
                            fedex_findings = explanations
                            # Remove the LaTeX formatting from the FedEx findings
                            fedex_findings = fedex_findings.values.tolist()
                            fedex_findings = [finding.replace("bf", "").replace("$", "").replace("\n", "") for finding
                                              in
                                              fedex_findings]
                        except Exception as e:
                            fedex_findings = f"Error: {str(e)}"
                            query_and_results[curr_index].fedex = None
                        try:
                            metainsight_findings = result_df.explain(
                                explainer="metainsight",
                                top_k=2,
                                do_not_visualize=True,
                                max_filter_columns=3,
                                max_aggregation_columns=3,
                            )
                            # Store the MetaInsight objects in the query and results mapping
                            query_and_results[curr_index].metainsight = result_df.last_used_explainer
                        except Exception as e:
                            metainsight_findings = f"Error: {str(e)}"
                            query_and_results[curr_index].metainsight = None
                        history = history._append({
                            "query": f"{result.index}: {result.generating_query}",
                            "fedex_explainer_findings": fedex_findings,
                            "metainsight_explainer_findings": metainsight_findings,
                            "error": None
                        }, ignore_index=True)
                        # Update the query tree with the new query
                        query_tree.add_node(result.index, result.generating_query, curr_index)
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
                                         f"according to the user query. This report, while it should be concise, should be detailed enough for the user to "
                                         f"understand and draw conclusions from.\n"
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
                return history, final_report, query_and_results, visualization_queries, query_tree
        finally:
            # Backup inside the finally block to ensure we restore all options even if an error occurs
            pd.set_option('display.max_rows', display_max_rows)
            pd.set_option('display.max_columns', display_max_columns)
            pd.set_option('display.width', display_width)
            pd.set_option('display.max_colwidth', display_max_colwidth)
            # Re-enable console output if it was disabled
            sys.stdout = current_output if current_output else sys.__stdout__

    def _create_query_string(self, query_idx: int, query_tree: QueryTree) -> str:
        """
        Create a string representation of the query ancestry for a given query index.
        :param query_idx: The index of the query in the query tree.
        :param query_tree: The query tree containing the queries and their ancestry.
        :return: A string representation of the query ancestry.
        """
        current_query_node = query_tree.get_node(query_idx)
        ancestor_idx = current_query_node.source
        query_ancestry = []
        # Traverse the query tree to find the ancestry of the current query
        while ancestor_idx is not None:
            query_ancestry.append(current_query_node)
            ancestor_node = query_tree.get_node(ancestor_idx)
            ancestor_idx = ancestor_node.source if ancestor_node else None
            current_query_node = ancestor_node
        # Reverse to show from root to current query
        query_ancestry.reverse()
        query_string = f"{self.source_name}"
        if query_ancestry:
            query_string += " -> " + " -> ".join(
                [f"{node.query}" for node in query_ancestry if node.query is not None]
            )
        return query_string

    def _create_important_visualizations_tab(self, query_and_results: dict[int, QueryResultObject],
                                             visualization_queries: list[str | int],
                                             query_tree_str: dict[int, str]) -> widgets.Tab:
        visualization_vbox = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                height='100%',  # Set the height to 90vh to fill most of the screen
                overflow_y='auto'  # Allow vertical scrolling if content is taller than 90vh
            )
        )
        disclaimer = widgets.HTML(
            value="<p style='font-weight: bold; padding-bottom: 10px; font-size: 14px;'>"
                  "The queries presented in this tab are the ones deemed most important by the LLM when drawing up the conclusions "
                  "from the analysis. <br>"
                  "As such, these may not truly be the most important queries performed during the analysis."
                  "</p>"
        )
        #  Create the “Main Query Visualizations” tab,
        #  which itself has subtabs for each query.
        visualization_subtabs = widgets.Tab(
            layout=widgets.Layout(
                width='100%',
                height='100%',  # <— let it inherit the parent's 90vh
                overflow_y='auto'  # <— scroll if the content is taller
            )
        )

        subtabs = []

        if not visualization_queries:
            no_viz = widgets.HTML(value="<p>No important queries were identified for visualization.</p>")
            subtabs.append((no_viz, "No Visualizations"))

        else:
            for i, query_idx in enumerate(visualization_queries):
                try:
                    query_idx = int(query_idx.strip())
                except ValueError:
                    continue

                query_info = query_and_results.get(query_idx, None)
                if query_info is None:
                    continue

                something_visualized = False

                query_string = query_tree_str[query_idx]
                query_title = widgets.HTML(value=f"<h2 style='margin-top: 20px; margin-bottom: 10px; align: center;'>"
                                                 f"Query {query_idx}: {query_string}</h2>")

                # Build a VBox that contains all plots for this single query_idx
                items_for_this_query = []
                items_for_this_query.append(query_title)

                # Visualize the FedEx and MetaInsight findings if they exist
                if query_info.fedex is not None and len(query_info.fedex) > 0:
                    fedex_title = widgets.HTML(value="<h3>Statistical Changes Analysis (FEDEx Explainer)</h3>")
                    fedex_output = widgets.Output(layout=widgets.Layout(width='100%'))
                    with fedex_output:
                        plt.close('all')
                        query_info.fedex.visualize(query_info.fedex._results)
                        plt.show()
                    items_for_this_query.append(fedex_title)
                    items_for_this_query.append(fedex_output)
                    something_visualized = True

                if query_info.metainsight is not None and len(query_info.metainsight) > 0:
                    meta_title = widgets.HTML(value="<h3>Pattern Detection (MetaInsight Explainer)</h3>")
                    meta_output = widgets.Output(layout=widgets.Layout(width='100%'))
                    with meta_output:
                        plt.close('all')
                        query_info.metainsight.visualize()
                        plt.show()
                    items_for_this_query.append(meta_title)
                    items_for_this_query.append(meta_output)
                    something_visualized = True

                # If no visualizations were generated, add a message indicating that
                if not something_visualized:
                    items_for_this_query.append(
                        widgets.HTML(value=f"<p>No visualizations available for this query</p>")
                    )

                # Wrap the plot outputs in a VBox _without_ imposing a fixed height here.
                # If you give them no height, each Output widget will be as tall as its figure.
                # The vertical scrollbar will come from the parent Tab, not from this VBox.
                query_vbox = widgets.VBox(
                    children=items_for_this_query,
                    layout=widgets.Layout(
                        width='100%',
                        # *** Notice: we do NOT set height='800px' here. We let the figure expand to its full height. ***
                        overflow_y='visible',  # <— let the children decide their own height
                        overflow_x='auto',
                        border='1px solid #ddd',
                        padding='10px'
                    )
                )

                subtabs.append((query_vbox, f"Query {query_idx}"))

        # Now actually assign the children of visualization_subtabs
        visualization_subtabs.children = [content for content, _ in subtabs]
        for i, (_, title) in enumerate(subtabs):
            visualization_subtabs.set_title(i, title)

        visualization_vbox.children = [disclaimer, visualization_subtabs]
        return visualization_vbox

    def _create_query_tree_tab(self, history: pd.DataFrame, query_tree: dict[int, QueryResultObject], query_tree_str: dict[int, str]) -> widgets.Tab:
        """
        Create a tab for the query tree visualization.
        :param history: The history DataFrame containing the queries and findings.
        :param query_tree: The query tree containing the queries and their ancestry.
        :param query_tree_str: A dictionary mapping query indices to their string representations.
        :return: A Tab widget containing the query tree visualization.
        """
        # Placeholder for the Query Tree tab
        query_tree_tab = widgets.HTML(value="<p>Query tree will be implemented in a future step.</p>")
        return query_tree_tab

    def visualize_deep_dive(self,
                            history,
                            final_report,
                            query_and_results,
                            visualization_queries,
                            query_tree) -> widgets.Tab:
        """
        Visualize the deep dive analysis in a Jupyter notebook with a tabbed interface.

        :param history: The history DataFrame containing the queries and findings.
        :param final_report: The final report generated by the LLM.
        :param query_and_results: A dictionary mapping query indices to their results.
        :param visualization_queries: A list of query indices that are important for visualization.
        :param query_tree: The query tree containing the queries and their ancestry.

        :return: A Tab widget containing the visualizations.
        """

        # Use the query tree to create a string representation of each query
        query_tree_str = {idx: self._create_query_string(idx, query_tree) for idx in query_tree.tree.keys()}

        #  Create the main tabs: Summary | Important Queries | Query Tree
        main_tabs = widgets.Tab(
            layout=widgets.Layout(
                width='100%',
                height='120vh',  # <— fix the height of the entire Tab widget to 90% of viewport
                overflow_y='auto'  # <— enable scrolling on the outer Tab if it overflows
            )
        )

        # Tab 0: Text Summary / HTML
        if final_report is None:
            summary_html = "<p>No summary was generated.</p>"
        else:
            formatted_report = final_report.replace('\n\n', '</p><p>')
            formatted_report = formatted_report.replace('\n', '<br>')
            # Replace ** with <strong> and ** with </strong> for bold text
            pattern = re.compile(r'\*\*(.*?)\*\*')
            formatted_report = pattern.sub(r'<strong>\1</strong>', formatted_report)
            summary_html = f"""
            <div style='padding:20px; max-width:800px; line-height:1.5; font-family:Arial,sans-serif;'>
                <p>{formatted_report}</p>
                <br>
                <br>
                <p><strong>This report was generated by a LLM, and may contain inaccuracies or errors.</strong></p>
                <p>Please review the findings and visualizations carefully, and use your own judgment to draw conclusions.</p>
                <p>For more information about the queries the LLM deemed the most important when finalizing the report, please refer to the "Important Queries" tab.</p>
                <p>To see the entire query tree and how the queries relate to each other, please refer to the "Query Tree" tab.</p>
            </div>
            """
        summary_tab = widgets.HTML(value=summary_html)

        visualizations_subtabs = self._create_important_visualizations_tab(
            query_and_results, visualization_queries, query_tree_str
        )

        # Create a placeholder for Query Tree
        query_tree_tab = self._create_query_tree_tab(history, query_tree, query_tree_str)

        # Hook everything into the main_tabs widget
        main_tabs.children = [summary_tab, visualizations_subtabs, query_tree_tab]
        main_tabs.set_title(0, "Summary")
        main_tabs.set_title(1, "Important Queries")
        main_tabs.set_title(2, "Query Tree")

        return main_tabs
