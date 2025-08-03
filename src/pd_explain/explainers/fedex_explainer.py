import pandas as pd
from matplotlib import pyplot as plt

from fedex_generator.Operations.GroupBy import GroupBy
from .explainer_interface import ExplainerInterface
from typing import List, Literal
from pandas import DataFrame
from copy import deepcopy
from IPython.display import display
import re
import warnings

from fedex_generator.Operations.Filter import Filter
from pd_explain.llm_integrations import ExplanationReasoning
from pd_explain.experimental.query_recommenders import QueryLogger
from pd_explain.experimental.query_recommenders.query_score_functions import score_queries
from pd_explain.visualizer_adaptations.carousel_adapter import CarouselAdapter
from pd_explain.llm_integrations.beta_integrations.visualization_beautifier import VisualizationBeautifier


class FedexExplainer(ExplainerInterface):
    """
    This class is responsible for interfacing between explainable data frames and the explainers implemented in the fedex
    package.
    Because pd-explain was originally designed with fedex in mind, this class is just a wrapper around the explain method
    of the explainer object.
    """

    def __init__(self, operation=None, schema: dict = None, attributes: List = None, top_k: int = None,
                 explainer='fedex', figs_in_row: int = 2, show_scores: bool = False, title: str = None,
                 corr_TH: float = 0.7, consider='right', value=None, attr=None, ignore=None,
                 use_sampling: bool = True, sample_size=5000, debug_mode: bool = False,
                 add_llm_context_explanations: bool = False,
                 do_not_visualize: bool = False,
                 log_query: bool = True,
                 display_mode: Literal['carousel', 'grid'] = 'grid',
                 beautify: bool = False,
                 beautify_max_fix_attempts: int = 10,
                 silent_beautify: bool = False,
                 return_beautify_code: bool = False,
                 generalize_beautify_code: bool = False,
                 *args, **kwargs):
        """
        Initialize the FedexExplainer object.
        The FedexExplainer works as an interface for calling the explain method of the fedex explainer objects.

        :param operation: The operation object to explain.
        :param schema: The schema of the data.
        :param attributes: The attributes to consider in the explanation.
        :param top_k: The number of top explanations to generate.
        :param explainer: The type of explainer to use. Can be 'fedex', 'outlier', or 'shapley'.
        :param target: The target value for the outlier explanation.
        :param dir: The direction of the outlier explanation. Can be 'high' or 'low'.
        :param figs_in_row: The number of figures to display in a row.
        :param show_scores: Whether to show the scores in the explanation.
        :param title: The title of the explanation.
        :param corr_TH: The correlation threshold. Attributes with a correlation above this threshold will be ignored.
        :param consider: The side of the join to consider in the explanation.
        :param use_sampling: Whether to use sampling to speed up the explanation generation process. Default is True.
        :param debug_mode: Developer option. Disables multiprocessing and enables debug prints. Defaults to False.
        :param add_llm_context_explanations: Whether to add LLM context explanations to the explanation. Defaults to False.
        :param do_not_visualize: If True, the visualizations will not be generated. This is useful for when the explainer
        is used in a context where visualizations are not needed, such as part of a pipeline.
        :param log_query: If True, the query will be logged to the query logger. Defaults to True.
        :param display_mode: The type of visualization to use. Can be 'carousel' or 'grid'. Defaults to 'grid'.
        :param beautify: If True, use a LLM to create new visualizations for the explanations, which should look (maybe) better
        and be easier to understand compared to the templates used. Defaults to False.
        :param beautify_max_fix_attempts: The maximum number of attempts to fix the code returned by the LLM beautifier.
        """

        if operation is None:
            raise ValueError('All fedex explainers require an operation object')

        if schema is None:
            schema = {}

        if attributes is None:
            attributes = []
            if top_k is None:
                top_k = 1
        else:
            if top_k is None:
                top_k = len(attributes)
        if ignore is None:
            ignore = []

        # Convert the source_df and result_df to DataFrame objects, to avoid overhead from overridden methods
        # in ExpDataFrame, as well as to avoid any bad interactions between those methods and the explainer.
        original_operation = operation
        operation = deepcopy(operation)
        if hasattr(operation, 'source_df'):
            operation.source_df = DataFrame(operation.source_df) if operation.source_df is not None else None
        elif hasattr(operation, 'left_df'):
            operation.left_df = DataFrame(operation.left_df) if operation.left_df is not None else None
            operation.right_df = DataFrame(operation.right_df) if operation.right_df is not None else None
        operation.result_df = DataFrame(operation.result_df) if operation.result_df is not None else None

        self._original_operation = original_operation
        self._schema = schema
        self._attributes = attributes
        self._top_k = top_k
        self._explainer = explainer
        self._figs_in_row = figs_in_row
        self._show_scores = show_scores
        self._title = title
        self._corr_TH = corr_TH
        self._consider = consider
        self._value = value
        self._attr = attr
        self._ignore = ignore
        self._operation = operation
        self._results = None
        self._use_sampling = use_sampling
        self._sample_size = sample_size
        self._debug_mode = debug_mode
        self._add_llm_context_explanations = add_llm_context_explanations
        self._logger = QueryLogger()
        self._log_query = log_query
        self._do_not_visualize = do_not_visualize
        self._added_explanations = None
        self._query = None
        self._query_type = None
        if display_mode not in ['carousel', 'grid']:
            warnings.warn(f"Visualization type {display_mode} is not supported. Defaulting to 'grid'.")
            display_mode = 'grid'
        self._display_mode = display_mode
        self._beautify = beautify
        self._beautify_max_fix_attempts = beautify_max_fix_attempts
        self._silent_beautify = silent_beautify
        self._return_beautify_code = return_beautify_code
        self._generalize_beautify_code = generalize_beautify_code
        self._do_not_visualize_beautify = False

    def generate_explanation(self):

        if self._operation is None:
            self._results = "No operation was found."
            return self._results

        else:
            self._results, scores = self._operation.explain(
                schema=self._schema, attributes=self._attributes, top_k=self._top_k,
                figs_in_row=self._figs_in_row, show_scores=self._show_scores, title=self._title, corr_TH=self._corr_TH,
                explainer=self._explainer, consider=self._consider, cont=self._value, attr=self._attr,
                ignore=self._ignore, use_sampling=self._use_sampling, sample_size=self._sample_size,
                debug_mode=self._debug_mode,
                # We always pass draw_figures as False, and move the logic of drawing the figures to the visualize method
                draw_figures=False,
                return_scores=True
            )

            self._query, self._query_type = self._create_query_string(operation=self._operation)

            if self._log_query:
                score = score_queries(scores)

                # Log the query to the query logger
                self._logger.log_query(
                    dataframe_name=self._operation.source_name,
                    query=self._query,
                    score=score
                )

        if isinstance(self._operation, Filter):
            self._original_operation.cor_deleted_atts = self._operation.cor_deleted_atts
            self._original_operation.not_presented = self._operation.not_presented
            self._original_operation.corr = self._operation.corr

        return self._results

    def can_visualize(self) -> bool:
        return not self._do_not_visualize

    def __len__(self):
        """
        Returns the number of explanations generated by the explainer.
        This is useful for checking if the explainer has generated any explanations.
        """
        if self._results is None:
            return 0
        if isinstance(self._results, str):
            return 0
        return len(self._results[4])


    def _visualize(self, title, scores, K, figs_in_row, explanations, bins,
                   influence_vals, source_name, show_scores, added_explanations, beautify_code: str = None)\
            -> None | str:
        """
        A helper method to avoid code duplication in the visualize method.
        This method is responsible for drawing the figures using the operation's draw_figures method.
        All parameters but added_explanations are simply an unpacking of the results tuple.
        The added_explanations parameter is a dictionary with explanations as keys and additional text as values.
        """
        # Draw the figures using the operation's draw_figures method.
        if self._display_mode == 'grid':
            _, fig = self._operation.draw_figures(
                title=title,
                scores=scores,
                K=K,
                figs_in_row=figs_in_row,
                explanations=explanations,
                bins=bins,
                influence_vals=influence_vals,
                source_name=source_name,
                show_scores=show_scores,
                added_text=added_explanations,
                added_text_name="LLM Reasoning" if self._add_llm_context_explanations else None
            )
            if self._beautify:
                plt.close(fig)  # Close the figure to avoid displaying it immediately
                requester_name = ""
                if self._generalize_beautify_code:
                    requester_name = "fedex-all"
                else:
                    if isinstance(self._operation, GroupBy):
                        requester_name = "fedex-gb"
                    else:
                        requester_name = "fedex"
                beautifier = VisualizationBeautifier(
                    visualization_object=fig,
                    data=self._operation.source_df,
                    visualization_params={
                        'title': title,
                        'scores': scores,
                        'K': K,
                        'figs_in_row': figs_in_row,
                        'explanations': explanations,
                        'bins': bins,
                        'influence_vals': influence_vals,
                        'source_name': source_name,
                        'show_scores': show_scores
                    },
                    must_generalize=self._generalize_beautify_code,
                    requester_name=requester_name,
                    max_fix_attempts=self._beautify_max_fix_attempts,
                    silent=self._silent_beautify,
                )
                if not beautify_code:
                    try:
                        tab, code = beautifier.do_llm_action()
                    except Exception as e:
                        print(f"Beautification failed with error: {e}. Displaying the original figure.")
                        tab = None
                else:
                    # If the beautify_code is provided, we will use it to create the tab.
                    tab = beautifier.beautify_from_code(beautify_code)
                    code = None
                if tab is not None:
                    # If the beautifier returns a tab, we will display it.
                    if not self._do_not_visualize_beautify:
                        display(tab)
                    if self._return_beautify_code:
                        return code
                else:
                    # If the beautifier returns None, we will display the original figure.
                    print("Beautifier failed to generate a new visualization. Displaying the original figure.")
                    if not self._do_not_visualize_beautify:
                        display(fig)
        elif self._display_mode == 'carousel':
            if self._beautify:
                print("Beautification is not supported in carousel display mode.")
            with CarouselAdapter() as adapter:
                for i in range(len(explanations)):
                    # These all still need to be iterables, with explanation in particular being a Series.
                    explanation = pd.Series(explanations.iloc[i])
                    bin = [bins.iloc[i]]
                    influence_val = [influence_vals.iloc[i]]
                    score = [scores.iloc[i]]
                    # If the added_explanations is not None, we will add the explanation to the carousel.
                    _, fig = self._operation.draw_figures(
                        title=title,
                        scores=score,
                        K=K,
                        figs_in_row=figs_in_row,
                        explanations=explanation,
                        bins=bin,
                        influence_vals=influence_val,
                        source_name=source_name,
                        show_scores=show_scores,
                        added_text=added_explanations,
                        added_text_name="LLM Reasoning" if self._add_llm_context_explanations else None
                    )
                    plt.close(fig)  # Close the figure to avoid displaying it immediately
                    adapter.capture_output(fig)  # Capture the output for the carousel
        return None

    def visualize(self, fedex_output: tuple = None, added_explanations=None, beautify_code: str = None) -> None | str:
        """
        Visualizes the explanation generated by the explainer.
        If the explanation was generated by the explainer, it will use the stored results.
        Alternatively, it is possible to pass the explainer's output as a parameter, which is useful if we want
        to visualize the explanation in a different context, such as in a pipeline.

        :param fedex_output: The output of the explainer's explain method. If None, it will use the stored results.
        :param added_explanations: Additional explanations to add to the visualization. Will be ignored if fedex_output None.

        :return: None
        """
        if fedex_output is None:
            title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = self._results
            # If the user has requested LLM explanations, we will generate them here.
            if self._add_llm_context_explanations:

                # Get the source dataframe (or left and right dataframes)
                if hasattr(self._operation, 'source_df'):
                    source_df = self._operation.source_df
                    right_df = None
                elif hasattr(self._operation, 'left_df'):
                    source_df = self._operation.left_df
                    right_df = self._operation.right_df
                else:
                    raise ValueError(
                        "The operation object does not have a source DataFrame. This should not happen with fedex operations.")

                if hasattr(self._operation, 'left_df'):
                    right_name = self._operation.right_name
                else:
                    right_name = None

                # Create an ExplanationReasoning object to generate the LLM explanations
                reasoner = ExplanationReasoning(
                    data=source_df,
                    source_name=source_name,
                    query=self._query,
                    explanations_found=explanations,
                    right_df=right_df if right_df is not None else None,
                    query_type=self._query_type,
                    right_name=right_name
                )
                # I tried making this async, but it didn't work. I think it's because the main thread exits before the
                # async function finishes, which makes it so the event loop is closed before the function finishes /
                # before the callback function is called, making it not draw the figures at all (and obviously waiting
                # for it to finish is not an option, since that would just make us synchronous again - the only blocking
                # call is the LLM call, which is what we want to avoid, everything else is fast).
                self._added_explanations = reasoner.do_llm_action()
                # The ExplanationReasoning object will return None if the API key is not set.
                if self._added_explanations is not None:
                    self._added_explanations = {
                        explanations.loc[i]: {
                            "added_text": self._added_explanations.loc[i],
                            "position": "bottom"
                        }
                        for i in explanations.index.values
                    }
                added_explanations = self._added_explanations

        else:
            # If we got the output from an external source, we will visualize it here. This may happen if the explainer was
            # used in a pipeline and the visualization was not done in the generate_explanation method.
            if isinstance(fedex_output, str):
                # If the fedex_output is a string, it means that no explanation was generated, so we will not visualize anything.
                return None
            title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = fedex_output

        # Draw the figures
        return self._visualize(
            title=title,
            scores=scores,
            K=K,
            figs_in_row=figs_in_row,
            explanations=explanations,
            bins=bins,
            influence_vals=influence_vals,
            source_name=source_name,
            show_scores=show_scores,
            added_explanations=added_explanations,
            beautify_code=beautify_code
        )

    def get_explanation_in_textual_description(self, index: int) -> str:
        """
        Get explanations after they have already been generated.
        If the explanations have not been generated yet, this method will raise an error.

        :param index: Which index of the explanation to return.

        :return: A string explaining which operation was performed, what was found, and the explanation itself.
        Also includes any additional explanation generated by the LLM, if available.
        """
        if self._results is None:
            raise ValueError("No explanations have been generated yet. Please call generate_explanation() first.")
        title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = self._results
        # Locate the explanation at the given index, and its added explanation if it exists.
        explanation_to_return = explanations.iloc[index]
        pattern = re.compile(r'\$\\+bf{(.*?)}\$')
        explanation_to_return_formatted = pattern.sub(r'\1', explanation_to_return)  # Remove LaTeX formatting
        explanation_to_return_formatted = explanation_to_return_formatted.replace("(in green)", "").replace("\n",
                                                                                                            " ").replace(
            "\\", "")
        added_explanation = None
        if self._added_explanations is not None and explanation_to_return in self._added_explanations:
            added_explanation = self._added_explanations[explanation_to_return]['added_text']

        explanation_string = (f"Among the most interesting statistical changes after running the query {self._query} "
                              f"on dataframe {source_name}, we found (using automated analysis):\n"
                              f"{explanation_to_return_formatted}.\n")
        if added_explanation is not None:
            explanation_string += (
                f"Additionally, a LLM with limited context and no ability to query the data suggested "
                f"that the cause of this change may be: {added_explanation.replace('\n', ' ')}.\n")
        return explanation_string
