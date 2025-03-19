from .explainer_interface import ExplainerInterface
from typing import List
from pandas import DataFrame
from copy import deepcopy

from fedex_generator.Operations.Filter import Filter
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Join import Join
from pd_explain.llm_integrations import ExplanationReasoning


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
                 add_llm_context_explanations: bool = True,
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

    def generate_explanation(self):

        if self._operation is None:
            self._results = "No operation was found."
            return self._results

        else:
            self._results = self._operation.explain(
                schema=self._schema, attributes=self._attributes, top_k=self._top_k,
                figs_in_row=self._figs_in_row, show_scores=self._show_scores, title=self._title, corr_TH=self._corr_TH,
                explainer=self._explainer, consider=self._consider, cont=self._value, attr=self._attr,
                ignore=self._ignore, use_sampling=self._use_sampling, sample_size=self._sample_size,
                debug_mode=self._debug_mode, draw_figures=not self._add_llm_context_explanations
            )

            if self._add_llm_context_explanations:
                title, scores, K, figs_in_row, explanations, bins, influence_vals, source_name, show_scores = self._results
                if hasattr(self._operation, 'source_df'):
                    source_df = self._operation.source_df
                    right_df = None
                elif hasattr(self._operation, 'left_df'):
                    source_df = self._operation.left_df
                    right_df = self._operation.right_df
                else:
                    raise ValueError(
                        "The operation object does not have a source DataFrame. This should not happen with fedex operations.")

                if hasattr(self._operation, 'source_name'):
                    source_name = self._operation.source_name
                elif hasattr(self._operation, 'left_df'):
                    source_name = self._operation.left_name + " and " + self._operation.right_name
                else:
                    raise ValueError(
                        "The operation object does not have a source name. This should not happen with fedex operations.")

                if isinstance(self._operation, Filter):
                    query = f"{source_name}[{self._operation.attribute} {self._operation.operation_str} {self._operation.value}]"
                elif isinstance(self._operation, GroupBy):
                    query = (f"{source_name}.groupby({', '.join(self._operation.group_attributes) 
                    if isinstance(self._operation.group_attributes, list) else self._operation.group_attributes})"
                             f".agg({self._operation.agg_dict})")
                elif isinstance(self._operation, Join):
                    query = f"{source_df}.join({right_df}, on={self._operation.attribute})"
                else:
                    raise ValueError(
                        "Unrecognized operation type. This may have happened if you added a new operation to Fedex without updating this method.")

                reasoner = ExplanationReasoning(
                    data=source_df,
                    source_name=source_name,
                    query=query,
                    explanations_found=explanations,
                    right_df=right_df if right_df is not None else None,
                )
                added_explanations = reasoner.explain()
                print(added_explanations)

        if isinstance(self._operation, Filter):
            self._original_operation.cor_deleted_atts = self._operation.cor_deleted_atts
            self._original_operation.not_presented = self._operation.not_presented
            self._original_operation.corr = self._operation.corr

        return self._results

    def can_visualize(self) -> bool:
        return True

    def visualize(self):
        return self._results
