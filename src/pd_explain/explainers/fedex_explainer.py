from .explainer_interface import ExplainerInterface
from typing import List
from pandas import DataFrame


class FedexExplainer(ExplainerInterface):
    """
    This class is responsible for interfacing between explainable data frames and the explainers implemented in the fedex
    package.
    Because pd-explain was originally designed with fedex in mind, this class is just a wrapper around the explain method
    of the explainer object.
    """

    def __init__(self, operation=None, schema: dict = None, attributes: List = None, top_k: int = None, explainer='fedex',
                 target=None,
                 dir=None,
                 figs_in_row: int = 2, show_scores: bool = False, title: str = None, corr_TH: float = 0.7,
                 consider='right', value=None, attr=None, ignore=[], *args, **kwargs):

        if schema is None:
            schema = {}

        if attributes is None:
            attributes = []
            if top_k is None:
                top_k = 1
        else:
            if top_k is None:
                top_k = len(attributes)

        # Convert the source_df and result_df to DataFrame objects, to avoid overhead from overridden methods
        # in ExpDataFrame, as well as to avoid any bad interactions between those methods and the explainer.
        if operation is not None:
            operation.source_df = DataFrame(operation.source_df) if operation.source_df is not None else None
            operation.result_df = DataFrame(operation.result_df) if operation.result_df is not None else None

        self._schema = schema
        self._attributes = attributes
        self._top_k = top_k
        self._explainer = explainer
        self._target = target
        self._dir = dir
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

    def generate_explanation(self):
        if self._operation is None:
            self._results = "No operation was found."
            return self._results

        self._results = self._operation.explain(schema=self._schema, attributes=self._attributes, top_k=self._top_k,
                                                figs_in_row=self._figs_in_row, show_scores=self._show_scores,
                                                title=self._title,
                                                corr_TH=self._corr_TH, explainer=self._explainer,
                                                consider=self._consider,
                                                cont=self._value, attr=self._attr, ignore=self._ignore)
        return self._results

    def can_visualize(self) -> bool:
        return True

    def visualize(self):
        return self._results
