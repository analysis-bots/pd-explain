from .explainer_interface import ExplainerInterface
from typing import List


class FedexExplainer(ExplainerInterface):
    """
    This class is responsible for interfacing between explainable data frames and the explainers implemented in the fedex
    package. </br>
    Because pd-explain was originally designed with fedex in mind, this class is just a wrapper around the explain method
    of the explainer object. </br>
    """

    def __init__(self, operation=None, schema: dict = None, attributes: List = None, top_k: int = None, explainer='fedex',
                 target=None,
                 dir=None,
                 figs_in_row: int = 2, show_scores: bool = False, title: str = None, corr_TH: float = 0.7,
                 consider='right', value=None, attr=None, ignore=[]):
        if operation is None:
            raise ValueError("Operation is required.")
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
