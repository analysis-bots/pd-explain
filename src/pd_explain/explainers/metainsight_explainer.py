from pd_explain.explainers.explainer_interface import ExplainerInterface

class MetaInsightExplainer(ExplainerInterface):
    """
    An implementation of MetaInsight from the paper MetaInsight: Automatic Discovery of Structured Knowledge for
    Exploratory Data Analysis by Ma et al. (2021).
    """

    def __init__(self, source_df, k=3, r=0.5, gamma=0.1, *args, **kwargs):
        """
        Initialize the MetaInsightExplainer with the provided arguments.
        """
        super().__init__(*args, **kwargs)
        self.explanation = None
        self.visualization = None
        self.source_df = source_df
        self.k = k
        self.r = r
        self.gamma = gamma

    def _search(self):
        raise NotImplementedError()

    def _query(self):
        raise NotImplementedError()

    def _evaluate(self):
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()

    def can_visualize(self) -> bool:
        raise NotImplementedError()

    def generate_explanation(self):
        raise NotImplementedError()
