from pd_explain.explainers.fedex_explainer import FedexExplainer
from pd_explain.explainers.many_to_one_explainer import ManyToOneExplainer
from pd_explain.explainers.outlier_explainer import OutlierExplainerInterface as OutlierExplainer
from pd_explain.explainers.beta_explainers.metainsight_explainer import MetaInsightExplainer

from singleton_decorator import singleton

@singleton
class ExplainerFactory:
    """
    A factory responsible for creating the appropriate explainer object based on the requested explainer.
    A singleton class.
    """

    def create_explainer(self, explainer, *args, **kwargs):
        """
        Create an explainer object based on the requested explainer.

        :param explainer: The name of the explainer to create.
        :param kwargs: The arguments to pass to the explainer.
        :return: The explainer object.
        """
        explainer = explainer.lower()
        if explainer == "fedex" or explainer == 'shapley':
            return FedexExplainer(explainer=explainer,*args, **kwargs)
        elif explainer == 'outlier':
            return OutlierExplainer(*args, **kwargs)
        elif explainer.replace("_", " ") == "many to one":
            return ManyToOneExplainer(*args, **kwargs)
        elif explainer.replace("_", "").startswith("metainsight"):
            return MetaInsightExplainer(*args, **kwargs)
        else:
            raise ValueError(f"Explainer {explainer} not supported.")