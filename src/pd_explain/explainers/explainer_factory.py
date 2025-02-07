from .fedex_explainer import FedexExplainer
from .many_to_one_explainer import ManyToOneExplainer
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
        if explainer == "fedex" or explainer == 'outlier' or explainer == 'shapley':
            return FedexExplainer(explainer=explainer,*args, **kwargs)
        elif explainer.replace("_", " ") == "many to one":
            return ManyToOneExplainer(*args, **kwargs)
        else:
            raise ValueError(f"Explainer {explainer} not supported.")