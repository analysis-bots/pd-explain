from abc import ABC, abstractmethod


class ExplainerInterface(ABC):
    """
    An interface for explainer classes.
    All explainer classes should inherit from this class and implement the abstract methods.
    All explainer classes should implement the 3 basic methods:
    1. generate_explanation: This method should generate the requested explanation, and return it in its raw form.
    2. can_visualize: this method should return a boolean value indicating whether the explanation can be visualized or not. </br>
    3. visualize: This method should return the visualized explanation.
    """

    @abstractmethod
    def generate_explanation(self):
        """
        Generate the requested explanation and return it in its raw form.
        Raw form meaning the explanation in the form of a dictionary, list, or any other data structure,
        without any visualization - whatever the explainer returns.
        """
        pass

    @abstractmethod
    def can_visualize(self) -> bool:
        """
        Return a boolean value indicating whether the explanation can be visualized or not.
        Example of when an explanation can't be visualized: User wants to visualize an explanation for clustering,
        but their dataframe has more than 3 dimensions and they chose to turn off dimensionality reduction for visualization.
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Handles the visualization of the explanation.
        Can return a plot, a widget, a string, or any other visualization, so long as they can be displayed in a Jupyter notebook.
        """
        pass
