from abc import ABC, abstractmethod
from typing import Any

from fedex_generator.Operations.Join import Join
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Filter import Filter


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

    @abstractmethod
    def get_explanation_in_textual_description(self, index: int) -> str:
        """
        Get the explanation for a specific index in a textual description format.
        If the explanations have not been generated yet, this method should raise an error.
        :param index: A single index to get the explanation for.
        :return: A human-readable string that explains the operation performed, what was found, and the explanation itself.
        """
        pass

    @staticmethod
    def _create_query_string(operation) -> tuple[str, str]:
        """
        Create a query string based on the operation object, which can be used to represent the operation in a human-readable format.
        Also writes a textual version of the query type, which can be used to identify the type of operation being performed (e.g., filter, groupby, join).
        This may be useful for logging, debugging, visualization, or any other purpose that requires a textual representation of the query.

        :param operation: The operation object to create the query string from.
        :return: A tuple containing the query string and the query type as a string.
        """
        if isinstance(operation, Filter):
            query = f"{operation.source_name}[{operation.attribute} {operation.operation_str} {operation.value}]"
            query_type = "filter"
        elif isinstance(operation, GroupBy):
            query = (f"{operation.source_name}.groupby({', '.join(operation.group_attributes)
            if isinstance(operation.group_attributes, list) else operation.group_attributes})"
                           f".agg({operation.agg_dict})")
            query_type = "groupby"
        elif isinstance(operation, Join):
            query = f"{operation.left_name}.join({operation.right_name}, on={operation.attribute})"
            query_type = "join"
        else:
            raise ValueError(
                "Unrecognized operation type. This may have happened if you added a new operation to Fedex without updating this method.")
        return query, query_type
