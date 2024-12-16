from abc import ABC, abstractmethod


class ExplainerInterface(ABC):
    """
    An interface for explainer classes. </br>
    All explainer classes should inherit from this class and implement the abstract methods. </br>
    All explainer classes should implement the 3 basic methods: </br>
    1. generate_explanation: This method should generate the explanation for the operation, and return it in its raw form. </br>
    2. can_visualize: this method should return a boolean value indicating whether the explanation can be visualized or not. </br>
    3. visualize: This method should return the visualized explanation. </br>
    """

    @abstractmethod
    def generate_explanation(self):
        pass

    @abstractmethod
    def can_visualize(self) -> bool:
        pass

    @abstractmethod
    def visualize(self):
        pass
