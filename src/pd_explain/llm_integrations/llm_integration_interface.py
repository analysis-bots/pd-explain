from abc import ABC, abstractmethod
import pandas as pd

class LLMIntegrationInterface(ABC):

    @abstractmethod
    def explain(self) -> pd.Series | None | str:
        """
        Abstract method for explaining a given input.
        Should be implemented by subclasses to provide specific functionality.
        Should utilize a LLM to generate explanations.
        Should return a pandas Series in case multiple explanations are expected or possible, string if only one explanation is expected.
        None if there was an error in the process.
        """
        raise NotImplementedError()