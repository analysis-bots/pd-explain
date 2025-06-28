from abc import ABC, abstractmethod
import ipywidgets as widgets
from typing import Any

class AdapterInterface(ABC):
    """
    Interface for adapters that modify the visualizer's behavior.
    """

    @abstractmethod
    def __enter__(self) -> 'AdapterInterface':
        """
        Enter the context of the adapter.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context of the adapter.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def capture_output(self, output: Any) -> None:
        """
        Method for the visualizer to 'send' its output to the adapter.
        """
        raise NotImplementedError("Subclasses must implement this method.")