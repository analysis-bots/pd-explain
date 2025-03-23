from abc import ABC, abstractmethod
import pandas as pd

class LLMIntegrationInterface(ABC):

    @abstractmethod
    def explain(self) -> pd.Series | None | str:
        raise NotImplementedError()