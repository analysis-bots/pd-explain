from abc import ABC, abstractmethod

class LLMIntegrationInterface(ABC):

    @abstractmethod
    def explain(self) -> str:
        raise NotImplementedError()