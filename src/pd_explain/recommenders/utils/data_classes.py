from dataclasses import dataclass

@dataclass(frozen=True)
class Query:
    attribute: str
    operation: str
    value: any

    def __repr__(self):
        return f"{self.attribute} {self.operation} {self.value}"

    def __eq__(self, other):
        if isinstance(other, Query):
            return self.attribute == other.attribute and self.operation == other.operation and self.value == other.value
        if isinstance(other, str):
            return str(self) == other or self.__repr__() == other

    def __str__(self):
        return self.__repr__()