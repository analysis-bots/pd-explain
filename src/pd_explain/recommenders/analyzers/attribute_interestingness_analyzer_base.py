from typing import Dict

import pandas as pd
from pandas import DataFrame
from abc import ABC, abstractmethod


class AttributeInterestingnessAnalyzerBase(ABC):

    def __init__(self):
        self._should_refresh_internals = True

    def strings_to_numbers(self, data: DataFrame) -> DataFrame:
        """
        Convert all string columns to numbers.

        :param data: The data to convert.

        :return: The data with all string columns converted to numbers.
        """
        data = data.copy()
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = pd.Categorical(data[column])
                data[column] = data[column].cat.codes
        return data

    def analyze(self, data: DataFrame) -> Dict[str, float]:
        """
        Analyze how interesting each attribute in the data is.
        Interestingness, in this case, is a measure of how likely an attribute is to produce interesting insights
        if we query it.

        :param data: The data to analyze.

        :return: A dictionary with the interestingness of each attribute, sorted in descending order.
        """
        self._should_refresh_internals = True
        data = self.strings_to_numbers(data)
        interestingness = {}
        for column in data.columns:
            interestingness[column] = self._calculate_interestingness(data, column)
        interestingness = dict(sorted(interestingness.items(), key=lambda item: item[1], reverse=True))
        return interestingness


    @abstractmethod
    def _calculate_interestingness(self, data, column) -> float:
        raise NotImplementedError