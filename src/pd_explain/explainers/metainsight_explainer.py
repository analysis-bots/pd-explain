from pd_explain.explainers.explainer_interface import ExplainerInterface
from enum import Enum


class PatternType(Enum):
    """
    An enumeration of the types of patterns.
    """
    NONE = 0
    OTHER = 1


class DataScope:
    """
    A data scope, as defined in the MetaInsight paper.
    Contains 3 elements: subspace, breakdown and measure.
    Example: for the query SELECT Month, SUM(Sales) FROM DATASET WHERE City==“Los Angeles” GROUP BY Month
    The subspace is {City: Los Angeles, Month: *}, the breakdown is {Month} and the measure is {SUM(Sales)}.
    """

    def __init__(self, subspace, breakdown, measure):
        """
        Initialize the DataScope with the provided subspace, breakdown and measure.

        :param df: The DataFrame containing the data.
        :param subspace: dict or list of filters, e.g., {'City': 'Los Angeles', 'Month': '*'}
        :param breakdown: str, the dimension for group-by
        :param measure: tuple, (measure_column_name, aggregate_function_name)
        """
        self.subspace = subspace
        self.breakdown = breakdown
        self.measure = measure

    def __hash__(self):
        # Need a hashable representation of subspace for hashing
        subspace_tuple = tuple(sorted(self.subspace.items())) if isinstance(self.subspace, dict) else tuple(
            self.subspace)
        return hash((subspace_tuple, self.breakdown, self.measure))

    def __repr__(self):
        return f"DataScope(subspace={self.subspace}, breakdown='{self.breakdown}', measure={self.measure})"


class BasicDataPattern:
    """
    A data pattern, as defined in the MetaInsight paper.
    Contains 3 elements: data scope, type (interpretation type) and highlight.
    """

    def __init__(self, data_scope: DataScope, type: str, highlight: str):
        """
        Initialize the DataPattern with the provided data scope, type and highlight.

        :param data_scope: The data scope of the pattern. a DataScope object.
        :param pattern_type: str, e.g., 'Unimodality', 'Trend', 'Other Pattern', 'No Pattern'
        :param highlight: depends on type, e.g., ('April', 'Valley') for Unimodality
        """
        self.data_scope = data_scope
        self.type = type
        self.highlight = highlight

    def __eq__(self, other):
        if not isinstance(other, BasicDataPattern):
            return False
        return self.type == other.type and \
            self.highlight == other.highlight and \
            self.type != "No Pattern" and \
            self.type != "Other Pattern"

    def __hash__(self):
        return hash((self.data_scope, self.type, self.highlight))

    def __repr__(self):
        return f"BasicDataPattern(ds={self.data_scope}, type='{self.type}', highlight={self.highlight})"


class MetaInsight:
    """
    Represents a MetaInsight (HDP, commonness_set, exceptions).
    """

    def __init__(self, hdp, commonness_set, exceptions, score=0):
        """
        :param hdp: list of BasicDataPattern objects
        :param commonness_set: list of lists of BasicDataPattern (each inner list is a commonness)
        :param exceptions: list of BasicDataPattern objects
        """
        # hdp: list of BasicDataPattern objects
        # commonness_set: list of lists of BasicDataPattern (each inner list is a commonness)
        # exceptions: list of BasicDataPattern objects
        self.hdp = hdp
        self.commonness_set = commonness_set
        self.exceptions = exceptions
        self.score = score  # Calculated score

    def __repr__(self):
        return f"MetaInsight(score={self.score:.4f}, #HDP={len(self.hdp)}, #Commonness={len(self.commonness_set)}, #Exceptions={len(self.exceptions)})"


class MetaInsightExplainer(ExplainerInterface):
    """
    An implementation of MetaInsight from the paper MetaInsight: Automatic Discovery of Structured Knowledge for
    Exploratory Data Analysis by Ma et al. (2021).
    """

    def __init__(self, source_df, k=3, r=0.5, gamma=0.1, *args, **kwargs):
        """
        Initialize the MetaInsightExplainer with the provided arguments.
        """
        super().__init__(*args, **kwargs)
        self.explanation = None
        self.visualization = None
        self.source_df = source_df
        self.k = k
        self.r = r
        self.gamma = gamma

    def _search(self):
        raise NotImplementedError()

    def _query(self):
        raise NotImplementedError()

    def _evaluate(self):
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()

    def can_visualize(self) -> bool:
        raise NotImplementedError()

    def generate_explanation(self):
        raise NotImplementedError()
