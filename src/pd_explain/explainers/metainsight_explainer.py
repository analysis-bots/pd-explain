from pd_explain.explainers.explainer_interface import ExplainerInterface
from enum import Enum
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from diptest import diptest
from scipy.stats import gaussian_kde


class PatternType(Enum):
    """
    An enumeration of the types of patterns.
    """
    NONE = 0
    OTHER = 1
    UNIMODALITY = 2
    TREND = 3
    OUTLIER = 4


class DataScope:
    """
    A data scope, as defined in the MetaInsight paper.
    Contains 3 elements: subspace, breakdown and measure.
    Example: for the query SELECT Month, SUM(Sales) FROM DATASET WHERE City==“Los Angeles” GROUP BY Month
    The subspace is {City: Los Angeles, Month: *}, the breakdown is {Month} and the measure is {SUM(Sales)}.
    """

    def __init__(self, source_df: pd.DataFrame, subspace: Dict[str, str], breakdown: str, measure: tuple):
        """
        Initialize the DataScope with the provided subspace, breakdown and measure.

        :param source_df: The DataFrame containing the data.
        :param subspace: dict of filters, e.g., {'City': 'Los Angeles', 'Month': '*'}
        :param breakdown: str, the dimension for group-by
        :param measure: tuple, (measure_column_name, aggregate_function_name)
        """
        self.source_df = source_df
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

    def _subspace_extend(self) -> List['DataScope']:
        """
        Extends the subspace of the DataScope into its sibling group by the dimension dim_to_extend.
        Subspaces with the same sibling group only differ from each other in 1 non-empty filter.

        :return: A list of new DataScope objects with the extended subspace.
        """
        new_ds = []
        if isinstance(self.subspace, dict):
            for dim_to_extend in self.subspace.keys():
                unique_values = self.source_df[dim_to_extend].dropna().unique()
                for value in unique_values:
                    # Ensure it's a sibling
                    if self.subspace.get(dim_to_extend) != value:
                        # Add the new DataScope with the extended subspace
                        new_subspace = self.subspace.copy()
                        new_subspace[dim_to_extend] = value
                        new_ds.append(DataScope(self.source_df, new_subspace, self.breakdown, self.measure))
        return new_ds

    def _measure_extend(self, measures: Dict[str,str]) -> List['DataScope']:
        """
        Extends the measure of the DataScope while keeping the same breakdown and subspace.

        :param measures: The measures to extend.
        :return: A list of new DataScope objects with the extended measure.
        """
        new_ds = []
        for measure_col, agg_func in measures.items():
            if (measure_col, agg_func) != self.measure:
                new_ds.append(DataScope(self.source_df, self.subspace, self.breakdown, (measure_col, agg_func)))
        return new_ds

    def _breakdown_extend(self, temporal_dimensions: List[str]) -> List['DataScope']:
        """
        Extends the breakdown of the DataScope while keeping the same subspace and measure.

        :param temporal_dimensions: The temporal dimensions to extend the breakdown with.
        :return: A list of new DataScope objects with the extended breakdown.
        """
        new_ds = []

        temporal_dimensions = [d for d in temporal_dimensions if
                               self.source_df[d].dtype in ['datetime64[ns]', 'period[M]', 'int64']]
        for breakdown_dim in temporal_dimensions:
            if breakdown_dim != self.breakdown:
                new_ds.append(DataScope(self.source_df, self.subspace, breakdown_dim, self.measure))
        return new_ds

    def create_hds(self, temporal_dimensions: List[str] = None, measures: Dict[str, str] = None) -> List['DataScope']:
        """
        Generates a Homogeneous Data Scope (HDS) from a base data scope, using subspace, measure and breakdown
        extensions as defined in the MetaInsight paper.

        :param temporal_dimensions: The temporal dimensions to extend the breakdown with. Expected as a list of strings.
        :param measures: The measures to extend the measure with. Expected to be a dict {measure_column: aggregate_function}.

        :return: A HDS in the form of a list of DataScope objects.
        """
        hds = []
        if temporal_dimensions is None:
            temporal_dimensions = []
        if measures is None:
            measures = {}

        # Subspace Extending
        hds.extend(self._subspace_extend())

        # Measure Extending
        hds.extend(self._measure_extend(measures))

        # Breakdown Extending
        hds.extend(self._breakdown_extend(temporal_dimensions))

        return hds


class PatternEvaluator:
    """
    A class to evaluate different patterns in a series.
    """

    @staticmethod
    def unimodality(series: pd.Series) -> (bool, Tuple[str, str]):
        """
        Evaluates if the series is unimodal using Hartigan's Dip test and returns the highlight.
        :param series: The series to evaluate.
        :return: (is_unimodal, highlight)
        """
        # Perform Hartigan's Dip test
        dip_statistic, p_value = diptest(series.dropna().values)
        is_unimodal = p_value > 0.05
        if not is_unimodal:
            return False, (None, None)
        # If there is unimodality, find the valley / peak
        # 2. Perform Kernel Density Estimation
        kde = gaussian_kde(series)

        # 3. Evaluate the KDE over a range of values
        # Create a range of points covering the data span
        x_range = np.linspace(series.min(), series.max(), 1000)
        density_values = kde(x_range)

        # 4. Find the index of the maximum (peak) and minimum (valley) density
        peak_index = np.argmax(density_values)
        valley_index = np.argmin(density_values)

        # 5. Map indices back to data values to get the estimated locations
        peak_location = x_range[peak_index]
        valley_location = x_range[valley_index]

        # Check which of the two is the bigger outlier, and return the one that is
        # furthest from the mean
        if abs(peak_location - series.mean()) > abs(valley_location - series.mean()):
            return True, (peak_location, 'Peak')
        else:
            return True, (valley_location, 'Valley')



    @staticmethod
    def trend(series: pd.Series) -> (bool, str):
        """
        Evaluates if the series has a trend and returns the highlight.
        :param series: The series to evaluate.
        :return: (has_trend, highlight)
        """
        # Placeholder for actual trend evaluation logic
        return False, None

    def __call__(self, series: pd.Series, pattern_type: PatternType) -> (bool, str):
        """
        Calls the appropriate pattern evaluation method based on the pattern type.
        :param series: The series to evaluate.
        :param pattern_type: The type of the pattern to evaluate.
        :return: (is_valid, highlight)
        """
        if pattern_type == PatternType.UNIMODALITY:
            return self.unimodality(series)
        elif pattern_type == PatternType.TREND:
            return self.trend(series)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")


class BasicDataPattern:
    """
    A data pattern, as defined in the MetaInsight paper.
    Contains 3 elements: data scope, type (interpretation type) and highlight.
    """

    def __init__(self, data_scope: DataScope, pattern_type: PatternType, highlight: str | None):
        """
        Initialize the BasicDataPattern with the provided data scope, type and highlight.

        :param data_scope: The data scope of the pattern. a DataScope object.
        :param pattern_type: str, e.g., 'Unimodality', 'Trend', 'Other Pattern', 'No Pattern'
        :param highlight: depends on type, e.g., ('April', 'Valley') for Unimodality
        """
        self.data_scope = data_scope
        self.pattern_type = pattern_type
        self.highlight = highlight
        self.pattern_cache = {}

    def __eq__(self, other):
        if not isinstance(other, BasicDataPattern):
            return False
        return self.pattern_type == other.pattern_type and \
            self.highlight == other.highlight and \
            self.data_scope == other.data_scope


    def sim(self, other) -> bool:
        """
        Computes the similarity between two BasicDataPattern objects.
        They are similar if they have the same pattern type and highlight, as well as neither having
        a pattern type of NONE or OTHER.

        :param other: The other BasicDataPattern object to compare with.
        :return: True if similar, False otherwise.
        """
        if not isinstance(other, BasicDataPattern):
            return False
        # There is no REAL need to check that both don't have NONE or OTHER pattern types, since if one
        # has it but the other doesn't, the equality will be false anyway. If they both have it, then
        # the equality conditions will be true but the inequality conditions will be false.
        return self.pattern_type == other.pattern_type and self.highlight == other.highlight and \
               self.pattern_type != PatternType.NONE and self.pattern_type != PatternType.OTHER

    def __hash__(self):
        return hash((self.data_scope, self.pattern_type, self.highlight))

    def __repr__(self):
        return f"BasicDataPattern(ds={self.data_scope}, type='{self.pattern_type}', highlight={self.highlight})"

    @staticmethod
    def evaluate_pattern(data_scope: DataScope, df: pd.DataFrame, pattern_type: PatternType) -> 'BasicDataPattern':
        """
        Evaluates a specific pattern type for the data distribution of a data scope.
        :param data_scope: The data scope to evaluate.
        :param df: The DataFrame containing the data.
        :param pattern_type: The type of the pattern to evaluate.
        """
        # Apply subspace filters
        filtered_df = df.copy()
        for dim, value in data_scope.subspace.items():
            if value != '*':
                filtered_df = filtered_df[filtered_df[dim] == value]

        # Group by breakdown dimension and aggregate measure
        if data_scope.breakdown not in filtered_df.columns:
            # Cannot group by breakdown if it's not in the filtered data
            return BasicDataPattern(data_scope, PatternType.NONE, None)

        measure_col, agg_func = data_scope.measure
        if measure_col not in filtered_df.columns:
            # Cannot aggregate if measure column is not in the data
            return BasicDataPattern(data_scope, PatternType.NONE, None)

        try:
            # Perform the aggregation
            aggregated_series = filtered_df.groupby(data_scope.breakdown)[measure_col].agg(agg_func)
        except Exception as e:
            print(f"Error during aggregation for {data_scope}: {e}")
            return BasicDataPattern(data_scope, PatternType.NONE, None)

        # Ensure series is sortable if breakdown is temporal
        if df[data_scope.breakdown].dtype in ['datetime64[ns]', 'period[M]', 'int64']:
            aggregated_series = aggregated_series.sort_index()

        # Evaluate the specific pattern type
        pattern_evaluator = PatternEvaluator()
        is_valid, highlight = pattern_evaluator(aggregated_series, pattern_type)
        if is_valid:
            return BasicDataPattern(data_scope, pattern_type, highlight)
        else:
            # Check for other pattern types
            for other_type in PatternType:
                if other_type != pattern_type:
                    other_is_valid, _ = pattern_evaluator(aggregated_series, other_type)
                    if other_is_valid:
                        return BasicDataPattern(data_scope, PatternType.OTHER, None)

        # If no pattern is found, return a 'No Pattern' type
        return BasicDataPattern(data_scope, PatternType.NONE, None)

    def create_hdp(self, pattern_type: PatternType, pattern_cache: Dict = None,
                   hds: List[DataScope] = None, temporal_dimensions: List[str] = None,
                   measures: Dict[str,str] = None) -> Tuple[List['BasicDataPattern'], Dict]:
        """
        Generates a Homogenous Data Pattern (HDP) either from a given HDS or from the current DataScope.

        :param pattern_type: The type of the pattern (e.g., 'Unimodality', 'Trend', etc.), provided as a PatternType enum.
        :param pattern_cache: A cache for the pattern, if available.
        :param hds: A list of DataScopes to create the HDP from. If None, it will be created from the current DataScope.
        :param temporal_dimensions: The temporal dimensions to extend the breakdown with. Expected as a list of strings. Only needed if hds is None.
        :param measures: The measures to extend the measure with. Expected to be a dict {measure_column: aggregate_function}. Only needed if hds is None.
        """
        if hds is None or len(hds) == 0:
            hds = self.data_scope.create_hds(temporal_dimensions=temporal_dimensions, measures=measures)
        # All the data scopes in the HDS should have the same source_df, and it should be
        # the same as the source_df of the current DataScope (otherwise, this pattern should not be
        # the one producing the HDP with this HDS).
        source_df = self.data_scope.source_df
        if not all(ds.source_df == source_df for ds in hds):
            raise ValueError("All DataScopes in the HDS must have the same source_df.")

        # Append the existing cache if available
        if pattern_cache is None:
            pattern_cache = {}
        pattern_cache.update(self.pattern_cache)

        # Create the HDP
        hdp = []
        for ds in hds:
            # Check pattern cache first
            cache_key = (ds, pattern_type)
            if cache_key in pattern_cache:
                dp = pattern_cache[cache_key]
            else:
                # Evaluate the pattern if not in cache
                dp = self.evaluate_pattern(ds, source_df, pattern_type)
                pattern_cache[cache_key] = dp  # Store in cache

            # Only add patterns that are not 'No Pattern' to the HDP for MetaInsight evaluation
            if dp.type != PatternType.NONE:
                hdp.append(dp)

        self.pattern_cache = pattern_cache

        return hdp, pattern_cache


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
