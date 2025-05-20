import math
import time
from collections import defaultdict
import numpy as np
from scipy import stats

import pandas as pd
from matplotlib import pyplot as plt, gridspec

from pd_explain.explainers.explainer_interface import ExplainerInterface
from external_explainers.metainsight_explainer.metainsight_mining import MetaInsightMiner, MetaInsight
from fedex_generator.Operations.Operation import Operation
from fedex_generator.Operations.Filter import Filter
from fedex_generator.Operations.GroupBy import GroupBy
from fedex_generator.Operations.Join import Join

from typing import List, Tuple, Literal

RANDOM_SEED = 42


class MetaInsightExplainer(ExplainerInterface):
    """
    An implementation of MetaInsight from the paper MetaInsight: Automatic Discovery of Structured Knowledge for
    Exploratory Data Analysis by Ma et al. (2021).
    """

    def __init__(self, source_df, top_k=4, min_commonness: float = 0.5,
                 actionability_regularizer=0.1, balance_factor: float = 1,
                 target_columns: List[str] | str = None, measures: List[Tuple[str, str]] = None,
                 operation: Operation = None, correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                 figs_in_row: int = 2, use_sampling: bool = True, sample_size: int | float = 5000,
                 *args, **kwargs):
        """
        Initialize the MetaInsightExplainer with the provided arguments.
        """
        self.metainsights = None
        self.source_df = pd.DataFrame(source_df)
        if top_k is None:
            top_k = 4
        self.top_k = top_k
        self.min_commonness = min_commonness
        self.actionability_regulizer = actionability_regularizer
        self.balance_factor = balance_factor
        self.figs_per_row = figs_in_row
        self.use_sampling = use_sampling
        self.sample_size = sample_size

        if self.source_df is None:
            raise ValueError("source_df cannot be None")

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("k must be a positive integer")

        if 1 < self.min_commonness <= 0:
            raise ValueError("min_commonness must be in the range (0, 1]")

        self.can_run_visualize = False

        # If target_columns is None, we try to draw them from the operation, if possible.
        if target_columns is None and operation is not None:
            if isinstance(operation, Filter):
                self.target_columns = operation.attribute
            elif isinstance(operation, GroupBy):
                self.target_columns = operation.group_attributes
            elif isinstance(operation, Join):
                self.target_columns = operation.attribute
            else:
                raise NotImplementedError(
                    "Unknown operation. The developers added an operation to Fedex_Generator and forgot to update here.")
            if isinstance(self.target_columns, str):
                self.target_columns = [self.target_columns]

        # Otherwise, use the user provided target_columns
        elif target_columns is not None:
            if isinstance(target_columns, str):
                self.target_columns = [target_columns]
            else:
                self.target_columns = target_columns

        else:
            raise ValueError(
                "If no target_columns are provided, an operation must have been performed prior to using the MetaInsight explainer.")

        # Check if target_columns are in the source_df
        if not all(col in self.source_df.columns for col in self.target_columns):
            raise ValueError("All target_columns must be present in the source dataframe")

        correlated_cols, numerical_cols = self._find_correlated_columns_multi(self.target_columns,
                                                                              method=correlation_aggregation_method)

        # If there are less than 3 target columns, we add the most correlated columns to the target_columns
        if len(self.target_columns) < 3:
            difference = 3 - len(self.target_columns)
            self.target_columns += [col for col in correlated_cols[:difference]]

        # Measures can be None, in which case we will figure out the measures automatically via the correlation.
        if measures is not None:
            self.measures = measures
        else:
            # If no measures are provided, we will use a mean on the 3 most correlated numerical columns
            best_numerical_cols = numerical_cols[:2]
            self.measures = [(col, 'mean') for col in best_numerical_cols]

    def visualize(self) -> None | str:
        if len(self.metainsights) == 0:
            return "No metainsights found"
        else:
            fig = plt.figure(figsize=(30, 35))
            main_grid = gridspec.GridSpec(nrows=self.top_k, ncols=1, figure=fig, wspace=0.2, hspace=1)

            for i, mi in enumerate(self.metainsights[:self.top_k]):
                mi.visualize(fig=fig, subplot_spec=main_grid[i, 0])

            return None

    def can_visualize(self) -> bool:
        return self.can_run_visualize

    def _find_correlated_columns(self, column_name: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Optimized version of correlated columns finder with faster eta calculation.
        """
        # Determine if target column is numeric
        target_is_numeric = self.source_df[column_name].dtype.kind in 'ifmM' and self.source_df[
            column_name].nunique() > 6

        # Pre-compute target column statistics if numeric
        if target_is_numeric:
            target_mean = self.source_df[column_name].mean()
            target_values = self.source_df[column_name]
            target_sst = np.sum((target_values - target_mean) ** 2)

        all_correlations = []
        numerical_columns = []

        for col in self.source_df.columns:
            if col == column_name:
                continue

            # Check if column is numeric
            is_numeric = self.source_df[col].dtype.kind in 'ifmM' and self.source_df[col].nunique() > 6

            start_time = time.time()

            # Case 1: Both columns are numeric - Pearson correlation
            if target_is_numeric and is_numeric:
                correlation = abs(self.source_df[[column_name, col]].corr().iloc[0, 1])
                numerical_columns.append(col)
                print(f"Pearson computation time: {time.time() - start_time}")

            # Case 2: Target is numeric, other column is categorical - Correlation ratio (eta)
            elif target_is_numeric and not is_numeric:
                if target_sst <= 0:
                    correlation = 0
                else:
                    # Vectorized implementation of correlation ratio
                    group_means = self.source_df.groupby(col)[column_name].mean()
                    group_counts = self.source_df.groupby(col)[column_name].count()
                    # Calculate SSB using vectorized operations
                    ssb = np.sum(group_counts * (group_means - target_mean) ** 2)
                    correlation = np.sqrt(ssb / target_sst)
                print(f"Eta computation time: {time.time() - start_time}")

            # Case 3: Target is categorical, other column is numeric. Use reverse correlation ratio.
            elif not target_is_numeric and is_numeric:
                col_mean = self.source_df[col].mean()
                col_values = self.source_df[col]
                col_sst = np.sum((col_values - col_mean) ** 2)

                if col_sst <= 0:
                    correlation = 0
                else:
                    # Vectorized implementation for categorical target, numeric feature
                    group_means = self.source_df.groupby(column_name)[col].mean()
                    group_counts = self.source_df.groupby(column_name)[col].count()
                    ssb = np.sum(group_counts * (group_means - col_mean) ** 2)
                    correlation = np.sqrt(ssb / col_sst)

                numerical_columns.append(col)

            # Case 4: Both columns are categorical - Cramer's V
            else:
                contingency = pd.crosstab(self.source_df[column_name], self.source_df[col])
                chi2, _, _, _ = stats.chi2_contingency(contingency)
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                correlation = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0

            all_correlations.append((col, correlation))

        # Sort both lists by correlation strength (descending)
        all_correlations.sort(key=lambda x: x[1], reverse=True)

        # Create the numerical columns list, sorted by their original correlation values
        numerical_corr = [(col, corr) for col, corr in all_correlations if col in numerical_columns]
        numerical_corr.sort(key=lambda x: x[1], reverse=True)

        return all_correlations, numerical_corr

    def _find_correlated_columns_multi(self, column_names: List[str], method='avg') -> Tuple[List[str], List[str]]:
        """
        Finds correlated columns for multiple target columns.

        :param column_names: List of column names to find correlations for
        :param method: Method to combine correlations - 'avg' for average, 'max' for maximum
        :return: A tuple containing two lists:
            - All correlated columns (in order of correlation)
            - Only the numerical correlated columns (in order of correlation)
        """

        # Dictionary to store correlations for each column
        all_column_correlations = defaultdict(list)
        all_numerical_columns = set()

        # Calculate correlations for each target column
        for target_column in column_names:
            all_cols, num_cols = self._find_correlated_columns(target_column)

            # Add to numerical columns set
            all_numerical_columns.update([col[0] for col in num_cols])

            # Get correlations for all columns
            for col, corr in all_cols:
                all_column_correlations[col].append(corr)

        # Combine correlations based on method
        combined_correlations = []
        for col, corr_values in all_column_correlations.items():
            if method == 'avg':
                combined_corr = np.mean(corr_values)
            elif method == 'max':
                combined_corr = max(corr_values)
            elif method == 'sum':
                combined_corr = sum(corr_values)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'avg', 'max' or 'sum'")

            combined_correlations.append((col, combined_corr))

        # Sort by correlation strength
        combined_correlations.sort(key=lambda x: x[1], reverse=True)
        all_cols = [col for col, _ in combined_correlations]

        # Filter and sort numerical columns
        numerical_corrs = [(col, corr) for col, corr in combined_correlations if col in all_numerical_columns]
        numerical_corrs.sort(key=lambda x: x[1], reverse=True)
        numerical_cols = [col for col, _ in numerical_corrs]

        return all_cols, numerical_cols

    @staticmethod
    def _sample(df: pd.DataFrame, sample_size: int | float):
        """
        Samples a dataframe.
        :param df: The dataframe to sample.
        :param sample_size: The number of samples to take. Can be an integer for an absolute amount or
        a float between 0 and 1 for a relative amount.
        """
        if isinstance(sample_size, float):
            if 0 < sample_size < 1:
                sample_size = df.shape[0] * sample_size
            else:
                raise ValueError("Sample size be an integer or a float in (0, 1)")
        if sample_size <= 0:
            raise ValueError("Sample size must be a positive number.")

        if df.shape[0] <= sample_size:
            return df
        else:
            # Convert the sample size to an integer, in case it was passed as a float above 1.
            sample_size = int(sample_size)
            # We use a set seed so that the user will always get the same explanation when using sampling.
            generator = np.random.default_rng(RANDOM_SEED)
            uniform_indexes = generator.choice(df.index, sample_size, replace=False)
            return df.loc[uniform_indexes]

    def generate_explanation(self) -> List[MetaInsight] | None:
        """
        Generate explanations using the MetaInsight algorithm.
        """
        if self.metainsights is None:
            miner = MetaInsightMiner(
                k=self.top_k,
                min_commonness=self.min_commonness,
                actionability_regularizer=self.actionability_regulizer,
                balance_factor=self.balance_factor
            )
            if self.use_sampling:
                source_df = self._sample(self.source_df, self.sample_size)
            else:
                source_df = self.source_df
            start_time = time.time()
            self.metainsights = miner.mine_metainsights(source_df=source_df,
                                                        dimensions=self.target_columns,
                                                        measures=self.measures
                                                        )
            print(f"Mining time: {time.time() - start_time}")
        self.can_run_visualize = True
        return self.metainsights
