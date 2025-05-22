from collections import defaultdict
import numpy as np
from scipy import stats
import itertools

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
                 filter_columns: List[str] | str = None, aggregations: List[Tuple[str, str]] = None,
                 groupby_columns: List[List[str]] | List[str] = None,
                 operation: Operation = None, correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                 figs_in_row: int = 2, use_sampling: bool = True, sample_size: int | float = 5000,
                 max_filter_columns: int = 3, max_aggregation_columns: int = 3,
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
        self.max_filter_columns = max_filter_columns
        self.max_aggregation_columns = max_aggregation_columns
        self.aggregations = None
        self.filter_columns = None
        self.groupby_columns = None

        if self.source_df is None:
            raise ValueError("source_df cannot be None")

        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("k must be a positive integer")

        if 1 < self.min_commonness <= 0:
            raise ValueError("min_commonness must be in the range (0, 1]")

        self.can_run_visualize = False

        handle_filter_or_join_or_provided_filters = False
        handle_groupby = False

        # If filter_columns is None, we try to draw them from the operation, if possible.
        if not filter_columns and operation is not None:
            if isinstance(operation, Filter) or isinstance(operation, Join):
                self.filter_columns = operation.attribute
                self.groupby_columns = None
                handle_filter_or_join_or_provided_filters = True
            elif isinstance(operation, GroupBy) and not groupby_columns:
                self.groupby_columns = operation.group_attributes
                if isinstance(self.groupby_columns, str):
                    self.groupby_columns = [self.groupby_columns]
                self.source_df = pd.DataFrame(operation.source_df)
                self.filter_columns = []
                handle_groupby = True
            else:
                raise NotImplementedError(
                    "Unknown operation. The developers added an operation to Fedex_Generator and forgot to update here.")
            if isinstance(self.filter_columns, str):
                self.filter_columns = [self.filter_columns]

        # Otherwise, use the user provided filter_columns
        elif filter_columns is not None:
            handle_filter_or_join_or_provided_filters = True
            if isinstance(filter_columns, str):
                self.filter_columns = [filter_columns]
            else:
                self.filter_columns = filter_columns

        else:
            raise ValueError(
                "If no target_columns are provided, an operation must have been performed prior to using the MetaInsight explainer.")

        if handle_filter_or_join_or_provided_filters:
            self._prepare_case_filter_join(
                groupby_columns=groupby_columns,
                correlation_aggregation_method=correlation_aggregation_method,
                aggregations=aggregations
            )
        elif handle_groupby:
            self._prepare_case_groupby(
                correlation_aggregation_method=correlation_aggregation_method,
                operation=operation
            )

        if self.aggregations is None:
            raise ValueError("No aggregations provided, and no viable aggregation options were found automatically.")


    def _prepare_case_groupby(self,
                                operation: GroupBy,
                                correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                                aggregations: List[Tuple[str, str]] = None
                              ) -> None:
        """
        Prepare the filter columsn, aggregations and groupby columns in the case we are automatically
        inferring from a groupby operation.

        :param correlation_aggregation_method: The method to use for aggregating the correlation values.
        :param operation: The GroupBy object to use.
        :param aggregations: The aggregations to use. If None, we will use the most correlated columns to the groupby columns.
        """
        # Check if groupby_columns are in the source_df
        if not all(col in self.source_df.columns for col in self.groupby_columns):
            raise ValueError("All groupby_columns must be present in the source dataframe")


        correlated_cols, numerical_cols = self._find_correlated_columns_multi(self.groupby_columns,
                                                                              method=correlation_aggregation_method)

        len_filters = len(self.filter_columns)
        # If there are less than k target columns, we add the most correlated columns to the target_columns
        if len_filters < self.max_filter_columns:
            difference = self.max_filter_columns - len_filters
            self.filter_columns += [col for col in correlated_cols[:difference]]

        # If the user provided a list of aggregations, we use them. Otherwise, we automatically infer them
        # from the operation.
        if aggregations:
            self.aggregations = aggregations
        else:
            agg_dict = operation.agg_dict
            self.aggregations = []
            if "All" in agg_dict:
                aggregations_on_all = [agg_method for agg_method in agg_dict["All"]]
            else:
                aggregations_on_all = []
            # Remove any "All" columns from the agg_dict, because they are not valid aggregations.
            agg_dict = {k: v for k, v in agg_dict.items() if k not in ["All", "all"]}
            # If there are aggregations on all columns, we will use them as the aggregations functions on the most highly
            # correlated columns.
            if len(aggregations_on_all) > 0:
                best_numerical_cols = [col for col in numerical_cols if col not in self.groupby_columns][:self.max_aggregation_columns]
                for agg_func in aggregations_on_all:
                    self.aggregations += [(col, agg_func) for col in best_numerical_cols]
            # If the operation has no aggregations at all, we will use the most correlated columns to the groupby columns,
            # with mean and std as the aggregation functions.
            if len(agg_dict) == 0 and len(aggregations_on_all) == 0:
                best_numerical_cols = [col for col in numerical_cols if col not in self.groupby_columns][:self.max_aggregation_columns]
                self.aggregations += [(col, 'mean') for col in best_numerical_cols] + [(col, 'std') for col in best_numerical_cols]
            # Else, we will use the aggregations from the operation.
            else:
                col_agg_tuples = []
                for col, agg_func_list in agg_dict.items():
                    # If the column is not in the source_df, we skip it.
                    if col not in self.source_df.columns:
                        continue
                    # If the column is in the source_df, we will use the aggregations from the operation.
                    for agg_func in agg_func_list:
                        col_agg_tuples.append((col, agg_func))
                self.aggregations += [col_agg_tuple for col_agg_tuple in col_agg_tuples if col_agg_tuple[0] not in self.groupby_columns]

        # Finally, set the groupby columns to be every combination of the groupby columns
        self.groupby_columns = list(itertools.combinations(self.groupby_columns, len(self.groupby_columns)))

        # Turn each tuple in that list into a list
        self.groupby_columns = [list(group) for group in self.groupby_columns]




    def _prepare_case_filter_join(self,
                                  groupby_columns: List[List[str]] = None,
                                  correlation_aggregation_method: Literal['avg', 'max', 'sum'] = 'avg',
                                  aggregations: List[Tuple[str, str]] = None) -> None:
        """
        Prepare the filter columns, aggregations and groupby columns in the case we are automatically
        inferring from a filter or join operation.

        :param groupby_columns: The groupby columns to use. If None, we will use the filter_columns.
        :param correlation_aggregation_method: The method to use for aggregating the correlation values.
        :param aggregations: The aggregations to use. If None, we will use the most correlated columns.
        """

        # Check if target_columns are in the source_df
        if not all(col in self.source_df.columns for col in self.filter_columns):
            raise ValueError("All target_columns must be present in the source dataframe")

        correlated_cols, numerical_cols = self._find_correlated_columns_multi(self.filter_columns,
                                                                              method=correlation_aggregation_method)

        # It is possible that filtering only left us with 1 value in the filtered columns. If that happened,
        # we want to remove those columns because we won't be able to get anything from them.
        # We do this after the correlation computing because we still need some columns to compute the correlation.
        self.filter_columns = [col for col in self.filter_columns if self.source_df[col].nunique() > 1]

        # If there are less than k target columns, we add the most correlated columns to the target_columns
        if len(self.filter_columns) < self.max_filter_columns:
            difference = self.max_filter_columns - len(self.filter_columns)
            self.filter_columns += [col for col in correlated_cols[:difference]]

        if groupby_columns is not None:
            if isinstance(groupby_columns, str):
                groupby_columns = [groupby_columns]
            self.groupby_columns = groupby_columns
        else:
            # If no groupby_columns are provided, we just use the filter_columns
            self.groupby_columns = self.filter_columns

        # Aggregations can be None, in which case we will figure out the measures automatically via the correlation.
        if aggregations is not None:
            self.aggregations = aggregations
        else:
            # If no aggregations are provided, we will use a mean on the k most correlated numerical columns
            best_numerical_cols = [col for col in numerical_cols if col not in self.groupby_columns][:self.max_aggregation_columns]
            self.aggregations = [(col, 'mean') for col in best_numerical_cols] + [(col, 'std') for col in best_numerical_cols]

    def visualize(self) -> None | str:
        if len(self.metainsights) == 0:
            return (f"""No common patterns detected for the given parameters:
                    Filter columns: {self.filter_columns}
                    Groupby columns: {self.groupby_columns}
                    Aggregations: {self.aggregations}""")
        else:
            num_rows = min(self.top_k, len(self.metainsights))
            fig = plt.figure(figsize=(30, 40))
            main_grid = gridspec.GridSpec(nrows=num_rows + 1, ncols=1, figure=fig,
                                          wspace=0.2, hspace=1, height_ratios=[1] + [50] * num_rows)
            if any([True for mi in self.metainsights if len(mi.exceptions) > 0]):
                need_second_column = True
            else:
                need_second_column = False
            # Create a title grid as the first row, spanning two columns
            n_cols = 2 if need_second_column else 1
            title_grid = gridspec.GridSpecFromSubplotSpec(
                nrows=1, ncols=n_cols, subplot_spec=main_grid[0, 0], wspace=0.2, hspace=0.2
            )
            # Left title : "Common patterns detected"
            ax_left = fig.add_subplot(title_grid[0, 0])
            ax_left.set_title("Common patterns detected", fontsize=30)
            ax_left.axis('off')
            if need_second_column:
                # Right title : "Exceptions to (matching) common pattern (left) detected"
                ax_right = fig.add_subplot(title_grid[0, 1])
                ax_right.set_title("Exceptions to (matching) common pattern (left) detected", fontsize=30)
                ax_right.axis('off')

            for i, mi in enumerate(self.metainsights[:self.top_k]):
                mi.visualize(fig=fig, subplot_spec=main_grid[i + 1, 0])

            return None

    def can_visualize(self) -> bool:
        return self.can_run_visualize

    def _find_correlated_columns(self, column_name: str) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Finds the correlated columns for a given column name with optimized performance.
        """
        # Determine if target column is numeric
        target_is_numeric = self.source_df[column_name].dtype.kind in 'ifmM' and self.source_df[
            column_name].nunique() > 6

        all_correlations = []
        numerical_columns = []

        # Pre-compute target data for categorical column
        if not target_is_numeric:
            target_categories = self.source_df[column_name].unique()
            category_indices = {cat: self.source_df[column_name] == cat for cat in target_categories}
            category_counts = {cat: indices.sum() for cat, indices in category_indices.items()}

        for col in self.source_df.columns:
            if col == column_name:
                continue

            # Check if column is numeric
            is_numeric = self.source_df[col].dtype.kind in 'ifmMcf' and self.source_df[col].nunique() > 6

            # Case 1: Both columns are numeric - Pearson correlation
            if target_is_numeric and is_numeric:
                correlation = abs(self.source_df[[column_name, col]].corr().iloc[0, 1])
                numerical_columns.append(col)

            # Case 2: Target is numeric, other column is categorical
            elif target_is_numeric and not is_numeric:
                target_mean = self.source_df[column_name].mean()
                target_sst = ((self.source_df[column_name] - target_mean) ** 2).sum()

                if target_sst <= 0:
                    correlation = 0
                else:
                    group_stats = self.source_df.groupby(col)[column_name].agg(['mean', 'count'])
                    ssb = (group_stats['count'] * ((group_stats['mean'] - target_mean) ** 2)).sum()
                    correlation = np.sqrt(ssb / target_sst)

            # Case 3: Target is categorical, other column is numeric
            elif not target_is_numeric and is_numeric:
                col_values = self.source_df[col].values
                col_mean = col_values.mean()
                col_sst = ((col_values - col_mean) ** 2).sum()

                if col_sst <= 0:
                    correlation = 0
                else:
                    ssb = 0
                    for cat in target_categories:
                        indices = category_indices[cat]
                        count = category_counts[cat]
                        if count > 0:
                            cat_mean = col_values[indices].mean()
                            ssb += count * ((cat_mean - col_mean) ** 2)
                    correlation = np.sqrt(ssb / col_sst)

                numerical_columns.append(col)

            # Case 4: Both columns are categorical - Cramer's V
            else:
                try:
                    # Handle GroupBy case - use Series objects directly to avoid index issues
                    series1 = self.source_df[column_name]
                    series2 = self.source_df[col]

                    # Ensure we have pandas Series with correct lengths
                    if len(series1) == len(series2):
                        contingency = pd.crosstab(series1, series2)

                        # Safety check for crosstab result
                        if contingency.size > 0:
                            chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
                            n = contingency.values.sum()
                            phi2 = chi2 / n
                            r, k = contingency.shape
                            correlation = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0
                        else:
                            correlation = 0
                    else:
                        correlation = 0
                except (TypeError, ValueError):
                    # Fallback if crosstab fails
                    correlation = 0

            all_correlations.append((col, correlation))

        # Sort both lists by correlation strength (descending)
        all_correlations.sort(key=lambda x: x[1], reverse=True)
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
            import time
            start_time = time.time()
            self.metainsights = miner.mine_metainsights(source_df=source_df,
                                                        filter_dimensions=self.filter_columns,
                                                        measures=self.aggregations,
                                                        breakdown_dimensions=self.groupby_columns,
                                                        extend_by_measure=False,
                                                        extend_by_breakdown=False
                                                        )
            print(f"Mining time: {time.time() - start_time}")
        self.can_run_visualize = True
        return self.metainsights
