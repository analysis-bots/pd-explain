from typing import Literal, List

from .explainer_interface import ExplainerInterface
from pd_explain.llm_integrations.explanation_reasoning import ExplanationReasoning

import pandas as pd
from pandas import DataFrame, Series
from cluster_explorer import Explainer, condition_generator, str_rule_to_list, rule_to_human_readable
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples

MAX_LABELS = 10
DEFAULT_SAMPLE_SIZE = 5000
RANDOM_SEED = 42
DEFAULT_ERROR_EXPLANATION_THRESHOLD = 0.05


class ManyToOneExplainer(ExplainerInterface):

    def __init__(self, source_df: DataFrame, labels: Series | str | list[int] | None, coverage_threshold: float = 0.7,
                 max_explanation_length: int = 3, separation_threshold: float = 0.3, p_value: int = 1,
                 explanation_form: Literal['conj', 'conjunction', 'disj', 'disjunction'] = 'conj',
                 attributes: List[str] = None, operation=None, use_sampling: bool = True,
                 prune_if_too_many_labels: bool = True, max_labels: int = MAX_LABELS, pruning_method: str = 'largest',
                 bin_numeric: bool = False, num_bins: int = 10, binning_method: str = 'quantile',
                 label_name: str = 'label', sample_size: int = DEFAULT_SAMPLE_SIZE, explain_errors=True,
                 error_explanation_threshold: float = DEFAULT_ERROR_EXPLANATION_THRESHOLD,
                 add_llm_context_explanations: bool = False,
                 *args, **kwargs):
        """
        Initialize the many-to-one explainer.
        The many to one explainer utilizes cluster-explorer to generate explanations for many-to-one relationships,
        be they clusters, groups created a groupby operation, existing labels, or any other kind of grouping.

        :param source_df: The dataframe to explain.
        :param labels: The labels to explain. This can be a Series object, a string representing the name of the column
        containing the labels, or a list of integers.
        :param coverage_threshold: The minimum coverage threshold for an explanation to be considered valid. Coverage is the
        proportion of the data that is covered by the rule with respect to the cluster.
        :param max_explanation_length: The maximum length of the explanation. This is the maximum number of conditions
        that can be in a rule.
        :param separation_threshold: The minimum separation threshold for an explanation to be considered valid. Separation
        is the proportion of the data that is covered by the rule with outside the cluster.
        :param p_value: A scaling value for the number of top attributes to consider when generating explanations. The
        formula for the number is explanation_length * p_value. This is done to optimize runtime, since the algorithm's
        runtime is dependent on the number of attributes. A higher p_value will result in more attributes being considered.
        Default is 1.
        :param explanation_form: The form of the explanations. This can be either 'conj' or 'conjunction' for conjunctions,
        or 'disj' or 'disjunction' for disjunctions.
        :param attributes: The attributes to consider when generating explanations. If not provided, all attributes will
        be considered.
        :param operation: The last operation performed on the DataFrame. This is used in case no labels are provided, in which
        case, if the last operation is a groupby operation, the labels will be the groups created by the groupby operation.
        :param use_sampling: Whether to sample the dataframe to reduce the number of rows and speed up explanation generation.
        This is especially useful when the dataframe is very large, but may affect the quality of the explanations, although
        the effect is usually negligible. Default is True.
        :param prune_if_too_many_labels: Whether to prune the labels if there are too many of them. If there are more than
        10 unique labels, the labels will be pruned to the top 10 most common labels. Default is True.
        :param max_labels: The maximum number of unique labels to keep. Default is 10. Only used if prune_if_too_many_labels
        is set to True.
        :param bin_numeric: Whether to bin numeric labels into discrete intervals. Labels are binned into num_bins intervals,
        using the specified binning method. Labels are considered numeric if they are of type int or float, and have more
        than num_bins unique values. Default is False.
        :param num_bins: The number of bins to create for the numeric labels. Default is 10.
        :param binning_method: The method to use for binning the numeric labels. This can be either 'uniform' or 'quantile'.
        Default is 'quantile'.
        :param label_name: A name to give the labels when binning them, if there is none to begin with. Default is 'label'.
        :param sample_size: The size of the sample to use when sampling the dataframe. Can be a percentage of the dataframe
        size if below 1. Default is 5000.
        :param explain_errors: Whether the explainer should add another column, explaining where the separation error comes from.
        """

        # Convert the source_df to a DataFrame object, to avoid overhead from overridden methods in ExpDataFrame,
        # as well as to avoid any bad interactions between those methods and the explainer.
        if type(source_df) != DataFrame:
            source_df = DataFrame(source_df)

        if attributes is not None and len(attributes) > 0:
            if isinstance(attributes, str):
                attributes = [attributes]
            # If the labels are in the dataframe, we add them to the columns to select, to not remove them.
            # They will be removed later, but removing them now would cause an error.
            if isinstance(labels, str) and labels not in attributes:
                attributes.append(labels)
            source_df = source_df[attributes]

        self._source_df = source_df
        self._num_bins = num_bins
        self._binning_method = binning_method
        self._bin_numeric = bin_numeric
        self._label_name = label_name

        if labels is None or len(labels) == 0:
            self._source_df, self._labels = self._create_groupby_labels(operation=operation)
        # If the labels are a string, we assume that the string is the name of the column that contains the labels.
        elif isinstance(labels, str):
            self._labels = source_df[labels]
            self._source_df = self._source_df.drop(labels, axis=1, inplace=False)
        # If the labels are a list of strings, we assume that the strings are the names of the columns that contain the labels,
        # and we treat it as a group-by on those columns.
        elif isinstance(labels, list) and all(isinstance(label, str) for label in labels):
            result_df = self._source_df.groupby(labels).mean(numeric_only=True)
            self._source_df, self._labels = self._create_groupby_labels(source_df=self._source_df, result_df=result_df)
        # In case the labels are a dataframe with multiple columns, we aggregate the labels into a single column.
        elif isinstance(labels, pd.DataFrame) and labels.shape[1] > 1:
            # Copy the labels to avoid modifying the original dataframe.
            labels = labels.copy()
            # This nonsensical column name is used to avoid conflicts with existing columns in the dataframe.
            # Because surely, no one would name a column in an actual dataset like this.
            labels['true_labels_for_many_to_one_explainer'] = labels.apply(lambda x: str(tuple(x)), axis=1)
            self._labels = labels['true_labels_for_many_to_one_explainer']
        # If the labels are any other list of values, we simply convert them to a series object.
        elif not isinstance(labels, Series):
            self._labels = Series(labels)
        # If the labels are already a Series object, we simply use them as they are.
        else:
            self._labels = labels

        if self._label_name is None:
            self._label_name = self._labels.name if (hasattr(self._labels, 'name') and self._labels.name is not None) else 'label'

        if self._source_df.shape[0] != len(self._labels):
            raise ValueError("The number of rows in the source DataFrame and the number of labels must be equal.")

        # Drop labels with missing values, as well as the corresponding rows in the source DataFrame.
        self._drop_na()

        # Check that the dataframe is not empty after all the operations.
        if self._source_df.empty:
            raise ValueError("The dataframe is empty. Please check that: \n"
                             "1. The dataframe is not empty to begin with.\n"
                             "2. The labels do not contain all missing values.\n"
                             "3. That you did not pass a dataframe / series with 1 attribute, and the labels as a string with the name of that attribute.\n")

        self._possible_to_visualize = True

        if bin_numeric:
            self._labels = self._bin_numeric_labels(self._labels)

        self._mapping = self.create_mapping()

        if p_value < 0:
            raise ValueError("The p-value must be a positive number.")
        if coverage_threshold < 0 or coverage_threshold > 1:
            raise ValueError("The coverage threshold must be between 0 and 1.")
        if max_explanation_length < 1:
            raise ValueError("The maximum explanation length must be at least 1.")
        if separation_threshold < 0 or separation_threshold > 1:
            raise ValueError("The separation threshold must be between 0 and 1.")

        self._coverage_threshold = coverage_threshold
        self._max_explanation_length = max_explanation_length
        self._separation_threshold = separation_threshold
        self._conciseness_threshold = 1 / self._max_explanation_length

        self._p_value = p_value

        self._explanations = None
        self.explanation_form = explanation_form
        self._explainer = None
        self._ran_explainer = False
        self._max_labels = max_labels
        self._pruning_method = pruning_method
        self._sample_size = sample_size
        self._explain_errors = explain_errors
        self._error_explanation_threshold = error_explanation_threshold
        self._add_llm_context_explanations = add_llm_context_explanations
        self._added_explanations = None
        self._query_string = None
        self.out_df = None

        if operation is not None:
            if hasattr(operation, 'source_name'):
                self._source_name = operation.source_name
            elif hasattr(operation, 'left_name') and hasattr(operation, 'right_name'):
                self._source_name = operation.left_name + " join " + operation.right_name
            else:
                self._source_name = "source"
        else:
            self._source_name = "source"

        # Optional operations to speed up explanation generation.
        if prune_if_too_many_labels:
            self._prune_labels()

        if use_sampling:
            self._sample(sample_size=self._sample_size)

    def _interval_to_str(self, intervals: Series, attribute_name: str) -> Series:
        """
        Convert the intervals to strings, so that they can be used in the explanations.
        """
        return intervals.apply(lambda
                                   x: f"{x.left} {'<=' if x.closed == 'left' else '<'} {attribute_name} {'<=' if x.closed == 'right' else '<'} {x.right}")

    def _bin_numeric_labels(self, labels: Series) -> Series:
        """
        Bins numeric labels into discrete intervals.
        Labels are binned into num_bins intervals, using the specified binning method.
        Labels are considered numeric if they are:
        1. Of type int or float.
        2. Have more than num_bins unique values.
        """
        # Check if the labels are numeric.
        dtype = str(labels.dtype)
        if dtype.startswith('int') or dtype.startswith('float'):
            unique_labels = labels.unique()
            # If the labels are numeric and have more than num_bins unique values, we bin them.
            if len(unique_labels) > self._num_bins:
                if self._binning_method == 'uniform':
                    bins = pd.cut(labels, bins=self._num_bins, retbins=False, duplicates='drop')
                elif self._binning_method == 'quantile':
                    bins = pd.qcut(labels, q=self._num_bins, retbins=False, duplicates='drop')
                else:
                    raise ValueError("The binning method must be either 'uniform' or 'quantile'.")
                # After binning, we convert the intervals to meaningful strings.
                attribute_name = labels.name if labels.name is not None else self._label_name
                str_intervals = self._interval_to_str(bins, attribute_name)
                # Then we return the binned labels.
                print(
                    f"Attribute '{attribute_name}' in labels has more than specified number of {self._num_bins} unique values. Binning the attribute.\n")
                return str_intervals
        return labels

    def _drop_na(self):
        """
        Drop labels with missing values, as well as the corresponding rows in the source DataFrame.
        """
        na_labels = self._labels[self._labels.isna()]
        if not na_labels.empty:
            print(f"Dropping {len(na_labels)} labels with missing values.\n")
            na_labels_indexes = na_labels.index
            self._source_df = self._source_df.drop(na_labels_indexes)
            self._labels = self._labels.drop(na_labels_indexes)
            self._labels.reset_index(drop=True, inplace=True)
            self._source_df.reset_index(drop=True, inplace=True)

    def _prune_labels(self):
        """
        Prune the labels if there are too many of them.
        If there are more than k unique labels, the labels will be pruned to the top k most common labels.
        """
        unique_labels = self._labels.unique()
        if len(unique_labels) > self._max_labels:
            print(
                f"There are more than the specified max number of {self._max_labels} unique labels, and the option `prune_if_too_many_labels` is set to True. Pruning the labels to the top {self._max_labels} most common labels using method {self._pruning_method}\n")
            value_counts = self._labels.value_counts()
            if self._pruning_method == 'largest':
                top_labels = value_counts.index[:self._max_labels]
                top_labels_indexes = self._labels.isin(top_labels)
            elif self._pruning_method == 'smallest':
                top_labels = value_counts.index[-self._max_labels:]
                top_labels_indexes = self._labels.isin(top_labels)
            elif self._pruning_method == 'random':
                top_labels = value_counts.sample(self._max_labels).index
                top_labels_indexes = self._labels.isin(top_labels)
            else:
                label_proprtional_size = value_counts / len(self._labels)
                decomposed = pd.DataFrame(
                    PCA(n_components=min(3, self._source_df.shape[1])).fit_transform(pd.get_dummies(self._source_df)))
                if self._pruning_method == 'max_dist' or self._pruning_method == 'min_dist':
                    decomposed['label'] = self._labels
                    # Perform dimensionality reduction to at most 3 dimensions, and compute the means of each cluster
                    cluster_means = decomposed.groupby('label').mean()
                    # Compute the average distance between each cluster's center to all other clusters
                    average_distances = np.zeros(len(cluster_means))
                    for i in range(len(cluster_means)):
                        other_clusters = cluster_means.loc[cluster_means.index != i]
                        distances = euclidean_distances(cluster_means.loc[i].values.reshape(1, -1), other_clusters)
                        average_distances[i] = distances.mean()
                    # Multiply by the size of the cluster, to give more weight to larger clusters and not give
                    # a large weight to a small cluster that is far away from the other clusters.
                    # average_distances *= label_proprtional_size
                    average_distances = pd.Series(average_distances)
                    if self._pruning_method == 'max_dist':
                        average_distances = average_distances.sort_values(ascending=False)
                    else:
                        average_distances = average_distances.sort_values(ascending=True)
                    top_labels = average_distances.index[:self._max_labels]
                    top_labels_indexes = self._labels.isin(top_labels)
                elif self._pruning_method == 'max_silhouette' or self._pruning_method == 'min_silhouette':
                    # Sample the data to speed up the computation of the silhouette scores, because otherwise
                    # it would take too long.
                    generator = np.random.default_rng(RANDOM_SEED)
                    random_indexes = generator.choice(decomposed.index, size=min(DEFAULT_SAMPLE_SIZE, len(decomposed)),
                                                      replace=False)
                    decomposed = decomposed.loc[random_indexes]
                    labels = self._labels.loc[random_indexes]
                    scores = silhouette_samples(decomposed, labels)
                    scores = pd.DataFrame(scores, index=decomposed.index)
                    # Compute the mean silhouette score for each cluster, and multiply by the size of the cluster.
                    scores['label'] = labels
                    scores = scores.groupby('label').mean()
                    scores = scores.squeeze()
                    # scores *= label_proprtional_size
                    if self._pruning_method == 'max_silhouette':
                        scores = scores.sort_values(ascending=False)
                    else:
                        scores = scores.sort_values(ascending=True)
                    top_labels = scores.index[:self._max_labels]
                    top_labels_indexes = self._labels.isin(top_labels)
                else:
                    raise ValueError("Pruning method must be either 'largest', 'silhouette' or 'max_dist'.")
            self._source_df = self._source_df[top_labels_indexes]
            self._labels = self._labels[top_labels_indexes]
            self._labels.reset_index(drop=True, inplace=True)
            self._source_df.reset_index(drop=True, inplace=True)

    def _sample(self, sample_size: int = DEFAULT_SAMPLE_SIZE):
        """
        Sample the dataframe, to reduce the number of rows and speed up explanation generation.
        This is especially useful when the dataframe is very large, but may affect the quality of the explanations.
        """
        if sample_size <= 0:
            raise ValueError("Sample size must be a positive number.")
        if 0 < sample_size < 1:
            sample_size = self._source_df.shape[0] * sample_size
        # If the sample size is below the default sample size, we use the default sample size.
        if sample_size < DEFAULT_SAMPLE_SIZE:
            print(
                f"Sample size is below the default sample size of {DEFAULT_SAMPLE_SIZE}. Using the default sample size.\n")
            sample_size = DEFAULT_SAMPLE_SIZE
        # Convert to int, just in case the sample size is a float.
        sample_size = int(sample_size)
        if self._source_df.shape[0] > sample_size:
            generator = np.random.default_rng(RANDOM_SEED)
            uniform_indexes = generator.choice(self._source_df.index, size=sample_size, replace=False)
            self._source_df = self._source_df.loc[uniform_indexes]
            self._source_df.reset_index(drop=True, inplace=True)
            self._labels = self._labels.loc[uniform_indexes]
            self._labels.reset_index(drop=True, inplace=True)

    def _create_groupby_labels(self, operation=None, source_df=None, result_df=None) -> (DataFrame, Series):
        """
        Create labels for the many-to-one explainer based on the last operation performed on the DataFrame.
        If the last operation is a groupby operation, the labels are the groups created by the groupby operation.
        Otherwise, the user must provide labels.

        :param operation: The last operation performed on the DataFrame.
        :param source_df: Alternatively to providing an operation, you can provide the source DataFrame and the result DataFrame
        of the groupby operation.
        :param result_df: The result DataFrame of the groupby operation.
        :return: The source DataFrame from the operation and the labels.
        We return the source dataframe as well, because the source dataframe from the operation is prior to the
        groupby operation, and that is what interests us.
        """
        if source_df is None and (operation is None or operation is not None and not "GroupBy" in operation.__repr__()):
            raise ValueError(
                "If this dataframe is not the result of a groupby operation, you must provide the labels.")
        else:
            if operation is not None:
                self._query, _ = self._create_query_string(operation=operation)
            # Extract the source and result df from the operation, if one is provided.
            if source_df is None:
                source_df = DataFrame(operation.source_df)
                result_df = DataFrame(operation.result_df)
            elif source_df is None or result_df is None:
                raise ValueError(
                    "You must provide either an operation or the source and result DataFrames of a groupby operation.")
            group_attributes = result_df.index.names
            # index_values = result_df.index.values
            # Create an array of all -1 values, the length of the original DataFrame, as a placeholder for the labels.
            labels = np.full(source_df.shape[0], -1, dtype='O')
            # Go over the index values and assign the group labels to the corresponding rows in the source DataFrame.
            source_group_attributes_only = source_df[group_attributes]

            if self._bin_numeric:
                # If any of the group attributes are numeric, we bin them into intervals.
                # The bin_numeric method does both the checking and the binning. Non numeric attributes are not affected.
                for attribute in group_attributes:
                    source_group_attributes_only.loc[:, attribute] = self._bin_numeric_labels(
                        source_group_attributes_only[attribute])
                # Change the groups to match the binning.
                groups = set(source_group_attributes_only[group_attributes].apply(lambda x: tuple(x), axis=1))
            else:
                groups = result_df.index.values

            for i, group in enumerate(groups):
                group_indices = source_group_attributes_only.index[
                    source_group_attributes_only[group_attributes].eq(group).all(axis=1)].values
                labels[group_indices] = str(group)

            # We drop the groupby attributes from the source dataframe, as otherwise, the explainer will return
            # an explanation saying the best rule is "groupby_attribute == group", which is not very informative.
            source_df = source_df.drop(group_attributes, axis=1, inplace=False)

            # Indexes where the labels are missing are rows where at-least 1 of the labels is missing, and thus
            # could not be associated with a group.
            missing_values_indexes = np.where(labels == -1)[0]
            if len(missing_values_indexes) > 0:
                print(
                    f"There are {len(missing_values_indexes)} rows with missing values in the labels. Dropping them, as they could not be associated with a group.\n")
                source_df = source_df.drop(missing_values_indexes)
                labels = labels[labels != -1]
                labels = Series(labels)
                labels.reset_index(drop=True, inplace=True)
                source_df.reset_index(drop=True, inplace=True)

            return source_df, Series(labels)

    def create_mapping(self) -> dict:
        """
        In the event that the user provided non-numeric labels, we need to map those labels to integers.
        This function checks if the labels are non-numeric, and if so, it creates a mapping from the labels to integers.
        Otherwise, it returns a dictionary where each label is mapped to itself.
        """
        if not self._labels.apply(lambda x: isinstance(x, (int, float))).all():
            unique_labels = self._labels.unique()
            mapping = {label: i for i, label in enumerate(unique_labels)}
            self._labels = self._labels.map(mapping)
            return mapping
        return {label: label for label in self._labels.unique()}

    def can_visualize(self) -> bool:
        return self._ran_explainer

    def _convert_rules(self, rules: DataFrame, categorical_mapping: dict) -> DataFrame:
        """
        Converts the rules from the explainer into forms that are more suitable to work with in code and for visualization.

        :param rules: The rules generated by the explainer.
        :return: The converted rules.
        """
        converted_rules = DataFrame(
            columns=['Cluster', 'Human Readable Rule', 'Coverage', 'Separation Error', 'Rule Binary Array',
                     'Error Explanation'])
        for explanation in rules.iterrows():
            # The explanation is a tuple, where the first element is the index and the second element is the explanation.
            explanation = explanation[1]
            # We first convert the stringified rule to a list of conditions, create a binary array from the rule,
            # and finally we also convert it into a human readable format, then save everything.
            rule = str_rule_to_list(explanation['rule'])
            if self._explain_errors:
                rule_binary_arr = condition_generator(data=self._source_df, rules=[rule], mode=self.explanation_form)
            else:
                rule_binary_arr = None
            human_readable_rule = rule_to_human_readable(rule, categorical_mapping, mode=self.explanation_form)
            cluster = explanation['Cluster']
            coverage = explanation['coverage']
            separation_error = explanation['separation_err']
            converted_rules = pd.concat([converted_rules, DataFrame({'Cluster': [cluster],
                                                                     'Human Readable Rule': [human_readable_rule],
                                                                     'Coverage': [coverage],
                                                                     'Separation Error': [separation_error],
                                                                     'Rule Binary Array': [rule_binary_arr],
                                                                     'Error Explanation': [None]
                                                                     })])
        return converted_rules

    def _map_categorical_features_to_one_hot_encoded_df(self, categorical_features: List[str],
                                                        one_hot_encoded_df: DataFrame) -> dict:
        """
        Maps the categorical features, after one-hot encoding, to the original dataframe, so that
        we can display the categorical features with their original names in the visualizations.

        :param categorical_features: The categorical features in the original dataframe.
        :param one_hot_encoded_df: The one-hot encoded dataframe.

        :return: A dictionary mapping the categorical features to the one-hot encoded dataframe. Each key is a column name
        in the one-hot encoded dataframe, and each value is the original categorical feature name.
        """
        mapping = {}
        for feature in categorical_features:
            for column in one_hot_encoded_df.columns:
                if column.startswith(feature):
                    mapping[column] = feature
        return mapping

    def _create_error_explanation_text(self, group_counts: DataFrame) -> str:
        """
        Creates the textual explanation for the separation error.

        :param group_counts: The counts of the points in each group.

        :return: The textual explanation for the separation error. Explanation is of the form "x% of error originates from group y, z% of the error originates from group w" etc.
        """
        group_counts = group_counts / group_counts.sum()
        # This name is used to sum up all the groups that have less than error_explanation_threshold % of the points.
        # It is a long and nonsensical name to avoid collisions with existing group names, and because it never leaves
        # this method, it does not matter what it is called.
        group_counts_low = group_counts[group_counts <= self._error_explanation_threshold]
        other_errors = group_counts_low.sum()
        group_counts_high = group_counts[group_counts > self._error_explanation_threshold]

        group_counts_high = group_counts_high.sort_values(ascending=False)
        error_explanation = ""
        for group, count in group_counts_high.items():
            error_explanation += f"{count * 100:.2f}% from group {group}, "
        if other_errors > 0:
            num_low_groups = group_counts_low.shape[0]
            error_explanation += f"{other_errors * 100:.2f}% from {num_low_groups} other group(s), each individually causing less than {self._error_explanation_threshold * 100:.2f}% of the error.  "
        error_explanation = error_explanation[:-2]
        return error_explanation

    def _create_error_explanations(self, converted_rules_df: DataFrame) -> DataFrame:
        """
        Provides an explanation for the separation error of the rules.
        Explanations are of the form "x% of error originates from group y, z% of the error originates from group w" etc.
        The returned DataFrame is the original dataframe, but modified to either have a new column called 'Error Explanation'
        or has the existing column modified to include the error explanation.

        :param converted_rules_df: The DataFrame with the converted rules, generated by the _convert_rules method.
        Alternatively, so long as the DataFrame has the 'Separation Error', 'Rule Binary Array' and 'Cluster' columns, it can be used, even if it was not
        generated by the _convert_rules method.

        :return: The DataFrame with the error explanations.
        """
        if "Rule Binary Array" not in converted_rules_df.columns:
            raise ValueError("The DataFrame must have a 'Rule Binary Array' column to explain the errors.")
        if "Cluster" not in converted_rules_df.columns:
            raise ValueError("The DataFrame must have a 'Cluster' column to explain the errors.")
        if "Error Explanation" not in converted_rules_df.columns:
            converted_rules_df['Error Explanation'] = None
        converted_rules_df = converted_rules_df.reset_index(drop=True)
        for idx, row in converted_rules_df.iterrows():
            separation_err = row['Separation Error']
            if separation_err == 0:
                converted_rules_df.loc[idx, 'Error Explanation'] = "Rule has no separation error."
            else:
                # Get the rule binary array from the row.
                rule = row['Rule Binary Array']
                cluster = row['Cluster']
                if rule is None:
                    continue
                # Get the indexes of points that are not in the cluster. We need this first because otherwise,
                # we would have an index and size mismatch when we try to apply the rule to the data or extract
                # the data that is not in the cluster.
                not_in_cluster = self._labels[self._labels != cluster].index
                # Keep the parts of the rule that correspond to points that are not in the cluster
                not_in_cluster_rule = rule[not_in_cluster]
                data_not_in_cluster = self._source_df.loc[not_in_cluster]
                # Apply the rule to the data that is not in the cluster
                data_not_in_cluster = data_not_in_cluster[not_in_cluster_rule]
                # Groupby and count the number of points that are in each group
                group_counts = data_not_in_cluster.groupby(self._labels[not_in_cluster]).size()
                # Create the error explanation
                error_explanation = self._create_error_explanation_text(group_counts)
                converted_rules_df.loc[idx, 'Error Explanation'] = error_explanation
        return converted_rules_df

    def visualize(self):

        if not self._ran_explainer:
            raise RuntimeError("You must run the explainer before visualizing the results.")

        # First, we one-hot encode the categorical features, as that is what the explainer did, and also what PCA needs.
        self._explainer.df = self._source_df
        categorical_features = self._explainer.one_hot()
        self._source_df = self._explainer.df
        # We map the one-hot encoded features to the original categorical features, so that we can display them in the visualizations.
        if categorical_features:
            categorical_mapping = self._map_categorical_features_to_one_hot_encoded_df(categorical_features,
                                                                                       self._source_df)
        else:
            categorical_mapping = {}

        converted_rules = self._convert_rules(self._explanations, categorical_mapping)

        unique_cluster_labels = list(self._labels.unique())
        unique_cluster_labels.sort()

        # Create a dataframe with a multi-index, where the first level is the cluster title, the second level is the
        # rule, and the values are the rule quality metrics.
        if self._explain_errors:
            columns = ['Coverage', 'Separation Error', 'Separation Error Origins']
        else:
            columns = ['Coverage', 'Separation Error']
        out_df = pd.DataFrame(columns=columns,
                              index=pd.MultiIndex(levels=[[], []], codes=[[], []],
                                                  names=['Group / Cluster', 'Explanation']))

        if self._explain_errors:
            converted_rules = self._create_error_explanations(converted_rules)

        # Fill in the dataframe with the rule quality metrics and rules.
        for idx, row in converted_rules.iterrows():
            out_df.loc[(row['Cluster'], row['Human Readable Rule']), 'Coverage'] = row['Coverage']
            out_df.loc[(row['Cluster'], row['Human Readable Rule']), 'Separation Error'] = row['Separation Error']
            if self._explain_errors:
                out_df.loc[(row['Cluster'], row['Human Readable Rule']), 'Separation Error Origins'] = row[
                    'Error Explanation']

        # For any cluster that does not have a rule, we fill in the dataframe with NaN values, and set rule
        # in that index to "No explanation found".
        for cluster in unique_cluster_labels:
            if cluster not in out_df.index.get_level_values(0):
                out_df.loc[(cluster, "No explanation found"), 'Coverage'] = np.nan
                out_df.loc[(cluster, "No explanation found"), 'Separation Error'] = np.nan
                if self._explain_errors:
                    out_df.loc[(cluster, "No explanation found"), 'Separation Error Origins'] = np.nan


        if self._add_llm_context_explanations:
            reasoning = ExplanationReasoning(
                data=self._source_df,
                labels=self._labels,
                explanations_found=out_df,
                source_name=self._source_name,
                query_type='many_to_one',
            )
            llm_explanations = reasoning.do_llm_action()
            out_df['LLM Explanation'] = llm_explanations

        self.out_df = out_df

        return out_df

    def generate_explanation(self):

        # Create the explainer object and generate the explanations.
        self._explainer = Explainer(self._source_df, self._labels)
        if self.explanation_form.startswith('conj'):
            self.explanation_form = 'conjunction'
        elif self.explanation_form.startswith('disj'):
            self.explanation_form = 'disjunction'
        else:
            raise ValueError("Explanations must be either conjunctions or disjunctions.")

        self._explanations = self._explainer.generate_explanations(coverage_threshold=self._coverage_threshold,
                                                                   conciseness_threshold=self._conciseness_threshold,
                                                                   separation_threshold=self._separation_threshold,
                                                                   p_value=self._p_value,
                                                                   mode=self.explanation_form)

        # Reverse the mapping, if a mapping was created.
        if self._mapping:
            self._explanations['Cluster'] = self._explanations['Cluster'].map({v: k for k, v in self._mapping.items()})
            self._labels = self._labels.map({v: k for k, v in self._mapping.items()})

        self._ran_explainer = True

        return self._explanations


    def get_explanation_in_textual_description(self, index: int) -> str:
        """
        Get the explanation for a specific index in a textual description format.
        If the explanations have not been generated yet, this method should raise an error.
        :param index: A single index to get the explanation for.
        :return: A human-readable string that explains the operation performed, what was found, and the explanation itself.
        """
        if not self._ran_explainer:
            raise RuntimeError("You must run the explainer before getting the explanations.")

        explanation = self.out_df.iloc[index]
        textual_description = (f"For the dataframe {self._source_name}, "
                               f"we used automated analysis to create rule based explanations for groupings in the data.\n")
        if self._label_name is not None:
            textual_description += (f"The labels used for the groupings were taken from a series called '{self._label_name}', "
                                    f"which may also correspond to a column in the dataframe.\n")
        textual_description += (
                               f"For the group {explanation.name[0]}, we found the following rule: {explanation.name[1]}.\n"
                               f"This explanation covers {(explanation['Coverage'] * 100):.2f}% of the data in the group, "
                               f"and has a separation error (coverage of points outside the group) of {(explanation['Separation Error'] * 100):.2f}%.\n"
        )
        if self._explain_errors:
            textual_description += (f"The separation error originates from the following groups: {explanation['Separation Error Origins']}.\n")
        if 'LLM Explanation' in explanation:
            textual_description += (f"Using a LLM, it suggested the following context to the explanation: {explanation['LLM Explanation']}\n")
        return textual_description

