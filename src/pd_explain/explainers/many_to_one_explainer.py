from typing import Literal, List

from .explainer_interface import ExplainerInterface
import pandas as pd
from pandas import DataFrame, Series
from warnings import warn
from cluster_explorer import Explainer, condition_generator, str_rule_to_list, rule_to_human_readable
import numpy as np

MAX_LABELS = 10


class ManyToOneExplainer(ExplainerInterface):

    def __init__(self, source_df: DataFrame, labels: Series | str | list[int], coverage_threshold: float = 0.7,
                 max_explanation_length: int = 3, separation_threshold: float = 0.3, p_value: int = 0,
                 explanation_form: Literal['conj', 'conjunction', 'disj', 'disjunction'] = 'conj',
                 attributes: List[str] = None, operation=None,
                 *args, **kwargs):

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

        if labels is None:
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
        # If the labels are a list of integers, we simply convert them to a Series object.
        elif not isinstance(labels, Series):
            self._labels = Series(labels)
        # If the labels are already a Series object, we simply use them as they are.
        else:
            self._labels = labels

        self._possible_to_visualize = True
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

        if p_value == 0:
            p_value = max_explanation_length

        self._p_value = p_value

        self._explanations = None
        self.explanation_form = explanation_form
        self._explainer = None
        self._ran_explainer = False

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
        if operation and not "GroupBy" in operation.__repr__():
            raise ValueError(
                "If the last operation performed on the DataFrame is not a groupby operation, you must provide labels.")
        else:
            # Extract the source and result df from the operation, if one is provided.
            if source_df is None:
                source_df = DataFrame(operation.unsampled_source_df) if hasattr(operation, 'unsampled_source_df') else DataFrame(operation.source_df)
                result_df = operation.unsampled_result_df if hasattr(operation, 'unsampled_result_df') else operation.result_df
            elif source_df is None or result_df is None:
                raise ValueError(
                    "You must provide either an operation or the source and result DataFrames of a groupby operation.")
            group_attributes = result_df.index.names
            index_values = result_df.index.values
            # Create an array of all -1 values, the length of the original DataFrame, as a placeholder for the labels.
            labels = np.full(source_df.shape[0], -1, dtype='O')
            # Go over the index values and assign the group labels to the corresponding rows in the source DataFrame.
            source_group_attributes_only = source_df[group_attributes]
            for i, group in enumerate(index_values):
                group_indices = source_group_attributes_only.index[
                    source_group_attributes_only[group_attributes].eq(group).all(axis=1)].values
                labels[group_indices] = str(group)

            # We drop the groupby attributes from the source dataframe, as otherwise, the explainer will return
            # an explanation saying the best rule is "groupby_attribute == group", which is not very informative.
            source_df = source_df.drop(group_attributes, axis=1, inplace=False)

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
        converted_rules = DataFrame(columns=['Cluster', 'Human Readable Rule', 'Coverage', 'Separation Error'])
        for explanation in rules.iterrows():
            # The explanation is a tuple, where the first element is the index and the second element is the explanation.
            idx = explanation[0]
            explanation = explanation[1]
            # We first convert the stringified rule to a list of conditions, create a binary array from the rule,
            # and finally we also convert it into a human readable format, then save everything.
            rule = str_rule_to_list(explanation['rule'])
            human_readable_rule = rule_to_human_readable(rule, categorical_mapping, mode=self.explanation_form)
            cluster = explanation['Cluster']
            coverage = explanation['coverage']
            separation_error = explanation['separation_err']
            converted_rules = pd.concat([converted_rules, DataFrame({'Cluster': [cluster],
                                                                    'Human Readable Rule': [human_readable_rule],
                                                                    'Coverage': [coverage],
                                                                    'Separation Error': [separation_error]})])
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

        unique_cluster_labels = self._labels.unique()
        unique_cluster_labels.sort()


        # Create a dataframe with a multi-index, where the first level is the cluster title, the second level is the
        # rule, and the values are the rule quality metrics.
        out_df = pd.DataFrame(columns=['Coverage', 'Separation Error'],
                              index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Group / Cluster', 'Explanation']))

        # Fill in the dataframe with the rule quality metrics and rules.
        for idx, row in converted_rules.iterrows():
            out_df.loc[(row['Cluster'], row['Human Readable Rule']), 'Coverage'] = row['Coverage']
            out_df.loc[(row['Cluster'], row['Human Readable Rule']), 'Separation Error'] = row['Separation Error']

        # For any cluster that does not have a rule, we fill in the dataframe with NaN values, and set rule
        # in that index to "No explanation found".
        for cluster in unique_cluster_labels:
            if cluster not in out_df.index.get_level_values(0):
                out_df.loc[(cluster, "No explanation found"), 'Coverage'] = np.nan
                out_df.loc[(cluster, "No explanation found"), 'Separation Error'] = np.nan

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
