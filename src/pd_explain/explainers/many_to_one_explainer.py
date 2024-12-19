from typing import Literal, List

from .explainer_interface import ExplainerInterface
import pandas as pd
from pandas import DataFrame, Series
from warnings import warn
from cluster_explorer import Explainer, condition_generator, str_rule_to_list, rule_to_human_readable
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import Tab, HTML, HTMLMath, Output, VBox, HBox, Box, Layout
import textwrap


class ManyToOneExplainer(ExplainerInterface):

    def __init__(self, source_df: DataFrame, labels: Series | str | list[int], coverage_threshold: float = 0.6,
                 max_explanation_length: int = 5, separation_threshold: float = 0.5, p_value: int = 0,
                 use_pca_for_visualization: bool = True, pca_components: Literal[2, 3] = 2,
                 explanation_form: Literal['conjunctive', 'disjunctive'] = 'conjunctive',
                 select_columns=None, operation=None,
                 *args, **kwargs):

        # Convert the source_df to a DataFrame object, to avoid overhead from overridden methods in ExpDataFrame,
        # as well as to avoid any bad interactions between those methods and the explainer.
        if type(source_df) != DataFrame:
            self._source_df = DataFrame(source_df)

        if select_columns is not None and len(select_columns) > 0:
            source_df = source_df[select_columns]

        if labels is None:
            self._source_df, self._labels = self._create_groupby_labels(operation)
        # If the labels are a string, we assume that the string is the name of the column that contains the labels.
        elif isinstance(labels, str):
            self._labels = source_df[labels]
            self._source_df = self._source_df.drop(labels, axis=1, inplace=False)
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
        if use_pca_for_visualization and pca_components > 3:
            raise ValueError("The number of PCA components must be at most 3. We do not support 4D or higher plots.")
        if not use_pca_for_visualization and source_df.shape[1] > 3:
            warn("The dataframe is too high dimensional to visualize. We recommend using PCA for visualization.")
            self._possible_to_visualize = False

        self._coverage_threshold = coverage_threshold
        self._max_explanation_length = max_explanation_length
        self._separation_threshold = separation_threshold
        self._conciseness_threshold = 1 / self._max_explanation_length

        if p_value == 0:
            p_value = max_explanation_length

        self._p_value = p_value

        self._use_pca_for_visualization = use_pca_for_visualization
        self._pca_components = pca_components
        self._explanations = None
        self.explanation_form = explanation_form
        self._explainer = None
        self._ran_explainer = False

    def _create_groupby_labels(self, operation) -> (DataFrame, Series):
        """
        Create labels for the many-to-one explainer based on the last operation performed on the DataFrame.
        If the last operation is a groupby operation, the labels are the groups created by the groupby operation.
        Otherwise, the user must provide labels.

        :param operation: The last operation performed on the DataFrame.
        :return: The source DataFrame from the operation and the labels.
        We return the source dataframe as well, because the source dataframe from the operation is prior to the
        groupby operation, and that is what interests us.
        """
        if not "GroupBy" in operation.__repr__():
            raise ValueError(
                "If the last operation performed on the DataFrame is not a groupby operation, you must provide labels.")
        else:
            # Get the groupby attributes and the index values of the resulting DataFrame.
            group_attributes = operation.result_df.index.names
            index_values = operation.result_df.index.values
            source_df = DataFrame(operation.source_df)
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
        return self._possible_to_visualize

    def _plot_clusters(self, to_visualize: np.ndarray, cluster_labels: np.ndarray | Series):
        """
        Visualizes all clusters in the data each in a different color, in one plot.
        """
        # Create a 3D plot if the data is 3D, otherwise create a 2D plot.
        if to_visualize.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()

        labels = cluster_labels.unique()
        labels.sort()

        for i, label in enumerate(labels):
            try:
                label_title = f"Cluster {int(label)}"
            except ValueError:
                label_title = f"{label}"
            current_datapoints = to_visualize[cluster_labels == label]
            if to_visualize.shape[1] == 3:
                ax.scatter(current_datapoints[:, 0], current_datapoints[:, 1], current_datapoints[:, 2],
                           label=label_title)
            else:
                ax.scatter(current_datapoints[:, 0], current_datapoints[:, 1], label=label_title)

        return fig, ax

    def _convert_rules(self, rules: DataFrame, categorical_mapping: dict) -> DataFrame:
        """
        Converts the rules from the explainer into forms that are more suitable to work with in code and for visualization.

        :param rules: The rules generated by the explainer.
        :return: The converted rules.
        """
        converted_rules = DataFrame(columns=['Rule', 'Cluster', 'Explanation', 'Idx', 'Human Readable Rule'])
        for explanation in rules.iterrows():
            # The explanation is a tuple, where the first element is the index and the second element is the explanation.
            idx = explanation[0]
            explanation = explanation[1]
            # We first convert the stringified rule to a list of conditions, create a binary array from the rule,
            # and finally we also convert it into a human readable format, then save everything.
            rule = str_rule_to_list(explanation['rule'])
            rule_as_binary_np_array = condition_generator(data=self._source_df, rules=[rule])
            human_readable_rule = rule_to_human_readable(rule, categorical_mapping)
            cluster = explanation['Cluster']
            converted_rules = pd.concat(
                [converted_rules, DataFrame({'Rule': [rule_as_binary_np_array], 'Cluster': cluster,
                                             'Explanation': explanation['rule'], 'Idx': idx,
                                             'Human Readable Rule': human_readable_rule})])
        return converted_rules

    def _create_general_tab(self, to_visualize: np.ndarray, cluster_labels: np.ndarray | Series) -> Tab:
        """
        Creates the initial tab of the visualizations, that contains:
        1. A plot of all clusters in the data, each in a different color.
        2. An explanation of the quality metrics used to evaluate the explanations.

        :param to_visualize: The data to visualize.
        :param cluster_labels: The labels of the clusters.
        :return: The tab containing the visualizations.
        """
        out = Output()
        with out:
            fig, ax = self._plot_clusters(to_visualize, cluster_labels)
            if len(cluster_labels.unique()) > 7:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                ax.legend(loc='upper right')
            ax.set_title("All Clusters")
            plt.show(fig)

        general_tab = Tab()
        metric_explanations_html = HTMLMath(f"""
                    <h2>Explanation quality metrics</h2>
                    <h4>Conciseness</h4>
                    <p>Conciseness is a measure of how concise the explanation is.<br>
                    It is calculated as the inverse of the number of conditions in the rule.
                    A rule with fewer conditions is considered more concise and thus better.
                    </p>
                    <h4>Separation error</h4>
                    <p>The ratio of points for which explanation $E_c$ holds, but those points are not in cluster / group $c$. 
                    Mathematically, it is defined as: $$\\frac{{|\\ x \\in X \\ | \\ E_c (x) = True \\wedge CL(x) \\neq c \\ |}}{{|\\ x \\in X \\ | \\ E_c(x) = True\\ |}}$$
                    The lower the separation error, the better the explanation is at separating the cluster from other points.
                    </p>
                    <h4>Coverage</h4>
                    <p>The ratio of points for which explanation $E_c$ holds and those points are in cluster / group $c$.
                    Mathematically, it is defined as: $$\\frac{{|\\ x \\in X \\ | \\ E_c (x) = True \\wedge CL(x) = c\\ |}}{{|\\ x \\in X \\ | \\ CL(x) = c \\ |}}$$
                    The higher the coverage, the better.
                    </p>
                """)
        general_tab.children = [Box(children=[out]), metric_explanations_html]
        general_tab.set_title(0, "All Clusters Plot")
        general_tab.set_title(1, "Explanation Metrics")

        return general_tab

    def _create_tab_for_cluster(self, cluster: str | int, rules: DataFrame, to_visualize: np.ndarray,
                                cluster_title: str, ) -> HTML | Tab:
        """
        Create a tab for a specific cluster, containing sub-tabs for each rule that explains the cluster.

        :param cluster: The cluster to create the tab for.
        :param rules: The rules generated by the explainer and converted to a more suitable format.
        :param to_visualize: The data to visualize.
        :param cluster_title: The title of the cluster.

        :return: The tab containing the explanations for the cluster, or an HTML element if no explanations were found.
        """
        cluster_tab = Tab()
        cluster_explanations = self._explanations[self._explanations['Cluster'] == cluster]
        cluster_rules = rules[rules['Cluster'] == cluster]
        cluster_outputs = []

        if cluster_explanations.empty:
            res = HTML(f"<h2>No explanation found for {cluster_title}</h2>")
            return res

        # Go over the rules for each cluster, and plot the data, emphasizing the data points that are explained by
        # each rule. Each of these plots go into a separate sub-tab.
        for rule in cluster_rules.iterrows():
            tab_hbox = HBox()
            text_vbox = VBox(layout=Layout(width='32%', left='2%'))
            rule_row = rule[1]
            idx = rule_row['Idx']
            rule = rule_row['Rule']
            explanation_row = self._explanations.iloc[idx]

            # Get the data points that are explained by the rule.
            explained_data_points = to_visualize[rule]

            out = Output()

            # Visualize all the data, then add an "X" marker for the data points that are explained by the rule.
            with out:
                fig, ax = self._plot_clusters(to_visualize, self._labels)
                ax.scatter(explained_data_points[:, 0], explained_data_points[:, 1], marker='X', c='black',
                           label='Covered by rule')
                # Add a legend with cluster labels + "X" for the explained data points.
                if len(self._labels.unique()) > 7:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax.legend(loc='upper right')
                plt.show(fig)

            human_readable_rule = rule_row['Human Readable Rule']
            # To make the rule more readable, we add line breaks after "AND" and "OR".
            # We also add a special token for splitting the rule into lines we can wrap.
            if "AND" in human_readable_rule:
                human_readable_rule = human_readable_rule.replace("AND", "AND<br>@@@")
            if "OR" in human_readable_rule:
                human_readable_rule = human_readable_rule.replace("OR", "OR<br>@@@")
            human_readable_rule_split = human_readable_rule.split("@@@")
            human_readable_rule_split = ["<br>".join(textwrap.wrap(line, width=40)) for line in
                                         human_readable_rule_split]
            human_readable_rule = "".join(human_readable_rule_split)

            text_vbox.children = [HTML(f"""
                        <h2>Rule:<br> {human_readable_rule}</h2><hr width='100%' size='2'>
                        <h3>Conciseness: {explanation_row['conciseness']}</h3><br>
                        <h3>Separation error: {explanation_row['separation_err']}</h3><br>
                        <h3>Coverage: {explanation_row['coverage']}</h3><br>
                        """)]
            left_box = Box(children=[out], layout=Layout(max_width='60%'))
            tab_hbox.children = [left_box, text_vbox]

            cluster_outputs.append(tab_hbox)

        # Add the explanations to the tab.
        cluster_tab.children = cluster_outputs
        for i, output in enumerate(cluster_outputs):
            cluster_tab.set_title(i, f"Explanation {i + 1}")

        return cluster_tab

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

        # Since most dataframes are too high dimensional to visualize, we use PCA to reduce the dimensionality.
        # The user can choose to use PCA for visualization or not.
        if self._use_pca_for_visualization:
            pca = PCA(n_components=self._pca_components)
            # We use one-hot encoding for categorical data, as PCA requires numerical data.
            to_visualize = pca.fit_transform(self._source_df)
        else:
            to_visualize = self._source_df

        converted_rules = self._convert_rules(self._explanations, categorical_mapping)

        cluster_titles = []
        cluster_titles.append("General")
        unique_cluster_labels = self._labels.unique()
        unique_cluster_labels.sort()
        # Get a list of cluster titles, which will be used as the titles of the tabs.
        # The title is either "Cluster {cluster_id}" if the cluster id is an int or the name of the cluster.
        for cluster in unique_cluster_labels:
            try:
                cluster_titles.append(f"Cluster {int(cluster)}")
            except ValueError:
                cluster_titles.append(f"{cluster}")

        # Create the outer tab that contains all the cluster tabs.
        cluster_tabs = Tab()
        # Add the first, general tab that contains the plot of all clusters and the explanation of the metrics.
        cluster_tabs_children = [self._create_general_tab(to_visualize, self._labels)]

        # Populate the outer tab with a tab for each cluster, each tab containing tabs for each rule.
        for i, cluster in enumerate(unique_cluster_labels):
            cluster_tabs_children.append(
                self._create_tab_for_cluster(cluster, converted_rules, to_visualize, cluster_titles[i + 1]))

        cluster_tabs.children = cluster_tabs_children
        # Give the tabs the appropriate titles.
        for i, title in enumerate(cluster_titles):
            cluster_tabs.set_title(i, title)

        return cluster_tabs

    def generate_explanation(self):

        # Create the explainer object and generate the explanations.
        self._explainer = Explainer(self._source_df, self._labels)
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
