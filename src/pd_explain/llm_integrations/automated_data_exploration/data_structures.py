from collections import namedtuple, defaultdict
from dataclasses import dataclass

from pd_explain.explainers.fedex_explainer import FedexExplainer
from pd_explain.explainers.beta_explainers.metainsight_explainer import MetaInsightExplainer

apply_result = namedtuple("apply_result", ["index",
                                           "result",
                                           "generating_query",
                                           "error_occurred"])


@dataclass(frozen=False)
class QueryResultObject:
    """
    A class to hold the results of a query applied to a DataFrame.
    It contains the findings from the FedEx and MetaInsight explainers, as well as any error that occurred.
    """
    fedex: FedexExplainer | None = None
    metainsight: MetaInsightExplainer | None = None
    fedex_findings = str | None
    metainsight_findings = str | None
    error: str | None = None


tree_node = namedtuple("tree_node", ["source", "query", "children"])


class QueryTree:
    """
    A class to built and maintain the query tree for the deep dive analysis.
    """

    def __init__(self, source_name: str = "Original DataFrame"):
        self.tree = defaultdict(tree_node)
        self.source_name = source_name
        self.tree[0] = tree_node(source=None, query=source_name, children=[])

    def add_node(self, source_idx: int, query: str, new_index: int) -> None:
        """
        :param source_idx: The index of the query that this query is based on.
        :param query: The query to add to the tree.
        :param new_index: The index of the new query in the tree.
        """
        if source_idx not in self.tree:
            raise ValueError(f"Source index {source_idx} not found in the tree.")
        self.tree[new_index] = tree_node(source=source_idx, query=query, children=[])
        self.tree[source_idx].children.append(new_index)

    def get_node(self, idx: int) -> tree_node | None:
        return self.tree.get(idx, None)

    def __getitem__(self, item):
        """
        Allows the QueryTree to be indexed like a dictionary.
        :param item: The index of the query in the tree.
        :return: The tree node at the given index.
        """
        return self.get_node(item)