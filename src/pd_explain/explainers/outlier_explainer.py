from pd_explain.explainers.explainer_interface import ExplainerInterface
from external_explainers import OutlierExplainer
from fedex_generator.Operations.GroupBy import GroupBy

from pandas import DataFrame, Series
from typing import List


class OutlierExplainerInterface(ExplainerInterface):

    def __init__(self, operation, target: str,
                 dir: int | str, control=None, hold_out: List = None, *args, **kwargs):
        """
        Initialize the OutlierExplainerInterface.
        This class is responsible for interfacing between the outlier explainer and the pd-explain package.

        :param operation: The operation to explain. Must be a GroupBy operation.
        :param target: The target value to explain whether it is an outlier or not and why.
        :param dir: The direction of the outlier. Can be 'low' or 'high' for low and high outliers respectively.
        """
        if target is None:
            raise ValueError("target must be specified")

        if hold_out is None:
            hold_out = []

        if not isinstance(operation, GroupBy):
            raise ValueError("Outlier explainer only works on the results of a groupby operation")

        res_col = None

        for attr, dataset_relation in operation.iterate_attributes():
            _, res_col = dataset_relation.get_source(attr), dataset_relation.get_result(attr)
            res_col = res_col[~res_col.isnull()]
            # Converting to a Series in case we get an ExpSeries (or some other overriden type), because those
            # can interact badly due to overriden methods changing the behavior of the object
            if res_col is not None:
                res_col = Series(res_col)
            break

        # Get the aggregation attribute and method
        try:
            agg = list(operation.agg_dict.items())[0]
        except:
            agg = operation.agg_dict.items()

        if type(operation.group_attributes) == list:
            g_attr = operation.group_attributes[0]
        else:
            g_attr = operation.group_attributes

        agg_attr, agg_method = agg[0], agg[1][0]

        self._df_agg = res_col
        # Like the above, converting to a DataFrame in case we get an ExpSeries or ExpDataFrame
        self._df_in = DataFrame(operation.source_df)
        self._g_att = g_attr
        if agg_attr in self._df_in.columns:
            self._g_agg = agg_attr
        else:
            agg_attr = agg_attr + "_" + agg_method
            if agg_attr in self._df_in.columns:
                self._g_agg = agg_attr
            else:
                raise ValueError(f"Could not find the aggregation attribute {agg_attr} in the DataFrame")
        self._agg_method = agg_method
        self._target = target
        if dir == "low":
            dir = -1
        elif dir == "high":
            dir = 1
        if dir not in [-1, 1]:
            raise ValueError("dir must be either 'low' or 'high' or -1 or 1")
        self._dir = dir
        self._control = control
        self._hold_out = hold_out
        self._explanation = None

    def generate_explanation(self):
        explainer = OutlierExplainer()
        self._explanation = explainer.explain(df_agg=self._df_agg, df_in=self._df_in, g_att=self._g_att,
                                              g_agg=self._g_agg,
                                              agg_method=self._agg_method, target=self._target, dir=self._dir,
                                              control=self._control, hold_out=self._hold_out)
        return self._explanation

    def can_visualize(self) -> bool:
        return True

    def visualize(self):
        return self._explanation
