import re

from pd_explain.explainers.explainer_interface import ExplainerInterface
from external_explainers import OutlierExplainer
from fedex_generator.Operations.GroupBy import GroupBy
from pd_explain.llm_integrations.explanation_reasoning import ExplanationReasoning
from fedex_generator.commons.utils import get_calling_params_name

from pandas import DataFrame, Series
from typing import List


class OutlierExplainerInterface(ExplainerInterface):

    def __init__(self, operation, target: str,
                 dir: int | str, control=None, hold_out: List = None,
                 add_llm_context_explanations: bool = False,
                 *args, **kwargs):
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
        self._add_llm_context_explanations = add_llm_context_explanations
        self._operation = operation
        self._explainer = None
        self._query = None
        self._added_text = None
        self._explanations = None

    def generate_explanation(self):
        self._explainer = OutlierExplainer()
        self._explanation = self._explainer.explain(df_agg=self._df_agg, df_in=self._df_in, g_att=self._g_att,
                                              g_agg=self._g_agg,
                                              agg_method=self._agg_method, target=self._target, dir=self._dir,
                                              control=self._control, hold_out=self._hold_out,
                                              # Always delay the drawing of the plot to the visualization step
                                              draw_plot=False)

        return self._explanation

    def can_visualize(self) -> bool:
        return self._explanation is not None and isinstance(self._explanation, tuple) and len(self._explanation) > 0

    def visualize(self):
        self._added_text = None
        self._explanations = self._explainer.pred_to_human_readable(self._explanation[4])[0]
        if self._add_llm_context_explanations and isinstance(self._explanation, tuple):
            self._query = self._operation.source_name + ".groupby('" + self._g_att + "').agg({'" + self._g_agg + "':'" + \
                    self._agg_method + "'})"
            reasoner = ExplanationReasoning(
                data=self._df_in,
                after_op_data=self._df_agg,
                dir="high" if self._dir == 1 else "low",
                target=self._target,
                query_type='outlier',
                query=self._query,
                explanations_found=self._explanations,
                source_name=self._operation.source_name
            )
            reasoning = reasoner.do_llm_action()
            self._added_text = {
                "text": reasoning,
                "position": "bottom"
            }
        return self._explainer.draw_bar_plot(*self._explanation, added_text=self._added_text)

    def get_explanation_in_textual_description(self, index = None) -> str:
        """
        Get the explanation for a specific index in a textual description format.

        :param index: Ignored, since there is only a single explanation for outliers. Needed to be compatible with the interface.
        :return: A human-readable string that explains the operation performed, what was found, and the explanation itself.
        """
        if self._explanation is None:
            raise ValueError("Explanations have not been generated yet. Please call generate_explanation() first.")

        textual_description = (f"After running the query {self._query} on the dataframe {self._operation.source_name}, "
                               f"the value {self._target} was suspected to be a a {'high' if self._dir == 1 else 'low'} outlier. "
                               f"Using automated analysis, we found that: ")
        pattern = re.compile(r"\$\\bf(.*?)\$")
        textual_description += f"{pattern.sub(r'\1', self._explanations).replace("\n", " ")}. \n"
        if self._added_text is not None:
            textual_description += (f"Using a LLM to reason about this explanation, without access to the data or ability to "
                                    f"query it, it suggested that the following may provide additional context to the findings: "
                                    f"{self._added_text['text'].replace('\n', ' ')}. \n")

        return textual_description