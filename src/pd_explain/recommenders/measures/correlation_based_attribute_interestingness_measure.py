from pd_explain.recommenders.measures.attribute_interestingness_measure_base import AttributeInterestingnessMeasureBase
import pandas as pd
from pd_explain.recommenders.utils.util_funcs import is_numeric

class CorrelationBasedAttributeInterestingnessMeasure(AttributeInterestingnessMeasureBase):

    def __init__(self):
        super().__init__()
        self._corr_matrix_pearson = None
        self._corr_matrix_spearman = None

    def _calculate_interestingness(self, data, column) -> float:
        """
        Calculate how interesting an attribute is based on its correlation with other attributes.
        An attribute is more interesting if it is highly correlated with many other attributes.
        """
        if self._should_refresh_internals:
            self._corr_matrix_pearson = data.corr(method='pearson')
            self._corr_matrix_spearman = data.corr(method='spearman')
            self._should_refresh_internals = False

        if is_numeric(data, column):
            # If the data is numeric, we use pearson correlation.
            corr = self._corr_matrix_pearson[column].abs()
        else:
            # If the data is not numeric, we use spearman correlation
            corr = self._corr_matrix_spearman[column].abs()

        # Drop the correlation with the attribute itself, as it will always be 1.
        corr = corr.drop(column)

        # We first normalize the correlation values, so that the effect of differences in scale is minimized.
        normalized_corr = (corr - corr.min()) / (corr.max() - corr.min())

        # To get a more nuanced interestingness score, we sum the correlation values and multiply by the maximum correlation value.
        # The goal is to reward attributes with high correlation with multiple attributes, and not attributes with either
        # high correlation with a single attribute or low correlation with many attributes.
        interestingness = normalized_corr.sum() * normalized_corr.max()

        return interestingness


if __name__ == '__main__':
    dataset = pd.DataFrame(pd.read_csv(r"..\..\..\..\Examples\Datasets\adult.csv"))
    analyzer = CorrelationBasedAttributeInterestingnessMeasure()

    interestingness = analyzer.compute_measure(dataset)
    print(interestingness)

