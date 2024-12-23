from pd_explain.recommenders.analyzers.attribute_interestingness_analyzer_base import AttributeInterestingnessAnalyzerBase
import pandas as pd

class CorrelationBasedAttributeInterestingnessAnalyzer(AttributeInterestingnessAnalyzerBase):

    def _calculate_interestingness(self, data, column) -> float:
        """
        Calculate how interesting an attribute is based on its correlation with other attributes.
        An attribute is more interesting if it is highly correlated with many other attributes.
        """
        if self.is_numeric(data, column):
            # If the data is numeric, we use pearson correlation.
            corr = data.corr(method='pearson')[column].abs()
        else:
            # If the data is not numeric, we use spearman correlation
            corr = data.corr(method='spearman')[column].abs()

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
    analyzer = CorrelationBasedAttributeInterestingnessAnalyzer()

    interestingness = analyzer.analyze(dataset)
    print(interestingness)

