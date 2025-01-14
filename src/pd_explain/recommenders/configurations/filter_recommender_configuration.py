from pd_explain.recommenders.configurations.configuration_base import ConfigurationBase

class FilterRecommenderConfiguration(ConfigurationBase):
    """
    A configuration specific to the Filter Recommender.
    Includes default values for the configuration, information about the values, and validation of the values.
    """

    def __init__(self):

        config = {
            "attributes": None,
            "top_k_attributes": 3,
            "top_k_recommendations": 1,
            "top_k_explanations": 4,
            "num_bins": 10
        }
        added_info = {
            "num_bins": "The number of bins to use when binning the data to generate recommendations."
                        "The higher the number, the more recommendation candidates will be generated, but it will also be slower to compute.",
            "attributes": "The attributes to recommend queries for. If None, the recommender will automatically select the attributes.",
            "top_k_attributes" : "The maximum number of attributes to recommend queries for.",
            "top_k_recommendations" : "The maximum number of recommendations to return for each attribute.",
            "top_k_explanations" : "The maximum number of explanations to provide for each recommendation."
        }
        self._valid_keys = config.keys()
        super().__init__(config, "Filter Recommender", added_info)

    def _is_valid_key_value(self, key, val):
        if key not in self._valid_keys:
            return False
        if key == "top_k_attributes" or key == "top_k_recommendations" or key == "top_k_explanations":
            if not isinstance(val, int) or val < 1:
                return False
        return True
