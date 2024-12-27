"""
These functions are used to set and get global configuration parameters.
They are made for convenience, so that the user can set and get global configurations easily, without
having to go through the recommender objects.
"""

from pd_explain.recommenders.configurations.global_configuration import GlobalConfiguration

from typing import Dict, Any

config = GlobalConfiguration()

def get_global_recommender_config_settings() -> Dict[str, Dict[str, Any]]:
    """
    Gets the current settings of the global recommender configurations.
    """
    return config.recommender_configurations

def set_global_recommender_config_settings(new_settings: Dict[str, Dict[str, Any]], apply_to_existing_recommenders: bool = True):
    """
    Sets the global recommender configurations to the given settings.
    Only the settings that are present in the new_settings dictionary will be updated.

    :param new_settings: The new settings to set. Expected format: {'recommender_name': {'config_key': config_value}}.
    That is, a dict where the keys are the recommender names, and the values are dictionaries with the configuration keys
    and their values. Example: {'FilterRecommender': {'num_bins': 30, 'top_k_attributes': 5}}

    :param apply_to_existing_recommenders: If True, the new settings will be applied to the existing recommender objects.
    If False, the new settings will only apply to new recommender objects.
    """
    config.set_configuration(new_settings, apply_to_existing_recommenders)


def enable_recommenders(recommender_names: list[str], apply_to_existing_recommenders: bool = True):
    """
    Enable the recommenders with the given names.

    :param recommender_names: The names of the recommenders to enable.
    :param apply_to_existing_recommenders: If True, the changes will be applied to the existing recommender objects.
    If False, the changes will only apply to new recommender objects.
    """
    config.enable_recommenders(recommender_names, apply_to_existing_recommenders)


def disable_recommenders(recommender_names: list[str], apply_to_existing_recommenders: bool = True):
    """
    Disable the recommenders with the given names.

    :param recommender_names: The names of the recommenders to disable.
    :param apply_to_existing_recommenders: If True, the changes will be applied to the existing recommender objects.
    If False, the changes will only apply to new recommender objects.
    """
    config.disable_recommenders(recommender_names, apply_to_existing_recommenders)

def get_global_config_info():
    """
    Get the information about the global configuration settings.
    """
    return config.config_info


