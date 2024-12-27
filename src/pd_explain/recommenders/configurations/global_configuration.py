from copy import copy

from singleton_decorator import singleton
from pd_explain.recommenders.configurations.filter_recommender_configuration import FilterRecommenderConfiguration


@singleton
class GlobalConfiguration:
    """
    The global configuration class is a singleton that holds a shared configuration for all recommenders.
    Settings in the global configuration can be applied to al recommenders, and will always be used as the default
    configuration for new recommenders.
    Global settings can be overridden by the user on a per-recommender basis.
    """

    def __init__(self):
        self._recommender_configurations = {
            'FilterRecommender': FilterRecommenderConfiguration()
        }
        self._listeners = []
        self._registered_recommenders = list(self._recommender_configurations.keys())
        self._enabled_recommenders = set(self._registered_recommenders)
        self._disabled_recommenders = set()


    @property
    def enabled_recommenders(self) -> set[str]:
        """
        Get the names of the enabled recommenders.
        """
        return copy(self._enabled_recommenders)

    @property
    def disabled_recommenders(self) -> set[str]:
        """
        Get the names of the disabled recommenders.
        """
        return copy(self._disabled_recommenders)


    def enable_recommenders(self, recommender_names: list[str], notify_listeners=True):
        """
        Enable the recommenders.

        :param recommender_names: The names of the recommenders to enable.
        :param notify_listeners: Whether to notify the listeners of the configuration change. Default is True.
        Setting to false will make it so only future recommender_engine objects will have the changes applied.
        """
        if isinstance(recommender_names, str):
            recommender_names = [recommender_names]

        enabled_recommenders = []

        for recommender_name in recommender_names:
            matched_name = self.match_recommender(recommender_name)
            if matched_name and matched_name in self._disabled_recommenders and matched_name not in self._enabled_recommenders:
                self._enabled_recommenders.add(matched_name)
                self._disabled_recommenders.remove(matched_name)
                enabled_recommenders.append(matched_name)
            if not matched_name:
                raise ValueError(f"Recommender '{recommender_name}' not found.")

        if notify_listeners:
            self.notify_listeners({
                'engine': {
                    'enable': enabled_recommenders
                }
            })

    def disable_recommenders(self, recommender_names: list[str], notify_listeners=True):
        """
        Disable the recommenders.

        :param recommender_names: The names of the recommenders to disable.
        :param notify_listeners: Whether to notify the listeners of the configuration change. Default is True.
        Setting to false will make it so only future recommender_engine objects will have the changes applied.
        """
        if isinstance(recommender_names, str):
            recommender_names = [recommender_names]

        disabled_recommenders = []

        for recommender_name in recommender_names:
            matched_name = self.match_recommender(recommender_name)
            if matched_name and matched_name in self._enabled_recommenders and matched_name not in self._disabled_recommenders:
                self._disabled_recommenders.add(matched_name)
                self._enabled_recommenders.remove(matched_name)
                disabled_recommenders.append(matched_name)
            if not matched_name:
                raise ValueError(f"Recommender '{recommender_name}' not found.")

        if notify_listeners:
            self.notify_listeners({
                'engine': {
                    'disable': disabled_recommenders
                }
            })

    @property
    def recommender_configurations(self):
        """
        Get the configurations of all recommenders.
        """
        config = {
            'Engine settings':
                {
                    'Enabled recommenders': self.enabled_recommenders,
                    'Disabled recommenders': self.disabled_recommenders
                }
        }
        config.update({
            name: config.config for name, config in self._recommender_configurations.items()
        })
        return config

    @property
    def config_info(self):
        """
        Gets the additional information about each attribute in the configuration, that each configuration class provides.
        """
        config = {
            'Engine settings':
                {
                    'Enabled recommenders': 'The recommenders that are currently enabled.',
                    'Disabled recommenders': 'The recommenders that are currently disabled.'
                }
        }
        config.update({
            name: config.config_info for name, config in self._recommender_configurations.items()
        })
        return config

    @property
    def registered_recommenders(self):
        """
        Get the names of all registered recommenders.
        """
        return copy(self._registered_recommenders)

    def set_configuration(self, configuration: dict, notify_listeners=True):
        """
        Set the configuration for the recommenders.
        Please note that you can not enable or disable recommenders using this method. See the enable_recommenders
        and disable_recommenders methods for that.

        :param configuration: The configuration to set. Expects a dictionary with the recommender name as the key, and
        another nested dictionary with the configuration keys as keys and the configuration values as values.
        :param notify_listeners: Whether to notify the listeners of the configuration change. Default is True.
        If set to true, all existing recommenders will have the changes applied to them. If set to false,
        the changes will only be applied to new recommenders.
        """
        replaced_keys = {}
        # Change the values of the configuration to the matched recommender names.
        # Matched names are the names of the registered recommenders, but may not be the same as the input names.
        # For example, if the input name is 'filter', the matched name will be 'FilterRecommender'.
        for recommender_name, config in configuration.items():
            matched_name = self.match_recommender(recommender_name)
            if matched_name:
                self._recommender_configurations[matched_name].config = config
                replaced_keys[recommender_name] = matched_name
            else:
                raise ValueError(f"Recommender '{recommender_name}' not found.")

        # Replace the keys in the configuration with the matched names.
        for key, value in replaced_keys.items():
            configuration.pop(key)
            configuration[value] = self._recommender_configurations[value].config

        if notify_listeners:
            self.notify_listeners(configuration)

    def get_recommender_configuration(self, recommender_name: str) -> dict:
        """
        Get the configuration of the recommender.

        :param recommender_name: The name of the recommender to get the configuration for.

        :return: The configuration of the recommender.
        """
        matched_name = self.match_recommender(recommender_name)
        if matched_name:
            return self._recommender_configurations[matched_name].config
        else:
            raise ValueError(f"Recommender '{recommender_name}' not found.")

    def match_recommender(self, recommender_name: str) -> str | None:
        """
        Match the input recommender name to the registered recommender names.

        :param recommender_name: The name of the recommender to match.

        :return: The matched recommender name, or None if no match was found.
        """
        recommender_name = recommender_name.lower().replace('recommender', '').strip()
        for name, config in self._recommender_configurations.items():
            if name.lower().replace('recommender', '').strip() == recommender_name:
                return name
        return None

    def add_listener(self, listener):
        """
        Add a listener to the global configuration.

        :param listener: The listener to add.
        """
        self._listeners.append(listener)

    def remove_listener(self, listener):
        """
        Remove a listener from the global configuration.

        :param listener: The listener to remove.
        """
        self._listeners.remove(listener)

    def notify_listeners(self, updates_values):
        """
        Notify all listeners of the configuration changes.

        :param updates_values: The updated values.
        """
        for listener in self._listeners:
            listener.on_event(updates_values)
