from abc import ABC, abstractmethod

class ConfigurationBase(ABC):
    """
    Base class for the configuration of a recommender.
    All configuration classes (but not the global configuration) should inherit from this class.
    """

    def __init__(self, config: dict, recommender_name: str, added_info: dict = None):
        self._config = config
        self._recommender_name = recommender_name
        self._added_info = added_info
        # Set the configuration values as attributes of the class, so they can be accessed directly.
        for key in config.keys():
            setattr(self, key, config[key])


    @property
    def config_info(self):
        """
        The configuration information.
        The config info gives human readable explanations of the configuration keys, as a dictionary with the configuration keys as keys and the explanations as values.
        """
        return self._added_info


    @property
    def config(self):
        """
        The configuration of the recommender.
        The configuration is a dictionary with the configuration keys as keys and the configuration values as values.
        """
        return self._config

    @config.setter
    def config(self, value: dict):
        """
        Set the configuration of the recommender.

        """
        for key, val in value.items():
            if self._is_valid_key_value(key, val):
                self._config[key] = val
                setattr(self, key, val)

    @abstractmethod
    def _is_valid_key_value(self, key, val):
        raise NotImplementedError