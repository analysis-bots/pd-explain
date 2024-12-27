import importlib
import pkgutil
import inspect
from pathlib import Path

from pandas import DataFrame, Series
from ipywidgets import Tab, HTML
from IPython.display import display

from pd_explain.recommenders.recommender_base import RecommenderBase
from pd_explain.recommenders.utils.consts import PACKAGE_NAME
from pd_explain.recommenders.utils.listener_interface import ListenerInterface
from typing import List, Dict, Any
from pd_explain.recommenders.configurations.global_configuration import GlobalConfiguration


class RecommenderEngine(ListenerInterface):
    """
    The RecommenderEngine class is the main class of the recommender module, and is the only one that is exposed by the module.
    This class is responsible for managing all recommenders, enabling and disabling them, and getting recommendations from them.
    """

    def __init__(self, df: DataFrame, disabled_recommenders=None):
        # We need a DataFrame to work with. If the user provides a different type, we will try to convert it to a DataFrame.
        # If that fails, an error will be raised by the DataFrame constructor.
        self._df = DataFrame(df)
        self._global_config = GlobalConfiguration()
        self._recommenders = []
        self.enabled_recommenders = self._global_config.enabled_recommenders
        self.disabled_recommenders = self._global_config.disabled_recommenders
        self.load_recommenders()
        self._global_config.add_listener(self)
        self._attribute_backup = None

        if disabled_recommenders is not None and len(disabled_recommenders) > 0:
            self.disable_recommenders(disabled_recommenders)


    def __str__(self):
        return f"RecommenderEngine with {len(self._recommenders)} recommenders. Enabled recommenders: {self.enabled_recommenders}. Disabled recommenders: {self.disabled_recommenders}."

    def __repr__(self):
        return self.__str__()

    def on_event(self, values: dict):
        for key, value in values.items():
            if key == "engine":
                if "enable" in value:
                    self.enable_recommenders(value["enable"])
                if "disable" in value:
                    self.disable_recommenders(value["disable"])

    def load_recommenders(self):
        """
        Dynamically load all recommenders from the 'recommenders' package.
        """
        package_dir = Path(__file__).resolve().parent
        self._recursive_load_modules(package_dir, PACKAGE_NAME)

    def _recursive_load_modules(self, directory: Path, package_name: str) -> None:
        """
        Recursively load all modules from the given directory.
        If a module is a package, recursively load its modules as well.
        """
        # Iterate over all modules in the directory.
        for _, module_name, is_pkg in pkgutil.iter_modules([str(directory)]):
            full_module_name = f"{package_name}.{module_name}"
            # If the module is a recommender, load its classes.
            if module_name.endswith('_recommender'):
                module = importlib.import_module(full_module_name)
                self._load_classes_from_module(module)
            # If the module is a package, recursively load its modules.
            if is_pkg:
                self._recursive_load_modules(directory / module_name, full_module_name)

    def _load_classes_from_module(self, module) -> None:
        """
        Load all classes from the given module that inherit from 'RecommenderBase'.
        """
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # If the class is a recommender, add it to the list of recommenders and enable it (unless it has been disabled globally).
            if name.endswith('Recommender') and issubclass(obj, RecommenderBase) and obj is not RecommenderBase:
                recommender_instance = obj()
                self._recommenders.append(recommender_instance)
                if recommender_instance.name not in self.disabled_recommenders:
                    self.enabled_recommenders.add(recommender_instance.name)

    def _disable_recommender(self, recommender_name: str) -> None:
        """
        Disable a recommender by name.
        Disabling is done by removing the recommender from the 'enabled_recommenders' list and adding it to the 'disabled_recommenders' list.

        :param recommender_name: The name of the recommender to disable.
        """
        recommender = self._get_recommender_by_name(recommender_name)
        # If the recommender is found and is enabled, disable it.
        if recommender and recommender.name in self.enabled_recommenders and recommender.name not in self.disabled_recommenders:
            self.enabled_recommenders.remove(recommender.name)
            self.disabled_recommenders.add(recommender.name)
        # If the recommender is already disabled, do nothing.
        # Double disabling should not throw an error, but should not change the state either.
        elif recommender and recommender.name in self.disabled_recommenders:
            pass
        # Only if the recommender is not found, raise an error.
        else:
            raise ValueError(f"Recommender '{recommender_name}' not found.")

    def _get_recommender_by_name(self, recommender_name: str) -> RecommenderBase | None:
        """
        Gets a recommender by name.

        :param recommender_name: The name of the recommender.
        :return: The recommender object.
        """
        if "Recommender" not in recommender_name:
            recommender_name += "Recommender"
        for recommender in self._recommenders:
            if recommender.name.lower() == recommender_name.lower():
                return recommender
        return None

    def _enable_recommender(self, recommender_name: str) -> None:
        """
        Enable a recommender by name.
        Enabling is done by removing the recommender from the 'disabled_recommenders' list and adding it to the 'enabled_recommenders' list.

        :param recommender_name: The name of the recommender to enable.
        """
        recommender = self._get_recommender_by_name(recommender_name)
        # If the recommender is found and is disabled, enable it.
        if recommender and recommender.name in self.disabled_recommenders and recommender.name not in self.enabled_recommenders:
            self.disabled_recommenders.remove(recommender.name)
            self.enabled_recommenders.add(recommender.name)
        # If the recommender is already enabled, do nothing.
        # Double enabling should not throw an error, but should not change the state either.
        elif recommender and recommender.name in self.enabled_recommenders:
            pass
        # Only if the recommender is not found, raise an error.
        else:
            raise ValueError(f"Recommender '{recommender_name}' not found.")

    def disable_recommenders(self, recommender_names: str | List[str]) -> None:
        """
        Disable multiple recommenders by name.

        :param recommender_names: The names of the recommenders to disable. Either a single name or a list of names.
        """
        if isinstance(recommender_names, str):
            recommender_names = [recommender_names]
        for recommender_name in recommender_names:
            self._disable_recommender(recommender_name)

    def enable_recommenders(self, recommender_names: str | List[str]) -> None:
        """
        Enable multiple recommenders by name.

        :param recommender_names: The names of the recommenders to enable. Either a single name or a list of names.
        """
        if isinstance(recommender_names, str):
            recommender_names = [recommender_names]
        for recommender_name in recommender_names:
            self._enable_recommender(recommender_name)

    def recommend(self) -> Tab | HTML:
        """
        Get recommendations from all enabled recommenders.

        :return: A Tab widget containing the recommendations of all enabled recommenders.
        """
        if len(self.enabled_recommenders) == 0:
            return HTML("No recommenders enabled. Unable to provide recommendations.")

        recommendations_tab = Tab()
        recommendations = []
        # Get the recommendations from all enabled recommenders.
        for recommender in self._recommenders:
            if recommender.name in self.enabled_recommenders:
                recommendations.append(recommender.recommend(self._df))

        # Set a tab and a title for each recommender.
        recommendations_tab.children = recommendations
        for i, recommender in enumerate(self.enabled_recommenders):
            title = recommender.replace("Recommender", "").replace("recommender", "") + " Recommendations"
            recommendations_tab.set_title(i, title)

        # This is a hack, to make it so that tabs use a flex layout and don't cut off long titles.
        display(
            HTML(
                """
        <style>
        .jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
            flex: 0 1 auto
        }
        </style>
        """
            )
        )
        return recommendations_tab

    @property
    def recommender_configurations(self):
        """
        Get the configurations of all recommenders.
        """
        config = {
            "Enabled recommenders": list(self.enabled_recommenders),
            "Disabled recommenders": list(self.disabled_recommenders),
        }
        config.update(
            {
                recommender.name: recommender.config for recommender in self._recommenders
            }
        )
        return config

    @property
    def recommender_configurations_descriptions(self):
        """
        Get the configuration descriptions of all recommenders.
        """
        return {recommender.name: recommender.config_info for recommender in self._recommenders}

    @recommender_configurations.setter
    def recommender_configurations(self, config: Dict[str, Dict[str, Any]]) -> None:
        """
        Set the configuration of a recommender.

        :param config: The configuration to set. Expected format: {'recommender_name': {'config_key': config_value}}.
        """
        for recommender_name, recommender_config in config.items():
            recommender = self._get_recommender_by_name(recommender_name)
            if recommender:
                recommender.config = recommender_config
            else:
                raise ValueError(f"Recommender '{recommender_name}' not found.")

    @property
    def global_config_values(self):
        """
        Get the values of the global configurations of all recommenders.
        """
        return self._recommenders[0].global_config_values

    @property
    def global_config(self):
        """
        Get the global configurations of all recommenders.
        """
        return self._recommenders[0].global_config

    def __del__(self):
        self._global_config.remove_listener(self)

    def set_attributes(self, attributes):
        """
        Set the attributes to recommend queries for on all recommenders.

        :param attributes: The attributes to recommend queries for.
        """
        self._attribute_backup = {}
        for recommender in self._recommenders:
            self._attribute_backup[recommender.name] = recommender.config["attributes"]
            recommender.config = {"attributes": attributes}


    def restore_attributes(self):
        """
        Restores the attributes to recommend queries for to their previous state on all recommenders.
        This should be used after setting attributes with the 'set_attributes' method.
        """
        for recommender in self._recommenders:
            recommender.config = {"attributes": self._attribute_backup[recommender.name]}
        self._attribute_backup = None
