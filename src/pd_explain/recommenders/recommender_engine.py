import importlib
import pkgutil
import inspect
from pathlib import Path

from pandas import DataFrame

from pd_explain.recommenders.recommender_base import RecommenderBase
from pd_explain.recommenders.utils.consts import PACKAGE_NAME
from typing import List

class RecommenderEngine:

    def __init__(self, df: DataFrame, disabled_recommenders=None):
        # If the input inherits from 'DataFrame', convert it to a 'DataFrame' object.
        self._df = DataFrame(df)
        self._recommenders = []
        self.enabled_recommenders = []
        self.disabled_recommenders = []
        self.load_recommenders()

        if disabled_recommenders is not None and len(disabled_recommenders) > 0:
            self.disable_recommenders(disabled_recommenders)

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
            # If the class is a recommender, add it to the list of recommenders and enable it by default.
            if name.endswith('Recommender') and issubclass(obj, RecommenderBase) and obj is not RecommenderBase:
                recommender_instance = obj()
                self._recommenders.append(recommender_instance)
                self.enabled_recommenders.append(recommender_instance.name)

    def _disable_recommender(self, recommender_name: str) -> None:
        """
        Disable a recommender by name.
        Disabling is done by removing the recommender from the 'enabled_recommenders' list and adding it to the 'disabled_recommenders' list.
        """
        recommender = self._get_recommender_by_name(recommender_name)
        # If the recommender is found and is enabled, disable it.
        if recommender and recommender.name in self.enabled_recommenders:
            self.enabled_recommenders.remove(recommender.name)
            self.disabled_recommenders.append(recommender.name)
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
        """
        recommender = self._get_recommender_by_name(recommender_name)
        # If the recommender is found and is disabled, enable it.
        if recommender and recommender.name in self.disabled_recommenders:
            self.disabled_recommenders.remove(recommender.name)
            self.enabled_recommenders.append(recommender.name)
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

    def recommend(self) -> List:
        """
        Recommend a list of items.

        :return: A list of recommendations.
        """
        recommendations = []
        for recommender in self._recommenders:
            if recommender.name in self.enabled_recommenders:
                recommendations.extend(recommender.recommend())
        return recommendations

# Example usage
if __name__ == "__main__":
    engine = RecommenderEngine()
    print(engine._recommenders)
    print(engine.enabled_recommenders)
    print(engine.disabled_recommenders)
    engine.disable_recommenders('filter')
    print(engine.enabled_recommenders)
    print(engine.disabled_recommenders)