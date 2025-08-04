from typing import Dict, List, Optional
import ast
import os
import importlib.util


class VisualizationCodeExtractor:
    KNOWLEDGE_BANK: Dict[str, List[Dict[str, Optional[str]]]] = {
        'MetaInsight': [
            {
                'package': 'external_explainers',
                'file': 'meta_insight.py',
                'class': 'MetaInsight',
                'function': '__init__'
            },
            {
                'package': 'external_explainers',
                'file': 'data_pattern.py',
                'class': 'BasicDataPattern',
                'function': '__init__'
            },
            {
                'package': 'external_explainers',
                'file': 'data_pattern.py',
                'class': 'BasicDataPattern',
                'function': '__init__'
            },
            {
                'package': 'external_explainers',
                'file': 'data_scope.py',
                'class': 'DataScope',
                'function': '__init__'
            },
            {
                'package': 'external_explainers',
                'file': 'data_scope.py',
                'class': 'HomogenousDataScope',
                'function': '__init__'
            },
            {
                'package': 'external_explainers',
                'file': 'pattern_base_classes.py',
                'class': 'PatternBase',
                'function': '__init__'
            },
        ],
        'fedex-gb': [
            {
                'package': 'pd_explain',
                'file': 'fedex_explainer.py',
                'class': 'FedexExplainer',
                'function': '_visualize'
            },
            {
                'package': 'fedex_generator',
                'file': 'GroupBy.py',
                'class': 'GroupBy',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'BaseMeasure.py',
                'class': 'BaseMeasure',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': 'build_operation_expression'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': 'draw_bar'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': '_find_max_group_value'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': '_select_top_columns'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': '_fix_explanation'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': None,  # No class, top-level function
                'function': 'draw_bar'
            }
        ],
        'fedex': [
            {
                'package': 'pd_explain',
                'file': 'fedex_explainer.py',
                'class': 'FedexExplainer',
                'function': '_visualize'
            },
            {
                'package': 'fedex_generator',
                'file': 'Filter.py',
                'class': 'Filter',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'BaseMeasure.py',
                'class': 'BaseMeasure',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'ExceptionalityMeasure.py',
                'class': 'ExceptionalityMeasure',
                'function': 'build_operation_expression'
            },
            {
                'package': 'fedex_generator',
                'file': 'ExceptionalityMeasure.py',
                'class': 'ExceptionalityMeasure',
                'function': 'draw_bar'
            },
            {
                'package': 'fedex_generator',
                'file': 'BaseMeasure.py',
                'class': 'BaseMeasure',
                'function': 'get_max_k'
            },
        ],
        'fedex-all': [
            {
                'package': 'pd_explain',
                'file': 'fedex_explainer.py',
                'class': 'FedexExplainer',
                'function': '_visualize'
            },
            {
                'package': 'fedex_generator',
                'file': 'Filter.py',
                'class': 'Filter',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'BaseMeasure.py',
                'class': 'BaseMeasure',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'ExceptionalityMeasure.py',
                'class': 'ExceptionalityMeasure',
                'function': 'build_operation_expression'
            },
            {
                'package': 'fedex_generator',
                'file': 'ExceptionalityMeasure.py',
                'class': 'ExceptionalityMeasure',
                'function': 'draw_bar'
            },
            {
                'package': 'fedex_generator',
                'file': 'BaseMeasure.py',
                'class': 'BaseMeasure',
                'function': 'get_max_k'
            },
            {
                'package': 'fedex_generator',
                'file': 'GroupBy.py',
                'class': 'GroupBy',
                'function': 'draw_figures'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': 'build_operation_expression'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': 'draw_bar'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': '_find_max_group_value'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': '_select_top_columns'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': 'DiversityMeasure',
                'function': '_fix_explanation'
            },
            {
                'package': 'fedex_generator',
                'file': 'DiversityMeasure.py',
                'class': None,  # No class, top-level function
                'function': 'draw_bar'
            }
        ],
    }

    def __init__(self):
        """
        Initializes the VisualizationCodeExtractor, which is responsible for extracting
        visualization code from the source files of the project.
        It uses the KNOWLEDGE_BANK to find the relevant files and functions to extract.
        """
        self.package_paths: Dict[str, str] = self._get_package_paths()

    def _get_package_paths(self) -> Dict[str, str]:
        """
        Finds the installation paths of the project's packages.
        This is to locate the source files for recursive code extraction, which is necessary
        because this package is intended to be installed via pip and used in various environments.

        :return: A dictionary mapping package names to their absolute paths.
        """
        # Packages are added with both snake_case and kebab-case names to support both local and pip installations.
        package_names = ["pd-explain", "fedex-generator", "cluster-explorer", "external-explainers",
                         "pd_explain", "fedex_generator", "cluster_explorer", "external_explainers"]
        paths = {}
        for name in package_names:
            try:
                spec = importlib.util.find_spec(name)
                if spec and spec.submodule_search_locations:
                    paths[name] = spec.submodule_search_locations[0]
            except (ModuleNotFoundError, ValueError):
                # Package might not be installed in the environment.
                pass
        return paths

    def get_visualization_code_from_source(self, requester_name: str) -> str:
        """
        Extracts the visualization code from the source files based on the requester's name.
        This is the entry point for the code extraction process.

        :param requester_name: The name of the requester, used as a key in the KNOWLEDGE_BANK.
        :return: A string containing all the extracted function code, concatenated together.
        """
        functions_to_extract: Optional[List[Dict[str, Optional[str]]]] = self.KNOWLEDGE_BANK.get(requester_name)
        if not functions_to_extract:
            return ""

        collected_code: List[str] = []

        for func_info in functions_to_extract:
            pkg_name = func_info.get('package')
            file_name = func_info.get('file')
            class_name = func_info.get('class')
            func_name = func_info.get('function')

            if not all([pkg_name, file_name]):
                continue

            pkg_path = self.package_paths.get(pkg_name)
            if not pkg_path:
                # Our local packages use snake_case, but the pip packages use kebab-case.
                # That might have been the reason for not finding the file.
                # So, we try to find the file with the kebab-case name.
                pkg_path = self.package_paths.get(pkg_name.replace('_', '-'))
                if not pkg_path:
                    # If we still can't find the package, skip this function.
                    continue

            file_path = self._find_file_in_package(pkg_path, file_name)
            if not file_path:
                continue

            func_code = self._extract_function_code(file_path, class_name, func_name)
            if func_code and func_code not in collected_code:
                collected_code.append(func_code)

        return "\n\n".join(collected_code)

    def _find_file_in_package(self, pkg_path: str, file_name: str) -> Optional[str]:
        """
        Finds the absolute path of a file within a package directory.

        :param pkg_path: The absolute path to the package.
        :param file_name: The name of the file to find.
        :return: The absolute path of the file, or None if not found.
        """
        for root, _, files in os.walk(pkg_path):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    def _extract_function_code(self, file_path: str, class_name: Optional[str], func_name: str) -> Optional[str]:
        """
        Extracts the source code of a single function from a file.

        :param file_path: The absolute path to the file.
        :param class_name: The name of the class containing the function (if any).
        :param func_name: The name of the function to extract.
        :return: The source code of the function, or None if not found.
        """
        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        try:
            file_ast: ast.Module = ast.parse(file_content, filename=file_path)
        except SyntaxError:
            return None  # Skip files with syntax errors.

        func_node = self._find_function_node(file_ast, class_name, func_name)
        if not func_node:
            return None

        if isinstance(func_node, ast.Module):
            # If the function node is a module, we return the entire file content.
            return file_content
        return ast.get_source_segment(file_content, func_node)

    def _find_function_node(self, file_ast: ast.Module, class_name: Optional[str], func_name: str) -> Optional[
        ast.FunctionDef | ast.Module | ast.ClassDef]:
        """
        Finds the AST node for a specific function within a file's AST.

        :param file_ast: The AST of the file to search in.
        :param class_name: The name of the class containing the function, or None if it's a top-level function.
        :param func_name: The name of the function to find.
        :return: The ast.FunctionDef node for the function, or None if not found.
        """
        if class_name:
            # Search within a specific class.
            for node in file_ast.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    if func_name is None:
                        # If no function name is provided, return the entire class AST.
                        return node
                    for sub_node in node.body:
                        if isinstance(sub_node, ast.FunctionDef) and sub_node.name == func_name:
                            return sub_node
        else:
            # If no class and func name is provided, we want to return the entire file AST.
            if func_name is None:
                return file_ast
            # Search for a top-level function.
            for node in file_ast.body:
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return node
        return None
