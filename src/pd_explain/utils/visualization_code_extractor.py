from typing import Dict, Any, List, Tuple, Optional, Set
import ast
import os
import importlib.util
import glob


class VisualizationCodeExtractor:

    KNOWLEDGE_BANK: Dict[str, Dict[str, Any]] = {
        'MetaInsight': {
            'file': 'metainsight_explainer.py',
            'class': 'MetaInsightExplainer',
            'functions': ['visualize']
        },
        'fedex': {
            'file': 'fedex_explainer.py',
            'class': 'FedexExplainer',
            'functions': ['_visualize']
        },
        'graph_visualizer': {
            'file': 'graph_visualizer.py',
            'class': 'GraphAutomatedExplorationVisualizer',
            'functions': ['_create_query_tree_tab']
        },
    }

    def __init__(self):
        """
        Initializes the VisualizationCodeExtractor, which is responsible for extracting
        visualization code from the source files of the project.
        It uses the KNOWLEDGE_BANK to find the relevant files and functions to extract.
        """
        self.package_paths: Dict[str, str] = self._get_package_paths()

    class FunctionCallVisitor(ast.NodeVisitor):
        """
        An AST visitor that collects all function calls within a node.
        This is used to find all function calls within a function's body to enable
        recursive code extraction.
        """
        def __init__(self) -> None:
            self.calls: List[Tuple[str, ...]] = []

        def visit_Call(self, node: ast.Call) -> None:
            """
            Visit a Call node in the AST and extract information about the function call.
            It handles calls to methods of 'self', direct function calls by name, and
            attribute calls on other objects.
            """
            func = node.func
            # Case 1: Method call on 'self' (e.g., self.some_method())
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'self':
                self.calls.append(('self', func.attr))
            # Case 2: Direct function call by name (e.g., some_function())
            elif isinstance(func, ast.Name):
                self.calls.append(('name', func.id))
            # Case 3: Attribute call on an object (e.g., some_object.some_method())
            elif isinstance(func, ast.Attribute):
                try:
                    # ast.unparse can fail on complex objects, so we wrap it in a try-except block.
                    obj_str = ast.unparse(func.value)
                    self.calls.append(('attribute', obj_str, func.attr))
                except:
                    pass
            self.generic_visit(node)

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
        This is the entry point for the recursive code extraction process.

        :param requester_name: The name of the requester, used as a key in the KNOWLEDGE_BANK.
        :return: A string containing all the extracted function code, concatenated together.
        """
        info: Optional[Dict[str, Any]] = self.KNOWLEDGE_BANK.get(requester_name)
        if not info:
            return ""

        # Find the file path using a glob pattern to be more flexible.
        file_to_find: str = info['file']
        file_path: Optional[str] = None
        for pkg_path in self.package_paths.values():
            if pkg_path:
                # Using glob to find the file recursively inside the package path
                file_pattern = os.path.join(pkg_path, '**', file_to_find)
                found_paths = glob.glob(file_pattern, recursive=True)
                if found_paths:
                    file_path = found_paths[0]
                    break

        if not file_path:
            return ""

        collected_code: List[str] = []
        processed_functions: Set[Tuple[str, Optional[str], str]] = set()  # To avoid processing the same function multiple times.

        # Start the recursive extraction for each entry-point function.
        for func_name in info['functions']:
            self._get_code_recursively(file_path, info.get('class'), func_name, processed_functions, collected_code)

        return "\n\n".join(collected_code)

    def _get_code_recursively(self, file_path: str, class_name: Optional[str], func_name: str,
                              processed_functions: Set[Tuple[str, Optional[str], str]], collected_code: List[str]) -> None:
        """
        Recursively extracts the source code of a function and all other project-internal functions it calls.

        :param file_path: The path to the file containing the function.
        :param class_name: The name of the class containing the function (if any).
        :param func_name: The name of the function to extract.
        :param processed_functions: A set to keep track of functions that have already been processed to avoid infinite loops.
        :param collected_code: A list to accumulate the source code of all extracted functions.
        """
        # Base case: if the function has already been processed, or the file doesn't exist, stop.
        if (file_path, class_name, func_name) in processed_functions:
            return
        if not file_path or not os.path.exists(file_path):
            return

        processed_functions.add((file_path, class_name, func_name))

        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        try:
            file_ast: ast.Module = ast.parse(file_content, filename=file_path)
        except SyntaxError:
            return  # Skip files with syntax errors.

        # Find the AST node for the target function.
        func_node: Optional[ast.FunctionDef] = self._find_function_node(file_ast, class_name, func_name)
        if not func_node:
            return

        # Get all imports from the file to help resolve function calls.
        imports: Dict[str, str] = self._get_imports(file_ast)
        # Use the visitor to find all function calls within the current function.
        visitor: VisualizationCodeExtractor.FunctionCallVisitor = self.FunctionCallVisitor()
        visitor.visit(func_node)

        # Recursively process each called function.
        for call in visitor.calls:
            callee: Optional[Tuple[str, Optional[str], str]] = self._resolve_call(call, file_path, class_name, imports)
            if callee:
                self._get_code_recursively(callee[0], callee[1], callee[2], processed_functions, collected_code)

        # Add the source code of the current function to the collected code.
        func_code: str = ast.get_source_segment(file_content, func_node)
        if func_code not in collected_code:
            collected_code.append(func_code)

    def _find_function_node(self, file_ast: ast.Module, class_name: Optional[str], func_name: str) -> Optional[ast.FunctionDef]:
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
                    for sub_node in node.body:
                        if isinstance(sub_node, ast.FunctionDef) and sub_node.name == func_name:
                            return sub_node
        else:
            # Search for a top-level function.
            for node in file_ast.body:
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return node
        return None

    def _get_imports(self, file_ast: ast.Module) -> Dict[str, str]:
        """
        Extracts all import statements from a file's AST.

        :param file_ast: The AST of the file.
        :return: A dictionary mapping imported names (or aliases) to their full import paths.
        """
        imports: Dict[str, str] = {}
        for node in file_ast.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Correctly handle imports like 'import a.b.c'
                    local_name = alias.asname or alias.name.split('.')[0]
                    imports[local_name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                # Handle relative imports by adding dots.
                module_path = '.' * node.level + (node.module or '')
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{module_path}.{alias.name}"
        return imports

    def _find_class_ast(self, file_path: str, class_name: str) -> Optional[ast.ClassDef]:
        """
        Finds and returns the AST node for a class in a given file.

        :param file_path: The path to the Python file.
        :param class_name: The name of the class to find.
        :return: The ast.ClassDef node or None if not found.
        """
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        try:
            file_ast = ast.parse(file_content, filename=file_path)
            for node in file_ast.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    return node
        except (SyntaxError, UnicodeDecodeError):
            return None
        return None

    def _find_attribute_type_from_assignment(self, class_ast: ast.ClassDef, attr_name: str) -> Optional[str]:
        """
        Finds the type of a class attribute by looking for its assignment in __init__
        or as a class variable.

        :param class_ast: The AST node of the class.
        :param attr_name: The name of the attribute to find the type of.
        :return: The type of the attribute as a string, or None if not found.
        """
        # First, check for class variable assignments
        for node in class_ast.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                            return node.value.func.id

        # Then, check for assignments in __init__
        for node in class_ast.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for sub_node in node.body:
                    if isinstance(sub_node, ast.Assign):
                        target = sub_node.targets[0]
                        if isinstance(target, ast.Attribute) and target.attr == attr_name and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            rhs = sub_node.value
                            if isinstance(rhs, ast.Call):
                                if isinstance(rhs.func, ast.Name):
                                    return rhs.func.id  # e.g., self.foo = MyClass() -> returns 'MyClass'
                            elif isinstance(rhs, ast.Name):
                                # self.foo = some_param. Find type hint of some_param in __init__.
                                for arg in node.args.args:
                                    if arg.arg == rhs.id:
                                        return ast.unparse(arg.annotation) if arg.annotation else ast.unparse(arg)
        return None

    def _resolve_call(self, call: Tuple[str, ...], current_file: str, current_class: Optional[str], imports: Dict[str, str]) -> Optional[Tuple[str, Optional[str], str]]:
        """
        Resolves a function call to its definition location (file, class, function name).

        :param call: A tuple representing the function call, from FunctionCallVisitor.
        :param current_file: The file where the call occurs.
        :param current_class: The class where the call occurs.
        :param imports: The dictionary of imports for the current file.
        :return: A tuple (file_path, class_name, func_name) or None if resolution fails.
        """
        call_type, call_info = call[0], call[1:]

        # A call to a method of the same class (e.g., self.method()).
        if call_type == 'self':
            return current_file, current_class, call_info[0]

        # A direct call to a function by its name (e.g., my_function()).
        if call_type == 'name':
            func_name = call_info[0]
            if func_name in imports:
                # The function is imported.
                return self._resolve_import(imports[func_name], current_file)
            else:
                # Check if it's a function defined in the same file.
                with open(current_file, 'r', encoding='utf-8') as f:
                    file_ast = ast.parse(f.read())
                if self._find_function_node(file_ast, None, func_name):
                    return current_file, None, func_name

        # A call to a method on an object (e.g., some_module.my_function()).
        if call_type == 'attribute':
            obj_name_str, method_name = call_info
            obj_parts = obj_name_str.split('.')
            base_obj = obj_parts[0]

            if base_obj in imports:
                # Handle complex object calls like 'a.b.c.method()' where 'a' is imported.
                if len(obj_parts) > 1:
                    full_path = f"{imports[base_obj]}.{'.'.join(obj_parts[1:])}.{method_name}"
                else:
                    full_path = f"{imports[base_obj]}.{method_name}"
                return self._resolve_import(full_path, current_file)

            if base_obj == 'self' and current_class and len(obj_parts) > 1:
                attr_name = obj_parts[1]
                class_ast = self._find_class_ast(current_file, current_class)
                if class_ast:
                    type_name = self._find_attribute_type_from_assignment(class_ast, attr_name)
                    if type_name:
                        if type_name in imports:
                            imported_path = imports[type_name]
                            remaining_path = ".".join(obj_parts[2:])
                            full_path = f"{imported_path}.{remaining_path}.{method_name}" if remaining_path else f"{imported_path}.{method_name}"
                            return self._resolve_import(full_path, current_file)
                        else:
                            remaining_path = ".".join(obj_parts[2:])
                            full_path = f"{type_name}.{remaining_path}.{method_name}" if remaining_path else f"{type_name}.{method_name}"
                            return self._resolve_import(full_path, current_file)

        return None

    def _resolve_import(self, full_path: str, current_file: str) -> Optional[Tuple[str, Optional[str], str]]:
        """
        Resolves an import path to the file and function/method it points to.
        This is limited to project-internal packages.

        :param full_path: The full import path (e.g., 'my_package.my_module.my_function').
        :param current_file: The path of the file containing the import.
        :return: A tuple (file_path, class_name, func_name) or None if resolution fails.
        """
        # Handle relative imports.
        if full_path.startswith('.'):
            base_dir = os.path.dirname(current_file)
            level = 0
            while full_path.startswith('.'):
                level += 1
                full_path = full_path[1:]

            # Go up the directory tree for each dot.
            base_dir = os.path.join(*([base_dir] + ['..'] * (level - 1)))

            module_path_parts = full_path.split('.')
            module_path = os.path.join(base_dir, *module_path_parts[:-1])

            if os.path.isfile(module_path + '.py'):
                return module_path + '.py', None, module_path_parts[-1]
            elif os.path.isdir(module_path):  # It's a class method
                if os.path.isfile(os.path.join(module_path, '__init__.py')):
                    return os.path.join(module_path, '__init__.py'), None, module_path_parts[-1]

        # Check if the import is from one of the project's packages.
        top_level_package = full_path.split('.')[0]
        if top_level_package not in self.package_paths:
            return None

        pkg_path = self.package_paths.get(top_level_package)
        if not pkg_path:
            return None

        path_parts = full_path.split('.')
        # The first part is the package name, which is the key for pkg_path.
        # The rest of the parts form the relative path inside the package dir.
        relative_path_parts = path_parts[1:]

        # Try to find the module file by progressively shortening the path from the end.
        temp_relative_path_parts = relative_path_parts.copy()
        module_file_path = None

        # Handle cases where module is a file or a directory with __init__.py
        while temp_relative_path_parts:
            potential_path_as_file = os.path.join(pkg_path, *temp_relative_path_parts) + '.py'
            potential_path_as_dir = os.path.join(pkg_path, *temp_relative_path_parts)

            if os.path.isfile(potential_path_as_file):
                module_file_path = potential_path_as_file
                break
            elif os.path.isdir(potential_path_as_dir) and os.path.isfile(os.path.join(potential_path_as_dir, '__init__.py')):
                 module_file_path = os.path.join(potential_path_as_dir, '__init__.py')
                 break
            temp_relative_path_parts.pop()

        if not module_file_path:
             # Fallback for when the module is the package itself (e.g. import fedex_generator)
            if os.path.isfile(os.path.join(pkg_path, '__init__.py')):
                module_file_path = os.path.join(pkg_path, '__init__.py')
                temp_relative_path_parts = []
            else:
                return None

        # Determine if the remaining parts of the path refer to a function or a class method.
        remaining_parts = relative_path_parts[len(temp_relative_path_parts):]
        if not remaining_parts:
             # This can happen if the call is on the module itself, e.g. module.function()
             # We can't determine the function name from here, so we assume it's in the __init__
             # and the method name from the call is the function name.
             # This part is tricky and might need more context. For now, we assume the method name is the function.
             return module_file_path, None, path_parts[-1]
        if len(remaining_parts) == 1:
            # It's a function in the module.
            return module_file_path, None, remaining_parts[0]
        elif len(remaining_parts) == 2:
            # It's a method in a class.
            return module_file_path, remaining_parts[0], remaining_parts[1]

        return None