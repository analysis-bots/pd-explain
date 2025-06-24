import base64
import io
import json
import os
from typing import Optional, Any, Dict
import traceback

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, Image
from matplotlib.figure import Figure

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client
from pd_explain.llm_integrations import consts
from pd_explain.utils.visualization_code_extractor import VisualizationCodeExtractor


class VisualizationBeautifier(LLMIntegrationInterface):
    """
    A class to create case specific visualizations using LLMs.
    The LLM is provided how the current visualization looks like and its code, and is
    asked to create a new visualization that is more appealing and informative.
    """

    def __init__(self, visualization_object: Any, data: pd.DataFrame,
                 visualization_params: Optional[Dict[str, Any]] = None, requester_name: Optional[str] = None, visualization_code: Optional[str] = None) -> None:
        """
        Initializes the VisualizationBeautifier.

        :param visualization_object: The visualization to be beautified. Can be a path to an image, a matplotlib Figure, or an ipywidget.
        :param data: The pandas DataFrame used to generate the visualization.
        :param visualization_params: A dictionary of parameters to be passed to the generated visualization function.
        :param requester_name: The name of the component requesting the beautification (e.g., 'MetaInsight').
                               This is used to look up the code in the KNOWLEDGE_BANK.
        :param visualization_code: A string containing the code that generates the visualization.
                                   If provided, it overrides the code extraction from the KNOWLEDGE_BANK.
        """
        code_extractor = VisualizationCodeExtractor()
        if requester_name:
            self.visualization_code: str = code_extractor.get_visualization_code_from_source(requester_name)
        elif visualization_code:
            self.visualization_code: str = visualization_code
        else:
            raise ValueError("Either requester_name or visualization_code must be provided.")

        self.visualization_object: Any = visualization_object
        self.data: pd.DataFrame = data
        self.llm_generated_code: Optional[str] = None
        self.visualization_params: Dict[str, Any] = visualization_params if visualization_params else {}
        # The maximum number of attempts to fix the code if it fails to execute.
        self.max_fix_attempts: int = 3

    def _define_task(self) -> str:
        """
        Defines the task for the LLM.
        """
        return ("You are a data visualization expert. Your task is to take Python code that generates a visualization "
                "and an image of that visualization, and produce new Python code for a more aesthetically pleasing and "
                "effective visualization. The new visualization must preserve all the crucial information from the original. "
                "It must also preserve all functionality of the original code, such as interactivity. "
                "The new code should be general and not be tied to the specific data it sees, but rather accept the data as a parameter. "
                "The goal is to address problems such as visual clutter, plots that get too big and long, and "
                "sometimes too much details that may not be needed.")

    def _describe_output_format(self) -> str:
        """
        Describes the expected output format from the LLM.
        """
        return ("The output must be a single Python code block, enclosed in <python> and </python> tags. "
                "The code should define a single entry function `create_visualization(...)` that accepts the following parameters:\n"
                f"{', '.join(self.visualization_params.keys()) if self.visualization_params else 'data: pd.DataFrame'}.\n"
                "as input and generates and generates the visualization. Do not include any other text or explanation outside the code block. "
                "Make sure to import all necessary libraries inside the function.\n"
                "The function should return the visualization object, which can be a matplotlib Figure, an ipywidget, or any other visualization object.\n"
                "You can not override existing classes or functions, and your code will be run in a context separate from the global context, so you must import all necessary libraries inside the function.\n"
                )


    def _handle_response(self, response: str):
        self.llm_generated_code = self._extract_response(response, "<python>", "</python>")
        if not self.llm_generated_code:
            # Fallback for the case where the LLM does not use the expected tags, because sometimes it just chooses to use
            # the ```python and ``` tags instead. Because LLM.
            self.llm_generated_code = self._extract_response(response, "```python", "```")
            if not self.llm_generated_code:
                print("Could not extract beautified code from the LLM response.")
                print("\nRaw response:\n")
                print(response)
                return None
        return None

    def do_llm_action(self) -> tuple[widgets.Tab | None, str]:
        """
        Main method to generate and display the beautified visualization.
        This method orchestrates the process of:
        1. Converting the input visualization to an image (if possible).
        2. Sending the code and image to the LLM.
        3. Executing the LLM's generated code.
        4. Displaying the original and beautified visualizations in a tabbed widget.

        :return: A tuple containing the tab widget with the original and beautified visualizations, and the code
        generated by the LLM.
        """
        client = Client(
            api_key=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_KEY, None),
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER, "google"),
            model=os.getenv(consts.DEFAULT_BEAUTIFICATION_LLM_VISION_MODEL, "gemini-1.5-flash"),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL,
                                   "https://generativelanguage.googleapis.com/v1beta/")
        )

        encoded_image: Optional[str] = None
        # Convert the visualization object to a base64 encoded image.
        # Case 1: The visualization object is a path to an image file.
        if isinstance(self.visualization_object, str) and os.path.exists(self.visualization_object):
            with open(self.visualization_object, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Case 2: The visualization object is a matplotlib Figure.
        elif isinstance(self.visualization_object, Figure):
            buf = io.BytesIO()
            self.visualization_object.savefig(buf, format='png')
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        # Case 3: The visualization object is an ipywidget.
        elif isinstance(self.visualization_object, widgets.Widget):
            # For widgets, we can't easily create a screenshot.
            # We will not send an image to the LLM in this case.
            # The LLM will have to rely on the code alone.
            pass

        system_message: str = self._define_task()
        user_message: str = (
            f"Here is the original code that produced the visualization:\n\n"
            f"```python{self.visualization_code}\n```\n\n"
            f"Attached as well is the visualization itself. Please improve it.\n\n"
            f"{self._describe_output_format()}\n"
            f"Remember again to place the code inside <python> and </python> tags, or the program will not be able to extract it.\n"
        )

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f'data:image/jpeg;base64,{encoded_image}',
                        },
                    },
                ],
            }
        ]

        response: str = client(
            system_messages=[system_message],
            user_messages=user_messages,
            override_user_messages_formatting=True
        )

        self._handle_response(response)
        if not self.llm_generated_code:
            return None, ""

        assistant_messages = [
            self.llm_generated_code,
        ]

        # Create the widget for the original visualization.
        original_vis_widget = widgets.Output()
        with original_vis_widget:
            if isinstance(self.visualization_object, str) and os.path.exists(self.visualization_object):
                display(Image(filename=self.visualization_object))
            else:
                display(self.visualization_object)

        # Create the widget for the beautified visualization.
        beautified_vis_widget = widgets.Output()
        code_execution_passed = False
        num_fix_attempts_made = 0
        if self.llm_generated_code:
            while not code_execution_passed and num_fix_attempts_made < self.max_fix_attempts:
                with beautified_vis_widget:
                    try:
                        # Execute the generated code in a controlled environment.
                        exec_globals: Dict[str, Any] = {'pd': pd, 'widgets': widgets}
                        exec(self.llm_generated_code, exec_globals)
                        create_visualization_func = exec_globals.get('create_visualization')
                        if callable(create_visualization_func):
                            # Call the generated function with the provided parameters.
                            create_visualization_func(**self.visualization_params)
                            code_execution_passed = True
                        else:
                            print("Could not find the 'create_visualization' function in the generated code.")
                    except Exception as e:
                        print(f"Error executing generated code: {e}")
                        print(f"Attempting to fix the code... ({num_fix_attempts_made + 1}/{self.max_fix_attempts})")
                        user_messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"There was an error executing the generated code: {traceback.format_exc()}\n"
                                                             f"Please fix the code and try again. "}
                                ]
                            }
                        )
                        response = client(
                            system_messages=[system_message],
                            user_messages=user_messages,
                            assistant_messages=assistant_messages,
                            override_user_messages_formatting=True
                        )
                        self._handle_response(response)
                        if not self.llm_generated_code:
                            print("Could not extract beautified code from the LLM response after fixing attempts.")
                            return None, ""
                        num_fix_attempts_made += 1
        else:
            with beautified_vis_widget:
                print("Could not extract beautified code from the LLM response.")
                print("\nRaw response:\n")
                print(response)

        # Create and return the tab widget.
        tab = widgets.Tab()
        tab.children = [original_vis_widget, beautified_vis_widget]
        tab.set_title(0, 'Original Visualization')
        tab.set_title(1, 'Beautified Visualization')

        return tab, self.llm_generated_code

    def save_generated_code(self, file_path: str, key: str) -> None:
        """
        Saves the generated visualization code to a file in JSON format.

        :param file_path: The path to the file to save the code to.
        :param key: The key to use for this specific code snippet in the JSON file.
        """
        if not self.llm_generated_code:
            print("No code has been generated yet.")
            return

        data_to_save: Dict[str, str] = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data_to_save = json.load(f)
                except json.JSONDecodeError:
                    # File is empty or corrupted, start fresh
                    pass

        data_to_save[key] = self.llm_generated_code

        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Code saved to {file_path} with key '{key}'")
