import base64
import io
import json
import os
from typing import Optional, Any, Dict, Tuple
import traceback

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import warnings

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client
from pd_explain.llm_integrations import consts
from pd_explain.utils.visualization_code_extractor import VisualizationCodeExtractor


def is_figure_empty(fig: Figure) -> bool:
    """
    Checks if a matplotlib Figure is empty (i.e., contains no plotted data).

    A figure is considered empty if all of its axes have no lines, patches,
    collections, or images.

    :param fig: The matplotlib Figure object to check.
    :return: True if the figure is empty, False otherwise.
    """
    if not fig.axes:
        # No axes on the figure at all
        return True

    for ax in fig.axes:
        # Check for common plotted elements (artists)
        has_lines = len(ax.get_lines()) > 0
        has_collections = len(ax.collections) > 0
        has_images = len(ax.images) > 0

        # You could also check ax.texts, but titles and labels are also texts.
        # The checks above are usually sufficient for data plots.

        if has_lines or has_collections or has_images:
            # Found an axis with data, so the figure is not empty
            return False

    # If we get here, no axis had any plotted data
    return True

class VisualizationBeautifier(LLMIntegrationInterface):
    """
    A class to create case specific visualizations using LLMs.
    The LLM is provided how the current visualization looks like and its code, and is
    asked to create a new visualization that is more appealing and informative.
    """

    def __init__(self, visualization_object: Any, data: pd.DataFrame,
                 visualization_params: Optional[Dict[str, Any]] = None,
                 requester_name: Optional[str] = None,
                 visualization_code: Optional[str] = None,
                 max_fix_attempts: int = 5,
                 must_generalize: bool = False,
                 silent: bool = True) -> None:
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
        self.max_fix_attempts: int = max_fix_attempts
        self.must_generalize: bool = must_generalize
        self.silent: bool = silent


    def _define_task(self) -> str:
        """
        Defines the task for the LLM.
        """
        task_str = (
            "You are a data visualization expert using Python's matplotlib and seaborn libraries. "
            "Your task is to take Python code that generates a visualization and an image of that visualization, and then "
            "produce new Python code for a more aesthetically pleasing and effective visualization.\n\n"
            "**Key Goals for Improvement:**\n"
            "1.  **Consolidation & Clarity:** The original visualization might have multiple, cluttered subplots. If possible, consolidate them into a single, well-organized figure. Use shared axes where appropriate. The goal is to reduce visual clutter and make comparisons easier.\n"
            "2.  **Aesthetics:** Use a professional color palette (e.g. from matplotlib). Ensure font sizes are legible and titles/labels are clear.\n"
            "3.  **Information Preservation:** The new visualization must preserve all the crucial information from the original, such as which groups are outliers and the values they represent.\n"
            "4.  **Limited Information**: If an object does not belong to a known library, and its code is not provided to you, you can only use its functions and properties that are already present in the original code. You cannot add new properties or functions to it.\n"
        )
        if not self.must_generalize:
            task_str += (
            "The code you create will be for one-time use and executed immediately. The user will only see the visualization, not the code. "
            "Therefore, you can make the code completely specific to this one visualization, without needing to generalize it for future use, "
            "and you can even hardcode specific values or data into the code.\n\n"
            )
        else:
            task_str += (
                "The code you create will not be for one-time use, but rather may be used to plot many different visualizations. "
                "As such, you must ensure that the code is general, relying solely on input parameters to the function you create, "
                "and not hardcoding any specific values or data. "
            )
        return task_str

    def _describe_output_format(self) -> str:
        """
        Describes the expected output format from the LLM.
        """
        return (
            "The output must be a single Python code block, enclosed in `<python>` and `</python>` tags.\n"
            "The code must define a single entry function `create_visualization(...)` that accepts the following parameters:\n"
            f"`{', '.join(self.visualization_params.keys()) if self.visualization_params else 'data: pd.DataFrame'}`.\n\n"
            "**CRITICAL INSTRUCTIONS:**\n"
            "1.  **Return the Figure:** The function **must** return the matplotlib `Figure` object that contains the plot.\n"
            "2.  **No Empty Plots:** The generated `Figure` **must not be empty**. It must have data plotted on its axes. Generating a blank canvas is a failure.\n"
            "3.  **Self-Contained:** All necessary libraries (e.g., `matplotlib.pyplot`, `seaborn`, `pandas`) must be imported *inside* the `create_visualization` function.\n"
            "4.  **No other text or explanation** should be included outside the code block.\n"
            "5.  **No comments** should be included in the code. No one will see the code, so comments are unnecessary and only waste tokens.\n"
            "6.  **Keep things simple**: Do not override the entire codebase, only write what is strictly necessary to create the improved visualization.\n"
        )


    def _handle_response(self, response: str):
        self.llm_generated_code = self._extract_response(response, "<python>", "</python>")
        if isinstance(self.llm_generated_code, str):
            self.llm_generated_code = self.llm_generated_code.strip()
        else:
            self.llm_generated_code = ""
        # If the length is less than 10 characters, something went very wrong, because that doesn't even cover the length of the expected
        # entry function.
        if not len(self.llm_generated_code) > 10:
            # Fallback for the case where the LLM does not use the expected tags, because sometimes it just chooses to use
            # the ```python and ``` tags instead. Because LLM.
            self.llm_generated_code = self._extract_response(response, "```python", "```")
            if isinstance(self.llm_generated_code, str):
                self.llm_generated_code = self.llm_generated_code.strip()
            else:
                self.llm_generated_code = ""
            # Final fallback - just take everything, remove whatever tags we find, and pray. Because LLMs are fun and it's not like
            # the LLM gets told the expected format or something twice.
            if not len(self.llm_generated_code) > 10:
                self.llm_generated_code = response
                # Remove any tags that might be present in the code.
                self.llm_generated_code = self.llm_generated_code.replace("<python>", "").replace("</python>", "")
                self.llm_generated_code = self.llm_generated_code.replace("```python", "").replace("```", "")
                self.llm_generated_code = self.llm_generated_code.strip()
                # If the length is still less than 10 characters, we have a problem, that problem probably being the LLM dying on us.
                if not len(self.llm_generated_code) > 10:
                    print("Could not extract code from the LLM response.")
                    return None
        return None


    def _encode_visualization(self, visualization_object) -> Optional[str]:
        """
        Encodes the visualization object to a base64 string if it is an image or a matplotlib Figure.
        Returns None if the visualization object is an ipywidget.
        """
        # Case 1: The visualization object is a path to an image file.
        if isinstance(visualization_object, str) and os.path.exists(visualization_object):
            with open(visualization_object, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        # Case 2: The visualization object is a matplotlib Figure.
        elif isinstance(visualization_object, Figure):
            buf = io.BytesIO()
            visualization_object.savefig(buf, format='png')
            buf.seek(0)
            encoded_image = base64.b64encode(buf.read()).decode('utf-8')
        # Case 3: The visualization object is an ipywidget.
        elif isinstance(visualization_object, widgets.Widget):
            # For widgets, we can't easily create a screenshot.
            # We will not send an image to the LLM in this case.
            # The LLM will have to rely on the code alone.
            encoded_image = None

        return encoded_image


    def execute(self, code) -> Tuple[Optional[Figure], Optional[str], Optional[str]]:
        """
        Executes the generated code in a controlled environment and checks the output.
        """
        beautified_figure = None
        error_message = None
        printed_error = None
        try:
            # Execute the generated code in a controlled environment.
            exec_globals: Dict[str, Any] = {'pd': pd, 'widgets': widgets, 'plt': plt}
            exec(code, exec_globals)
            create_visualization_func = exec_globals.get('create_visualization')

            if callable(create_visualization_func):
                # Call the generated function to get the figure
                beautified_figure = create_visualization_func(**self.visualization_params)

                if not isinstance(beautified_figure, Figure):
                    error_message = (
                        "The `create_visualization` function did not return a matplotlib Figure object. "
                        "Please ensure your function concludes with `return fig`.")
                    printed_error = "Incorrect return type"

            else:
                error_message = "Could not find the 'create_visualization' function in the generated code. Code can not be executed without this entry point."
                printed_error = "Missing entry function"

        except Exception as e:
            error_message = f"There was an error executing the generated code: {traceback.format_exc()}"
            printed_error = "Execution error - " + str(e)

        return beautified_figure, error_message, printed_error

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
            model=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_VISION_MODEL, "gemini-2.0-flash"),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL,
                                   "https://generativelanguage.googleapis.com/v1beta/")
        )

        encoded_image: Optional[str] = self._encode_visualization(self.visualization_object)

        system_message: str = self._define_task()
        # Create a data summary
        data_summary = io.StringIO()
        self.data.info(buf=data_summary)
        data_summary_str = data_summary.getvalue()

        user_message: str = (
            f"Here is the original code that produced the visualization:\n\n"
            f"<python>\n{self.visualization_code}\n</python>\n\n"
            f"For context, here is a summary of the pandas DataFrame that will be passed to your function:\n"
            f"```\n{data_summary_str}\n```\n"
            f"And here is the head of the data:\n"
            f"```\n{self.data.head().to_string()}\n```\n\n"
            f"Attached as well is the visualization itself. Please improve it by making it clearer and more consolidated.\n\n"
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

        # Create the widget for the original visualization.
        original_vis_widget = widgets.Output()
        with original_vis_widget:
            if isinstance(self.visualization_object, str) and os.path.exists(self.visualization_object):
                display(Image(filename=self.visualization_object))
            else:
                display(self.visualization_object)


        # Create the widget for the beautified visualization.
        beautified_vis_widget = widgets.Output()

        last_working_code = None
        last_loop_provided_code = False
        beautified_figure = None
        # Suppress warnings, because the LLM code may raise warnings galore.
        warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
        if self.llm_generated_code:
            for i in range(self.max_fix_attempts):
                plt.close('all')  # Close any previous plots
                beautified_figure, error_message, printed_error = self.execute(self.llm_generated_code)

                # If there was any error, try to fix it
                if error_message:
                    if not self.silent:
                        print(f"Error encountered in LLM generated code - {printed_error}")
                        print(f"Attempting to fix the code... ({i + 1}/{self.max_fix_attempts})")

                    # Append the previous attempt and the error to the message history
                    user_messages.append(
                        {"role": "assistant", "content": f"<python>{self.llm_generated_code}</python>"})
                    user_messages.append({"role": "user",
                                          "content": f"{error_message}\nPlease fix the code and provide the full, corrected code block."})

                    response = client(
                        system_messages=[system_message],
                        user_messages=user_messages,
                        override_user_messages_formatting=True
                    )
                    self._handle_response(response)
                    if not self.llm_generated_code:
                        break  # Exit the loop if we can't get new code
                # If there was no error, we instead prompt the LLM for two things:
                # 1. To approve or disapprove the generated visualization.
                # 2. If it disapproves, to provide a new code that fixes the issues.
                else:
                    last_working_code = self.llm_generated_code
                    if not self.silent:
                        print("The generated code executed successfully.")
                        print(f"Approving or improving the generated visualization... {i + 1}/{self.max_fix_attempts}")
                    user_messages.append(
                        {"role": "assistant", "content": f"<python>{self.llm_generated_code}</python>"}
                    )
                    encoded_image = self._encode_visualization(beautified_figure)
                    user_message_text = (
                        "Please review the generated visualization and either approve it, in which case it will be immediately displayed, "
                        "or disapprove it and provide new code that improves the visualization.\n"
                        "Approve the visualization if you think it is clear, consolidated, and aesthetically pleasing, while also "
                        "relaying all of the important information from the original visualization.\n"
                        "Provide your approval status between <approve and </approve> and as a single boolean value of 'True' or 'False'.\n"
                        "Provide the new code inside <python> and </python> tags, or the program will not be able to extract it.\n"
                    )
                    user_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f'data:image/jpeg;base64,{encoded_image}',
                                    },
                                },
                            ],
                        }
                    )
                    response = client(
                        system_messages=[system_message],
                        user_messages=user_messages,
                        override_user_messages_formatting=True
                    )
                    approval_status = self._extract_response(response, "<approve>", "</approve>").strip().lower()
                    approval_status = True if approval_status == 'true' else False
                    # If the LLM approves the generated visualization, we can stop here.
                    if approval_status:
                        if not self.silent:
                            print("The LLM approved the generated visualization.")
                        break
                    # If the LLM disapproves, we will try to fix the code.
                    else:
                        print("The LLM disapproved the generated visualization. It will attempt to improve it.")
                        self._handle_response(response)
                        if not self.llm_generated_code:
                            if not self.silent:
                                print("Could not extract beautified code from the LLM response after approval attempt.")
                            break
                        if i == self.max_fix_attempts - 1:
                            last_loop_provided_code = True

        else:
            with beautified_vis_widget:
                print("Could not extract beautified code from the LLM response.")
                print("\nRaw response:\n")
                print(response)

        # Special case: the LLM generated code in the last loop that was provided, but it was not executed
        # In theory, if this code runs, it should be the best code, so we will use it.
        # If it does not run, we will fall back to last_working_code.
        if last_loop_provided_code:
            beautified_figure, error_message, _ = self.execute(self.llm_generated_code)
            if error_message:
                beautified_figure, error_message, _ = self.execute(last_working_code)

        with beautified_vis_widget:
            if beautified_figure:
                # Display the beautified figure
                display(beautified_figure)
            else:
                print(f"No valid beautified figure was generated within the allowed {self.max_fix_attempts} attempts.")

        plt.close('all')  # Close all figures to avoid displaying them outside the widget.

        # Create and return the tab widget.
        tab = widgets.Tab()
        tab.children = [original_vis_widget, beautified_vis_widget]
        tab.set_title(0, 'Original Visualization')
        tab.set_title(1, 'Beautified Visualization')

        # Restore warnings to default behavior
        warnings.resetwarnings()

        # If some figures are still open, close them to avoid displaying them outside the widget.
        # This can happen if the LLM failed in all its attempts to generate a valid visualization.
        plt.close('all')

        return tab, last_working_code

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
