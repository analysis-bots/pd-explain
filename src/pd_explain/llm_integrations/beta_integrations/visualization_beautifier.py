"""
This feature is still in Beta, and is not yet ready for production use.
While it works mostly fine for fedex, it is not yet ready for other use cases.
"""

import base64
import io
import json
import os
from typing import Optional, Any, Dict, Tuple
import traceback

import ipywidgets as widgets
import pandas as pd
from IPython.display import display, Image
from ipywidgets import Output
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import warnings

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
                 visualization_params: Optional[Dict[str, Any]] = None,
                 requester_name: Optional[str] = None,
                 visualization_code: Optional[str] = None,
                 max_fix_attempts: int = 10,
                 must_generalize: bool = False,
                 give_llm_source_code: bool = True,
                 show_llm_original_image: bool = True,
                 visualization_description: Optional[list[str]] = None,
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
        if not show_llm_original_image and visualization_description is None:
            raise ValueError("If show_llm_original_image is False, visualization_description must be provided.")
        if give_llm_source_code:
            if requester_name:
                self.visualization_code: str = code_extractor.get_visualization_code_from_source(requester_name)
            elif visualization_code:
                self.visualization_code: str = visualization_code
            else:
                raise ValueError("Either requester_name or visualization_code must be provided.")
        else:
            self.visualization_code: str = ""

        self.visualization_object: Any = visualization_object
        self.data: pd.DataFrame = data
        self.llm_generated_code: Optional[str] = None
        self.visualization_params: Dict[str, Any] = visualization_params if visualization_params else {}
        # The maximum number of attempts to fix the code if it fails to execute.
        self.max_fix_attempts: int = max_fix_attempts
        self.must_generalize: bool = must_generalize
        self.silent: bool = silent
        self.requester_name: Optional[str] = requester_name
        self.give_llm_source_code: bool = give_llm_source_code
        self.show_llm_original_image: bool = show_llm_original_image
        self.visualization_description: Optional[list] = visualization_description

    def _define_task(self) -> str:
        """
        Defines the task for the LLM.
        """
        task_str = ""
        task_str += (
            "You are a data visualization expert using Python's matplotlib and pandas libraries.\n"
            "Your task is to accepts either:"
            "1.  Python code that generates a visualization, as well as an image of the visualization itself.\n"
            "2.  Only the image of a visualization, without the code the generates it.\n"
            "3.  A textual description of what to visualize, without any code or image.\n\n"
            "Using this input, you must create new Python code that generates a new visualization that is more appealing, "
            "consolidated, and aesthetically pleasing than the original, or a visualization that matches the description, "
            "using an iterative process of improvement.\n\n"
            "**Key Goals for Improvement:**\n"
            "1.  **Consolidation & Clarity:** The original visualization might have multiple, cluttered subplots. If possible, consolidate them into a single, well-organized figure. Use shared axes where appropriate. The goal is to reduce visual clutter and make comparisons easier.\n"
            "2.  **Aesthetics:** Use a professional color palette (e.g. from matplotlib). Ensure font sizes are legible and titles/labels are clear.\n"
            "3.  **Information Preservation:** The new visualization must preserve all the crucial information from the original, such as which groups are outliers and the values they represent. Additionally, if the visualization had added text to it, such as LLM generated text, this text MUST be in the beautified visualization too.\n"
            "4.  **No Visual Overload:**: Make sure the visualization is not overloaded with information. If there are too many data points or categories, consider aggregating or pruning them to focus on the most important aspects.\n"
            "5.  **Reasonable Size:** The visualization should be of a reasonable size, not too large or too small. It should fit well within the context it will be displayed in, and not stretched out or squished beyond what it should be for clear viewing.\n"
        )
        if self.give_llm_source_code:
            task_str += (
                "6.  **Limited Implementation Information:** We can not provide you with the full implementation code of the visualization. For all objects in the code, unless they are from a known library,"
                " you must assume that they are defined in the code, and you can only use functions and properties of those objects that you explicitly see in the code.\n"
            )
        if self.requester_name.lower().startswith("fedex"):
            task_str += (
                "\n The visualizations you are improving are meant to point out the most interesting statistical "
                "changes in the data after it has been queried.\n"
            )
        if self.requester_name.lower().startswith("metainsight"):
            task_str += (
                "\n The visualizations you are created are meant to point out the most interesting common patterns found "
                "in the data.\n"
                "You will only be provided with the __init__ function of our custom objects, you must come up with the visualization code yourself.\n"
            )
        if not self.must_generalize:
            task_str += (
                "The code you create will be for one-time use and executed immediately. The user will only see the visualization, not the code. "
                "Therefore, you can make the code completely specific to this one visualization, without needing to generalize it for future use, "
                "and you can even hardcode specific values or data into the code.\n\n"
            )
        else:
            task_str += (
                "The code you create will not be for one-time use, but rather may be used many times with different data. "
                "As such, you must ensure that the code is general, relying solely on input parameters to the function you create, "
                "and not hardcoding any specific values or data. "
                "You may also be provided visualization code that is more general and for more cases than just this one visualization "
                "you see, so you must ensure that the code you create is compatible with that code.\n\n"
            )
            if self.requester_name.lower().startswith("fedex"):
                task_str += "The code you create needs to handle visualization cases for the insights found via filter, groupby and join queries."
            if self.requester_name.lower().startswith("metainsight"):
                task_str += "The code you create needs to handle visualization cases for the patterns: trends, outliers and unimodalities."
        task_str += "\nYou must not cause any permanent changes, such as changing the default style of matplotlib, or changing the default settings of pandas. "
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
        """
        Handles the response from the LLM, extracting the generated code.
        """
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
        # Case 3: The visualization object may be an ipywidget (e.g., a plotly widget or other interactive widget).
        # or something else unexpected.
        else:
            encoded_image = None

        return encoded_image

    def beautify_from_code(self, code):
        """
        Uses the provided code, saved previously, to beautify the visualization.
        """
        beautified_figure, _, _ = self._execute(code)
        beautified_output = widgets.Output()
        with beautified_output:
            if beautified_figure:
                display(beautified_figure)
            else:
                print("No valid beautified figure was generated.")

        original_output = widgets.Output()
        with original_output:
            if isinstance(self.visualization_object, str) and os.path.exists(self.visualization_object):
                display(Image(filename=self.visualization_object))
            else:
                display(self.visualization_object)

        # Create and return the tab widget.
        tab = widgets.Tab()
        tab.children = [original_output, beautified_output]
        tab.set_title(0, 'Original Visualization')
        tab.set_title(1, 'Beautified Visualization')
        plt.close('all')  # Close all figures to avoid displaying them outside the widget.

        return tab

    def _execute(self, code) -> Tuple[Optional[Figure], Optional[str], Optional[str]]:
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

                if beautified_figure is None or not isinstance(beautified_figure, Figure):
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
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER,
                               consts.DEFAULT_BEAUTIFICATION_LLM_PROVIDER),
            model=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_VISION_MODEL,
                            consts.DEFAULT_BEAUTIFICATION_LLM_VISION_MODEL),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL,
                                   consts.DEFAULT_BEAUTIFICATION_LLM_PROVIDER_URL)
        )

        original_encoded_image: Optional[str] = self._encode_visualization(self.visualization_object)

        system_message: str = self._define_task()
        # Create a data summary
        data_summary = io.StringIO()
        self.data.info(buf=data_summary)
        data_summary_str = data_summary.getvalue()
        code_score_dict = {}

        user_message = ""

        if self.give_llm_source_code:
            user_message += (
                f"Here is part of the original code that produced the visualization:\n\n"
                f"<python>\n{self.visualization_code}\n</python>\n\n"
            )
        else:
            user_message += (
                "You will not be provided with the original code that produced the visualization."
            )

        user_message += (
            f"For context, here is a summary of the pandas DataFrame that will be passed to your function:\n"
            f"```\n{data_summary_str}\n```\n"
            f"And here is the head of the data:\n"
            f"```\n{self.data.head().to_string()}\n```\n\n"
            f"{self._describe_output_format()}\n"
            f"Remember again to place the code inside <python> and </python> tags, or the program will not be able to extract it.\n"
        )

        if self.show_llm_original_image:
            user_message += f"Attached as well is the visualization itself. Please improve it by making it clearer and more consolidated.\n\n"
        else:
            user_message += (
                "The original visualization is not provided as an image. "
                f"Instead, create a new visualization based on the following descriptions of the insights you need to visualize:"
                f"\n{self.visualization_description}\n\n"
            )

        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f'data:image/jpeg;base64,{original_encoded_image}',
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

        user_messages.append({"role": "assistant", "content": response})

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
        best_visualization_tab = None
        highest_score = 0.0
        highest_scoring_code = None
        beautified_figure = None
        # Suppress warnings, because the LLM code may raise warnings galore.
        warnings.filterwarnings("ignore")
        if self.llm_generated_code:
            for i in range(self.max_fix_attempts):
                plt.close('all')  # Close any previous plots
                beautified_figure, error_message, printed_error = self._execute(self.llm_generated_code)

                # If there was any error, try to fix it
                if error_message or printed_error:
                    if not self.silent:
                        print(f"Error encountered in LLM generated code - {printed_error}")
                        print(f"Attempting to fix the code... ({i + 1}/{self.max_fix_attempts})")

                    user_messages.append({"role": "user",
                                          "content": f"This is iteration {i + 1} / {self.max_fix_attempts} of the iterative improvement process.\n"
                                                     f"{error_message}\nPlease fix the code and provide the full, corrected code block."})

                    response = client(
                        system_messages=[system_message],
                        user_messages=user_messages,
                        override_user_messages_formatting=True
                    )
                    user_messages.append({"role": "assistant", "content": response})
                    self._handle_response(response)
                    if not self.llm_generated_code:
                        break  # Exit the loop if we can't get new code
                # If there was no error, we instead prompt the LLM for two things:
                # 1. To approve or disapprove the generated visualization.
                # 2. If it disapproves, to provide a new code that fixes the issues.
                else:
                    last_working_code = self.llm_generated_code[:]
                    if not self.silent:
                        print("The generated code executed successfully.")
                        print(f"Approving or improving the generated visualization... {i + 1}/{self.max_fix_attempts}")
                    encoded_image = self._encode_visualization(beautified_figure)
                    user_message_text = (
                        f"This is iteration {i + 1} / {self.max_fix_attempts} of the iterative improvement process.\n"
                        f"You will receive images, in this order: the original visualization (as a reminder) and the generated visualization "
                        f"after running the code you generated.\n"
                        f"You have three tasks, in this order:\n"
                        f"1. Describe in detail what you see in the visualization, including any issues or areas for improvement. "
                        f"If there are inconsistencies between the original visualization and the generated one or important information that is missing, "
                        f"very clearly point them out in this description.\n"
                        f"This should be in between <description> and </description> tags.\n"
                        f"2. Rate the visualization on a scale from 0 to 10, where 10 is perfect and 0 is completely unusable. "
                        f"A score of 9.5 or higher will immediately approve the visualization, while a score below that will require you to improve it. "
                        f"This should be in between <score> and </score> tags.\n"
                        f"A score of 9.0 should mean the the visualization is good (but not completely perfect), any further improvements to it are minor, and "
                        f"are a waste of tokens and paying for LLM API calls.\n"
                        f"The score should be based on figure readability, aesthetics, and most importantly how well it preserves and conveys the information from the original visualization.\n"
                        f"3. If you disapprove of the visualization, provide a new code that fixes the issues you found. "
                        f"This code should be in between <python> and </python> tags, and it should be a complete code block that can be executed on its own.\n"
                    )
                    user_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_message_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f'data:image/jpeg;base64,{original_encoded_image}',
                                    },
                                },
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
                    user_messages.append({"role": "assistant", "content": response})
                    score = self._extract_response(response, "<score>", "</score>")
                    if isinstance(score, str):
                        score = score.strip().lower()
                    else:
                        score = "0.0"  # Default to 0.0 if we can't extract a score
                    score = float(score)
                    code_score_dict[self.llm_generated_code] = score
                    # While the instruction is that 10 is approval, we will consider a more lenient threshold of 9.5
                    approved = score >= 9.0
                    # If the LLM approves the generated visualization, we can stop here.
                    if approved:
                        if not self.silent:
                            print(f"The LLM approved the generated visualization, giving it a score of {score} / 10")
                        highest_score = score
                        # Copy the code. Not doing this seems to result in highest_scoring_code getting a pointer to the
                        # self.llm_generated_code, which is not what we want as this will change in the next iteration.
                        highest_scoring_code = self.llm_generated_code[:]
                        best_visualization_tab = Output()
                        with best_visualization_tab:
                            display(beautified_figure)
                        break
                    # If the LLM disapproves, we will try to fix the code.
                    else:
                        if not self.silent:
                            print(
                                f"The LLM disapproved the generated visualization and scored it {score} / 10. It will attempt to improve it.")
                        self._handle_response(response)
                        if not self.llm_generated_code:
                            if not self.silent:
                                print("Could not extract beautified code from the LLM response after approval attempt.")
                            break
                        # We assume that even if the score is the same, newer code is better, hence the greater-equal comparison.
                        if score >= highest_score:
                            best_visualization_tab = Output()
                            with best_visualization_tab:
                                display(beautified_figure)
                            highest_score = score
                            highest_scoring_code = self.llm_generated_code[:]

        else:
            with beautified_vis_widget:
                print("Could not extract beautified code from the LLM response.")
                print("\nRaw response:\n")
                print(response)

        # Check best_visualization_tab to see if we have a tab with the best visualization.
        if best_visualization_tab:
            beautified_vis_widget = best_visualization_tab

        else:
            # We use the highest scoring code to execute the final beautified visualization.
            if highest_scoring_code is not None:
                beautified_figure, error_message, _ = self._execute(highest_scoring_code)
            # First condition is not really needed, but it is here for clarity.
            elif highest_scoring_code is None and last_working_code is not None:
                beautified_figure, error_message, _ = self._execute(last_working_code)
            # If for some reason both didn't work, we check the code_score dictionary for the highest scoring code.
            elif code_score_dict:
                # sort the dictionary by score, and iteratively try to execute the code with the highest score.
                sorted_codes = sorted(code_score_dict.items(), key=lambda item: item[1], reverse=True)
                for code, score in sorted_codes:
                    beautified_figure, error_message, _ = self._execute(code)
                    if beautified_figure:
                        break
            else:
                beautified_figure = None
            # If we have a beautified figure, display it.
            with beautified_vis_widget:
                if beautified_figure is not None:
                    display(beautified_figure)
                else:
                    print(
                        f"No valid beautified figure was generated within the allowed {self.max_fix_attempts} attempts.")

        plt.close('all')  # Close all figures to avoid displaying them outside the widget.

        # Create and return the tab widget.
        tab = widgets.Tab()
        tab.children = [original_vis_widget, beautified_vis_widget]
        tab.set_title(0, 'Original Visualization')
        tab.set_title(1, 'Beautified Visualization')

        # Restore warnings to default behavior
        warnings.filterwarnings("default")

        # If some figures are still open, close them to avoid displaying them outside the widget.
        # This can happen if the LLM failed in all its attempts to generate a valid visualization.
        plt.close('all')

        return tab, highest_scoring_code if highest_scoring_code else last_working_code

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
