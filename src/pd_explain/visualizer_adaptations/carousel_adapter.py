from IPython.display import display, clear_output
import ipywidgets as widgets
from typing import List, Any

from pd_explain.visualizer_adaptations.adapter_interface import AdapterInterface

class CarouselAdapter(AdapterInterface):
    """
    CarouselAdapter is an adapter for visualizing multiple outputs in a carousel format.
    """
    def __init__(self):
        self._captured_outputs: List[widgets.Output] = []
        self._carousel_ui: widgets.VBox | None = None
        self._num_plots: int = 0
        self._plot_index: widgets.IntSlider | None = None
        self._plot_display_output: widgets.Output | None = None

    def __enter__(self) -> 'CarouselAdapter':
        """
        Prepares the adapter to capture outputs.
        """
        # Create a new Output widget for capturing
        # This will be the context manager for clear_output/display
        self._current_output_widget = widgets.Output()
        self._captured_outputs = []
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Finalizes the adapter, assembling and displaying the carousel.
        """

        self._num_plots = len(self._captured_outputs)
        if self._num_plots == 0:
            display(widgets.HTML("No plots captured by CarouselAdapter."))
            return

        # Initialize core widgets for the carousel
        self._plot_index = widgets.IntSlider(
            value=0,
            min=0,
            max=self._num_plots - 1,
            step=1,
            description='Plot Index:',
            continuous_update=False
        )
        self._plot_display_output = widgets.Output() # This will display the selected plot

        self._plot_display_output = widgets.Output(
            layout=widgets.Layout(
                display='flex',          # Enable flex container
                justify_content='center', # Center horizontally
                align_items='center',     # Center vertically
                min_height='450px',       # Give it some minimum height to see centering
                border='1px solid #ddd',  # Optional: visualize the output area
                margin='10px 0',          # Optional: space it out,
            )
        )

        prev_button = widgets.Button(description="Previous")
        next_button = widgets.Button(description="Next")

        # Link button actions to slider
        prev_button.on_click(self._on_prev_button_clicked)
        next_button.on_click(self._on_next_button_clicked)

        # Link slider change to update the displayed plot
        self._plot_index.observe(self._update_plot_display, names='value')

        # Initial display of the first plot
        self._update_plot_display()

        # Assemble the UI
        if self._num_plots > 1:
            navigation_buttons = widgets.HBox([prev_button, next_button])
            self._carousel_ui = widgets.VBox([navigation_buttons, self._plot_index, self._plot_display_output])
        else:
            # If there's only one plot, no need for navigation buttons
            self._carousel_ui = widgets.VBox([self._plot_display_output])

        display(self._carousel_ui)

    def capture_output(self, output: Any) -> None:
        """
        Captures the output from the visualization method.
        This method is called by the visualizer when it produces a plot.
        """
        if self._current_output_widget is None:
            raise RuntimeError("capture_output called outside of CarouselAdapter context.")

        with self._current_output_widget:
            clear_output(wait=True) # Clear any previous content in this output widget
            display(output) # Display the plot into the current Output widget

        self._captured_outputs.append(self._current_output_widget)

        # After capturing, prepare a new Output widget for the *next* capture
        # This is key to capturing multiple separate plots
        self._current_output_widget = widgets.Output()


    # Internal methods for carousel functionality
    # The unused parameters are kept for compatibility
    def _update_plot_display(self, change=None):
        if self._plot_display_output and self._plot_index:
            with self._plot_display_output:
                clear_output(wait=True)
                # The actual content is already in the captured Output widget,
                # we just need to display that widget's content
                display(self._captured_outputs[self._plot_index.value])

    def _on_prev_button_clicked(self, b):
        if self._plot_index and self._plot_index.value > 0:
            self._plot_index.value -= 1
        # If we are at the first plot and click previous, wrap around to the last plot
        elif self._plot_index and self._plot_index.value == 0:
            self._plot_index.value = self._num_plots - 1

    def _on_next_button_clicked(self, b):
        if self._plot_index and self._plot_index.value < self._num_plots - 1:
            self._plot_index.value += 1
        # If we are at the last plot and click next, wrap around to the first plot
        elif self._plot_index and self._plot_index.value == self._num_plots - 1:
            self._plot_index.value = 0




