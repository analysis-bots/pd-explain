import pandas as pd
import os

from pd_explain.llm_integrations.llm_integration_interface import LLMIntegrationInterface
from pd_explain.llm_integrations.client import Client
from pd_explain.llm_integrations import consts

class VisualizationBeautifier(LLMIntegrationInterface):
    """
    A class to create case specific visualizations using LLMs.
    The LLM is provided how the current visualization looks like and its code, and is
    asked to create a new visualization that is more appealing and informative.
    """

    def do_llm_action(self) -> pd.Series | None | str:
        client = Client(
            api_key=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_KEY, None),
            provider=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER, "google"),
            model=os.getenv(consts.DEFAULT_BEAUTIFICATION_LLM_VISION_MODEL, "gemini-2.5-flash"),
            provider_url=os.getenv(consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL, "https://generativelanguage.googleapis.com/v1beta/openai/")
        )
        pass
