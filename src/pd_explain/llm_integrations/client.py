import os
import openai
import together
import warnings
from typing import List
from singleton_decorator import singleton


@singleton
class Client:
    """
    A general client for interfacing with LLM services.
    Handles the API key, provider, model, and provider URL.
    """

    def __init__(self, api_key: str = None, provider: str = None, model: str = None, provider_url: str = None):
        if api_key is None:
            api_key = os.getenv("PD_EXPLAIN_LLM_KEY")
        if provider is None:
            provider = os.getenv("PD_EXPLAIN_LLM_PROVIDER")
        if model is None:
            model = os.getenv("PD_EXPLAIN_LLM_MODEL")

        self.api_key = api_key
        self._provider = provider
        self.model = model
        self._provider_url = provider_url
        if provider_url is None:
            match provider:
                case "openai":
                    self._provider_url = "https://api.openai.com"
                case "together":
                    self._provider_url = "https://api.together.xyz/v1"
                case _:
                    raise ValueError(
                        "Unknown LLM service provider. We support 'openai' and 'together'. If you wish to use a different provider, please provide the base_url parameter.")
        if self._provider == "openai" or "openai.com" in self._provider_url:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self._provider_url,
            )
        else:
            self.client = together.Together(
                api_key=self.api_key,
                base_url=self._provider_url,
            )

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, value):
        self._provider = value
        self._set_client()

    @property
    def provider_url(self):
        return self._provider_url

    @provider_url.setter
    def provider_url(self, value):
        self._provider_url = value
        self._set_client()

    def _set_client(self):
        if self._provider != "together" or "together.xyz" not in self._provider_url:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self._provider_url,
            )
        else:
            self.client = together.Together(
                api_key=self.api_key
            )

    def __call__(self, system_messages: List[str], user_messages: List[str],
                 assistant_messages: List[str] = None, *args, **kwargs) -> str | None:
        """
        Call the API with the given messages.
        :return: The response from the API. If no API key is provided, return None.
        """
        if not self.api_key or self.api_key == 'YOUR_API_KEY':
            warnings.warn(
                "You have not set your API key for a LLM API provider. If you wish to use the LLM functions, please set the API key using the write_llm_api_key function. "
                "All usage of LLM functions will not work until the API key is set.")
            return None
        messages = [{"role": "user", "content": message} for message in user_messages]
        # if assistant messages are not None, i.e. the user is continuing a conversation, interleave the user
        # messages and assistant messages.
        if assistant_messages is not None:
            for i in range(1, len(assistant_messages) + 1):
                messages.insert(i * 2 - 1, {"role": "assistant", "content": assistant_messages[i - 1]})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                *[{"role": "system", "content": message} for message in system_messages],
                *messages,
            ],
        )
        return response.choices[0].message.content
