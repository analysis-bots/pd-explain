import os
import openai
import together
import warnings
from typing import List
from pd_explain.llm_integrations import consts


class Client:
    """
    A general client for interfacing with LLM services.
    Handles the API key, provider, model, and provider URL.
    """

    def __init__(self, api_key: str, provider: str, model: str,
                 provider_url: str):

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
                case "google" | "gemini":
                    self._provider_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                case _:
                    self._provider_url = os.getenv(consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL)
        if (self._provider == "openai" or "openai.com" in self._provider_url)\
                or (self._provider in ["google", "gemini"] or "generativelanguage.googleapis.com" in self._provider_url):
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

    def __call__(self, system_messages: List[str], user_messages: List[str] | list[dict],
                 assistant_messages: List[str] = None, override_user_messages_formatting: bool = False,
                 *args, **kwargs) -> str | None:
        """
        Call the API with the given messages.
        :param system_messages: List of system messages to set the context for the conversation.
        :param user_messages: List of user messages to send to the API. If override_user_messages_formatting is True,
            the user messages can be in any format, otherwise they should be strings.
        :param assistant_messages: List of assistant messages to interleave with user messages if the user is continuing a conversation.
        :param override_user_messages_formatting: If True, user messages can be in any format, otherwise they should be strings. Useful for
        using vision models where the user messages can contain images or other non-text content.
        :return: The response from the API. If no API key is provided, return None.
        """
        if not self.api_key or self.api_key == 'YOUR_API_KEY':
            warnings.warn(
                "You have not set your API key for a LLM API provider. If you wish to use the LLM functions, please set the API key using the write_llm_api_key function. "
                "All usage of LLM functions will not work until the API key is set.")
            return None
        if not override_user_messages_formatting:
            messages = [{"role": "user", "content": message} for message in user_messages]
        else:
            messages = user_messages
        # if assistant messages are not None, i.e. the user is continuing a conversation, interleave the user
        # messages and assistant messages.
        if assistant_messages is not None and len(assistant_messages) > 0:
            for i in range(1, len(assistant_messages) + 1):
                messages.insert(i * 2 - 1, {"role": "assistant", "content": str(assistant_messages[i - 1])})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                *[{"role": "system", "content": message} for message in system_messages],
                *messages,
            ],
        )
        return response.choices[0].message.content if response.choices else None


