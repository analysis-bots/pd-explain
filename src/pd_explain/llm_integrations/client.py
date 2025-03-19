import os
import openai
import warnings


class Client:
    """
    A general client for interfacing with LLM services.
    Handles the API key, provider, model, and provider URL.
    """

    def __init__(self, api_key:str = None, provider: str = None, model: str = None, provider_url: str = None):
        if api_key is None:
            api_key = os.getenv("PD_EXPLAIN_LLM_KEY")
        if provider is None:
            provider = os.getenv("PD_EXPLAIN_LLM_PROVIDER")
        if model is None:
            model = os.getenv("PD_EXPLAIN_LLM_MODEL")

        self.api_key = api_key
        self._api_key_provided = True if self.api_key is not None else False
        if self.api_key == 'YOUR_API_KEY':
            warnings.warn("You have not set your API key for a LLM API provider. If you wish to use the LLM functions, please set the API key using the write_llm_api_key function. "
                          "All usage of LLM functions will not work until the API key is set.")
            self._api_key_provided = False
        self.provider = provider
        self.model = model
        self.provider_url = provider_url
        if provider_url is None:
            match provider:
                case "openai":
                    self.provider_url = "https://api.openai.com"
                case "together":
                    self.provider_url = "https://api.together.xyz/v1"
                case _:
                    raise ValueError("Unknown LLM service provider. We support 'openai' and 'together'. If you wish to use a different provider, please provide the base_url parameter.")
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.provider_url,
        )


    def __call__(self, system_messages: [str], user_messages: [str], *args, **kwargs) -> str | None:
        """
        Call the API with the given messages.
        :return: The response from the API. If no API key is provided, return None.
        """
        if not self._api_key_provided:
            return None
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_messages},
                {"role": "user", "content": user_messages},
            ],
        )
        return response.choices[0].message.content