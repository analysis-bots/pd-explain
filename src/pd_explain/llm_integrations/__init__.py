import os
import dotenv
import pd_explain.llm_integrations.consts as consts
from pd_explain.llm_integrations.client import Client

# Determine the directory of the current file
package_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the .env file within the package directory
env_path = os.path.join(package_dir, ".env")

# Check if a .env file exists right inside the package. If not, create one with a TOGETHER_API_KEY variable.
if not os.path.exists(env_path):
    with open(env_path, "w") as f:
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_REASONiNG_LLM_KEY}={consts.DEFAULT_KEY}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER}={consts.DEFAULT_REASONING_LLM_PROVIDER}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_MODEL}={consts.DEFAULT_REASONING_LLM_MODEL}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL}={consts.DEFAULT_REASONING_LLM_PROVIDER_URL}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_VISION_MODEL}={consts.DEFAULT_BEAUTIFICATION_LLM_VISION_MODEL}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_KEY}={consts.DEFAULT_KEY}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER}={consts.DEFAULT_BEAUTIFICATION_LLM_PROVIDER}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL}={consts.DEFAULT_BEAUTIFICATION_LLM_PROVIDER_URL}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_KEY}={consts.DEFAULT_KEY}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER}={consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_PROVIDER}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER_URL}={consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_PROVIDER_URL}\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_MODEL}={consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_MODEL}\n")

# Load the .env file
dotenv.load_dotenv(dotenv_path=env_path, override=False)

class LlmSetupMethods:
    """
    A collection of static methods to set up the LLM API key, provider, model, and provider URL.
    """

    @staticmethod
    def write_reasoning_llm_api_key(api_key: str):
        """
        Write the LLM API key to the .env file and the environment variables.
        The API key is used to authenticate with the LLM service.
        Can be obtained from the LLM service provider, such as OpenAI, Google or Together.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_REASONiNG_LLM_KEY] = api_key
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_REASONiNG_LLM_KEY, api_key)
        client.api_key = api_key

    @staticmethod
    def write_reasoning_llm_provider(provider: str):
        """
        Write the LLM provider to the .env file and the environment variables.
        We support 'openai', 'google' and 'together'.
        Default is 'google'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER] = provider
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER, provider)
        client._provider = provider

    @staticmethod
    def write_reasoning_llm_model(model: str):
        """
        Write the LLM model to the .env file and the environment variables.
        The model is the name of the LLM model to use for generating explanations.
        Default is 'gemini-2.5-flash'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_MODEL] = model
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_MODEL, model)
        client.model = model

    @staticmethod
    def write_reasoning_provider_url(provider_url: str):
        """
        Write the LLM provider URL to the .env file and the environment variables.
        The provider URL is the base URL for the LLM service provider.
        Default is 'https://generativelanguage.googleapis.com/v1beta/openai/'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL] = provider_url
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL, provider_url)
        client._provider_url = provider_url


    @staticmethod
    def write_beautification_model_api_key(api_key: str):
        """
        Write the LLM beautification API key to the .env file and the environment variables.
        The API key is used to authenticate with the LLM service for beautification.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_KEY] = api_key
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_KEY, api_key)
        client.beautification_api_key = api_key


    @staticmethod
    def write_beautification_llm_provider(provider: str):
        """
        Write the LLM beautification provider to the .env file and the environment variables.
        We support 'openai', 'google' and 'together'.
        Default is 'google'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER] = provider
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER, provider)
        client.beautification_provider = provider


    @staticmethod
    def write_beautification_model_provider_url(provider_url: str):
        """
        Write the LLM beautification provider URL to the .env file and the environment variables.
        The provider URL is the base URL for the LLM service provider for beautification.
        Default is 'https://generativelanguage.googleapis.com/v1beta/openai/'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL] = provider_url
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL, provider_url)
        client.beautification_provider_url = provider_url


    @staticmethod
    def write_beautification_vision_model(vision_model: str):
        """
        Write the LLM vision model to the .env file and the environment variables.
        The vision model is the name of the LLM model to use for generating explanations from images.
        Default is 'gemini-2.5-flash'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_VISION_MODEL] = vision_model
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_VISION_MODEL, vision_model)
        client.vision_model = vision_model

    @staticmethod
    def write_automated_exploration_llm_api_key(api_key: str):
        """
        Write the automated exploration LLM API key to the .env file and the environment variables.
        The API key is used to authenticate with the LLM service for automated exploration.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_KEY] = api_key
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_KEY, api_key)
        client.automated_exploration_api_key = api_key

    @staticmethod
    def write_automated_exploration_llm_provider(provider: str):
        """
        Write the automated exploration LLM provider to the .env file and the environment variables.
        We support 'openai' and 'together'.
        Default is 'together'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER] = provider
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER, provider)
        client.automated_exploration_provider = provider

    @staticmethod
    def write_automated_exploration_llm_provider_url(provider_url: str):
        """
        Write the automated exploration LLM provider URL to the .env file and the environment variables.
        The provider URL is the base URL for the LLM service provider for automated exploration.
        Default is 'https://api.together.xyz/v1/'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER_URL] = provider_url
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER_URL, provider_url)
        client.automated_exploration_provider_url = provider_url

    @staticmethod
    def write_automated_exploration_llm_model(model: str):
        """
        Write the automated exploration LLM model to the .env file and the environment variables.
        The model is the name of the LLM model to use for generating automated exploration.
        Default is 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_MODEL] = model
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_MODEL, model)
        client.automated_exploration_model = model


# If any of the variables are not set, set them to default values.
if consts.DOT_ENV_PD_EXPLAIN_REASONiNG_LLM_KEY not in os.environ:
    LlmSetupMethods.write_reasoning_llm_api_key(consts.DEFAULT_KEY)
if consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER not in os.environ:
    LlmSetupMethods.write_reasoning_llm_provider(consts.DEFAULT_REASONING_LLM_PROVIDER)
if consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_MODEL not in os.environ:
    LlmSetupMethods.write_reasoning_llm_model(consts.DEFAULT_REASONING_LLM_MODEL)
if consts.DOT_ENV_PD_EXPLAIN_REASONING_LLM_PROVIDER_URL not in os.environ:
    LlmSetupMethods.write_reasoning_provider_url(consts.DEFAULT_REASONING_LLM_PROVIDER_URL)
if consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_KEY not in os.environ:
    LlmSetupMethods.write_beautification_model_api_key(consts.DEFAULT_KEY)
if consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER not in os.environ:
    LlmSetupMethods.write_beautification_llm_provider(consts.DEFAULT_BEAUTIFICATION_LLM_PROVIDER)
if consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_VISION_MODEL not in os.environ:
    LlmSetupMethods.write_beautification_vision_model(consts.DEFAULT_BEAUTIFICATION_LLM_VISION_MODEL)
if consts.DOT_ENV_PD_EXPLAIN_BEAUTIFICATION_LLM_PROVIDER_URL not in os.environ:
    LlmSetupMethods.write_beautification_model_provider_url(consts.DEFAULT_BEAUTIFICATION_LLM_PROVIDER_URL)
if consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_KEY not in os.environ:
    LlmSetupMethods.write_automated_exploration_llm_api_key(consts.DEFAULT_KEY)
if consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER not in os.environ:
    LlmSetupMethods.write_automated_exploration_llm_provider(consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_PROVIDER)
if consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_PROVIDER_URL not in os.environ:
    LlmSetupMethods.write_automated_exploration_llm_provider_url(consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_PROVIDER_URL)
if consts.DOT_ENV_PD_EXPLAIN_AUTOMATED_EXPLORATION_LLM_MODEL not in os.environ:
    LlmSetupMethods.write_automated_exploration_llm_model(consts.DEFAULT_AUTOMATED_EXPLORATION_LLM_MODEL)


from pd_explain.llm_integrations.explanation_reasoning import ExplanationReasoning