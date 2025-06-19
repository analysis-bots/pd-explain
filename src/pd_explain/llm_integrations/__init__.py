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
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_KEY}=YOUR_API_KEY\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER}=google\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL}=gemini-2.5-flash\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER_URL}=https://generativelanguage.googleapis.com/v1beta/openai/\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_VISION_MODEL}=gemini-2.5-flash\n")

# Load the .env file
dotenv.load_dotenv(dotenv_path=env_path, override=False)

client = Client()

class LlmSetupMethods:
    """
    A collection of static methods to set up the LLM API key, provider, model, and provider URL.
    """

    @staticmethod
    def write_llm_api_key(api_key: str):
        """
        Write the LLM API key to the .env file and the environment variables.
        The API key is used to authenticate with the LLM service.
        Can be obtained from the LLM service provider, such as OpenAI or Together.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_KEY] = api_key
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LLM_KEY, api_key)
        client.api_key = api_key

    @staticmethod
    def write_llm_provider(provider: str):
        """
        Write the LLM provider to the .env file and the environment variables.
        We support 'openai' and 'together'.
        Default is 'together'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER] = provider
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER, provider)
        client._provider = provider

    @staticmethod
    def write_llm_model(model: str):
        """
        Write the LLM model to the .env file and the environment variables.
        The model is the name of the LLM model to use for generating explanations.
        Default is 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL] = model
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL, model)
        client.model = model

    @staticmethod
    def write_provider_url(provider_url: str):
        """
        Write the LLM provider URL to the .env file and the environment variables.
        The provider URL is the base URL for the LLM service provider.
        Default is 'https://api.together.xyz/v1'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER_URL] = provider_url
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER_URL, provider_url)
        client._provider_url = provider_url


    @staticmethod
    def write_vision_model(vision_model: str):
        """
        Write the LLM vision model to the .env file and the environment variables.
        The vision model is the name of the LLM model to use for generating explanations from images.
        Default is 'gemini-2.5-flash'.
        """
        os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_VISION_MODEL] = vision_model
        dotenv.set_key(env_path, consts.DOT_ENV_PD_EXPLAIN_LLM_VISION_MODEL, vision_model)
        client.vision_model = vision_model


# If any of the variables are not set, set them to default values.
if consts.DOT_ENV_PD_EXPLAIN_LLM_KEY not in os.environ:
    LlmSetupMethods.write_llm_api_key("YOUR_API_KEY")
if consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER not in os.environ:
    LlmSetupMethods.write_llm_provider("google")
if consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL not in os.environ:
    LlmSetupMethods.write_llm_model("gemini-2.5-flash")
if consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER_URL not in os.environ:
    LlmSetupMethods.write_provider_url("https://generativelanguage.googleapis.com/v1beta/openai/")
if consts.DOT_ENV_PD_EXPLAIN_LLM_VISION_MODEL not in os.environ:
    LlmSetupMethods.write_vision_model("gemini-2.5-flash")


from pd_explain.llm_integrations.explanation_reasoning import ExplanationReasoning