import os
import dotenv
import pd_explain.llm_integrations.consts as consts

# Check if a .env file exists right inside the package. If not, create one with a TOGETHER_API_KEY variable.
if not os.path.exists(".env"):
    with open(".env", "w") as f:
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_KEY}=YOUR_API_KEY\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER}=together\n")
        f.write(f"{consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL}=deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free\n")

# Load the .env file
dotenv_path = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_path=dotenv_path, override=False)

def write_llm_api_key(api_key: str):
    """
    Write the LLM API key to the .env file and the environment variables.
    The API key is used to authenticate with the LLM service.
    Can be obtained from the LLM service provider, such as OpenAI or Together.
    """
    os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_KEY] = api_key
    dotenv.set_key(dotenv_path, consts.DOT_ENV_PD_EXPLAIN_LLM_KEY, api_key)

def write_llm_provider(provider: str):
    """
    Write the LLM provider to the .env file and the environment variables.
    We support 'openai' and 'together'.
    Default is 'together'.
    """
    os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER] = provider
    dotenv.set_key(dotenv_path, consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER, provider)

def write_llm_model(model: str):
    """
    Write the LLM model to the .env file and the environment variables.
    The model is the name of the LLM model to use for generating explanations.
    Default is 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'.
    """
    os.environ[consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL] = model
    dotenv.set_key(dotenv_path, consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL, model)


# If any of the variables are not set, set them to default values.
if consts.DOT_ENV_PD_EXPLAIN_LLM_KEY not in os.environ:
    write_llm_api_key("YOUR_API_KEY")
if consts.DOT_ENV_PD_EXPLAIN_LLM_PROVIDER not in os.environ:
    write_llm_provider("together")
if consts.DOT_ENV_PD_EXPLAIN_LLM_MODEL not in os.environ:
    write_llm_model("deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free")


from pd_explain.llm_integrations.explanation_reasoning import ExplanationReasoning