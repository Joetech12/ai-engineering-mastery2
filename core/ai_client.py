
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_ai_client():
    """
    Initialize an AI client based on AI_PROVIDER.
    """

    provider = os.getenv("AI_PROVIDER", "gemini").lower()

    if provider == "gemini":

        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL")

        base_url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        )

    elif provider == "groq":

        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL")

        base_url = "https://api.groq.com/openai/v1"

    elif provider == "openai":

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-5.5")

        base_url = None

    else:
        raise ValueError(
            f"Unsupported AI provider: {provider}"
        )

    if not api_key:
        raise ValueError(
            f"Missing API key for provider: {provider}"
        )

    if not model:
        raise ValueError(
            f"Missing model configuration for provider: {provider}"
        )

    client_options = {
        "api_key": api_key,
        "timeout": 30.0,
        "max_retries": 2
    }

    if base_url:
        client_options["base_url"] = base_url

    client = OpenAI(**client_options)

    return client, model, provider
