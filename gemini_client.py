import os
import time
from google import genai

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiQuotaError(RuntimeError):
    pass


class GeminiTransientError(RuntimeError):
    pass


def get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")


def get_model():
    return os.getenv("GEMINI_MODEL") or DEFAULT_MODEL


def get_fallback_model():
    return os.getenv("GEMINI_FALLBACK_MODEL")


def get_client():
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _is_quota_error(message):
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower()


def _is_transient_error(message):
    return "503" in message or "UNAVAILABLE" in message or "high demand" in message.lower()


def _generate_once(client, prompt, model):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    if not response or not response.text:
        return ""
    return response.text.strip()


def generate_text(prompt, model=None, retries=3):
    client = get_client()
    if client is None:
        raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY or GEMINI_KEY in .env")

    models_to_try = [model or get_model()]
    fallback_model = get_fallback_model()
    if fallback_model and fallback_model not in models_to_try:
        models_to_try.append(fallback_model)

    last_error = None
    for model_name in models_to_try:
        for attempt in range(retries + 1):
            try:
                return _generate_once(client, prompt, model_name)
            except Exception as e:
                message = str(e)
                if _is_quota_error(message):
                    raise GeminiQuotaError("Gemini quota exhausted. Try a key/project with available quota.") from e
                if not _is_transient_error(message):
                    raise

                last_error = e
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))

    raise GeminiTransientError("Gemini is temporarily unavailable or overloaded. Please retry shortly.") from last_error
