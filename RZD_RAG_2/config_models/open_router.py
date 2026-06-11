import os

from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OR_TOKEN")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

OPENROUTER_HEADERS = {
    # Необязательно, но OpenRouter рекомендует для идентификации приложения
    "HTTP-Referer": "http://localhost",
    "X-OpenRouter-Title": "TrainBot RAG Agent",
}


def make_openrouter_llm(
    model: str,
    context_window: int,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    is_function_calling_model: bool = False,
) -> OpenAILike:
    """
    Создаёт LlamaIndex OpenAILike LLM для OpenRouter.

    Args:
        model: slug модели OpenRouter, например "openai/gpt-oss-120b:free".
        context_window: контекстное окно модели.
        max_tokens: максимальное количество генерируемых токенов.
        temperature: температура генерации.
        is_function_calling_model: True, если модель используется с OpenAI-style tools/function calling.

    Returns:
        OpenAILike LLM.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Не задана переменная окружения OPENROUTER_API_KEY")

    return OpenAILike(
        model=model,
        api_base=OPENROUTER_API_BASE,
        api_key=OPENROUTER_API_KEY,
        context_window=context_window,
        max_tokens=max_tokens,
        temperature=temperature,
        is_chat_model=True,
        is_function_calling_model=is_function_calling_model,
        default_headers=OPENROUTER_HEADERS,
        timeout=90.0,
        max_retries=2,
    )

name_answer_llm = "openai/gpt-oss-20b:free"
name_selector_llm = "nvidia/nemotron-nano-9b-v2:free"
# name_normalizer_llm = "nvidia/nemotron-nano-9b-v2:free"
name_normalizer_llm = "openai/gpt-oss-20b:free"
name_evaluator_llm = "openai/gpt-oss-20b:free"
# name_evaluator_llm = "nvidia/nemotron-nano-9b-v2:free" # Для отладки

answer_llm = make_openrouter_llm(
    model=name_answer_llm,
    context_window=131_072,
    max_tokens=2048,
    temperature=0.1,
    is_function_calling_model=True,
)

selector_llm = make_openrouter_llm(
    model=name_selector_llm,
    context_window=131_072,
    max_tokens=512,
    temperature=0.0,
    is_function_calling_model=False,
)

normalizer_llm = make_openrouter_llm(
    model=name_normalizer_llm,
    context_window=131_072,
    max_tokens=250,
    temperature=0.0,
    is_function_calling_model=False,
)

evaluator_llm = make_openrouter_llm(
    model=name_evaluator_llm,
    context_window=131_072,
    max_tokens=1024,
    temperature=0.0,
    is_function_calling_model=False,
)