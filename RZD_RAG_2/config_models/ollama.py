"""Настройка LLM-моделей LlamaIndex для работы с локальной Ollama."""
from llama_index.llms.openai_like import OpenAILike


OLLAMA_API_BASE = "http://192.168.3.154:11434/v1"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_API_KEY = "ollama"


normalizer_llm = OpenAILike(
    model=OLLAMA_MODEL,
    api_base=OLLAMA_API_BASE,
    api_key=OLLAMA_API_KEY,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.0,
    max_tokens=250,
)

selector_llm = OpenAILike(
    model=OLLAMA_MODEL,
    api_base=OLLAMA_API_BASE,
    api_key=OLLAMA_API_KEY,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.0,
    max_tokens=256,
    timeout=90.0,
    max_retries=2,
)

answer_llm = OpenAILike(
    model=OLLAMA_MODEL,
    api_base=OLLAMA_API_BASE,
    api_key=OLLAMA_API_KEY,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.1,
    max_tokens=450,
)

evaluator_llm = OpenAILike(
    model=OLLAMA_MODEL,
    api_base=OLLAMA_API_BASE,
    api_key=OLLAMA_API_KEY,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.0,
    max_tokens=128,
)