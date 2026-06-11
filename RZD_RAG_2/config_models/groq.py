import os

from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv

load_dotenv()

TOKEN_GROQ = os.environ["GROQ_TOKEN_1"]


normalizer_llm = OpenAILike(
    model="llama-3.1-8b-instant",
    api_base="https://api.groq.com/openai/v1",
    api_key=TOKEN_GROQ,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.0,
    max_tokens=250,
)

selector_llm = OpenAILike(
    model="llama-3.1-8b-instant",
    api_base="https://api.groq.com/openai/v1",
    api_key=TOKEN_GROQ,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.0,
    max_tokens=256,
    timeout=90.0,
    max_retries=2,
)

answer_llm = OpenAILike(
    model="llama-3.3-70b-versatile",
    api_base="https://api.groq.com/openai/v1",
    api_key=TOKEN_GROQ,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.1,
    max_tokens=450,
)

evaluator_llm = OpenAILike(
    model="llama-3.3-70b-versatile",
    api_base="https://api.groq.com/openai/v1",
    api_key=TOKEN_GROQ,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=131072,
    temperature=0.0,
    max_tokens=128,
)