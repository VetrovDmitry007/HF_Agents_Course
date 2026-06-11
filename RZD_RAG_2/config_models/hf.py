import os

from llama_index.llms.openai_like import OpenAILike
from RZD_RAG_2.fallback_llm import FallbackHuggingFaceLLM
from dotenv import load_dotenv

load_dotenv()

TOKEN_GROQ = os.environ["GROQ_TOKEN_1"]

TOKEN_ENVS = [
    "HF_TOKEN_0",
    "HF_TOKEN",
    "HF_TOKEN_2",
    "HF_TOKEN_0",
    "HF_TOKEN",
    "HF_TOKEN_2",
    "HF_TOKEN_0",
    "HF_TOKEN",
    "HF_TOKEN_2",
    ]


normalizer_llm = OpenAILike(
    model="Qwen/Qwen2.5-32B-Instruct:featherless-ai",
    api_base="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=32768,
    temperature=0.1,
    max_tokens=500,
)


# normalizer_llm = FallbackHuggingFaceLLM(
#     token_envs=TOKEN_ENVS,
#     # model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
#     model_name="Qwen/Qwen2.5-32B-Instruct",
#     temperature=0.01,
#     max_tokens=128,
#     provider="auto",
# )



selector_llm = FallbackHuggingFaceLLM(
    token_envs=TOKEN_ENVS,
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.1,
    max_tokens=128,
    provider="auto",
)

answer_llm = FallbackHuggingFaceLLM(
    token_envs=TOKEN_ENVS,
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.1,
    max_tokens=450,
    provider="auto",
)

evaluator_llm = FallbackHuggingFaceLLM(
    token_envs=TOKEN_ENVS,
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    temperature=0.01,
    max_tokens=128,
    provider="auto",
)
