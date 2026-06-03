"""

pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
"""

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto"
)

response = llm.complete("Здравствуйте, как дела?")
print(response)
# Здравствуйте! Спасибо, что спросили. У меня всё в порядке, я здесь, чтобы помочь вам с любыми
# вопросами или задачами. Как у вас дела? Чем могу быть полезен?
