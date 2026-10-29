import os
import requests
import litellm
from litellm import completion
from dotenv import load_dotenv

# Убираем возможные прокси, которые Python/httpx/openai-sdk может подхватывать из окружения
for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(key, None)

litellm._turn_on_debug()

load_dotenv()

MY_IP = os.environ["OLLAMA_IP"]
os.environ["NO_PROXY"] = f"{MY_IP},localhost,127.0.0.1"
os.environ["no_proxy"] = f"{MY_IP},localhost,127.0.0.1"

BASE_URL = f"http://{MY_IP}:11434"
MODEL = "qwen2:7b"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Привет. Ответь одним коротким предложением.",
        }
    ],
    "temperature": 0.1,
    "max_tokens": 100,
}

print("\n=== 1. Проверка через requests ===")
r = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json=payload,
    timeout=120,
)
print("status:", r.status_code)
print("text:", r.text)

r.raise_for_status()

print("\n=== 2. Проверка через LiteLLM openai-compatible ===")
response = completion(
    model=f"openai/{MODEL}",
    api_base=f"{BASE_URL}/v1",
    api_key="ollama",
    messages=payload["messages"],
    temperature=0.1,
    max_tokens=100,
    timeout=120,
    max_retries=0,
    drop_params=True,
)

print(response.choices[0].message.content)