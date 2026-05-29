"""
Пример одноагентного Agentic RAG-маршрутизатора

По материалам курса HF и статьи:
https://weaviate.io/blog/what-is-agentic-rag

"""
import datetime
import os

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

os.environ["NO_PROXY"] = "188.116.172.185,localhost,127.0.0.1"
os.environ["no_proxy"] = "188.116.172.185,localhost,127.0.0.1"

import re
from smolagents import tool, ToolCallingAgent, LiteLLMModel, OpenAIModel
from dotenv import load_dotenv

load_dotenv()


import re
from smolagents import CodeAgent, OpenAIModel, tool


PRODUCT_DOCS = [
    (
        "Speech2Subtitles",
        "Модуль Speech2Subtitles преобразует речь из видео в субтитры. "
        "Поддерживает автоматическое распознавание речи, таймкоды и экспорт SRT."
    ),
    (
        "Мультивьювер",
        "Мультивьювер позволяет отображать несколько live-плееров на одном экране."
    ),
]

SUPPORT_DOCS = [
    (
        "Оплата подписки",
        "Если подписка не активировалась после оплаты, нужно проверить статус платежа, "
        "ID пользователя и вручную обновить подписку в административной панели."
    ),
    (
        "Возврат средств",
        "Для возврата средств пользователь должен предоставить свой ID. "
        "После проверки остаток денежных средств возвращается пользователю."
    ),
]

ENGINEERING_DOCS = [
    (
        "FastAPI + Gunicorn + Nginx",
        "FastAPI-приложение запускается через Gunicorn/Uvicorn workers, "
        "а Nginx принимает внешние HTTP-запросы и проксирует их на backend."
    ),
    (
        "Docker-развёртывание",
        "Сервис разворачивается через Docker Compose. Обычно используются контейнеры "
        "backend, frontend, postgres и nginx."
    ),
]


def simple_search(docs: list[tuple[str, str]], query: str, top_k: int = 2) -> str:
    query_words = set(re.findall(r"[а-яА-Яa-zA-Z0-9]+", query.lower()))
    results = []

    for title, text in docs:
        haystack = f"{title} {text}".lower()
        score = sum(1 for word in query_words if word in haystack)

        if score > 0:
            results.append((score, title, text))

    if not results:
        return "Ничего релевантного в этом источнике не найдено."

    results.sort(reverse=True, key=lambda item: item[0])

    return "\n\n".join(
        f"Источник: {title}\nФрагмент: {text}"
        for _, title, text in results[:top_k]
    )


@tool
def search_product_docs(query: str) -> str:
    """
    Ищет информацию в продуктовой документации: функции продукта, модули, возможности системы.

    Args:
        query: Поисковый запрос пользователя.
    """
    return simple_search(PRODUCT_DOCS, query)


@tool
def search_support_docs(query: str) -> str:
    """
    Ищет информацию в базе поддержки: оплаты, подписки, ошибки пользователей, возвраты.

    Args:
        query: Поисковый запрос пользователя.
    """
    return simple_search(SUPPORT_DOCS, query)


@tool
def search_engineering_docs(query: str) -> str:
    """
    Ищет информацию в инженерной документации: Docker, FastAPI, Nginx, backend, deployment.

    Args:
        query: Поисковый запрос пользователя.
    """
    return simple_search(ENGINEERING_DOCS, query)


model = OpenAIModel(
    model_id="qwen2.5:7b",
    api_base="http://188.116.172.185:11434/v1",
    api_key="ollama",
    temperature=0.0,
    max_tokens=500,
)

agent = CodeAgent(
    tools=[
        search_product_docs,
        search_support_docs,
        search_engineering_docs,
    ],
    model=model,
    max_steps=3,
    verbosity_level=2,
    instructions="""
                Ты одноагентный AgentRAG-маршрутизатор.

                Инструменты:
                - search_product_docs — продукт, модули, функции.
                - search_support_docs — оплата, подписки, поддержка, возвраты.
                - search_engineering_docs — Docker, FastAPI, Nginx, backend, deployment.
                
                Правила:
                1. Выбери подходящий retrieval-инструмент по смыслу вопроса.
                2. Если контекста мало, вызови второй подходящий инструмент.
                3. Отвечай только по Observation.
                4. Не добавляй примеры, команды, конфиги и технические детали, которых нет в Observation.
                5. Не возвращай сырые фрагменты Observation — сформулируй краткий ответ своими словами.
                6. Если в Observation нет точного ответа, прямо скажи, что найденного контекста недостаточно.
                7. После получения релевантного контекста сразу вызови final_answer(...).
                8. Отвечай на русском языке.
                """,
                )


questions = [
    "Как работает модуль автоматических субтитров?",
    "Что делать, если подписка не активировалась после оплаты?",
    "Как развернуть FastAPI-сервис через Docker и Nginx?",
]

all_answer = []
time_start = datetime.datetime.now()
for question in questions:
    print("\n" + "=" * 80)
    print("Вопрос:", question)
    answer = agent.run(f"""
    Вопрос пользователя: {question}
    
    Сделай 2 строки:
    Ответ: <краткий ответ только по Observation>
    Источник: <название вызванного инструмента>
    
    Не добавляй информацию вне Observation.
    Финальный ответ верни через final_answer(...).
    """,
        reset=True,
        max_steps=3,
    )
    print("Ответ:", answer)
    all_answer.append({"Вопрос:": question, "Ответ:": answer})

print(f'{"="*50}')
print('Продолжительность:', datetime.datetime.now() - time_start)
print(all_answer)

"""
1.
OpenAIModel - qwen2.5:7b
Продолжительность: 0:02:19.446097
[{'Вопрос:': 'Как работает модуль автоматических субтитров?', 'Ответ:': 'Модуль автоматических субтитров преобразует речь из видео в текстовые субтитры. Он поддерживает автоматическое распознавание речи, таймкоды и экспорт файлов формата SRT.'}, {'Вопрос:': 'Что делать, если подписка не активировалась после оплаты?', 'Ответ:': 'Ответ: Если подписка не активировалась после оплаты, нужно проверить статус платежа, ID пользователя и вручную обновить подписку в административной панели.\nИсточник: Оплата подписки'}, {'Вопрос:': 'Как развернуть FastAPI-сервис через Docker и Nginx?', 'Ответ:': {'ответ': 'Для развертывания FastAPI-сервиса через Docker и Nginx можно использовать Docker Compose. Backend запускается с помощью Gunicorn/Uvicorn, а Nginx принимает внешние запросы и проксирует их на backend.', 'источник': 'FastAPI + Gunicorn + Nginx, Docker-развёртывание'}}]

2.
OpenAIModel - qwen3:8b
Продолжительность: 0:13:07.216798
[{'Вопрос:': 'Как работает модуль автоматических субтитров?', 'Ответ:': 'Модуль автоматических субтитров Speech2Subtitles работает через автоматическое распознавание речи из видео, генерацию таймкодов и экспорт субтитров в формат SRT. Источник: Speech2Subtitles'}, {'Вопрос:': 'Что делать, если подписка не активировалась после оплаты?', 'Ответ:': 'Источник: Оплата подписки\nФрагмент: Если подписка не активировалась после оплаты, нужно проверить статус платежа, ID пользователя и вручную обновить подписку в административной панели.\n\nИсточник: Возврат средств\nФрагмент: Для возврата средств пользователь должен предоставить свой ID. После проверки остаток денежных средств возвращается пользователю.'}, {'Вопрос:': 'Как развернуть FastAPI-сервис через Docker и Nginx?', 'Ответ:': "Для развертывания FastAPI через Docker и Nginx выполните следующие шаги:\n\n1. **Docker Compose**: Создайте файл `docker-compose.yml` с конфигурацией контейнеров. Пример:\n   ```yaml\n   version: '3'\n  "}]

"""