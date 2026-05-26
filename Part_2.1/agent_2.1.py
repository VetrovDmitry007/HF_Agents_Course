# Использование моделей через InferenceClientModel
#
# InferenceClientModel -- интерфейс для облачных моделей на стороне разных провайдеров через Hugging Face или на HF,
# DuckDuckGoSearchTool -- инструмент позволяет искать информацию в интернете через DuckDuckGo
# pip install ddgs

import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
from dotenv import load_dotenv

load_dotenv()


instructions = """
    Всегда отвечай пользователю на русском языке.
    Финальный ответ final_answer должен быть только на русском языке.
    Если поиск удобнее выполнять на английском, можешь искать на английском,
    но итоговый ответ обязательно переводи и оформляй по-русски.
    """

model = InferenceClientModel(
    model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
    token=os.environ['HF_TOKEN']
)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()],
                  model=model,
                  instructions=instructions
                  )

agent.run("Поиск лучших музыкальных рекомендаций для вечеринки в особняке Уэйнов.")
