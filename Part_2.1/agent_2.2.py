# Использование пользовательского инструмента для подготовки меню
# https://huggingface.co/learn/agents-course/unit2/smolagents/code_agents#using-a-custom-tool-to-prepare-the-menu
#
# InferenceClientModel -- интерфейс для облачных моделей на стороне разных провайдеров через Hugging Face или на HF,
# DuckDuckGoSearchTool -- инструмент позволяет искать информацию в интернете через DuckDuckGo
# pip install ddgs

import os

from smolagents import CodeAgent, tool, InferenceClientModel
from dotenv import load_dotenv

load_dotenv()

# Инструмент для предложения меню в зависимости от случая
@tool
def suggest_menu(occasion: str) -> str:
    """
    Предлагает меню в зависимости от случая.
    Args:
        occasion (str): Тип повода для вечеринки. Допустимые значения:
                        -«casual»: Меню для непринужденной вечеринки.
                        - "formal": Меню для официальной вечеринки.
                        - «супергерой»: Меню для супергеройской вечеринки.
                        - «custom»: Индивидуальное меню.
    """
    if occasion == "casual":
        return "Пицца, закуски и напитки."
    elif occasion == "formal":
        return "Ужин из 3 блюд с вином и десертом."
    elif occasion == "superhero":
        return "Шведский стол с высокоэнергетической и здоровой едой."
    else:
        return "Индивидуальное меню для дворецкого."

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

agent = CodeAgent(tools=[suggest_menu],
                  model=model,
                  instructions=instructions
                  )

agent.run("Подготовьте официальное меню для вечеринки.")
