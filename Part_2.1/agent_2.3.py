# Интеграция нескольких инструментов с использованием интерфейса модели OpenAIModel

"""
Инструменты
-------------
1. DuckDuckGoSearchTool -- (ищет ссылки) Инструмент позволяет искать информацию в интернете через DuckDuckGo
2. VisitWebpageTool -- (заходит на ссылки) Позволяет агенту открыть конкретную веб-страницу по URL и прочитать её содержимое
3. FinalAnswerTool -- Упрощает определение агентом явной точки завершения.
"""


import os
from smolagents import (
                        CodeAgent,
                        DuckDuckGoSearchTool,
                        FinalAnswerTool,
                        Tool,
                        tool,
                        VisitWebpageTool,
                        OpenAIModel)
from dotenv import load_dotenv

load_dotenv()

@tool
def suggest_menu(occasion: str) -> str:
    """ Предлагает меню в зависимости от случая.
    Args:
        occasion: Тип повода для вечеринки.
    """
    if occasion == "casual":
        return "Пицца, закуски и напитки."
    elif occasion == "formal":
        return "Ужин из 3 блюд с вином и десертом."
    elif occasion == "superhero":
        return "Шведский стол с высокоэнергетической и здоровой пищей."
    else:
        return "Специальное меню для дворецкого."


@tool
def catering_service_tool(query: str) -> str:
    """ Этот инструмент возвращает услуги общественного питания с самым высоким рейтингом в Готэм-сити.

    Args:
        query: Поисковый запрос для поиска услуг общественного питания.
    """
    # Примерный перечень услуг общественного питания и их рейтинги
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Найдите кейтеринговую службу с самым высоким рейтингом (имитируя фильтрацию поисковых запросов)
    best_service = max(services, key=services.get)

    return best_service


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator" # генератор тем супергеройской вечеринки
    description = """
            Этот инструмент предлагает креативные идеи для вечеринок на тему супергероев в зависимости от категории.
            Он возвращает уникальную идею темы вечеринки."""

    inputs = {
        "category": {
            "type": "string",
            "description": "Тип супергеройской вечеринки (например, «классические герои», «маскарад злодеев»,"
                           " «футуристический Готэм»).",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "классические герои": "Гала-концерт Лиги справедливости: гости приходят в костюмах своих любимых героев DC "
                              "и угощаются тематическими коктейлями, такими как «Криптонитовый пунш».",
            "злодей-маскарад": "Бал разбойников Готэма: загадочный маскарад, на котором гости одеваются "
                                  "как классические злодеи Бэтмена.",
            "футуристический Готэм": "Neo-Gotham Night: вечеринка в стиле киберпанк, вдохновленная Batman Beyond, с "
                                 "неоновыми декорациями и футуристическими гаджетами."
        }

        return themes.get(category.lower(),
                          "Идея тематической вечеринки не найдена. Попробуйте «классических героев», «маскарад злодеев» "
                          "или «футуристический Готэм».")


# Альфред, дворецкий, готовит меню для вечеринки.

model = OpenAIModel(
    # model_id="deepseek/deepseek-v3.2", # 17-66 сек.
    model_id="poolside/laguna-m.1:free", # 49 сек.
    # model_id="deepseek/deepseek-v4-flash:free", # 17.65 сек.
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ['OR_TOKEN'],
)

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(max_output_length=10_000),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
        FinalAnswerTool()
    ],
    model=model,
    max_steps=10,
    verbosity_level=2
)

agent.run(
    "Назовите мне лучший плейлист для вечеринки в особняке Уэйнов. Идея вечеринки — тема «злодейского маскарада».")