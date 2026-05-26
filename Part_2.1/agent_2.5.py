"""
Использование Инструментов фреймворка smolagents
"""

import os
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    Tool,
    VisitWebpageTool,
    OpenAIModel)
from dotenv import load_dotenv

load_dotenv()


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """Этот инструмент предлагает креативные идеи для вечеринок на тему супергероев в зависимости от
     категории. Он возвращает уникальную идею темы вечеринки."""

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
            "маскарад злодеев": "Бал разбойников Готэма: загадочный маскарад, на котором гости одеваются как классические"
                               " злодеи Бэтмена.",
            "футуристический Готэм": "Neo-Gotham Night: вечеринка в стиле киберпанк, вдохновленная Batman Beyond, с "
                                     "неоновыми декорациями и футуристическими гаджетами."
        }

        return themes.get(category.lower(),
                          "Идея тематической вечеринки не найдена. Попробуйте «классических героев», «маскарад злодеев»"
                          " или «футуристический Готэм».")


model = OpenAIModel(
    model_id="poolside/laguna-m.1:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ['OR_TOKEN'],
)

instructions = 'Ищи креативное идеи в сети. Придумай и опиши детали вечеринки.'

party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool,
                         DuckDuckGoSearchTool(),
                         VisitWebpageTool(),
                         FinalAnswerTool()],
                  model=model,
                  instructions=instructions)

# Запустите агент, чтобы сгенерировать идею темы вечеринки.
result = agent.run("Какой была бы хорошая идея для супергеройской вечеринки на тему «маскарад злодеев»?")

print(result)
