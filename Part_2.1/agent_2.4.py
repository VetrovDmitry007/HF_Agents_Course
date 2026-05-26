"""
Мониторинг работы агента
======================
1. Старт сервера мониторинга, переход на http://localhost:6006/v1/traces
   python -m phoenix.server.main serve
2. Запуск основного скрипта на выполнение

pip install 'smolagents[telemetry,toolkit]'

======================
ToolCallingAgent -- тип агента работающий через JSON

Инструменты
-------------
1. WebSearchTool -- это более общий инструмент web поиска, у которого есть параметр engine
                    WebSearchTool(max_results: int = 10, engine: str = "duckduckgo")
2. VisitWebpageTool -- (заходит на ссылки) Позволяет агенту открыть конкретную веб-страницу по URL и прочитать её содержимое
"""

import os
from dotenv import load_dotenv
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
    OpenAIModel,
)

load_dotenv()

# 1. Регистрируем Phoenix/OpenTelemetry
tracer_provider = register(
    project_name="default",
    batch=True,
)

# 2. Включаем инструментирование smolagents
SmolagentsInstrumentor().instrument()

model = OpenAIModel(
    model_id="poolside/laguna-m.1:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ['OR_TOKEN'],
)

# Демонстрирует использование архитектуры “агент-менеджер + подчинённый агент”
search_agent = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,
    name= "search_agent" ,
    description= "Это агент, который может выполнять веб-поиск." ,
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)

manager_agent.run("Если темпы роста США сохранятся на уровне 2024 года, сколько лет потребуется для удвоения ВВП?")
