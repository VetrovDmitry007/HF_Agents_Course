"""
Использование MCP инструментов

pip install "smolagents[mcp]"
pip install uv

Порядок работы
------------
Подключись к MCP-серверу pubmedmcp,
загрузи его инструменты,
преврати их в smolagents Tool,
положи их в tool_collection.tools,
и держи соединение открытым, пока агент работает.

pubmedmcp -- Сервер MCP, который позволяет искать и получать статьи из PubMed.
https://pypi.org/project/pubmedmcp/?utm_source=chatgpt.com

StdioServerParameters — это объект настройки(конфигурации) запуска MCP-сервера через stdio
uvx — Это команда из пакетного менеджера uv.
      Она запускает Python CLI-инструменты во временном изолированном окружении.
ToolCollection.from_mcp() -- подключается к MCP-серверу, загружает из него инструменты и передаёт их агенту.

"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from smolagents import ToolCollection, CodeAgent, OpenAIModel
from mcp import StdioServerParameters

load_dotenv()

model = OpenAIModel(
    model_id="poolside/laguna-m.1:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OR_TOKEN"],
)

# Берём uvx.exe из того же venv, из которого запущен Python
uvx_path = Path(sys.executable).with_name("uvx.exe")

server_parameters = StdioServerParameters(
    command=str(uvx_path),
    args=["--quiet", "pubmedmcp@0.1.3"],
    # args=["pubmedmcp@0.1.3"],
    env={
        **os.environ,
        "UV_PYTHON": "3.12",
        "UV_PRERELEASE": "allow",
    },
)

# ToolCollection.from_mcp() -- подключается к MCP-серверу, загружает из него инструменты и передаёт их агенту.
with ToolCollection.from_mcp(server_parameters, trust_remote_code=True, structured_output=False, ) as tool_collection:
    agent = CodeAgent(
        tools=[*tool_collection.tools],
        model=model,
        add_base_tools=True,
    )

    agent.run("Пожалуйста, найдите средство от похмелья.")
