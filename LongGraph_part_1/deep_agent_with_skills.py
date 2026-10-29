"""
Пример использования Deep Agents с паттерном Skills.

Агент получает каталог Skills, выбирает подходящие навыки и выполняет
составную арифметическую задачу. Ход выполнения выводится через streaming.

!!! Вывод
Штатный prompt-driven Skills Deep Agents плохо совместим с протестированными Groq-моделями для этой демонстрации.
llama-3.3-70b-versatile работает лучше, но съедает TPD; Qwen 3 32B работает нестабильно.
"""

import os

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OR_TOKEN")
os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN_1"]
OLLAMA_IP = os.environ["OLLAMA_IP"]

SKILLS_SYSTEM_PROMPT = """
## Система Skills

Доступные источники Skills:
{skills_locations}

{skills_load_warnings}

Доступные Skills:
{skills_list}

Skills — это инструкции, а не инструменты.

Имена Skills из списка выше никогда не являются именами функций
и не могут использоваться как tool calls.

Для использования Skill:

1. Определи подходящий Skill по имени и описанию.
2. Вызови настоящий инструмент `read_file` и прочитай SKILL.md
   по точному пути, указанному рядом со Skill.
3. Следуй инструкциям из SKILL.md.
4. Выполняй зависимые операции последовательно: сначала дождись
   результата текущего инструмента, затем используй этот результат
   в следующем действии.

Вызывай только инструменты, присутствующие в схеме доступных tools.
Не создавай вложенные вызовы инструментов.
"""

register_harness_profile(
    "groq:qwen/qwen3-32b",
    HarnessProfile(
        excluded_middleware={
            "TodoListMiddleware",
            "SummarizationMiddleware",
        },
        general_purpose_subagent=GeneralPurposeSubagentProfile(
            enabled=False,
        ),
    ),
)


def create_llm():
    """Создаёт и настраивает LLM через OpenRouter."""
    # llm = ChatOpenRouter(
    #     # model="openai/gpt-4o-mini",
    #     model="openai/gpt-oss-20b:free",
    #     temperature=0,
    #     max_tokens=1024,
    # )

    # llm = ChatOllama(
    #     model="qwen2.5:7b",
    #     base_url=f"http://{OLLAMA_IP}:11434",
    #     temperature=0.1,
    #     num_predict=800,
    # )

    llm = ChatGroq(
        # model="llama-3.3-70b-versatile",
        model="qwen/qwen3-32b",
        # model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1024,
        model_kwargs={"parallel_tool_calls": False},
        )

    return llm


def create_agent():
    """Создаёт Deep Agent с поддержкой локальных Skills."""
    llm = create_llm()
    backend = LocalShellBackend(
        root_dir=".",
        virtual_mode=True,
        inherit_env=True,
    )

    skills_middleware = SkillsMiddleware(
        backend=backend,
        sources=["skills/"],
        system_prompt=SKILLS_SYSTEM_PROMPT,
    )

    return create_deep_agent(
        model=llm,
        backend=backend,
        middleware=[skills_middleware],
    )


def run_agent(agent) -> None:
    """Запускает агента и выводит изменения состояния по шагам."""
    input_data = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Разделите 6790 на 5, затем результат "
                    "умножьте на 2 и прибавьте 100."
                ),
            }
        ]
    }

    for chunk in agent.stream(
        input_data,
        config={"recursion_limit": 20},
        stream_mode="updates",
    ):
        print(chunk)


def main() -> None:
    """Точка запуска примера."""
    agent = create_agent()
    run_agent(agent)


if __name__ == "__main__":
    main()