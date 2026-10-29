"""
Пример AI-агента на LangGraph с подключением через CLIProxyAPI.

Агент выполняет арифметические операции с помощью инструментов и сохраняет
историю состояния между вызовами графа через MemorySaver. Использование одного
thread_id позволяет агенту учитывать результаты предыдущих запросов и продолжать
диалог в рамках одной сессии.
"""

import os
from typing import TypedDict, Annotated
import httpx
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# 1.1 Загрузка модели
def init_llm():
    api_key = os.getenv("CLIPROXY_API_KEY")
    if not api_key:
        raise RuntimeError("Укажи CLIPROXY_API_KEY в .env")

    return ChatOpenAI(
        model=os.getenv("CLIPROXY_MODEL"),
        base_url=os.getenv("CLIPROXY_BASE_URL", "http://127.0.0.1:8317/v1"),
        api_key=api_key,
        # Обходим системный прокси Windows для прямого доступа к локальному SSH-туннелю.
        http_client=httpx.Client(trust_env=False),
        http_socket_options=(),
        use_responses_api=True,
        timeout=120,
        max_retries=2,
    )

# 1.2 Объявление состояния
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 2.1 Объявление инструментов
@tool
def add(a: int, b: int) -> int:
    """Складывает два числа."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Умножает два числа."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Делит первое число на второе."""
    return a / b


# 3. Создание ноды assistant
def make_assistant(llm_with_tools, sys_msg: SystemMessage):
    def assistant(state: AgentState):
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    return assistant


# 4. Создание системного промпта
def get_system_message():
    sys_msg = SystemMessage(
        content=(
            "Вы полезный помощник, которому поручено выполнять арифметические действия с набором входных данных.\n"
            "Сформируй финальный ответ пользователю.\n"
            "Инструменты:\n"
            "1. add(a: int, b: int) -> int -- Складывает два числа.\n"
            "2. multiply(a: int, b: int) -> int -- Умножает два числа.\n"
            "3. divide(a: int, b: int) -> float -- Делит первое число на второе.\n"
        )
    )
    return sys_msg


# 5. Создание графа
def creare_graph(tools: list, llm_with_tools, sys_msg: SystemMessage):

    graf = StateGraph(AgentState)

    graf.add_node("assistant", make_assistant(llm_with_tools, sys_msg))
    graf.add_node("tools", ToolNode(tools))  # ToolNode — это исполнитель инструментов внутри LangGraph

    graf.add_edge(START, "assistant")
    graf.add_conditional_edges(
        "assistant",
        # Если для последнего сообщения (state.messages.AIMessage) требуется инструмент, направьте его к инструментам.
        # В противном случае дайте прямой ответ END.
        tools_condition,
    )
    # Для формирования финального ответа, заполненный state передаём обратно в assistant
    graf.add_edge("tools", "assistant")
    compiled_graph = graf.compile()
    return compiled_graph


def get_graph():
    """
    Создаёт и возвращает скомпилированный LangGraph-граф анализа документов.

    Инициализирует LLM, создаёт инструменты, подключает их к модели через bind_tools()
    и собирает граф с двумя основными нодами: assistant и tools.
    Нода assistant формирует ответ или tool_calls, а ToolNode выполняет выбранные инструменты.

    :return: Скомпилированный граф, готовый к запуску через invoke().
    """
    sys_msg = get_system_message()
    llm = init_llm()

    # Список доступных инструментов
    tools = [add, multiply, divide]
    # Подключает к LLM список доступных инструментов
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    compiled_graph = creare_graph(tools=tools, llm_with_tools=llm_with_tools, sys_msg=sys_msg)
    return compiled_graph


def run_graph():
    config = {"configurable": {"thread_id": "1"}}
    print('Компиляция графа...')
    compiled_graph = get_graph()
    print('Граф скомпилирован.')

    messages = [HumanMessage(content="Разделите 6790 на 5. Затем это умножь на 2 и прибавь 100")]
    agent_state: AgentState = compiled_graph.invoke({"messages": messages}, config=config)

    # Показать сообщения.
    for m in agent_state['messages']:
        m.pretty_print()


if __name__ == '__main__':
    run_graph()