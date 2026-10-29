"""
Пример AI-агента на LangGraph с краткосрочной памятью и паттерном Skills.

Агент выполняет арифметические операции и использует Skills с прогрессивным
раскрытием инструкций: сначала модель получает краткий каталог доступных навыков,
затем выбирает подходящий навык и загружает его полную инструкцию через load_skill.

Граф разделён на этап выбора и выполнения навыка. Selector определяет подходящий
Skill по краткому описанию, а executor следует загруженной инструкции, запускает
указанные скрипты и при необходимости читает дополнительную документацию навыка.

Для сохранения состояния диалога используется MemorySaver. Один thread_id
позволяет сохранять историю сообщений между последовательными вызовами графа
и использовать результаты предыдущих операций в следующих запросах.

## Вывод ##
Данная реализация паттерна Skills, с точки зрения расхода токенов для трёх арифметических операций,
существенно хуже обычных tools.

Skills экономят не «токены вообще». Они уменьшают стартовый и постоянно присутствующий контекст при большом объёме
специализированных инструкций.
"""

import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_core.messages import AIMessage, ToolMessage

from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from skill_menager import build_skill_catalog, get_sys_tools

load_dotenv()
os.environ["OPENROUTER_API_KEY"] = os.getenv("OR_TOKEN")
os.environ["GROQ_API_KEY"] = os.environ["GROQ_TOKEN_1"]
OLLAMA_IP = os.environ["OLLAMA_IP"]


# 1 Загрузка модели
def init_llm():
    """

    """
    # llm = ChatOpenRouter(
    #     # model="openai/gpt-4o-mini",
    #     model="openai/gpt-oss-20b:free",
    #     temperature=0,
    #     max_tokens=450,
    # )

    # llm = ChatOllama(
    #     model="qwen2.5:7b",
    #     base_url=f"http://{OLLAMA_IP}:11434",
    #     temperature=0.1,
    #     num_predict=540,
    # )

    llm = ChatGroq(
        # model="llama-3.3-70b-versatile",
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=450,
    )

    return llm


# 2 Объявление состояния
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 3. Создание ноды
def make_selector_node(llm_with_tools, sys_msg: SystemMessage):
    def selector_node(state: AgentState):
        """
        selector видит только сообщения: HumanMessage и финальные AIMessage
        """
        messages = [
            message
            for message in state["messages"]
            if not isinstance(message, ToolMessage)
            and not (
                isinstance(message, AIMessage)
                and message.tool_calls
            )
        ]

        response = llm_with_tools.invoke(
            [sys_msg] + messages
        )

        return {"messages": [response]}

    return selector_node


def make_executor_node(llm_with_tools, sys_msg: SystemMessage):
    def executor_node(state: AgentState):
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    return executor_node


# 4. Создание системного промпта
def get_selector_system_message():
    sys_msg = SystemMessage(
        content=(
            "Ты выбираешь подходящий навык для выполнения запроса пользователя.\n"
            "Выбирай один навык только для последнего сообщения пользователя.\n"
            "Предыдущую историю используй только для понимания контекста и ссылок "
            "на предыдущий результат.\n"
            "Не загружай навыки, относящиеся к уже выполненным сообщениям пользователя.\n"
            "Выбери навык только на основании его имени и краткого описания.\n"
            "После выбора обязательно вызови load_skill.\n"
            "Имя выбранного навыка всегда передавай в аргументе skill_name.\n"
            "Пример: load_skill(skill_name='division').\n"
            "Не используй аргумент name.\n"
            "Не выполняй задачу самостоятельно и не формируй финальный ответ.\n"
            f"Доступные навыки:\n{build_skill_catalog()}"
        )
    )
    return sys_msg


def get_executor_system_message():
    return SystemMessage(
        content=(
            "Ты исполнитель выбранного навыка.\n"
            "Полные инструкции выбранного навыка уже находятся в истории сообщений "
            "как результат вызова load_skill.\n"
            "Строго следуй этим инструкциям.\n"
            "Используй указанные в инструкции скрипты и дополнительную документацию.\n"
            "Не угадывай имена файлов и пути. Используй только пути, явно указанные "
            "в инструкции навыка.\n"
            "Если инструкция требует прочитать дополнительную документацию, "
            "вызови read_skill_resource.\n"
            "Если инструкция требует выполнить скрипт, вызови run_skill_script.\n"
            "После выполнения всех требований навыка сформируй финальный ответ пользователю.\n"
            "Отвечай обычным текстом без Markdown-разметки."
        )
    )


# 5. Создание графа
def creare_graph(dc_tools: dict, llm):
    memory = MemorySaver()

    selector_sys_msg = get_selector_system_message()
    executor_sys_msg = get_executor_system_message()

    # Подключает к LLM список доступных инструментов
    # skill_selector_llm = llm.bind_tools(dc_tools['selector'], parallel_tool_calls=False)
    # skill_executor_llm = llm.bind_tools(dc_tools['executor'], parallel_tool_calls=False)

    # tool_choice указывает модели, какой именно tool она обязана вызвать
    skill_selector_llm = llm.bind_tools(
        dc_tools["selector"],
        tool_choice={
            "type": "function",
            "function": {"name": "load_skill"},
        },
    )
    skill_executor_llm = llm.bind_tools(dc_tools['executor'])

    graf = StateGraph(AgentState)

    graf.add_node("selector", make_selector_node(skill_selector_llm, sys_msg=selector_sys_msg))
    graf.add_node("selector_tools", ToolNode(dc_tools['selector']))
    graf.add_node("executor", make_executor_node(skill_executor_llm, sys_msg=executor_sys_msg))
    graf.add_node("executor_tools", ToolNode(dc_tools['executor']))  # ToolNode — это исполнитель инструментов внутри LangGraph

    graf.add_edge(START, "selector")
    graf.add_conditional_edges(
        "selector",
        tools_condition,
        {
            "tools": "selector_tools",
            "__end__": END,
        }
    )
    graf.add_edge('selector_tools', "executor")
    graf.add_conditional_edges(
        "executor",
        # Если для последнего сообщения (state.messages.AIMessage) требуется инструмент, направьте его к инструментам.
        # В противном случае дайте прямой ответ END.
        tools_condition,
        {
            "tools": "executor_tools",
            "__end__": END,
        }
    )
    # Для формирования финального ответа, заполненный state передаём обратно в assistant
    graf.add_edge("executor_tools", "executor")
    compiled_graph = graf.compile(checkpointer=memory)
    return compiled_graph


def get_graph():
    """
    Создаёт и возвращает скомпилированный LangGraph-граф анализа документов.

    Инициализирует LLM, создаёт инструменты, подключает их к модели через bind_tools()
    и собирает граф с двумя основными нодами: assistant и tools.
    Нода assistant формирует ответ или tool_calls, а ToolNode выполняет выбранные инструменты.

    :return: Скомпилированный граф, готовый к запуску через invoke().
    """
    llm = init_llm()

    # Список доступных инструментов
    dc_tools = get_sys_tools()

    compiled_graph = creare_graph(dc_tools=dc_tools, llm=llm)
    return compiled_graph


def run_graph():
    config = {"configurable": {"thread_id": "1"}}
    compiled_graph = get_graph()

    messages = [HumanMessage(content="Разделите 6790 на 5")]
    agent_state: AgentState = compiled_graph.invoke({"messages": messages}, config=config)

    messages = [HumanMessage(content="Затем это умножь на 2")]
    agent_state: AgentState = compiled_graph.invoke({"messages": messages}, config=config)

    messages = [HumanMessage(content="и прибавь 100")]
    agent_state: AgentState = compiled_graph.invoke({"messages": messages}, config=config)

    # Показать сообщения.
    for m in agent_state['messages']:
        m.pretty_print()


if __name__ == '__main__':
    run_graph()
