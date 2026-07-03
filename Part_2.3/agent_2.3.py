"""
Агент анализа документов
https://huggingface.co/learn/agents-course/en/unit2/langgraph/document_analysis_agent

1. Обрабатывать изображения документов
2. Извлекать текст с помощью визуальных моделей (Vision Language Model)
3. При необходимости выполнять вычисления (для демонстрации обычных инструментов)
4. Анализировать контент и предоставлять краткие обзоры
5. Выполнять конкретные инструкции, связанные с документами

Подключает к LLM список доступных инструментов, чтобы модель могла выбрать нужный инструмент и запросить его вызов
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

ToolNode — это исполнитель инструментов внутри LangGraph

#######################
Важное примечание к работе логики LangGraf
--------------------

1. Поле state["messages"] рекомендуется использовать, потому что это стандартная структура для message-based графов в LangGraph.

2. Если используется другое поле, например state["chat_history"], его нужно передать через messages_key.

3. tools_condition берёт последнее сообщение из списка сообщений и проверяет, есть ли в AIMessage содержиться tool_calls.

4. Если tool_calls есть, tools_condition возвращает "tools".
   Если tool_calls нет, возвращает END.

5. Если mapping не указан, подразумевается, что в графе есть нода с именем "tools".

6. Если используется несколько инструментов, модель сама выбирает, какой инструмент нужен,
   и возвращает AIMessage с именем инструмента:

   AIMessage(
    content="",
    tool_calls=[
            {
                "name": "extract_text",
                "args": {
                    "img_path": "Batman_training_and_meals.png"
                }
            }
        ]
   )

#########################


pip install langgraph langchain_openrouter langchain_core
"""

import os
import time
import base64
from typing import TypedDict, List, Dict, Any, Optional, Literal, Annotated, Optional

from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

from io import BytesIO
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OR_TOKEN")

# 1.1 Загрузка модели

def init_llm():
    """
    Время выполнения :
    Время выполнения :
    """
    llm = ChatOpenRouter(
        # model="openai/gpt-4o-mini",
        model="openai/gpt-oss-20b:free",
        temperature=0,
        max_tokens=1024,
    )

    """
    Время выполнения :  
    Время выполнения :  
    """


    # llm  = ChatOllama(
    #     model="qwen2.5:7b",
    #     base_url="http://188.116.172.185:11434",
    #     temperature=0.1,
    #     num_predict=800,
    # )

    return llm

def init_vision_llm():
    """Мультимодальная модель для извлечения текста из изображений."""
    return ChatOpenRouter(
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=1024,
    )

# 1.2 Объявление состояния
# Add_messages -- cохраняет контекстную осведомленность предыдущих взаимодействий.
# Add_messages -- это инструкция LangGraph указывающая добавлять новые сообщения к уже существующим.
class AgentState(TypedDict):
    input_file: Optional[str]  # Содержит путь к файлу (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]


# 2.1 Объявление инструмента extract_text()
def make_extract_text_tool(vision_llm):
    @tool
    def extract_text(img_path: str) -> str:
        """ Извлечение текста из файла изображения с помощью мультимодальной модели.

        :param img_path: Путь к локальному файлу изображения (строка).
        :return: Единая строка, содержащая объединенный текст, извлеченный из каждого изображения.
        """
        all_text = ""
        try:
            # Прочитайте изображение и закодируйте его как base64.
            with open(img_path, "rb") as image_file:
                image_bytes = image_file.read()

            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Подготовьте запрос, включая данные изображения в формате Base64.
            message = [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Извлеките весь текст из этого изображения. "
                                "Возвращать только извлеченный текст, без пояснений."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            },
                        },
                    ]
                )
            ]

            # Вызов модели с возможностями машинного зрения
            response = vision_llm.invoke(message)
            # Добавить извлеченный текст
            all_text += response.content + "\n\n"

            return all_text.strip()
        except Exception as e:
            # Инструмент должен корректно обрабатывать ошибки
            error_msg = f"Ошибка извлечения текста: {str(e)}"
            print(error_msg)
            return ""

    return extract_text

# 2.2 Объявление инструмента divide()
@tool
def divide(a: int, b: int) -> float:
    """Разделите a и b - для периодических вычислений."""
    return a / b


# 3. Создание ноды
def make_assistant(llm_with_tools):
    def assistant(state: AgentState):
        """ Нода "Ассистент" использующая llm_with_tools.

            Заполняет поле state.messages значением AIMessage определённым инструментом(tool_calls)
        """
        image = state["input_file"]

        sys_msg = SystemMessage(
            content=(
                "Вы — услужливый секретарь. "
                "Вы можете анализировать документы и выполнять вычисления с помощью инструментов.\n"
                "Если инструмент уже вернул достаточный результат, не вызывай инструмент повторно, "
                "а сформируй финальный ответ пользователю. В ответе дай пояснение результату.\n"
                "Инструменты:\n"
                "1. extract_text(img_path: str) -> str\n"
                "2. divide(a: int, b: int) -> float\n"
                f"В настоящее время загруженное изображение: {image}"
            )
        )

        response = llm_with_tools.invoke([sys_msg] + state["messages"])

        return {
            "messages": [response],
            "input_file": state["input_file"],
        }

    return assistant

# 4. Создание графа
def creare_graph(tools, llm_with_tools):
    graf = StateGraph(AgentState)

    graf.add_node("assistant", make_assistant(llm_with_tools))
    graf.add_node("tools", ToolNode(tools)) # ToolNode — это исполнитель инструментов внутри LangGraph

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
    llm = init_llm()
    vision_llm = init_vision_llm()

    extract_text_tool = make_extract_text_tool(vision_llm)

    # Список доступных инструментов
    tools = [
        divide,
        extract_text_tool
    ]
    # Подключает к LLM список доступных инструментов, чтобы модель могла выбрать нужный инструмент и запросить его вызов
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    compiled_graph = creare_graph(tools=tools, llm_with_tools=llm_with_tools)
    return compiled_graph


def show_graf(graph):
    png_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(png_data))
    image.show()


def run_example_1():
    compiled_graph = get_graph()
    show_graf(compiled_graph)

    messages = [HumanMessage(content="Разделите 6790 на 5")]

    """
    compiled_graph.invoke(...) возвращает финальное состояние графа.
    То есть не просто ответ модели, а весь state целиком
    
    {
        "input_file": None,
        "messages": [
            HumanMessage(...),
            AIMessage(...),
            ToolMessage(...),
            AIMessage(...),
        ]
    }
    """
    messages:AgentState = compiled_graph.invoke({"messages": messages, "input_file": None})

    # Показать сообщения.
    # m — это объект сообщения: HumanMessage, AIMessage, ToolMessage и т.д.
    # Метод pretty_print() выводит его в читаемом виде:
    for m in messages['messages']:
        m.pretty_print()


def run_example_2():
    compiled_graph = get_graph()

    messages = [HumanMessage(
        content="Согласно примечанию, предоставленному г-ном Уэйном на предоставленных изображениях. Какой список продуктов мне следует купить для меню на ужин??")]
    messages:AgentState = compiled_graph.invoke({"messages": messages, "input_file": "Batman_training_and_meals.png"})

    # Показать сообщения.
    for m in messages['messages']:
        m.pretty_print()


if __name__ == '__main__':
    run_example_1()
    # run_example_2()