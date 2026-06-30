"""
Строительные блоки LangGraph
https://huggingface.co/learn/agents-course/en/unit2/langgraph/building_blocks
"""

from typing_extensions import TypedDict
import random
from typing import Literal
from langgraph.graph import StateGraph, START, END
from io import BytesIO
from PIL import Image as PILImage


# 1. Объявление состояния
class State(TypedDict):
    graph_state: str


# 2. Объявление нод
def node_1(state):
    print("---Node 1---")
    return {"graph_state": state['graph_state'] +" Я"}


def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" счастлив!"}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" грущу!"}


# 3. Объявление рёбер соединяющие узлы
def decide_mood(state) -> Literal["node_2", "node_3"]:
    user_input = state['graph_state']
    # Здесь давайте просто разделим узлы 2 и 3 поровну
    if random.random() < 0.5:
        # В 50 % случаев мы возвращаем узел 2
        return "node_2"

    # В 50 % случаев мы возвращаем узел 3
    return "node_3"


# 4. Создание графа
def creare_graf():
    # Build graph
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Описание логики графа
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    # Add
    graph = builder.compile()
    return graph


def show_graf(graph):
    png_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(png_data))
    image.show()


if __name__ == '__main__':
    graph = creare_graf()
    show_graf(graph)

    final_state = graph.invoke({"graph_state": "Привет, это Лэнс."})
    print(final_state)
    print(final_state["graph_state"])



