"""
Создание первого языкового графа. Инструменты не используются.
https://huggingface.co/learn/agents-course/en/unit2/langgraph/first_graph

1. Читать входящие письма
2. Классифицировать их как спам или легитимные письма
3. Составлять предварительный ответ на легитимные письма
"""

import os
import time
from contextlib import contextmanager
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from io import BytesIO
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OR_TOKEN")

# 1.1 Загрузка модели

"""
Время выполнения span_classify:  1.6038610935211182
Время выполнения draft response:  1.510807991027832
"""
# model = ChatOpenRouter(
#     model="openai/gpt-4o-mini",
#     temperature=0,
#     max_tokens=1024,
# )

"""
Время выполнения span classify:  24.22611427307129
Время выполнения draft response:  27.50546622276306
"""
# model = ChatOllama(
#     model="qwen2.5:7b",
#     base_url="http://188.116.172.185:11434",
#     temperature=0.1,
#     num_predict=800,
# )

"""
Время выполнения span_classify:  2.201591968536377
Время выполнения draft response:  2.5656380653381348
"""
model = ChatOllama(
    model="gpt-oss:20b",
    base_url="http://192.168.3.154:11434",
    temperature=0.1,
    num_predict=800,
)

# 1.2 Объявление состояния
class EmailState(TypedDict):
    # Обрабатываемое электронное письмо
    email: Dict[str, Any]  #  Содержит тему, отправителя, текст письма и т. д.
    # Категория письма (запрос, жалоба и т. д.)
    email_category: str
    # Причина, по которой письмо было помечено как спам
    spam_reason: str
    # Признак спама
    is_spam: bool
    # Генерация ответа
    email_draft: str
    # Обработка метаданных
    messages: List[Dict[str, Any]]  # Отслеживание диалога с LLM для анализа


# 2. Объявление нод

def read_email(state: EmailState):
    """Чтение и регистрация входящей электронной почты"""
    email = state["email"]

    # Здесь мы могли бы выполнить некоторую первоначальную предварительную обработку
    print(f"Обрабатывается письмо от {email['sender']} заголовок: {email['subject']}")

    # Никаких изменений состояния здесь не требуется.
    return {}


def classify_email(state: EmailState):
    """Использование LLM, чтобы определить, является ли письмо спамом или законным."""
    email = state["email"]

    prompt_system = (
        "Ты секретарь, твоя задача обрабатывать письма и определять спам в них. Письма со спамом маркируй как spam. "
        "Законные письма помечай как ham. После параметра 'reason:' укажи почему оно помечено как спам."
        " Пиши по русски.")

    prompt_message = f"""Проанализируйте это письмо и определите, является ли оно спамом или законным.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    Сначала определите, является ли это письмо спамом. Если это спам, объясните почему.
    Если оно законно, классифицируйте его (запрос, жалоба, благодарность и т. д.).
    """

    messages = [SystemMessage(content=prompt_system),
                HumanMessage(content=prompt_message)]

    with lead_time('span_classify'):
        response = model.invoke(messages)

    # Упрощённая логика для анализа ответа
    response_text = response.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    # Укажите причину, если это спам
    spam_reason = None
    if is_spam and "reason:" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

    # Определите категорию, если это законно
    email_category = None
    if not is_spam:
        categories = ["запрос", "жалоба", "благодарность", "запрос", "информация"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    # Обновить сообщения для отслеживания
    new_messages = state.get("messages", []) + [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_message},
        {"role": "assistant", "content": response.content}
    ]

    # Возвращать обновления состояния.
    # !!! Важно. В state будут обновлены только возвращаемы поля, остальные поля останутся без изменений.
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


def handle_spam(state: EmailState):
    """Отклоняет спам с запиской"""
    print(f"Помечено письмо как спам.\nПричина: {state['spam_reason']}")
    print("Письмо было перемещено в папку «Спам»..")

    # Закончили обработку этого письма
    return {}


def draft_response(state: EmailState):
    """Подготовка предварительного ответ на законные(не спам) электронные письма"""
    email = state["email"]
    category = state["email_category"] or "general"

    prompt_system = "Ты секретарь, твоя задача обрабатывать письма и подготавливать краткий ответ на эти письма."
    prompt_message = f"""Напишите вежливый предварительный ответ на это письмо.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    Это письмо было отнесено к категории: {category}
    ##########
    Составьте краткий профессиональный ответ, который администратор сможет просмотреть и персонализировать перед отправкой.
    """

    messages = [SystemMessage(content=prompt_system),
                HumanMessage(content=prompt_message)]

    with lead_time('draft response'):
        response = model.invoke(messages)

    # Обновить сообщения для отслеживания
    new_messages = state.get("messages", []) + [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_message},
        {"role": "assistant", "content": response.content}
    ]

    # Возвращать обновления состояния
    # !!! Важно. В state будут обновлены только возвращаемы поля, остальные поля останутся без изменений.
    return {
        "email_draft": response.content,
        "messages": new_messages
    }


def notify_mr_hugg(state: EmailState):
    """Уведомляет администратора об электронном письме и представляет проект ответа."""
    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Сэр, вы получили письмо от {email['sender']}.")
    print(f"Описание: {email['subject']}")
    print(f"Категория: {state['email_category']}")
    print("\nЯ подготовила черновик ответа на ваш отзыв:")
    print("-" * 50)
    print(state["email_draft"])
    print("=" * 50 + "\n")

    # Мы закончили обработку этого письма
    return {}


# 3. Объявление логики маршрутизации
def route_email(state: EmailState) -> Literal["spam", "legitimate"]:
    """Определите следующий шаг на основе классификации спама"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


# 4. Создание графа
def creare_graf():
    email_graph = StateGraph(EmailState)

    # Add nodes
    email_graph.add_node("read_email", read_email)
    email_graph.add_node("classify_email", classify_email)
    email_graph.add_node("handle_spam", handle_spam)
    email_graph.add_node("draft_response", draft_response)
    email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

    # Add edge
    email_graph.add_edge(START, "read_email")
    email_graph.add_edge("read_email", "classify_email")

    # Добавить условное ветвление из classify_email
    email_graph.add_conditional_edges(
        "classify_email",
        route_email,
        {
            "spam": "handle_spam",
            "legitimate": "draft_response"
        }
    )

    # Добавьте последние рёбра
    email_graph.add_edge("handle_spam", END)
    email_graph.add_edge("draft_response", "notify_mr_hugg")
    email_graph.add_edge("notify_mr_hugg", END)

    # Compile the graph
    compiled_graph = email_graph.compile()
    return compiled_graph


def run_graf(compiled_graph):
    # Пример законного электронного письма
    legitimate_email = {
        "sender": "john.smith@example.com",
        "subject": "Вопрос о ваших услугах",
        "body": "Уважаемый г-н Хагг, Меня порекомендовал к вам коллега, и мне интересно узнать больше о ваших "
                "консультационных услугах. Можем ли мы запланировать звонок на следующей неделе? С уважением, Джон Смит"
    }

    # Пример спам-письма
    spam_email = {
        "sender": "winner@lottery-intl.com",
        "subject": "ВЫ ВЫИГРАЛИ $5,000,000!!!",
        "body": "ПОЗДРАВЛЯЕМ! Вы выбраны победителем нашей международной лотереи! Чтобы получить приз в размере "
                "5 000 000 долларов США, пришлите нам свои банковские реквизиты и уплатите комиссию за обработку "
                "в размере 100 долларов США."
    }

    # Process the legitimate email
    print("\nProcessing legitimate email...")
    legitimate_result = compiled_graph.invoke({
        "email": legitimate_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": []
    })
    print(legitimate_result)

    # Process the spam email
    print("\nProcessing spam email...")
    spam_result = compiled_graph.invoke({
        "email": spam_email,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "email_draft": None,
        "messages": []
    })
    print(spam_result)


def show_graf(graph):
    png_data = graph.get_graph().draw_mermaid_png()
    image = PILImage.open(BytesIO(png_data))
    image.show()

@contextmanager
def lead_time(task: str):
    tm_1 = time.time()
    yield
    print(f'Время выполнения {task}: ',time.time() - tm_1)


if __name__ == '__main__':
    compiled_graph = creare_graf()
    # show_graf(compiled_graph)
    run_graf(compiled_graph)
