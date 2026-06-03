"""
rag_agent.py

RAG-агент для TrainBot на базе smolagents.

Что делает скрипт:
1. Загружает Q&A-базу из trainbot_rag_qa_updated.json.
2. Превращает каждую пару "вопрос пользователя — ответ ассистента" в LangChain Document.
3. Создаёт или загружает локальное FAISS-векторное хранилище.
4. Оборачивает векторное хранилище в smolagents Tool.
5. Создаёт агента, который перед ответом ищет релевантные фрагменты в базе TrainBot.

Логика соответствует примеру Hugging Face Agentic RAG:
агент получает retriever tool и сам решает, какой поисковый запрос сформулировать,
какие документы получить и как собрать финальный ответ.

pip install sentence-transformers
pip install faiss-cpu
"""
from __future__ import annotations
import os

# Убираем возможные прокси, которые Python/httpx/openai-sdk может подхватывать из окружения
for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(key, None)

os.environ["NO_PROXY"] = "188.116.172.185,localhost,127.0.0.1"
os.environ["no_proxy"] = "188.116.172.185,localhost,127.0.0.1"

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

try:
    # Новый рекомендуемый пакет для HuggingFace embeddings в LangChain.
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback для старых окружений.
    from langchain_community.embeddings import HuggingFaceEmbeddings

from smolagents import Tool, OpenAIModel, CodeAgent
from dotenv import load_dotenv

load_dotenv()

# Альтернатива, если используете Hugging Face Inference API:
# from smolagents import InferenceClientModel


JSON_PATH = Path("trainbot_rag_qa_updated.json")
INDEX_DIR = Path("trainbot_faiss_index")

# Для русскоязычных Q&A лучше брать мультиязычную модель эмбеддингов.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_trainbot_qa(json_path: Path) -> list[Document]:
    """
    Загружает Q&A-базу TrainBot из JSON и превращает её в список Document.

    Ожидаемый формат JSON:
    [
        {
            "question_user": "...",
            "answer_assistant": "..."
        },
        ...
    ]

    Почему вопрос и ответ кладутся в один документ:
    - вопрос помогает retriever'у найти запись по похожей пользовательской формулировке;
    - ответ даёт агенту готовый канонический текст, на который нужно опираться.
    """
    records: list[dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))

    documents: list[Document] = []

    for qa_id, item in enumerate(records):
        question = str(item.get("question_user", "")).strip()
        answer = str(item.get("answer_assistant", "")).strip()

        if not question or not answer:
            continue

        page_content = (
            f"Вопрос пользователя:\n{question}\n\n"
            f"Канонический ответ TrainBot:\n{answer}"
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": json_path.name,
                    "qa_id": qa_id,
                    "question_user": question,
                },
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Разбивает длинные Q&A-документы на фрагменты.

    Для коротких Q&A разбиение почти не сработает — документ останется целым.
    Для длинных клиентских кейсов разбиение полезно, потому что:
    - эмбеддинг длинного текста может хуже отражать конкретную суть;
    - retriever сможет вернуть более точный фрагмент ответа.

    В каждый фрагмент дополнительно добавляется исходный вопрос,
    чтобы не потерять контекст при разбиении длинного ответа.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []

    for document in documents:
        parts = splitter.split_text(document.page_content)

        for chunk_id, text in enumerate(parts):
            question = document.metadata.get("question_user", "")

            # Если ответ был разбит на несколько частей,
            # добавляем исходный вопрос к каждому чанку.
            chunk_text = (
                f"Исходный вопрос:\n{question}\n\n"
                f"Фрагмент базы знаний:\n{text}"
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_id": chunk_id,
                    },
                )
            )

    return chunks


def build_or_load_vector_store(
    json_path: Path = JSON_PATH,
    index_dir: Path = INDEX_DIR,
    rebuild: bool = False,
) -> FAISS:
    """
    Создаёт или загружает FAISS-векторное хранилище.

    Если index_dir уже существует и rebuild=False:
        загружается готовый индекс.

    Если index_dir нет или rebuild=True:
        JSON читается заново, документы индексируются, индекс сохраняется на диск.

    Важно:
    allow_dangerous_deserialization=True безопасно использовать только для индекса,
    который вы сами создали локально и которому доверяете.
    """
    print('Загрузка эмбэдинга ...')

    embeddings = HuggingFaceEmbeddings(
        cache_folder="./models/embeddings",
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={"normalize_embeddings": True},
    )
    print('Модель эмбэдинга загружена.')

    if index_dir.exists() and not rebuild:
        print('Загрузка сохранённого векторного хранилища.')
        return FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    source_documents = load_trainbot_qa(json_path)
    processed_documents = split_documents(source_documents)

    vector_store = FAISS.from_documents(
        documents=processed_documents,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )

    vector_store.save_local(str(index_dir))
    return vector_store


class RetrieverFAQ(Tool):
    """
    Инструмент поиска по базе знаний TrainBot.

    Этот Tool будет доступен агенту.
    Агент сможет вызвать его с текстовым запросом, получить несколько
    релевантных Q&A-фрагментов и на их основе сформировать ответ пользователю.
    """

    name = "retriever_fag"

    description = """
    Ищет релевантную информацию в базе знаний TrainBot.
    Используй этот инструмент, когда нужно ответить на вопрос о TrainBot,
    подписках, бронировании, отслеживании, GrandTrain, РЖД, балансе,
    пассажирах, возвратах, уведомлениях или настройках бота.

    Лучше формулировать запрос не как вопрос, а как короткое утверждение
    с ключевыми словами. Например:
    "возврат денежных средств баланс ID пользователя",
    "не удалось заброниовать билеты на Гранд Сервис",
    "не отображается список пассажиров тест аккаунта РЖД".
    """

    inputs = {
        "query": {
            "type": "string",
            "description": "Поисковый запрос к базе знаний TrainBot.",
        }
    }

    output_type = "string"

    def __init__(self, vector_store: FAISS, k: int = 5, **kwargs):
        """
        Args:
            vector_store:
                Готовое FAISS-векторное хранилище.
            k:
                Количество фрагментов, которые нужно вернуть агенту.
        """
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.k = k

    def forward(self, query: str) -> str:
        """
        Выполняет семантический поиск по FAISS и возвращает найденные документы.

        Args:
            query:
                Текстовый поисковый запрос.

        Returns:
            Строка с найденными фрагментами базы знаний.
        """
        if not isinstance(query, str) or not query.strip():
            return "Ошибка: поисковый запрос должен быть непустой строкой."

        docs_with_scores = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.k,
        )

        if not docs_with_scores:
            return "По базе знаний TrainBot ничего не найдено."

        result_parts = ["Найденные фрагменты базы знаний TrainBot:"]

        for index, (doc, score) in enumerate(docs_with_scores, start=1):
            qa_id = doc.metadata.get("qa_id")
            chunk_id = doc.metadata.get("chunk_id")
            source = doc.metadata.get("source")

            result_parts.append(
                f"""
                ===== Фрагмент {index} =====
                source: {source}
                qa_id: {qa_id}
                chunk_id: {chunk_id}
                score: {score}
                
                {doc.page_content}
                """.strip()
            )

        return "\n\n".join(result_parts)


def build_trainbot_agent(rebuild_index: bool = False) -> CodeAgent:
    """
    Создаёт RAG-агента TrainBot.

    Агент получает один инструмент:
    - retriever_fag

    В системных инструкциях агенту явно запрещается придумывать ответы,
    если в базе знаний нет подходящей информации.
    """
    vector_store = build_or_load_vector_store(rebuild=rebuild_index)
    print('FAISS-векторное хранилище создано.')

    retriever_faq = RetrieverFAQ(vector_store=vector_store, k=3)

    model = OpenAIModel(
        model_id="qwen2.5:7b",
        api_base="http://188.116.172.185:11434/v1",
        api_key="ollama",
        temperature=0.0,
        max_tokens=500,
    )

    agent = CodeAgent(
        tools=[retriever_faq],
        model=model,
        planning_interval=3,
        max_steps=5,
        verbosity_level=2,
        instructions="""
        Ты — RAG-ассистент поддержки TrainBot.
        
        Правила:
        1. Перед ответом на вопрос о TrainBot всегда используй инструмент retriever_fag.
        2. Отвечай только на основе найденных фрагментов базы знаний.
        3. Если найденные фрагменты не содержат ответа, честно скажи:
           "В базе знаний TrainBot нет точной информации по этому вопросу."
        4. Не выдумывай тарифы, статусы подписок, правила РЖД, сроки возврата и технические детали.
        5. Отвечай на русском языке.
        6. Для пользовательской поддержки формулируй ответ вежливо и кратко.
        7. Если для решения проблемы нужен ID пользователя, попроси указать ID и объясни,
           где его найти: меню бота «Инфо.» → «Тарифы» → «ID».
        """,
            )

    return agent


if __name__ == "__main__":
    # При первом запуске можно поставить rebuild_index=True,
    # чтобы принудительно создать FAISS-индекс из JSON.
    agent = build_trainbot_agent(rebuild_index=False)

    question = "Как вывести деньги со счёта TrainBot?"
    # question = "Почему я не вижу список пассажиров в боте?"

    answer_msg = agent.run(question)

    print("\nОтвет агента:")
    print(answer_msg)