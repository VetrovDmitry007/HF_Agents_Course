"""
Модуль реализует RAG-агента поддержки TrainBot на базе LlamaIndex.

pip install llama-index-llms-ollama
pip install llama-index-llms-openai-like
"""

import asyncio
from pathlib import Path
from typing import Any

import chromadb

from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config_models.open_router import answer_llm, selector_llm, normalizer_llm, evaluator_llm
# from config_models.groq import answer_llm, selector_llm, normalizer_llm, evaluator_llm
# from config_models.hf import answer_llm, selector_llm, normalizer_llm, evaluator_llm

from rag_workflows import FaithfulRAGWorkflow
from verified_answer_memory import VerifiedAnswerMemory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "models" / "embeddings"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TRAINBOT_CHROMA_PATH = "./trainbot_chroma_db"
TRAINBOT_COLLECTION_NAME = "trainbot"
PDF_CHROMA_PATH = "./rzd_pdf_chroma_db"
PDF_COLLECTION_NAME = "rzd_pdf"
ANSWER_MEMORY_PATH = PROJECT_ROOT / "answer_memory_chroma_db"
SIMILARITY_TOP_K = 3
RESPONSE_MODE = "compact"

MAIN_ENGINE_TOOL_DESCRIPTION = (
    "Главный поисковый инструмент TrainBot. "
    "Сам выбирает подходящий источник: короткую Q&A-базу поддержки или PDF-документацию. "
    "Используй для любого вопроса о TrainBot, РЖД-аккаунте, подписках, бронировании, "
    "возвратах, пассажирах, уведомлениях, тарифах и ошибках. "
    "В запрос передавай полный вопрос пользователя и 3-7 смысловых ключевых слов."
)


embed_model: HuggingFaceEmbedding | None = None
vam: VerifiedAnswerMemory | None = None
index: VectorStoreIndex | None = None
index_pdf: VectorStoreIndex | None = None
query_engine: Any | None = None
query_engine_pdf: Any | None = None
rag_tool: QueryEngineTool | None = None
pdf_document_tool: QueryEngineTool | None = None
router_query_engine: RouterQueryEngine | None = None
main_engine_query_tool: QueryEngineTool | None = None
main_engine_tool: FunctionTool | None = None
rag_agent: ReActAgent | None = None
normalizer_agent: ReActAgent | None = None
workflow: FaithfulRAGWorkflow | None = None


def build_embed_model() -> HuggingFaceEmbedding:
    """
    Создаёт модель эмбеддингов HuggingFace для всех векторных хранилищ проекта.

    Модель загружается по имени EMBEDDING_MODEL_NAME, а локальный кэш сохраняется
    в директории EMBEDDINGS_DIR.
    """
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDINGS_DIR,
    )


def build_verified_answer_memory(active_embed_model: HuggingFaceEmbedding) -> VerifiedAnswerMemory:
    """
    Создаёт хранилище проверенных ответов VerifiedAnswerMemory.

    Хранилище использует тот же embed_model, что и основные Chroma-индексы,
    чтобы поиск по сохранённым ответам работал в общем векторном пространстве.
    """
    return VerifiedAnswerMemory(
        persist_dir=ANSWER_MEMORY_PATH,
        embed_model=active_embed_model,
    )


def build_vector_index(
    chroma_path: str,
    collection_name: str,
    active_embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """
    Создаёт VectorStoreIndex поверх существующей Chroma-коллекции.

    Args:
        chroma_path: Путь к директории Chroma PersistentClient.
        collection_name: Имя коллекции Chroma.
        active_embed_model: Модель эмбеддингов для восстановления индекса.

    Returns:
        Индекс LlamaIndex, подключённый к указанной Chroma-коллекции.
    """
    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=active_embed_model)


def build_query_engine(active_index: VectorStoreIndex) -> Any:
    """
    Создаёт query_engine для поиска и генерации ответа по одному индексу.

    Используется общий LLM ответа, similarity_top_k=3 и response_mode="compact".
    """
    # Интерфейс поиска со стратегией обработки ответа "compact"
    return active_index.as_query_engine(
        llm=answer_llm,
        similarity_top_k=SIMILARITY_TOP_K,
        response_mode=RESPONSE_MODE,
    )


def build_source_tools(
    active_query_engine: Any,
    active_query_engine_pdf: Any,
) -> tuple[QueryEngineTool, QueryEngineTool]:
    """
    Создаёт два инструмента-источника для RouterQueryEngine.

    Первый инструмент работает с краткой Q&A-базой TrainBot, второй — с PDF-документацией.
    Описания инструментов используются LLM-селектором для выбора подходящего источника.
    """
    trainbot_rag_tool = QueryEngineTool.from_defaults(
        query_engine=active_query_engine,
        name="rag_tool",
        description=(
            "Короткая база знаний поддержки TrainBot на основе Q&A и канонических ответов. "
            "Используй в первую очередь для практических пользовательских обращений: "
            "возврат средств, ID пользователя, подписки, бронь, уведомления, пассажиры, "
            "аккаунт РЖД, ошибки бота, тарифы и типовые проблемы. "
            "Лучше подходит для точного краткого ответа в стиле службы поддержки."
        ),
    )

    trainbot_pdf_document_tool = QueryEngineTool.from_defaults(
        query_engine=active_query_engine_pdf,
        name="pdf_document_tool",
        description=(
            "PDF-документация TrainBot с подробными инструкциями по разделам сервиса. "
            "Используй для пошаговых инструкций. "
            "Важно: не смешивай разные разделы. "
            "Подписка «Бронирование», подписка «Удержание» и подписка «Резерв с отсрочкой» "
            "являются разными сценариями. "
            "Для вопроса про оформление подписки «Бронирование» используй раздел "
            "создания подписки «Бронирование», а не разделы про удержание или резерв."
        ),
    )

    return trainbot_rag_tool, trainbot_pdf_document_tool




def get_selected_selector(raw_response: Any) -> int | None:
    """Возвращает index выбранного источника RouterQueryEngine: 0=Q&A, 1=PDF."""

    metadata = getattr(raw_response, "metadata", None) or {}
    selector_result = metadata.get("selector_result")

    selections = getattr(selector_result, "selections", None)
    if not selections:
        return None

    return getattr(selections[0], "index", None)


def get_selected_source(raw_response: Any) -> str:
    """Возвращает имя выбранного источника RouterQueryEngine."""

    selected_index = get_selected_selector(raw_response)
    sources = {
        0: "rag_tool",
        1: "pdf_document_tool",
    }
    selector = sources.get(selected_index, "unknown")
    return selector


async def save_rag_context(ctx: Context, raw_response: Any, selected_source: str) -> None:
    """
    Сохраняет найденные контексты и служебную информацию в workflow Context.

    В Context помещаются выбранный источник, уникальные тексты source_nodes,
    сырой строковый ответ и количество найденных контекстов.
    """
    contexts = [
        source_node.node.get_content()
        for source_node in getattr(raw_response, "source_nodes", [])
    ]

    await ctx.store.set("rag_selected_source", selected_source)
    await ctx.store.set("rag_contexts", list(dict.fromkeys(contexts)))
    await ctx.store.set("rag_raw_response", str(raw_response) if raw_response else "")
    await ctx.store.set("rag_contexts_count", len(contexts))


def build_main_engine_search(active_router_query_engine: RouterQueryEngine):
    """
    Создаёт context-aware async-функцию поиска для FunctionTool.

    Функция вызывает RouterQueryEngine, печатает отладочную информацию и сохраняет
    source_nodes вместе с выбранным источником в workflow Context.
    """

    async def main_engine_search(ctx: Context, query: str) -> str:
        """
        Context-aware обёртка над router_query_engine.
        Сохраняет source_nodes и выбранный RouterQueryEngine источник в workflow Context.
        """
        print(f"\n=== main_engine_tool CALLED ===")
        print(f"query={query}")

        raw_response = await active_router_query_engine.aquery(query)

        print(f"raw_response type={type(raw_response)}")
        print(f"source_nodes count={len(getattr(raw_response, 'source_nodes', []))}")

        selected_source = get_selected_source(raw_response)
        print(f"selected_source={selected_source}")

        await save_rag_context(ctx, raw_response, selected_source)

        return str(raw_response)

    return main_engine_search



def get_system_prompt() -> str:
    """
    Возвращает системный промпт основного RAG-агента поддержки TrainBot.

    Промпт ограничивает агента найденными фрагментами, запрещает выдумывать детали
    и задаёт правила краткого ответа на русском языке.
    """
    return """
        Ты — RAG-ассистент поддержки TrainBot.

        Правила:
        1. Всегда используй main_engine_tool для вопросов о TrainBot.
        2. В инструмент передавай полный вопрос пользователя и 3–7 ключевых слов.
        3. Отвечай только по найденным фрагментам.
        4. Если точной информации нет, скажи: «В базе знаний TrainBot нет точной информации по этому вопросу».
        5. Не выдумывай тарифы, правила РЖД, статусы подписок, сроки возврата и технические детали.
        6. Если нужен ID пользователя, попроси ID и укажи путь: «Инфо.» → «Тарифы» → «ID».
        7. Отвечай кратко, вежливо, на русском языке.
        8. Не смешивай инструкции из разных разделов.
        9. Если источник содержит только близкую инструкцию, пиши: «В найденных материалах указано...».
        10. Не обрывай ответ на середине предложения. 
        11. Если пользователь явно не упоминает Гранд Сервис или Гранд, в таком случае считай вопрос общим вопросом по TrainBot/РЖД. 
        12. Не добавляй технические причины, которых нет в найденном ответе или source_nodes.
        """


def get_normalizer_prompt() -> str:
    """
    Возвращает системный промпт агента нормализации пользовательского вопроса.

    Нормализатор не отвечает пользователю, а только переписывает исходный текст
    в короткий самодостаточный поисковый запрос для RAG-системы.
    """
    return """
            Ты — агент нормализации пользовательских вопросов для RAG-системы TrainBot.
            
            Твоя задача — переписать исходный вопрос пользователя в короткий, точный и самодостаточный
            поисковый запрос для передачи RAG-агенту.
             
            Правила:
            
            1. Не отвечай на вопрос пользователя.
            2. Не добавляй факты, которых нет в исходном вопросе.
            3. Не объясняй свои действия.
            4. Не используй Markdown, списки и комментарии.
            5. Сохраняй важные детали: даты, ID, суммы, станции, номера поездов, типы подписок, статусы,
               ошибки, действия пользователя.
            6. Удаляй только лишние приветствия, эмоции, повторы и несущественные подробности.
            7. Если вопрос уже короткий и понятный, верни его почти без изменений.
            8. Итог должен быть в виде вопроса одной строкой.
            
            Запрещено давать рекомендации, инструкции или решение проблемы пользователя.
            Твоя задача — только переписать вопрос для поиска.
            Верни только нормализованный вопрос.
            """

def initialize_components() -> FaithfulRAGWorkflow:
    """
    Инициализирует все компоненты RAG-системы и сохраняет их в глобальные переменные.

    Функция сохраняет совместимость с исходным кодом: после вызова доступны переменные
    embed_model, vam, index, index_pdf, query_engine, query_engine_pdf, rag_tool,
    pdf_document_tool, router_query_engine, main_engine_query_tool, main_engine_tool,
    rag_agent, normalizer_agent и workflow.
    """
    global embed_model
    global vam
    global index
    global index_pdf
    global query_engine
    global query_engine_pdf
    global rag_tool
    global pdf_document_tool
    global router_query_engine
    global main_engine_query_tool
    global main_engine_tool
    global rag_agent
    global normalizer_agent
    global workflow

    embed_model = build_embed_model()
    vam = build_verified_answer_memory(embed_model)

    index = build_vector_index(
        chroma_path=TRAINBOT_CHROMA_PATH,
        collection_name=TRAINBOT_COLLECTION_NAME,
        active_embed_model=embed_model,
    )
    index_pdf = build_vector_index(
        chroma_path=PDF_CHROMA_PATH,
        collection_name=PDF_COLLECTION_NAME,
        active_embed_model=embed_model,
    )

    query_engine = build_query_engine(index)
    query_engine_pdf = build_query_engine(index_pdf)

    # Отладка
    # response = query_engine_pdf.query("Как оформить подписку?")
    # for node in response.source_nodes:
    #     print("=" * 80)
    #     print(node.node.get_content()[:1000])
    #

    rag_tool, pdf_document_tool = build_source_tools(
        active_query_engine=query_engine,
        active_query_engine_pdf=query_engine_pdf,
    )

    router_query_engine = RouterQueryEngine(
                                selector=LLMSingleSelector.from_defaults(llm=selector_llm),
                                query_engine_tools=[rag_tool, pdf_document_tool],
                                llm=answer_llm,
                                verbose=True,
                                )

    # Создаёт обычный QueryEngineTool поверх RouterQueryEngine.
    main_engine_query_tool = QueryEngineTool.from_defaults(
                                query_engine=router_query_engine,
                                name="main_engine_tool",
                                description=MAIN_ENGINE_TOOL_DESCRIPTION,
                                )

    # Создаёт FunctionTool main_engine_tool для ReActAgent
    main_engine_tool = FunctionTool.from_defaults(
                                        async_fn=build_main_engine_search(router_query_engine),
                                        name="main_engine_tool",
                                        description=MAIN_ENGINE_TOOL_DESCRIPTION,
                                        )

    # Создаёт основной ReActAgent для ответов пользователям TrainBot.
    rag_agent = ReActAgent(
                            name="rag_agent",
                            description="RAG-ассистент поддержки TrainBot",
                            tools=[main_engine_tool],
                            llm=answer_llm,
                            system_prompt=get_system_prompt(),
                            streaming=False,
                            )

    # Создаёт ReActAgent для нормализации пользовательских вопросов.
    normalizer_agent = ReActAgent(name="normalizer_agent",
                                  description="""Нормализует вопрос пользователя в короткий точный запрос для RAG, сохраняя ключевые
                                               детали и не добавляя новых фактов""",
                                  tools=[],
                                  llm=normalizer_llm,
                                  system_prompt=get_normalizer_prompt(),
                                  streaming=False,
                                  )

    # Создаёт FaithfulRAGWorkflow для нормализации, запуска агента и проверки ответа.
    workflow = FaithfulRAGWorkflow(
                                agent=rag_agent,
                                normalizer_agent=normalizer_agent,
                                normalizer_llm=normalizer_llm,
                                vam=vam,
                                evaluator_llm=evaluator_llm,
                                max_retries=1,
                                timeout=300,
                                )

    return workflow


workflow = initialize_components()


async def run_workflow(active_workflow: FaithfulRAGWorkflow | None = None) -> None:
    """
    Запускает workflow на тестовом пользовательском запросе и печатает результат.

    Тестовые варианты запросов сохранены из исходного кода в комментариях.
    Если workflow не передан явно, используется глобальный workflow, созданный при инициализации модуля.
    """
    current_workflow = active_workflow or workflow
    if current_workflow is None:
        raise RuntimeError("Workflow не инициализирован")

    result = await current_workflow.run(
        # query="Как вернуть деньги со счёта TrainBot?"
        # query="Как мне получит назад деньги"
        # query="Оплатил 1000 чтоб пополнить баланс. А баланс не пополняется."
        # query="Почему не видно списка пассажиров."
        # query="""Добрый день. Я вчера поставил подписку на бронирование, бот прислал уведомление,
        #            что места есть, но бронь не появилась. Деньги списались. Как мне теперь вернуть остаток?
        #         """
        # query = """
        #         Добрый день! Создал подписку, исправно выдает сообщения: 2026-07-19 16:35
        #         Керчь - Москва 464С: 249->242 мест. Но не бронирует! В чем дело?!
        #         """
        query='Почему не бронируется билеты'
        )
    print(result)


if __name__ == '__main__':
    asyncio.run(run_workflow())
