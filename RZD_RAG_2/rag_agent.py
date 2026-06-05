"""
Модуль реализует RAG-агента поддержки TrainBot на базе LlamaIndex.

Агент использует два источника знаний: краткую Q&A-базу поддержки TrainBot и PDF-документацию с подробными инструкциями
по работе сервиса. Оба источника подключаются как Chroma-векторные хранилища с эмбеддингами HuggingFace.

Для выбора подходящего источника используется RouterQueryEngine с LLMSingleSelector: типовые пользовательские обращения
направляются в Q&A-базу, а вопросы, требующие пошаговых инструкций, — в PDF-документацию. Поверх роутера создан
ReActAgent, который всегда обращается к главному поисковому инструменту main_engine_tool и формирует краткий ответ
на русском языке только на основе найденных фрагментов.

Ответ агента дополнительно проверяется в FaithfulRAGWorkflow с помощью FaithfulnessEvaluator. Если ответ недостаточно
подтверждён найденными источниками, workflow повторно запускает агента с обратной связью от evaluator.
"""

import asyncio
import os
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

from RZD_RAG_2.rag_workflows import FaithfulRAGWorkflow

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "models" / "embeddings"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                   cache_folder=EMBEDDINGS_DIR,
                                   )

db = chromadb.PersistentClient(path="./trainbot_chroma_db")
chroma_collection = db.get_or_create_collection("trainbot")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

db_pdf = chromadb.PersistentClient(path="./rzd_pdf_chroma_db")
chroma_collection_pdf = db_pdf.get_or_create_collection("rzd_pdf")
vector_store_pdf = ChromaVectorStore(chroma_collection=chroma_collection_pdf)
index_pdf = VectorStoreIndex.from_vector_store(vector_store_pdf, embed_model=embed_model)


selector_llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.1,
    max_tokens=128,
    token=hf_token,
    provider="auto",
)

answer_llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.1,
    max_tokens=900,
    token=hf_token,
    provider="auto",
)

# Интерфейс поиска со стратегией обработки ответа "tree_summarize"
query_engine = index.as_query_engine(llm=answer_llm,
                                     similarity_top_k=5,
                                     response_mode="tree_summarize")

query_engine_pdf = index_pdf.as_query_engine(llm=answer_llm,
                                             similarity_top_k=3,
                                             response_mode="compact",)

# Отладка
# response = query_engine_pdf.query("Как оформить подписку?")
# for node in response.source_nodes:
#     print("=" * 80)
#     print(node.node.get_content()[:1000])
#

rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="rag_tool",
    description=(
        "Короткая база знаний поддержки TrainBot на основе Q&A и канонических ответов. "
        "Используй в первую очередь для практических пользовательских обращений: "
        "возврат средств, ID пользователя, подписки, бронь, уведомления, пассажиры, "
        "аккаунт РЖД, ошибки бота, тарифы и типовые проблемы. "
        "Лучше подходит для точного краткого ответа в стиле службы поддержки."
    ),
)

pdf_document_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine_pdf,
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

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=selector_llm),
    query_engine_tools=[
        rag_tool,
        pdf_document_tool,
        ],
    llm=answer_llm,
    verbose=True,
    )

main_engine_tool = QueryEngineTool.from_defaults(
    query_engine=router_query_engine,
    name="main_engine_tool",
    description=(
        "Главный поисковый инструмент TrainBot. "
        "Сам выбирает подходящий источник: короткую Q&A-базу поддержки или PDF-документацию. "
        "Используй для любого вопроса о TrainBot, РЖД-аккаунте, подписках, бронировании, "
        "возвратах, пассажирах, уведомлениях, тарифах и ошибках. "
        "В запрос передавай полный вопрос пользователя и 3-7 смысловых ключевых слов."
    ),
)

system_prompt = """
        Ты — RAG-ассистент поддержки TrainBot.

        Правила:
        1. Перед ответом на вопрос о TrainBot всегда используй инструмент main_engine_tool.
        2. При вызове main_engine_tool НЕ сокращай вопрос до 2-3 общих слов.
        3. В поисковый запрос передавай:
           - исходный вопрос пользователя полностью;
           - затем добавь 3-7 ключевых слов по смыслу.
        4. Пример плохого запроса: "список пассажиров бот".
        5. Пример хорошего запроса:
           "Почему я не вижу список пассажиров в боте? не отображается список пассажиров аккаунт РЖД тест аккаунта обновить"
        6. Отвечай только на основе найденных фрагментов базы знаний.
        7. Если найденные фрагменты не содержат ответа, честно скажи:
           "В базе знаний TrainBot нет точной информации по этому вопросу."
        8. Не выдумывай тарифы, статусы подписок, правила РЖД, сроки возврата и технические детали.
        9. Отвечай на русском языке.
        10. Для пользовательской поддержки формулируй ответ вежливо и кратко.
        11. Если для решения проблемы нужен ID пользователя, попроси указать ID и объясни,
            где его найти: меню бота «Инфо.» → «Тарифы» → «ID».
        12. Если ответ содержит пошаговую инструкцию, дай её кратко: обычно 3–5 пунктов.
        13. Не обрывай ответ на середине предложения. Лучше дай более короткий, но завершённый ответ.
        14. Не смешивай инструкции из разных разделов в одном ответе.
        15. Не добавляй общие рекомендации вроде «обратитесь в поддержку», если в найденном контексте есть конкретная инструкция.
        16. Если в источнике несколько близких шагов, объединяй их без потери смысла.
        17. Не расписывай очевидные подшаги отдельно, если их можно объединить.    
        """

rag_agent = ReActAgent(
    name="rag_agent",
    description="RAG-ассистент поддержки TrainBot",
    tools=[main_engine_tool],
    llm=answer_llm,
    system_prompt=system_prompt,
    streaming=False,
)

workflow = FaithfulRAGWorkflow(
    agent=rag_agent,
    evaluator_llm=answer_llm,
    max_retries=2,
    timeout=120,
)


async def run_workflow():
    result = await workflow.run(
        # query="Как вернуть денежные средства со счёта TrainBot?"
        query="Как оформить подписку бронирования?"
    )
    print(result)


if __name__ == '__main__':
    asyncio.run(run_workflow())
