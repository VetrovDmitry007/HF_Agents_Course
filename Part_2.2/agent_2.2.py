"""
Создание обычного RAG-пайплайна (НЕ RAG-Agent) с использованием компонентов LlamaIndex
Здесь нет Tool/Action/Observation-цикла.
Есть: документы → chunks → embeddings → Chroma → поиск похожих фрагментов → LLM → ответ

Установка векторного хранилища
pip install llama-index-vector-stores-chroma

Принцип работы (если Chroma пустая)
---------------
6. Загружаются документы из ./data/qa
   ↓
7. Документы разбиваются на chunks через SentenceSplitter
   ↓
8. Для chunks считаются embeddings
   ↓
9. Chunks + embeddings записываются в Chroma
   ↓
10. Создаётся индекс из Chroma
   ↓
11. Создаётся query_engine
   ↓
12. Пользовательский вопрос → поиск похожих chunks → LLM формирует ответ
"""
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import chromadb
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                   cache_folder="./models/embeddings",
                                   )

# 1. Создание/открытие векторного хранилища
db = chromadb.PersistentClient(path="./trainbot_chroma_db")
chroma_collection = db.get_or_create_collection("trainbot")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 2. Разбивка и преобразование документов в эмбэддинги
if chroma_collection.count() == 0:

    # 3. Загрузка документов
    reader = SimpleDirectoryReader(input_dir="./data/qa")
    documents = reader.load_data()

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
            ),
            embed_model,
        ],
        vector_store=vector_store,
    )
    # Выполняет весь ingestion-процесс и записывает результат в vector_store
    nodes = pipeline.run(documents=documents)

    print("Документов загружено:", len(documents))
    print("Nodes создано:", len(nodes))
    print("Записей в Chroma:", chroma_collection.count())
else:
    print("Chroma уже заполнена. Используем готовый индекс.")
    print("Записей в Chroma:", chroma_collection.count())

# 4. Индекс векторного хранилища
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# 5. Преобразование индекса в интерфейс запросов QueryEngine
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.1,
    max_tokens=100,
    token=hf_token,
    provider="auto"
)
# response_mode -- стратегия обработки ответа (refine, compact, tree_summarize)
query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")

response = query_engine.query("Как вывести деньги со счёта TrainBot?")
print(response)
"""
Вывести деньги со счёта в TrainBot напрямую невозможно. Средства можно использовать только для покупки билетов. 
Если у вас остались вопросы или вам нужна дополнительная помощь, рекомендую обратиться в поддержку сервиса.
"""

# 6. Оценка и наблюдаемость при помощи встроенных инструментов оценки качества ответов
# FaithfulnessEvaluator -- Оценивает достоверность ответа, проверяя, подтверждается ли он контекстом
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(llm=llm)
eval_result = evaluator.evaluate_response(response=response)
print('Evaluator: ', eval_result.passing)
"""
Evaluator: False
"""
