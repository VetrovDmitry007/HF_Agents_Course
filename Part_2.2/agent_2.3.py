"""
Четыре вида инструментов
Глава: https://huggingface.co/learn/agents-course/unit2/llama-index/tools

Установка инструментария google
pip install llama-index-tools-google
"""
import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.google import GmailToolSpec
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")


# 1. Инструмент FunctionTool -- инструменты на основе произвольных функций
def get_info_user(user_id: int) -> str:
    """Получение информации о заданном пользователе."""
    dc_info = {'user_id': 123456, 'name': 'Иванов Иван', 'balans': 85.06}
    print(f"Информация о пользователе {user_id}: {dc_info}")
    return f"Информация о пользователе {user_id}: {dc_info}"


info_user_tool = FunctionTool.from_defaults(
    get_info_user,
    name="get_info_user",
    description="Получение информации о заданном пользователе.",
)
info_user_tool.call(123456)

# 2. Инструмент QueryEngineTool -- Инструменты использующие механизмы обработки запросов
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                   cache_folder="./models/embeddings",
                                   )

db = chromadb.PersistentClient(path="./trainbot_chroma_db")
chroma_collection = db.get_or_create_collection("trainbot")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
                              temperature=0.1,
                              max_tokens=100,
                              token=hf_token,
                              provider="auto"
                              )
query_engine = index.as_query_engine(llm=llm)
trainbot_rag_tool = QueryEngineTool.from_defaults(query_engine,
                                                  name="trainbot_rag_tool",
                                                  description="RAG конвейер поиска информации о TrainBot в базе FAQ")

# 3. Специфические инструменты -- инструменты сообщества созданные для конкретных сервисов
# Загрузка спецификации инструментов Google и преобразование её в список инструментов
tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()
ls_spec_tool = [(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
print(ls_spec_tool)

# 4. Вспомогательные инструменты -- Специальные инструменты, помогающие обрабатывать большие объемы данных,
#    поступающих из других инструментов.
"""
4.1 OnDemandToolLoader -- преобразует загрузчик в инструмент загрузки

4.2 LoadAndSearchToolSpec -- в качестве входных данных любой существующий инструмент, 
    возвращаются два инструмента: инструмент загрузки и инструмент поиска
"""

