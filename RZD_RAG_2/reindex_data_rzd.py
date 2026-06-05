"""
Пересоздаёт Chroma так, чтобы каждая пара question_user + answer_assistant была отдельным документом.
"""
import json
import os

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                   cache_folder="./models/embeddings",
                                   )

db = chromadb.PersistentClient(path="./trainbot_chroma_db")

# ВАЖНО: удалить старую коллекцию, если она уже была создана крупными JSON-чанками
try:
    db.delete_collection("trainbot")
except Exception:
    pass

chroma_collection = db.get_or_create_collection("trainbot")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

with open("./data/qa/trainbot_rag_qa_updated.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

documents = []

for i, item in enumerate(faq_data):
    question = str(item.get("question_user", "")).strip()
    answer = str(item.get("answer_assistant", "")).strip()

    if not question or not answer:
        continue

    text = (
        f"Вопрос пользователя:\n{question}\n\n"
        f"Канонический ответ TrainBot:\n{answer}"
    )

    documents.append(
        Document(
            text=text,
            metadata={
                "qa_id": i,
                "question_user": question,
            }
        )
    )

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)