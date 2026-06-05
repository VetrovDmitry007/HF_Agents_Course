"""
Пересоздаёт Chroma-векторное хранилище из содержимого PDF-файла.

Отличие от reindex_data_rzd.py:
- исходные данные загружаются не из JSON, а из PDF;
- загрузка PDF выполняется средствами LlamaIndex через SimpleDirectoryReader;
- текст PDF дополнительно режется на чанки через SentenceSplitter;
- в Chroma сохраняются чанки с метаданными исходного файла.

pip install llama-index-readers-file pymupdf
"""

from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


MODULE_DIR = Path(__file__).resolve().parent

PDF_PATH = MODULE_DIR / "data" / "pdf" / "Описание_работы_с_TrainBot.pdf"
CHROMA_PATH = MODULE_DIR / "rzd_pdf_chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_CACHE_DIR = "./models/embeddings"


def load_pdf_with_llamaindex(pdf_path: Path):
    """
    Загружает PDF через LlamaIndex SimpleDirectoryReader.

    SimpleDirectoryReader сам выбирает подходящий reader по расширению файла.
    Для PDF обычно возвращаются документы, соответствующие страницам или частям файла,
    в зависимости от используемого PDF-reader внутри LlamaIndex.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF-файл не найден: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Ожидался PDF-файл, получен: {pdf_path}")

    reader = PyMuPDFReader()

    documents = reader.load_data(
        file_path=pdf_path,
        metadata=True,
        extra_info={
            "source_type": "pdf",
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
        },
    )

    for doc in documents:
        doc.metadata.update(
            {
                "source_type": "pdf",
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
            }
        )

    return documents


def recreate_chroma_collection(chroma_path: str, collection_name: str) -> ChromaVectorStore:
    """
    Удаляет старую коллекцию Chroma и создаёт новую.

    Это важно, чтобы при переиндексации не смешивались старые и новые чанки.
    """
    db = chromadb.PersistentClient(path=chroma_path)

    try:
        db.delete_collection(collection_name)
        print(f"Старая коллекция Chroma удалена: {collection_name}")
    except Exception:
        print(f"Коллекция {collection_name} ещё не существовала, удаление пропущено")

    chroma_collection = db.get_or_create_collection(collection_name)
    return ChromaVectorStore(chroma_collection=chroma_collection)


def main() -> None:
    """Основной сценарий переиндексации PDF в Chroma."""
    load_dotenv()

    pdf_path = Path(PDF_PATH).resolve()
    embedding_cache_dir = DEFAULT_EMBEDDING_CACHE_DIR
    chunk_size = 700
    chunk_overlap = 100
    chroma_path = CHROMA_PATH
    collection = 'rzd_pdf'

    print(f"Загрузка embedding-модели: {EMBEDDING_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=embedding_cache_dir,
    )

    print(f"Загрузка PDF через LlamaIndex: {pdf_path}")
    documents = load_pdf_with_llamaindex(pdf_path)

    if not documents:
        raise RuntimeError("LlamaIndex не извлёк документы из PDF. Проверь, что PDF содержит текстовый слой.")

    print(f"Загружено документов из PDF: {len(documents)}")

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    if not nodes:
        raise RuntimeError("После разбиения не получилось ни одного node/chunk.")

    print(f"Создано чанков для индексации: {len(nodes)}")

    vector_store = recreate_chroma_collection(
        chroma_path=chroma_path,
        collection_name=collection,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Создание VectorStoreIndex и сохранение данных в Chroma...")
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("Готово")
    print(f"Chroma path: {chroma_path}")
    print(f"Collection: {collection}")


if __name__ == "__main__":
    main()
