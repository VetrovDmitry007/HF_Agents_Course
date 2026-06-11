"""
Модуль verified_answer_memory.py

Хранилище проверенных прошлых ответов RAG-агента.

Идея:
1. Если агент дал ответ.
2. Ответ прошёл RelevancyEvaluator.
3. Ответ прошёл FaithfulnessEvaluator.
4. Тогда этот ответ можно сохранить в отдельное Chroma-хранилище.
5. При следующих похожих вопросах можно сначала искать здесь,
   чтобы не запускать полный RAG-пайплайн.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class VerifiedAnswerMemory:
    """
    Векторное хранилище проверенных ответов агента.

    Хранит только те ответы, которые прошли проверки:
    - RelevancyEvaluator
    - FaithfulnessEvaluator

    На первом этапе этот класс можно использовать отдельно от агента:
    - создать пустое хранилище;
    - сохранить тестовый ответ;
    - вывести содержимое;
    - проверить поиск.
    """

    def __init__(
            self,
            persist_dir: str | Path = "./answer_memory_chroma_db",
            collection_name: str = "verified_answers",
            embed_model: HuggingFaceEmbedding | None = None,
    ) -> None:
        """
        Инициализирует подключение к Chroma-хранилищу.

        Args:
            persist_dir:
                Папка, где Chroma будет хранить данные.

            collection_name:
                Название коллекции с проверенными ответами.

            embed_model:
                Модель эмбеддингов. Лучше передавать ту же модель,
                которая используется в основном RAG-агенте.
        """

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        self.embed_model = embed_model

    def create_empty_storage(self, recreate: bool = False) -> None:
        """
        Создаёт пустое хранилище.

        Args:
            recreate:
                Если False — просто создаёт коллекцию, если её ещё нет.
                Если True — удаляет старую коллекцию и создаёт новую пустую.

        Важно:
            recreate=True полностью удалит все ранее сохранённые ответы.
        """

        if recreate:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                # Если коллекции ещё нет — это не ошибка.
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        print(
            f"Пустое хранилище готово: "
            f"path='{self.persist_dir}', collection='{self.collection_name}'"
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Получает embedding для текста."""

        if self.embed_model is None:
            raise ValueError(
                "embed_model не передан. "
                "Передай HuggingFaceEmbedding в VerifiedAnswerMemory."
            )

        return self.embed_model.get_text_embedding(text)

    def save_answer(
                    self,
                    query: str,
                    answer: str,
                    relevance_score: float | None = None,
                    faithfulness_score: float | None = None,
                    ) -> str:
        """  Сохраняет проверенный ответ.

        Args:
            query:
                Вопрос пользователя.

            answer:
                Ответ агента, который прошёл проверки.

            relevance_score:
                Оценка RelevancyEvaluator.

            faithfulness_score:
                Оценка FaithfulnessEvaluator.

        Returns:
            ID сохранённой записи.
        """

        answer_id = str(uuid.uuid4())
        embedding = self._get_embedding(query)

        metadata = {
            "query": query,
            "answer": answer,
            "relevance_score": relevance_score,
            "faithfulness_score": faithfulness_score,
            "status": "passed",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "usage_count": 0, # счётчик использования
        }

        self.collection.add(
            ids=[answer_id],
            documents=[query],
            embeddings=[embedding],
            metadatas=[metadata],
            )

        return answer_id

    def search(
            self,
            query: str,
            top_k: int = 3,
            ) -> list[dict[str, Any]]:
        """
        Ищет похожие проверенные ответы.

        Пока это базовая версия поиска.
        Позже сюда можно добавить:
        - порог similarity;
        - проверку RelevancyEvaluator;
        - увеличение usage_count;
        - фильтрацию по source_tool;
        - фильтрацию по дате.
        """

        query_embedding = self._get_embedding(query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        found_items: list[dict[str, Any]] = []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for item_id, document, metadata, distance in zip(
                ids,
                documents,
                metadatas,
                distances,
        ):
            found_items.append(
                {
                    "id": item_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return found_items

    def mark_used(self, answer_id: str) -> None:
        """
        Отмечает, что сохранённый ответ был использован повторно.

        Увеличивает usage_count и записывает last_used_at.

        Args:
            answer_id:
                ID записи в Chroma-хранилище.
        """

        result = self.collection.get(
            ids=[answer_id],
            include=["metadatas"],
        )

        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])

        if not ids or not metadatas:
            print(f"Ответ с id='{answer_id}' не найден в VerifiedAnswerMemory.")
            return

        metadata = metadatas[0] or {}

        current_usage_count = metadata.get("usage_count", 0)

        try:
            current_usage_count = int(current_usage_count)
        except (TypeError, ValueError):
            current_usage_count = 0

        metadata["usage_count"] = current_usage_count + 1
        metadata["last_used_at"] = datetime.now().isoformat(timespec="seconds")

        self.collection.update(
            ids=[answer_id],
            metadatas=[metadata],
        )

    def get_best_answer(
            self,
            query: str,
            max_distance: float = 0.25,
            ) -> dict[str, Any] | None:
        """ Получения лучшего ответа.

        max_distance:
            Чем меньше distance, тем ближе найденный ответ.
            Точный порог нужно будет подобрать экспериментально.

        Returns:
            dict с найденным ответом или None.
        """

        results = self.search(query=query, top_k=1)

        if not results:
            return None

        best = results[0]

        if best["distance"] > max_distance:
            return None

        metadata = best.get("metadata") or {}

        return {
            "id": best["id"],
            "answer": metadata.get("answer", ""),
            "query": metadata.get("query", ""),
            "distance": best["distance"],
            "metadata": metadata,
        }

    def print_storage_contents(self, limit: int = 20) -> None:
        """
        Выводит содержимое хранилища в консоль.

        Удобно для отладки:
        - проверить, что хранилище создано;
        - посмотреть, какие ответы сохранены;
        - проверить metadata.
        """

        result = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )

        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not ids:
            print("Хранилище проверенных ответов пустое.")
            return

        print("=" * 100)
        print(f"Содержимое хранилища: {self.collection_name}")
        print(f"Показано записей: {len(ids)}")
        print("=" * 100)

        for index, item_id in enumerate(ids, start=1):
            metadata = metadatas[index - 1] or {}
            document = documents[index - 1] or ""

            print(f"\n[{index}] ID: {item_id}")
            print("-" * 100)
            print(f"Вопрос: {metadata.get('query', '')}")
            print(f"Ответ: {metadata.get('answer', '')}")
            print(f"Relevancy score: {metadata.get('relevance_score')}")
            print(f"Faithfulness score: {metadata.get('faithfulness_score')}")
            print(f"Источник: {metadata.get('source_tool', '')}")
            print(f"Дата создания: {metadata.get('created_at', '')}")
            print(f"Использований: {metadata.get('usage_count', 0)}")

            print("\nТекст документа для embedding:")
            print(document[:1000])

            if len(document) > 1000:
                print("...")

            print("-" * 100)

    def count(self) -> int:
        """
        Возвращает количество записей в хранилище.
        """

        return self.collection.count()

def debug_vam():
    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    EMBEDDINGS_DIR = PROJECT_ROOT / "models" / "embeddings"

    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME,
                                       cache_folder=EMBEDDINGS_DIR
                                       )

    vam = VerifiedAnswerMemory(persist_dir=PROJECT_ROOT / "answer_memory_chroma_db",
                               embed_model=embed_model)
    vam.create_empty_storage(recreate=True)
    vam.print_storage_contents()


if __name__ == '__main__':
    debug_vam()
