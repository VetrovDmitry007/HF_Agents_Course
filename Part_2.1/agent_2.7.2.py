"""
Агенты поиска.

Agentic RAG -- CodeAgent получает пользовательский вопрос, при необходимости вызывает инструмент поиска по базе знаний,
получает найденные фрагменты и на их основе формирует ответ.

Агент может сам формулировать поисковые запросы, делать несколько retrieval-запросов, рассуждать по найденным данным
и уточнять поиск при необходимости

BM25Retriever -- это лексический поиск, то есть поиск по совпадению слов и близких терминов, а не семантический поиск через embeddings.
Document (langchain_community) -- это стандартный объект LangChain для хранения текста и метаданных.
RecursiveCharacterTextSplitter -- это компонент, который разбивает длинные документы на небольшие фрагменты

pip install rank_bm25
"""
import os
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, OpenAIModel, Tool
from dotenv import load_dotenv

load_dotenv()


class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = ("Использует лексический поиск для поиска соответствующих идей по планированию вечеринки Альфреда "
                   "в супер-геройском стиле в поместье Уэйнов.")
    inputs = {
        "query": {
            "type": "string",
            "description": "Запрос для выполнения. Это должен быть запрос, связанный с планированием вечеринки или"
                           " темой супергероев.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # Строиться поисковый индекс по документам и возвращать 5 наиболее релевантных фрагментов
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Ваш поисковый запрос должен быть строкой"

        # происходит поиск по базе знаний
        docs = self.retriever.invoke(query)
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Смоделировать базу знаний о планировании вечеринок
party_ideas = [
    {"text": "Бал-маскарад в стиле супергероев с роскошным декором, включая золотые акценты и бархатные шторы.", "source": "Идеи для вечеринки 1"},
    {"text": "Наймите профессионального диджея, который сможет играть тематическую музыку для таких супергероев, как Бэтмен и Чудо-женщина.", "source": "Идеи для развлечений"},
    {"text": "В качестве кейтеринга подавайте блюда, названные в честь супергероев, например «Зеленый коктейль Халка» и «Сильный стейк Железного человека».'", "source": "Идеи общественного питания"},
    {"text": "Украсьте место проведения знаковыми логотипами супергероев и проекциями Готэма и других городов-супергероев.", "source": "Идеи украшения"},
    {"text": "Интерактивные впечатления от виртуальной реальности, где гости могут участвовать в симуляциях супергероев или соревноваться в тематических играх.", "source": "Идеи для развлечений"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Разделите документы на более мелкие части для более эффективного поиска.
# Размер документа 500 символов с перекрытием 50 символов
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

# Создайте инструмент retriever
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

model = OpenAIModel(
    model_id="poolside/laguna-m.1:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OR_TOKEN"],
)

agent = CodeAgent(tools=[party_planning_retriever], model=model)

response = agent.run(
    "Найдите идеи для роскошной вечеринки в стиле супергероев, включая развлечения, питание и варианты оформления."
)

print(response)