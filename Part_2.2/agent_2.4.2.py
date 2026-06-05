"""
Создание RAG-агента с одним retriever-инструментом
https://huggingface.co/learn/agents-course/unit2/llama-index/agents#creating-rag-agents-with-queryenginetools

При использовании QueryEngineTool может появиться дополнительная генерация внутри query engine:
Agent LLM → index.as_query_engine() LLM → QueryEngineTool → Agent LLM

Если return_direct=False, агент может переписать ответ query_engine или вызвать другой инструмент.
Если установить return_direct=True, ответ query_engine будет возвращён напрямую, а цикл агента завершится.
-------------------------

Агентность минимальная, потому что:

- инструмент только один;
- использование инструмента обязательно;
- нет выбора между разными инструментами;
- нет проверки качества найденных фрагментов;
- нет rerank;
- нет гибридного поиска BM25 + vector;
- нет автоматического повторного поиска, если первый запрос слабый;
- нет отдельного classifier intent;
- нет memory;
- нет набора регрессионных тестов.

"""
import asyncio
import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import AgentWorkflow, ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

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
                              max_tokens=300,
                              token=hf_token,
                              provider="auto"
                              )

# Интерфейс поиска со стратегией обработки ответа "tree_summarize"
query_engine = index.as_query_engine(llm=llm,
                                     similarity_top_k=5,
                                     response_mode="tree_summarize")

trainbot_rag_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="trainbot_rag_tool",
    description=(
        "Ищет релевантные фрагменты в базе знаний TrainBot. "
        "Используй для вопросов о TrainBot, подписках, бронировании, "
        "отслеживании, балансе, возвратах, пассажирах, GrandTrain и РЖД. "
        "В query передавай исходный вопрос пользователя полностью и добавляй ключевые слова. "
        "Не используй слишком короткие запросы вроде 'список пассажиров бот'."
    ),
    return_direct= False, # !!! Агент может переписать ответ query engine
)

system_prompt = """
        Ты — RAG-ассистент поддержки TrainBot.
        
        Правила:
        1. Перед ответом на вопрос о TrainBot всегда используй инструмент trainbot_rag_tool.
        2. При вызове trainbot_rag_tool НЕ сокращай вопрос до 2-3 общих слов.
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
        """

react_agent = ReActAgent(
    name="trainbot_agent",
    description="RAG-ассистент поддержки TrainBot",
    tools=[trainbot_rag_tool],
    llm=llm,
    system_prompt=system_prompt,
    streaming=False,
)

# Оркестратор, для управления системой из одного или нескольких агентов
agent = AgentWorkflow(
    agents=[react_agent],
    root_agent="trainbot_agent",
)


async def run_agent(agent):
    answer = await agent.run(user_msg="Как вывести деньги со счёта TrainBot?")
    # answer = await agent.run(user_msg="Почему я не вижу список пассажиров в боте?")
    """
    Чтобы вывести деньги со счёта в TrainBot, нужно оформить заявку через форму обратной связи на сайте trainbot.ru.
     В заявке необходимо указать ваш ID пользователя, который можно найти в боте в разделе
     «Информация» → «Тарифы» → «ID». После проверки запроса остаток средств будет возвращён на вашу карту
     за вычетом комиссии эквайринга.
    """
    return answer



if __name__ == '__main__':
    res = asyncio.run(run_agent(agent))
    print(res)
