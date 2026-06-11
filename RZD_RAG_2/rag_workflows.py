from typing import Any
from pydantic import Field
from llama_index.core.agent.workflow import ReActAgent, ToolCallResult
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)

from RZD_RAG_2.query_normalizer import get_normalize_query
from RZD_RAG_2.verified_answer_memory import VerifiedAnswerMemory


class RunAgentEvent(Event):
    """Событие для запуска или повторного запуска агента."""

    original_query: str
    normalized_query: str
    search_query: str # запрос для конкретной попытки поиска
    attempt: int = 0
    evaluator_feedback: str = ""
    relevance_feedback: str = ""
    tried_sources: list[str] = Field(default_factory=list) # управляет когда остановить retry


class AgentAnswerEvent(Event):
    """Событие с ответом агента и использованными RAG-контекстами."""

    original_query: str
    normalized_query: str
    search_query: str # запрос для конкретной попытки поиска
    answer: str
    contexts: list[str]
    attempt: int
    selected_source: str = ""
    tried_sources: list[str] = Field(default_factory=list) # управляет когда остановить retry


class RetryAgentEvent(Event):
    """Событие, запускающее новую попытку после неудачной проверки."""

    original_query: str
    normalized_query: str
    search_query: str # запрос для конкретной попытки поиска
    attempt: int
    evaluator_feedback: str
    relevance_feedback: str
    tried_sources: list[str] = Field(default_factory=list) # управляет когда остановить retry


class FaithfulRAGWorkflow(Workflow):
    """Workflow, проверяющий ответ RAG-агента на подтверждённость источниками."""

    def __init__(
            self,
            agent: ReActAgent,
            normalizer_agent: ReActAgent,
            normalizer_llm,
            vam: VerifiedAnswerMemory,
            evaluator_llm: Any,
            max_retries: int = 1, # сколько повторных попыток workflow может сделать после первой попытки
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.evaluator_llm=evaluator_llm
        self.normalizer_llm = normalizer_llm
        self.vam = vam
        self.agent = agent
        self.normalizer_agent = normalizer_agent
        self.max_retries = max_retries
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.evaluator_llm)
        self.relevance_evaluator = RelevancyEvaluator(llm=self.evaluator_llm)

    @staticmethod
    def build_retry_search_query(normalized_query: str, previous_source: str) -> str:
        """
        Формирует search_query для следующей попытки.
        Не указывает имя инструмента напрямую.
        Только меняет поисковый ракурс для RouterQueryEngine.

        :param normalized_query: Нормализованный вопрос
        :param previous_source: Наименование источника данных
        """

        if previous_source == "rag_tool":
            return f"Найди в документации TrainBot информацию по вопросу: {normalized_query}"

        if previous_source == "pdf_document_tool":
            return f"Найди похожее обращение пользователя TrainBot по вопросу: {normalized_query}"

        return normalized_query

    @step
    async def start(self, ev: StartEvent) -> StopEvent | RunAgentEvent:
        """Преобразует входной запрос в событие запуска агента."""

        print(f'Step start, {ev=}')
        original_query = ev.query

        # Нормализация пользовательского вопроса
        normalized_query = await get_normalize_query(query=original_query, llm=self.normalizer_llm)
        print(f'{normalized_query=}')

        cached_answer = self.vam.get_best_answer(query=normalized_query, max_distance=0.15)
        if cached_answer is not None:
            self.vam.mark_used(cached_answer["id"])

        if cached_answer is not None:
            return StopEvent(
                result={
                    "answer": cached_answer["answer"],
                    "status": "cached",
                    "cache_hit": True,
                    "matched_query": cached_answer["query"],
                    "distance": cached_answer["distance"],
                    "source": "verified_answer_memory",
                }
            )

        return RunAgentEvent(original_query=original_query,
                             normalized_query=normalized_query,
                             search_query=normalized_query,
                             attempt=0,
                             tried_sources=[],)

    @step
    async def run_agent(self, ctx: Context, ev: RunAgentEvent) -> AgentAnswerEvent:
        """Запускает агента и получает контексты через workflow Context."""

        print(f"Step run_agent, {ev=}")

        await ctx.store.set("rag_contexts", [])
        await ctx.store.set("rag_contexts_count", 0)
        await ctx.store.set("rag_raw_response", "")
        await ctx.store.set("rag_selected_source", "unknown")

        agent_query = f"""
                    Вопрос пользователя:
                    {ev.original_query}
                
                    Поисковый запрос:
                    {ev.search_query}
                
                    Обязательно используй инструмент main_engine_tool.
                    В инструмент передай только текст из поля «Поисковый запрос».
                
                    После получения результата сформируй краткий ответ только на основе найденных фрагментов.
                    Если точной информации нет, скажи: «В базе знаний TrainBot нет точной информации по этому вопросу».
                    """

        agent_ctx = Context(self.agent)

        handler = self.agent.run(
            user_msg=agent_query,
            ctx=agent_ctx,
            max_iterations=3,
            early_stopping_method="generate",
        )

        agent_response = await handler
        answer = str(agent_response).strip()

        if not answer:
            raw_response = await agent_ctx.store.get("rag_raw_response", default="")
            answer = str(raw_response).strip()

        contexts = await agent_ctx.store.get("rag_contexts", default=[])
        contexts = list(dict.fromkeys(contexts))

        contexts_count = await agent_ctx.store.get("rag_contexts_count", default=0)
        print(f"Contexts from agent_ctx: {contexts_count}")

        selected_source = await agent_ctx.store.get("rag_selected_source", default="unknown")
        print(f"Selected source: {selected_source}")

        return AgentAnswerEvent(
            original_query=ev.original_query,
            normalized_query=ev.normalized_query,
            search_query=ev.search_query,
            answer=answer,
            contexts=contexts,
            attempt=ev.attempt,
            selected_source=selected_source,
            tried_sources=ev.tried_sources,
        )

    @step
    async def evaluate_answer(self, ev: AgentAnswerEvent) -> StopEvent | RetryAgentEvent:
        """Проверяет ответ: сначала релевантность, затем подтверждённость источниками."""

        fh_passed = False
        rl_passed = False
        fh_checked = False
        fh_score = None
        rl_score = None
        fh_feedback = ""

        print(f"Step evaluate_answer, {ev=}")

        # Блок управления когда остановить retry, если оба источника уже проверены
        known_sources = {"rag_tool", "pdf_document_tool"}
        tried_sources = list(ev.tried_sources)

        if ev.selected_source in known_sources and ev.selected_source not in tried_sources:
            tried_sources.append(ev.selected_source)

        both_sources_checked = known_sources.issubset(set(tried_sources))
        print(f"Tried sources: {tried_sources}")

        if not ev.contexts:
            rl_feedback = (
                "Агент не получил ни одного source_node из RAG-инструмента. "
                "Ответ нельзя проверить на релевантность и подтверждённость источникам."
            )
        else:
            relevance = await self.relevance_evaluator.aevaluate(
                query=ev.original_query,
                response=ev.answer,
                contexts=ev.contexts,
            )
            rl_passed = bool(relevance.passing)
            rl_score = relevance.score
            rl_feedback = relevance.feedback or ""

            print(f"Результат проверки RelevancyEvaluator: {rl_passed=}, {rl_score=}")

            if rl_passed:
                faithfulness = await self.faithfulness_evaluator.aevaluate(
                    query=ev.original_query,
                    response=ev.answer,
                    contexts=ev.contexts,
                )
                fh_checked = True
                fh_passed = bool(faithfulness.passing)
                fh_score = faithfulness.score
                fh_feedback = faithfulness.feedback or ""

                print(f"Результат проверки FaithfulnessEvaluator: {fh_passed=}, {fh_score=}")

        # Блок успешного ответа
        if rl_passed and fh_passed:
            self.vam.save_answer(
                query=ev.normalized_query,
                answer=ev.answer,
                relevance_score=rl_score,
                faithfulness_score=fh_score,
            )

            return StopEvent(
                result={
                    "answer": ev.answer,
                    "faithfulness_score": fh_score,
                    "relevance_score": rl_score,
                    "attempts": ev.attempt + 1,
                    "status": "passed",
                    "cache_hit": False,
                    "selected_source": ev.selected_source,
                    "search_query": ev.search_query,
                }
            )

        # Блок НЕ успешного ответа
        # если ответ не прошёл проверку, но оба источника уже были выбраны роутером, дальше retry не нужен
        if both_sources_checked and not rl_passed:
            return StopEvent(
                result={
                    "answer": (
                        "В базе знаний TrainBot нет точной информации по этому вопросу. "
                        "Пожалуйста, уточните детали вопроса."
                    ),
                    "last_agent_answer": ev.answer,
                    "checked_sources": tried_sources,
                    "faithfulness_score": fh_score,
                    "relevance_score": rl_score,
                    "evaluator_feedback": fh_feedback,
                    "relevance_feedback": rl_feedback,
                    "attempts": ev.attempt + 1,
                    "status": "needs_clarification",
                    "selected_source": ev.selected_source,
                    "search_query": ev.search_query,
                }
            )

        if fh_checked and (not fh_feedback or fh_feedback.strip().upper() in {"NO", "NO."}):
            fh_feedback = (
                "Ответ содержит утверждения, которые не были явно подтверждены найденными "
                "контекстами. Переформулируй ответ короче: используй только факты из source_nodes. "
                "Не добавляй предположения, если они прямо не указаны в источниках."
            )

        if not rl_feedback or rl_feedback.strip().upper() in {"NO", "NO."}:
            rl_feedback = (
                "Найденные контексты или ответ недостаточно релевантны вопросу пользователя."
            )

        # Блок ограничения кол-ва повторов
        if ev.attempt >= self.max_retries:
            if rl_passed and not fh_passed:
                return StopEvent(
                    result={
                        "answer": ev.answer,
                        "warning": (
                            "Ответ релевантен вопросу, но не все формулировки "
                            "строго подтверждены найденными источниками."
                        ),
                        "faithfulness_score": fh_score,
                        "relevance_score": rl_score,
                        "evaluator_feedback": fh_feedback,
                        "relevance_feedback": rl_feedback,
                        "attempts": ev.attempt + 1,
                        "status": "partially_supported",
                        "selected_source": ev.selected_source,
                        "search_query": ev.search_query,
                    }
                )

            return StopEvent(
                result={
                    "answer": (
                        "В базе знаний TrainBot нет точной информации по этому вопросу. "
                        "Пожалуйста, уточните детали вопроса."
                    ),
                    "last_agent_answer": ev.answer,
                    "faithfulness_score": fh_score,
                    "relevance_score": rl_score,
                    "evaluator_feedback": fh_feedback,
                    "relevance_feedback": rl_feedback,
                    "attempts": ev.attempt + 1,
                    "status": "needs_clarification",
                    "selected_source": ev.selected_source,
                    "search_query": ev.search_query,
                }
            )

        next_search_query = self.build_retry_search_query(
            normalized_query=ev.normalized_query,
            previous_source=ev.selected_source,
        )

        print(f"Next search_query: {next_search_query}")

        return RetryAgentEvent(
            original_query=ev.original_query,
            normalized_query=ev.normalized_query,
            search_query=next_search_query,
            attempt=ev.attempt + 1,
            evaluator_feedback="" if fh_passed else fh_feedback,
            relevance_feedback="" if rl_passed else rl_feedback,
            tried_sources=tried_sources,
        )

    @step
    async def retry_agent(self, ev: RetryAgentEvent) -> RunAgentEvent:
        """Возвращает выполнение к шагу run_agent и создаёт цикл."""

        print(f'Step retry_agent, {ev=}')
        return RunAgentEvent(
            original_query=ev.original_query,
            normalized_query=ev.normalized_query,
            search_query=ev.search_query,
            attempt=ev.attempt,
            evaluator_feedback=ev.evaluator_feedback,
            relevance_feedback=ev.relevance_feedback,
            tried_sources=ev.tried_sources,
        )
