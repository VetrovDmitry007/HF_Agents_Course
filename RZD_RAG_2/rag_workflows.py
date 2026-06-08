from typing import Any

from llama_index.core.agent.workflow import ReActAgent, ToolCallResult
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class RunAgentEvent(Event):
    """Событие для запуска или повторного запуска агента."""

    query: str
    attempt: int = 0
    evaluator_feedback: str = ""
    relevance_feedback: str = ""


class AgentAnswerEvent(Event):
    """Событие с ответом агента и использованными RAG-контекстами."""

    query: str
    answer: str
    contexts: list[str]
    attempt: int


class RetryAgentEvent(Event):
    """Событие, запускающее новую попытку после неудачной проверки."""

    query: str
    attempt: int
    evaluator_feedback: str
    relevance_feedback: str


class FaithfulRAGWorkflow(Workflow):
    """Workflow, проверяющий ответ RAG-агента на подтверждённость источниками."""

    def __init__(
            self,
            agent: ReActAgent,
            evaluator_llm: Any,
            max_retries: int = 2,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.agent = agent
        self.max_retries = max_retries
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=evaluator_llm)
        self.relevance_evaluator = RelevancyEvaluator(llm=evaluator_llm)

    @step
    async def start(self, ev: StartEvent) -> RunAgentEvent:
        """Преобразует входной запрос в событие запуска агента."""

        print(f'Step start, {ev=}')
        return RunAgentEvent(
            query=ev.query,
            attempt=0,
        )

    @step
    async def run_agent(self, ev: RunAgentEvent) -> AgentAnswerEvent:
        """Запускает агента и извлекает контексты из вызовов QueryEngineTool."""

        print(f'Step run_agent, {ev=}')
        if ev.evaluator_feedback or ev.relevance_feedback:
            agent_query = f"""
                            Исходный вопрос пользователя:
                            {ev.query}
                            """

            if ev.evaluator_feedback:
                agent_query += f"""
                                Предыдущий ответ не прошёл проверку FaithfulnessEvaluator.
                                
                                Комментарий проверяющей модели:
                                {ev.evaluator_feedback}
                                """

            if ev.relevance_feedback:
                agent_query += f"""
                                Предыдущий ответ не прошёл проверку RelevancyEvaluator.

                                Комментарий проверяющей модели:
                                {ev.relevance_feedback}
                                """

            agent_query += """
                            Повторно используй инструмент базы знаний.
                            Не добавляй факты, которые не подтверждаются найденными источниками.
                            Если информации недостаточно, прямо сообщи об этом.
                           """

        else:
            agent_query = ev.query

        contexts: list[str] = []

        handler = self.agent.run(user_msg=agent_query)
        """
        ToolCallResult — это событие LlamaIndex, которое появляется в момент, когда агент уже вызвал инструмент
        и получил от него результат.
        
        1. Запускаешь агента, но не ждёшь сразу финальный ответ.
        2. Получаешь handler.
        3. Через handler.stream_events() слушаешь все события выполнения агента.
        4. Когда появляется ToolCallResult, значит какой-то инструмент завершил работу.
        5. Из event.tool_output достаёшь результат инструмента.
        6. Из tool_output.raw_output достаёшь настоящий Response от QueryEngineTool.
        7. Из Response.source_nodes достаёшь найденные RAG-фрагменты.
        8. Эти фрагменты кладёшь в contexts.
        9. После завершения событий ждёшь финальный ответ агента через await handler.
        10. Возвращаешь AgentAnswerEvent с answer и contexts.
        """
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                tool_output = event.tool_output
                raw_response = getattr(tool_output, "raw_output", None)

                for source_node in getattr(raw_response, "source_nodes", []):
                    contexts.append(source_node.node.get_content())

        agent_response = await handler
        contexts = list(dict.fromkeys(contexts))

        return AgentAnswerEvent(
            query=ev.query,
            answer=str(agent_response),
            contexts=contexts,
            attempt=ev.attempt,
        )

    @step
    async def evaluate_answer(
            self,
            ev: AgentAnswerEvent,
    ) -> StopEvent | RetryAgentEvent:
        """Проверяет ответ: сначала релевантность, затем подтверждённость источниками."""

        fh_passed = False
        rl_passed = False
        fh_checked = False
        fh_score = None
        rl_score = None

        fh_feedback = ""
        rl_feedback = ""

        print(f"Step evaluate_answer, {ev=}")

        if not ev.contexts:
            rl_feedback = (
                "Агент не получил ни одного source_node из RAG-инструмента. "
                "Ответ нельзя проверить на релевантность и подтверждённость источникам."
            )
        else:
            # 1. Сначала проверяем, релевантен ли ответ вопросу и найденным контекстам.
            relevance = await self.relevance_evaluator.aevaluate(
                query=ev.query,
                response=ev.answer,
                contexts=ev.contexts,
            )
            rl_passed = bool(relevance.passing)
            rl_score = relevance.score
            rl_feedback = relevance.feedback or ""

            print(f"Результат проверки RelevancyEvaluator: {rl_passed=}, {rl_score=}")

            # 2. Faithfulness имеет смысл проверять только если ответ релевантен.
            if rl_passed:
                faithfulness = await self.faithfulness_evaluator.aevaluate(
                    query=ev.query,
                    response=ev.answer,
                    contexts=ev.contexts,
                )
                fh_passed = bool(faithfulness.passing)
                fh_score = faithfulness.score
                fh_feedback = faithfulness.feedback or ""

                print(f"Результат проверки FaithfulnessEvaluator: {fh_passed=}, {fh_score=}")

        if rl_passed and fh_passed:
            return StopEvent(
                result={
                    "answer": ev.answer,
                    "faithfulness_score": fh_score,
                    "relevance_score": rl_score,
                    "attempts": ev.attempt + 1,
                    "status": "passed",
                }
            )

        if fh_checked and (not fh_feedback or fh_feedback.strip().upper() in {"NO", "NO."}):
            fh_feedback = (
                "Ответ содержит утверждения, которые не были явно подтверждены найденными "
                "контекстами. Переформулируй ответ короче: используй только факты из source_nodes. "
                "Не добавляй предположения, если они прямо не указаны в источниках."
            )

        if (not rl_feedback or rl_feedback.strip().upper() in {"NO", "NO."}):
            rl_feedback = (
                "Найденные контексты или ответ недостаточно релевантны вопросу пользователя. "
                f"Повторно используй RAG-инструмент и ищи ответ именно на вопрос: {ev.query}"
            )

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
                    }
                )

            return StopEvent(
                result={
                    "answer": (
                        "В базе знаний TrainBot нет точной информации по этому вопросу."
                    ),
                    "last_agent_answer": ev.answer,
                    "faithfulness_score": fh_score,
                    "relevance_score": rl_score,
                    "evaluator_feedback": fh_feedback,
                    "relevance_feedback": rl_feedback,
                    "attempts": ev.attempt + 1,
                    "status": "failed",
                }
            )

        return RetryAgentEvent(
            query=ev.query,
            attempt=ev.attempt + 1,
            evaluator_feedback=fh_feedback,
            relevance_feedback=rl_feedback,
        )

    @step
    async def retry_agent(self, ev: RetryAgentEvent) -> RunAgentEvent:
        """Возвращает выполнение к шагу run_agent и создаёт цикл."""

        print(f'Step retry_agent, {ev=}')
        return RunAgentEvent(
            query=ev.query,
            attempt=ev.attempt,
            evaluator_feedback=ev.evaluator_feedback,
            relevance_feedback=ev.relevance_feedback,
        )
