from typing import Any

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.evaluation import FaithfulnessEvaluator
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
        if ev.evaluator_feedback:
            agent_query = f"""
                            Исходный вопрос пользователя:
                            {ev.query}
                            
                            Предыдущий ответ не прошёл проверку FaithfulnessEvaluator.
                            
                            Комментарий проверяющей модели:
                            {ev.evaluator_feedback}
                            
                            Повторно используй инструмент базы знаний.
                            Не добавляй факты, которые не подтверждаются найденными источниками.
                            Если информации недостаточно, прямо сообщи об этом.
                            """
        else:
            agent_query = ev.query

        agent_response = await self.agent.run(agent_query)

        contexts: list[str] = []

        # Агент мог вызвать несколько инструментов или один инструмент несколько раз.
        for tool_call in getattr(agent_response, "tool_calls", []):
            tool_output = getattr(tool_call, "tool_output", None)
            raw_response = getattr(tool_output, "raw_output", None)

            # Для QueryEngineTool raw_output является объектом Response.
            for source_node in getattr(raw_response, "source_nodes", []):
                contexts.append(source_node.node.get_content())

        # Удаляем дубликаты, сохраняя порядок.
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
        """Проверяет ответ и выбирает: завершение или новую попытку."""

        print(f'Step evaluate_answer, {ev=}')
        if not ev.contexts:
            passed = False
            score = None
            feedback = (
                "Агент не получил ни одного source_node из RAG-инструмента. "
                "Ответ нельзя проверить на соответствие источникам."
            )
        else:
            evaluation = await self.faithfulness_evaluator.aevaluate(
                query=ev.query,
                response=ev.answer,
                contexts=ev.contexts,
            )

            passed = bool(evaluation.passing)
            score = evaluation.score
            feedback = evaluation.feedback or ""

        if passed:
            return StopEvent(
                result={
                    "answer": ev.answer,
                    "faithfulness_score": score,
                    "attempts": ev.attempt + 1,
                }
            )

        if ev.attempt >= self.max_retries:
            return StopEvent(
                result={
                    "answer": (
                        "Не удалось сформировать ответ, полностью "
                        "подтверждённый найденными источниками."
                    ),
                    "last_agent_answer": ev.answer,
                    "evaluator_feedback": feedback,
                    "attempts": ev.attempt + 1,
                }
            )

        return RetryAgentEvent(
            query=ev.query,
            attempt=ev.attempt + 1,
            evaluator_feedback=feedback,
        )

    @step
    async def retry_agent(self, ev: RetryAgentEvent) -> RunAgentEvent:
        """Возвращает выполнение к шагу run_agent и создаёт цикл."""

        print(f'Step retry_agent, {ev=}')
        return RunAgentEvent(
            query=ev.query,
            attempt=ev.attempt,
            evaluator_feedback=ev.evaluator_feedback,
        )