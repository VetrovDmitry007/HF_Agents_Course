"""
Создание рабочих процессов для агентов в LlamaIndex
https://huggingface.co/learn/agents-course/unit2/llama-index/workflows#creating-agentic-workflows-in-llamaindex

для построения схемы рабочего процесса
pip install pyvis
"""
import asyncio

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event
import random

from llama_index.core.workflow.drawing import draw_all_possible_flows
from workflows.events import StopEvent


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)



async def run_work_flow():
    w = MultiStepWorkflow(verbose=False)
    result = await w.run()
    print(result)
    # Изображение схемы рабочих процессов
    draw_all_possible_flows(w, "flow.html")


if __name__ == '__main__':
    asyncio.run(run_work_flow())
