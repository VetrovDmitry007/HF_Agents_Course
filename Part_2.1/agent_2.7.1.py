"""
Агенты поиска.
"""
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, FinalAnswerTool, VisitWebpageTool
import os
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel

load_dotenv()

model = OpenAIModel(
    model_id="poolside/laguna-m.1:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ["OR_TOKEN"],
)

agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(),
           VisitWebpageTool(),
           FinalAnswerTool()],
    max_steps=10,
)

response = agent.run(
    "Найдите идеи для посещения достопримечательностей в Токио при туристической 7-и дневной поездки."
)
print(response)