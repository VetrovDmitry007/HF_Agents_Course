# Ваш агент, которого зовут Альфред

# pip install 'smolagents[litellm]
# pip install boto3

from smolagents import LiteLLMModel, GradioUI, CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml


@tool
def my_custom_tool(arg1:str, arg2:int)-> str: # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that does nothing yet
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Создать объект часового пояса
        tz = pytz.timezone(timezone)
        # Определите текущее время в этом часовом поясе
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

final_answer = FinalAnswerTool()

model = LiteLLMModel(
    model_id="ollama_chat/qwen2:7b",
    api_base="http://192.168.3.88:11434",
    num_ctx=8192,
)

# Инструмент импорта из Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

# Загрузка системного запроса из файла prompt.yaml
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


agent = CodeAgent(
    model=model,
    tools=[final_answer],  # Добавьте свои инструменты здесь (не удаляйте final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates  # Передайте системный запрос в CodeAgent
)

# GradioUI -- это готовая обёртка для запуска агента smolagents в веб-чате через Gradio
GradioUI(agent).launch()
