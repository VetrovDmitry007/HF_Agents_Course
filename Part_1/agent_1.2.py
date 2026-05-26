# Использование моделей провайдера OpenRouter
import os
from typing import Dict, Any
from smolagents import CodeAgent, LiteLLMModel, tool, OpenAIModel

from Learn_1.prompt_1 import agent_instructions
from dotenv import load_dotenv

load_dotenv()

@tool
def find_city_coordinates(city: str) -> Dict[str, Any]:
    """Ищет координаты города по его названию.

    Args:
        city: Название города, например Tokyo, Paris, Amsterdam, Sanya.
    """
    import requests

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "ru",
            "format": "json",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return {
            "ok": False,
            "error": f"Город не найден: {city}",
        }

    place = data["results"][0]

    return {
        "ok": True,
        "city": place["name"],
        "country": place.get("country", ""),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
    }


@tool
def get_weather_by_coordinates(
    latitude: float,
    longitude: float,
    city: str,
    country: str,
) -> Dict[str, Any]:
    """Получает текущую погоду по координатам.

    Инструмент НЕ решает, что человеку одевать.
    Инструмент только возвращает факты о погоде и простую оценку температуры.

    Args:
        latitude: Широта города.
        longitude: Долгота города.
        city: Название города.
        country: Название страны.
    """
    import requests

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,precipitation,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    current = data["current"]

    temperature = current["temperature_2m"]
    precipitation = current["precipitation"]
    wind_speed = current["wind_speed_10m"]

    if temperature >= 30:
        temperature_category = "жарко"
    elif temperature >= 20:
        temperature_category = "тепло"
    elif temperature >= 15:
        temperature_category = "прохладно"
    elif temperature >= 5:
        temperature_category = "холодно"
    else:
        temperature_category = "очень холодно"

    return {
        "ok": True,
        "city": city,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": temperature,
        "temperature_category": temperature_category,
        "precipitation_mm": precipitation,
        "has_precipitation": precipitation > 0,
        "wind_speed_kmh": wind_speed,
        "is_windy": wind_speed >= 15,
    }


model = OpenAIModel(
    # model_id="deepseek/deepseek-v3.2", # 17-66 сек.
    # model_id="poolside/laguna-m.1:free", # 49 сек.
    model_id="deepseek/deepseek-v4-flash:free", # 17.65 сек.
    api_base="https://openrouter.ai/api/v1",
    api_key=os.environ['OR_TOKEN'],
)

agent = CodeAgent(
    tools=[
        find_city_coordinates,
        get_weather_by_coordinates,
    ],
    model=model,
    instructions=agent_instructions,
    max_steps=4,
    additional_authorized_imports=["requests"],
)


agent.run(
    "Я сейчас в Токио. Какая там погода и как лучше одеться для прогулки?"
)