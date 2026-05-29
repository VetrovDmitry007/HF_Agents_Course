"""
Пример создания мульти-агента финансового аналитика по облигациям.
"""

import os

# Убираем возможные прокси, которые Python/httpx/openai-sdk может подхватывать из окружения
for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]:
    os.environ.pop(key, None)

os.environ["NO_PROXY"] = "188.116.172.185,localhost,127.0.0.1"
os.environ["no_proxy"] = "188.116.172.185,localhost,127.0.0.1"

from typing import Any
from smolagents import CodeAgent, ToolCallingAgent, tool, OpenAIModel, DuckDuckGoSearchTool, VisitWebpageTool
from dotenv import load_dotenv

load_dotenv()


@tool
def edisclosure_search(query: str, max_results: int = 5) -> str:
    """
    Ищет информацию только на официальном сайте https://e-disclosure.ru/ Центра раскрытия корпоративной информации.

    Args:
        query: Поисковый запрос без оператора site.
            Например: "Полипласт бухгалтерская отчетность 2025 выручка чистая прибыль".
        max_results: Максимальное количество результатов поиска.

    Returns:
        Текст с результатами поиска по сайту e-disclosure.ru.
    """
    search_tool = DuckDuckGoSearchTool(max_results=max_results)

    priority_query = f"site:e-disclosure.ru {query}"

    return search_tool(priority_query)

@tool
def calculate_bond_metrics(
    revenue: float,
    net_income: float,
    ebitda: float,
    net_debt: float,
    ocf: float,
    capex: float,
) -> dict:
    """
    Рассчитывает ключевые финансовые метрики эмитента облигации.

    Args:
        revenue: Выручка за период.
        net_income: Чистая прибыль за период.
        ebitda: EBITDA за период.
        net_debt: Чистый долг на отчётную дату.
        ocf: Операционный денежный поток за период.
        capex: Капитальные затраты за период. Если в отчётности CAPEX указан как отрицательное значение, передавай модуль числа.

    Returns:
        Словарь с рассчитанными метриками: Чистый долг / EBITDA, FCF, чистая маржинальность и OCF / EBITDA.
    """
    if ebitda == 0:
        net_debt_to_ebitda = None
        ocf_to_ebitda = None
    else:
        net_debt_to_ebitda = net_debt / ebitda
        ocf_to_ebitda = ocf / ebitda

    if revenue == 0:
        net_margin = None
    else:
        net_margin = net_income / revenue

    fcf = ocf - capex

    return {
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "fcf": fcf,
        "net_margin": net_margin,
        "ocf_to_ebitda": ocf_to_ebitda,
    }


@tool
def interpret_bond_metrics(
    net_debt_to_ebitda: float,
    fcf: float,
    net_margin: float,
    ocf_to_ebitda: float,
) -> str:
    """
    Даёт простую качественную интерпретацию финансовых метрик эмитента облигации.

    Args:
        net_debt_to_ebitda: Отношение чистого долга к EBITDA.
        fcf: Свободный денежный поток.
        net_margin: Чистая маржинальность.
        ocf_to_ebitda: Отношение операционного денежного потока к EBITDA.

    Returns:
        Текстовая интерпретация метрик с предварительной оценкой финансового качества.
    """
    conclusions = []

    if net_debt_to_ebitda is None:
        conclusions.append("Чистый долг / EBITDA не рассчитан: EBITDA равна нулю или отсутствует.")
    elif net_debt_to_ebitda <= 2:
        conclusions.append("Долговая нагрузка выглядит умеренной: Чистый долг / EBITDA ≤ 2.")
    elif net_debt_to_ebitda <= 3.5:
        conclusions.append("Долговая нагрузка средняя: Чистый долг / EBITDA находится в зоне 2–3.5.")
    else:
        conclusions.append("Долговая нагрузка повышенная: Чистый долг / EBITDA > 3.5.")

    if fcf > 0:
        conclusions.append("FCF положительный: компания генерирует свободный денежный поток после CAPEX.")
    else:
        conclusions.append("FCF отрицательный: денежный поток после CAPEX требует дополнительного внимания.")

    if net_margin is None:
        conclusions.append("Чистая маржинальность не рассчитана: выручка равна нулю или отсутствует.")
    elif net_margin > 0.10:
        conclusions.append("Чистая маржинальность хорошая: выше 10%.")
    elif net_margin > 0:
        conclusions.append("Чистая маржинальность положительная, но умеренная.")
    else:
        conclusions.append("Чистая маржинальность отрицательная: компания убыточна по чистой прибыли.")

    if ocf_to_ebitda is None:
        conclusions.append("OCF / EBITDA не рассчитан: EBITDA равна нулю или отсутствует.")
    elif ocf_to_ebitda > 0.8:
        conclusions.append("OCF / EBITDA > 0.8: качество прибыли выглядит хорошим.")
    else:
        conclusions.append("OCF / EBITDA ≤ 0.8: качество прибыли вызывает вопросы.")

    return "\n".join(conclusions)


def _memory_to_text(agent_memory: Any) -> str:
    """
    Преобразует память агента в текст для проверки факта вызова managed-agent.
    """
    if agent_memory is None:
        return ""

    if hasattr(agent_memory, "get_succinct_steps"):
        try:
            return str(agent_memory.get_succinct_steps())
        except Exception:
            pass

    return str(agent_memory)


def check_bond_report_quality(final_answer: Any, agent_memory: Any = None, agent: Any = None) -> bool:
    """
    Проверяет качество финального отчёта по облигации.

    Проверка допускает два корректных результата:
    1. данные найдены и метрики рассчитаны;
    2. данные не найдены, но агент явно перечислил недостающие показатели
       и не стал выдумывать числа.

    Args:
        final_answer: Финальный ответ агента.
        agent_memory: Память агента, используется для проверки, что web-agent действительно вызывался.
        agent: Экземпляр агента, используется для диагностической информации о конфигурации.

    Returns:
        True, если ответ можно вернуть пользователю; иначе False.
    """
    text = str(final_answer).lower()
    memory_text = _memory_to_text(agent_memory).lower()

    agent_max_steps = getattr(agent, "max_steps", None)
    agent_name = getattr(agent, "name", "manager_agent")

    forbidden_phrases = [
        "продолжу поиск",
        "нужно дополнительно поискать",
        "требуется дальнейший поиск",
        "further investigation is required",
        "now initiating a web search",
        "let's proceed",
    ]

    if any(phrase in text for phrase in forbidden_phrases):
        print(
            f"Финальный ответ не прошёл проверку: ответ не завершён, "
            f"а предлагает продолжать поиск. agent={agent_name}, max_steps={agent_max_steps}"
        )
        return False

    required_sections = [
        "облигация",
        "эмитент",
        "исходные данные",
        "метрик",
        "интерпретац",
        "риск",
        "источник",
        "огранич",
    ]

    missing_sections = [part for part in required_sections if part not in text]
    if missing_sections:
        print(f"Финальный ответ не прошёл проверку. Не хватает разделов: {missing_sections}")
        return False

    used_web_agent = "financial_web_research_agent" in memory_text
    if not used_web_agent:
        print("Финальный ответ не прошёл проверку: не найден вызов financial_web_research_agent в памяти агента.")
        return False

    missing_data_mode = any(
        marker in text
        for marker in [
            "data_missing",
            "не хватает данных",
            "данные не найдены",
            "не удалось найти",
            "недостаточно данных",
            "не рассчитано",
        ]
    )

    metrics_terms = [
        "чистый долг",
        "ebitda",
        "fcf",
        "ocf",
        "маржинальность",
    ]

    missing_metrics_terms = [term for term in metrics_terms if term not in text]

    if missing_data_mode:
        # Если данных нет, не требуем численных расчётов.
        # Главное — чтобы агент явно сказал, чего не хватает.
        required_missing_fields = [
            "выруч",
            "чист",
            "ebitda",
            "долг",
            "ocf",
            "capex",
        ]
        absent_missing_fields = [field for field in required_missing_fields if field not in text]

        if absent_missing_fields:
            print(f"В режиме DATA_MISSING не перечислены недостающие поля: {absent_missing_fields}")
            return False

        return True

    if missing_metrics_terms:
        print(f"Финальный ответ не прошёл проверку. Не хватает метрик: {missing_metrics_terms}")
        return False

    return True


# model = OpenAIModel(
#     # model_id="poolside/laguna-m.1:free",
#     # model_id="deepseek/deepseek-v3.2",
#     model_id="deepseek/deepseek-v4-flash:free",
#     api_base="https://openrouter.ai/api/v1",
#     api_key=os.environ["OR_TOKEN"],
# )

model = OpenAIModel(
    model_id="qwen2.5:7b",
    api_base="http://188.116.172.185:11434/v1",
    api_key="ollama",
    temperature=0.1,
    max_tokens=800,
)

WEB_AGENT_INSTRUCTIONS = """
            Ты специализированный агент поиска финансовой информации по эмитенту облигации.
            
            Правила:
            1. Сначала ищи только по приоритетному источнику:
               site:e-disclosure.ru Полипласт отчётность выручка EBITDA чистый долг OCF CAPEX
            2. Затем допускается не более одного дополнительного общего web-поиска.
            3. Не делай бесконечные уточнения.
            4. Если точные значения выручки, чистой прибыли, EBITDA, чистого долга, OCF и CAPEX не найдены,
               верни статус DATA_MISSING.
            5. Запрещено писать: "продолжу поиск", "нужно дополнительно поискать", "further investigation is required".
            6. В последнем шаге обязательно вызови final_answer(...).
            
            Формат ответа:
            СТАТУС: DATA_FOUND или DATA_MISSING
            
            ЭМИТЕНТ:
            ...
            
            НАЙДЕННЫЕ ДАННЫЕ:
            - Выручка:
            - Чистая прибыль:
            - EBITDA:
            - Чистый долг:
            - OCF:
            - CAPEX:
            
            НЕДОСТАЮЩИЕ ДАННЫЕ:
            ...
            
            ИСТОЧНИКИ:
            - название источника — ссылка
            """

web_agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
    ],
    model=model,
    max_steps=4,
    name="financial_web_research_agent",
    description=(
        "Однократно ищет финансовую информацию по эмитенту облигации. "
        "Возвращает найденные данные или статус DATA_MISSING. "
        "Не продолжает поиск бесконечно."
    ),
    verbosity_level=1,
    instructions=WEB_AGENT_INSTRUCTIONS,
    provide_run_summary=False,
)


manager_agent = CodeAgent(
    tools=[
        calculate_bond_metrics,
        interpret_bond_metrics,
    ],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["json", "pandas", "re"],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_bond_report_quality],
    max_steps=6,
    )


bond_query = "RU000A10F7V9 -- Полипласт П02-БО-16 "

task = f"""
Ты финансовый аналитик по облигациям.

Нужно проанализировать облигацию: {bond_query}

Порядок работы:
   
1. Через financial_web_research_agent найди:
   - эмитента облигации;
   - последнюю доступную финансовую отчётность;
   - значения выручки, чистой прибыли, EBITDA, чистого долга, OCF и CAPEX;
   - источники этих данных.

2. Если точные значения не найдены, явно напиши, каких данных не хватает.
   Не выдумывай числа.

3. Если данные найдены, вызови calculate_bond_metrics и рассчитай:
   - "Чистый долг" / EBITDA;
   - FCF = OCF - CAPEX;
   - Чистую маржинальность = Чистая прибыль / Выручка;
   - OCF / EBITDA.

4. Вызови interpret_bond_metrics и добавь качественную оценку.

5. Итоговый ответ оформи так:
   - Облигация и эмитент;
   - Найденные исходные данные;
   - Таблица рассчитанных метрик;
   - Интерпретация;
   - Основные риски;
   - Источники;
   - Ограничения анализа.

Важно:
- Делай именно кредитный/финансовый анализ эмитента.
- OCF / EBITDA > 0.8 считай хорошим показателем качества прибыли.
- Если "Чистый долг" / EBITDA тогда: 
   < 1x — отлично.
   3x–4x — норма для капиталоемких отраслей (нефть, телеком).
   5x — зона риска (банкротство при первом кризисе).
   
Важно для управления циклом:
- financial_web_research_agent можно вызвать только один раз.
- Если financial_web_research_agent вернул DATA_MISSING или не нашёл точные значения,
  не вызывай его повторно.
- В этом случае сразу сформируй финальный отчёт:
  значения отсутствующих показателей укажи как "не найдено",
  расчётные метрики укажи как "не рассчитано из-за отсутствия данных".
- Не пытайся продолжать поиск после DATA_MISSING.
   
"""

result = manager_agent.run(task)
print(result)