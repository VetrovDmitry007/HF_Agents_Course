"""
Нормализация пользовательского вопроса для RAG TrainBot.

LLM-вызов через acomplete().
"""
from typing import Any


NORMALIZER_PROMPT = """
Сожми обращение пользователя в короткий самодостаточный вопрос для RAG-системы TrainBot.

Строгие правила:
- верни только итоговый вопрос;
- не рассуждай;
- не пиши объяснения;
- не отвечай пользователю;
- не добавляй новые факты;
- не превращай вопрос в набор ключевых слов;
- сохрани проблему пользователя и желаемое действие;
- сохрани важные детали: сумма, дата, ID, тип подписки, ошибка, списание денег, отсутствие брони.

Пример:
Исходное обращение: Добрый день. Я вчера поставил подписку на бронирование, бот прислал уведомление, что места есть, но бронь не появилась. Деньги списались. Как мне теперь вернуть остаток?
Сжатый вопрос: Как вернуть остаток средств, если бот прислал уведомление о наличии мест, деньги списались, но бронь не появилась?
""".strip()


async def get_normalize_query(query: str, llm: Any) -> str:
    """Возвращает нормализованный вопрос или исходный вопрос при ошибке."""

    query = (query or "").strip()

    if not query:
        return query

    prompt = f"""
            {NORMALIZER_PROMPT}
            
            Вопрос пользователя:
            {query}
            
            Сжатый вопрос:
            """.strip()

    try:
        response = await llm.acomplete(prompt)
        # print(f'{response=}')
        normalized = str(response).strip()
    except Exception as e:
        print(f"Ошибка нормализации: {e}")
        return query


    normalized = normalized.replace("\n", " ").strip()
    normalized = " ".join(normalized.split())
    normalized = normalized.strip('"').strip("'").strip("`")

    return normalized or query

if __name__ == '__main__':
    import asyncio
    from config_models.open_router import normalizer_llm

    # query="""Добрый день. Я вчера поставил подписку на бронирование, бот прислал уведомление,
    #            что места есть, но бронь не появилась. Деньги списались. Как мне теперь вернуть остаток?
    #        """
    query = "Добрый день. Вчера оплатил 1000 чтоб пополнить баланс. А баланс не пополняется."


    res = asyncio.run(get_normalize_query(query=query, llm=normalizer_llm))
    print(res)