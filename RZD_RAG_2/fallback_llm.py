import os
from typing import Any
from pydantic import PrivateAttr
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI


def is_fallback_llm_error(exc: BaseException) -> bool:
    """
    Возвращает True, если ошибку можно обработать переходом на другой ключ.
    """
    fallback_statuses = {402, 429, 500, 502, 503, 504}
    fallback_markers = (
        "payment required",
        "depleted",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
    )

    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None:
        if id(current) in seen:
            break

        seen.add(id(current))

        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None)

        if status_code in fallback_statuses:
            return True

        text = str(current).lower()

        if any(marker in text for marker in fallback_markers):
            return True

        current = current.__cause__ or current.__context__

    return False


class FallbackHuggingFaceLLM(HuggingFaceInferenceAPI):
    """
    LlamaIndex-совместимый LLM с fallback между несколькими HF-токенами.

    Важно:
    класс наследуется от HuggingFaceInferenceAPI,
    поэтому проходит проверку isinstance(llm, LLM).
    """

    _clients: list[HuggingFaceInferenceAPI] = PrivateAttr(default_factory=list)
    _token_envs: list[str] = PrivateAttr(default_factory=list)
    _current_index: int = PrivateAttr(default=0)

    def __init__(
        self,
        *,
        token_envs: list[str],
        model_name: str,
        temperature: float,
        max_tokens: int,
        provider: str = "auto",
        **kwargs: Any,
    ) -> None:
        token_pairs: list[tuple[str, str]] = []

        for token_env in token_envs:
            token = os.getenv(token_env)

            if token:
                token_pairs.append((token_env, token))

        if not token_pairs:
            raise RuntimeError(
                f"Не найден ни один токен из списка переменных окружения: {token_envs}"
            )

        first_token_env, first_token = token_pairs[0]

        # Инициализируем родительский HuggingFaceInferenceAPI,
        # чтобы объект был полноценным LlamaIndex LLM.
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            token=first_token,
            provider=provider,
            **kwargs,
        )

        self._token_envs = [env for env, _ in token_pairs]

        self._clients = [
            HuggingFaceInferenceAPI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                token=token,
                provider=provider,
                **kwargs,
            )
            for _, token in token_pairs
        ]

        self._current_index = 0

    @property
    def current_client(self) -> HuggingFaceInferenceAPI:
        return self._clients[self._current_index]

    @property
    def current_token_env(self) -> str:
        return self._token_envs[self._current_index]

    def _move_to_next_client(self) -> None:
        self._current_index = (self._current_index + 1) % len(self._clients)

    async def _async_call_with_fallback(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_error: Exception | None = None
        start_index = self._current_index

        for _ in range(len(self._clients)):
            client = self.current_client
            token_env = self.current_token_env

            try:
                method = getattr(client, method_name)
                return await method(*args, **kwargs)

            except Exception as e:
                if not is_fallback_llm_error(e):
                    raise

                print(
                    f"LLM fallback: {method_name} упал на {token_env}: "
                    f"{type(e).__name__}: {e}"
                )

                last_error = e
                self._move_to_next_client()

                if self._current_index == start_index:
                    break

        raise RuntimeError(
            "Все доступные HuggingFace LLM-ключи недоступны для текущего запроса."
        ) from last_error

    def _sync_call_with_fallback(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        last_error: Exception | None = None
        start_index = self._current_index

        for _ in range(len(self._clients)):
            client = self.current_client
            token_env = self.current_token_env

            try:
                method = getattr(client, method_name)
                return method(*args, **kwargs)

            except Exception as e:
                if not is_fallback_llm_error(e):
                    raise

                print(
                    f"LLM fallback: {method_name} упал на {token_env}: "
                    f"{type(e).__name__}: {e}"
                )

                last_error = e
                self._move_to_next_client()

                if self._current_index == start_index:
                    break

        raise RuntimeError(
            "Все доступные HuggingFace LLM-ключи недоступны для текущего запроса."
        ) from last_error

    async def achat(self, *args: Any, **kwargs: Any) -> Any:
        return await self._async_call_with_fallback("achat", *args, **kwargs)

    async def acomplete(self, *args: Any, **kwargs: Any) -> Any:
        return await self._async_call_with_fallback("acomplete", *args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync_call_with_fallback("chat", *args, **kwargs)

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync_call_with_fallback("complete", *args, **kwargs)