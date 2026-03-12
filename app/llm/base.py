from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        stream: bool = False,
        system_prompt: str = "",
        user_message: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: str = "text",
        **kwargs,
    ) -> Any:
        """Генерация текста. Если stream=True — возвращает async-генератор."""
        pass

    @abstractmethod
    def count_tokens(self, prompt: str) -> int:
        pass
