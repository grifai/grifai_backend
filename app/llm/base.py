from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: str = "text",
    ) -> str:
        """Генерация текста. response_format: 'text' или 'json'"""
        pass

    @abstractmethod
    def generate_with_history(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Генерация с историей [{role: "user"|"assistant", content: "..."}]"""
        pass
