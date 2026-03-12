"""
BaseTool ABC.
All tools inherit from this class and implement execute().
ToolRegistry checks category before execution per architecture mechanic #14.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from grif.models.enums import ToolCategory


class ToolInput(BaseModel):
    """Base model for tool inputs — subclass per tool."""
    pass


class ToolResult(BaseModel):
    """Standard tool execution result."""

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    tokens_used: int = 0
    metadata: dict[str, Any] = {}

    def to_observation(self) -> str:
        """Format as a text observation for the ReAct loop."""
        if not self.success:
            return f"[{self.tool_name}] ERROR: {self.error}"
        if isinstance(self.output, str):
            return f"[{self.tool_name}] {self.output}"
        return f"[{self.tool_name}] {str(self.output)}"


class BaseTool(ABC):
    """
    Abstract base for all GRIF tools.

    Subclasses MUST implement:
      - name: str
      - description: str
      - category: ToolCategory
      - execute(**kwargs) -> ToolResult
    """

    name: str
    description: str
    category: ToolCategory

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        ...

    def to_function_schema(self) -> dict[str, Any]:
        """
        Return OpenAI-compatible function schema for use in LLM tool-calling.
        Subclasses can override to provide richer parameter schemas.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema(),
            },
        }

    def _get_parameters_schema(self) -> dict[str, Any]:
        """Default parameter schema — subclasses override for strict typing."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The input query or instruction for this tool.",
                }
            },
            "required": ["query"],
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, category={self.category})"
