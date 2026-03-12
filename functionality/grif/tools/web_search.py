"""
Web Search Tool — Tavily API.
Category: READ (no confirmation required).
"""

from typing import Any

import structlog

from grif.models.enums import ToolCategory
from grif.tools.base import BaseTool, ToolResult

log = structlog.get_logger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for current information. "
        "Use this to find recent news, prices, facts, or any information not in your training data."
    )
    category = ToolCategory.READ

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient  # type: ignore[import]
            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        log.debug("web_search_start", query=query[:100])
        try:
            import asyncio
            client = self._get_client()
            # Tavily is sync — run in thread pool
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_answer=include_answer,
                ),
            )

            # Format output
            answer = raw.get("answer", "")
            results = raw.get("results", [])
            output_parts: list[str] = []

            if answer:
                output_parts.append(f"Summary: {answer}")

            for i, r in enumerate(results[:max_results], 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:500]
                output_parts.append(f"\n[{i}] {title}\nURL: {url}\n{content}")

            output = "\n".join(output_parts) if output_parts else "No results found."
            log.info("web_search_done", results_count=len(results), query=query[:60])
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as exc:
            log.error("web_search_failed", error=str(exc), query=query[:60])
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (1-10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        }
