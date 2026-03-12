"""
Tool Registry.
Central catalog of all available tools.
Enforces permission checks before execution (mechanic #14).
Implements Trust Escalation: WRITE_PUBLIC → auto after N approvals (mechanic #14).
"""

from typing import Any

import structlog

from grif.config import get_settings
from grif.models.agent_config import AgentConfig, ToolPermission
from grif.models.enums import ToolCategory
from grif.tools.base import BaseTool, ToolResult

log = structlog.get_logger(__name__)
settings = get_settings()


class PermissionDeniedError(Exception):
    """Raised when a tool action is blocked pending user approval."""

    def __init__(self, tool_name: str, category: ToolCategory) -> None:
        self.tool_name = tool_name
        self.category = category
        super().__init__(
            f"Tool '{tool_name}' ({category}) requires user approval before execution."
        )


class ToolRegistry:
    """
    Manages tool instances and enforces execution permissions.

    Permission rules (mechanic #14):
    - READ / WRITE_SAFE: execute immediately, no approval needed.
    - WRITE_PUBLIC: show draft first; auto-approved after N user approvals (Trust Escalation).
    - WRITE_IRREVERSIBLE: always require explicit user confirmation — cannot be disabled.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._initialized = False

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        log.debug("tool_registered", name=tool.name, category=tool.category)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def get_schemas(self, tool_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas for specified (or all) tools."""
        tools = (
            [self._tools[n] for n in tool_names if n in self._tools]
            if tool_names
            else list(self._tools.values())
        )
        return [t.to_function_schema() for t in tools]

    # ── Permission-aware execution ─────────────────────────────────────────────

    async def execute(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        agent_config: AgentConfig,
        draft_mode: bool = False,
    ) -> ToolResult:
        """
        Execute a tool after permission check.

        draft_mode=True: for WRITE_PUBLIC, return the draft text without posting.
        For WRITE_IRREVERSIBLE: always raises PermissionDeniedError.
        For WRITE_PUBLIC not yet auto-approved: raises PermissionDeniedError.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found in registry.",
            )

        perm = self._find_permission(tool_name, agent_config)
        category = perm.category if perm else tool.category

        # WRITE_IRREVERSIBLE: never auto-execute
        if category == ToolCategory.WRITE_IRREVERSIBLE:
            raise PermissionDeniedError(tool_name, category)

        # WRITE_PUBLIC: check auto_approved or draft_mode
        if category == ToolCategory.WRITE_PUBLIC:
            if draft_mode:
                # Return draft without executing
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    output=f"[DRAFT] Would execute {tool_name} with: {kwargs}",
                    metadata={"is_draft": True},
                )
            if perm and not perm.auto_approved:
                raise PermissionDeniedError(tool_name, category)

        # READ / WRITE_SAFE / auto-approved WRITE_PUBLIC: execute
        result = await tool.execute(**kwargs)
        log.info(
            "tool_executed",
            tool=tool_name,
            category=category,
            success=result.success,
        )
        return result

    def record_approval(
        self, tool_name: str, agent_config: AgentConfig
    ) -> bool:
        """
        Record one user approval. Returns True if trust threshold reached.
        On reaching threshold, WRITE_PUBLIC becomes auto-approved (Trust Escalation).
        """
        perm = self._find_permission(tool_name, agent_config)
        if perm and perm.category == ToolCategory.WRITE_PUBLIC:
            perm.approval_count += 1
            if perm.approval_count >= perm.trust_threshold:
                perm.auto_approved = True
                log.info(
                    "trust_escalation_reached",
                    tool=tool_name,
                    approvals=perm.approval_count,
                )
                return True
        return False

    def _find_permission(
        self, tool_name: str, agent_config: AgentConfig
    ) -> ToolPermission | None:
        return next(
            (p for p in agent_config.tool_permissions if p.tool_name == tool_name),
            None,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, agent_config: AgentConfig) -> "ToolRegistry":
        """
        Build a ToolRegistry for a specific agent, initialising only the tools
        it is allowed to use (as listed in agent_config.tools).
        """
        registry = cls()
        registry._initialize_tools(agent_config.tools)
        return registry

    def _initialize_tools(self, tool_names: list[str]) -> None:
        """Lazy-init tool instances based on available credentials."""
        from grif.tools.web_search import WebSearchTool
        from grif.tools.telegram_bot import TelegramBotTool
        from grif.tools.email_client import EmailClientTool

        tool_builders: dict[str, Any] = {
            "web_search": lambda: WebSearchTool(api_key=settings.tavily_api_key)
            if settings.tavily_api_key else None,
            "telegram_bot": lambda: TelegramBotTool(bot_token=settings.telegram_bot_token)
            if settings.telegram_bot_token else None,
            "post_telegram": lambda: TelegramBotTool(bot_token=settings.telegram_bot_token)
            if settings.telegram_bot_token else None,
            "email_client": lambda: EmailClientTool(),
            "send_email": lambda: EmailClientTool(),
            "fetch": lambda: _FetchTool(),
            "analyze": lambda: _AnalyzeTool(),
            "draft": lambda: _DraftTool(),
        }

        for name in tool_names:
            if name in tool_builders:
                tool = tool_builders[name]()
                if tool is not None:
                    self.register(tool)
                else:
                    log.warning("tool_init_skipped_no_creds", tool=name)
            else:
                log.warning("tool_unknown", tool=name)


# ── Lightweight built-in tools ─────────────────────────────────────────────────

class _FetchTool(BaseTool):
    """Fetches content from a URL."""
    name = "fetch"
    description = "Fetch the content of a URL."
    category = ToolCategory.READ

    async def execute(self, url: str, **kwargs: Any) -> ToolResult:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url)
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    output=resp.text[:3000],
                    metadata={"status_code": resp.status_code, "url": url},
                )
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _get_parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to fetch."}},
            "required": ["url"],
        }


class _AnalyzeTool(BaseTool):
    """Placeholder analyze tool — returns its input for in-context processing."""
    name = "analyze"
    description = "Analyze provided data or text."
    category = ToolCategory.READ

    async def execute(self, data: str, **kwargs: Any) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, output=f"Analysis input: {data[:500]}")


class _DraftTool(BaseTool):
    """Creates a draft — saved locally, no external side effects."""
    name = "draft"
    description = "Save a draft of content for later review."
    category = ToolCategory.WRITE_SAFE

    async def execute(self, content: str, title: str = "Draft", **kwargs: Any) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=f"Draft saved: {title}",
            metadata={"title": title, "length": len(content)},
        )
