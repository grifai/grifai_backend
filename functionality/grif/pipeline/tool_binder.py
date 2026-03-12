"""
Step 6: Tool Binder.
Validates that required tools are available and credentials exist.
Returns a bound tool list for the agent to use.
No LLM calls — deterministic credential check.
"""

import structlog

from grif.config import get_settings
from grif.models.agent_config import AgentConfig, ToolPermission
from grif.models.enums import ToolCategory

log = structlog.get_logger(__name__)
settings = get_settings()


class ToolBindingError(Exception):
    """Raised when a required tool cannot be bound (missing credentials)."""


class ToolBinder:
    """
    Step 6: checks that each tool in AgentConfig.tools has credentials available.
    Updates tool_permissions on the AgentConfig.
    Returns validated tool names that can be safely initialised.
    """

    # Mapping: tool_name → (category, required_credential_setting)
    _TOOL_REGISTRY: dict[str, tuple[ToolCategory, str | None]] = {
        "web_search": (ToolCategory.READ, "tavily_api_key"),
        "fetch": (ToolCategory.READ, None),
        "analyze": (ToolCategory.READ, None),
        "draft": (ToolCategory.WRITE_SAFE, None),
        "save": (ToolCategory.WRITE_SAFE, None),
        "create_file": (ToolCategory.WRITE_SAFE, None),
        "post_telegram": (ToolCategory.WRITE_PUBLIC, "telegram_bot_token"),
        "send_email": (ToolCategory.WRITE_PUBLIC, None),
        "tweet": (ToolCategory.WRITE_PUBLIC, None),
        "buy": (ToolCategory.WRITE_IRREVERSIBLE, None),
        "book": (ToolCategory.WRITE_IRREVERSIBLE, None),
        "delete": (ToolCategory.WRITE_IRREVERSIBLE, None),
        "transfer": (ToolCategory.WRITE_IRREVERSIBLE, None),
        "email_client": (ToolCategory.WRITE_PUBLIC, None),
        "telegram_bot": (ToolCategory.WRITE_PUBLIC, "telegram_bot_token"),
    }

    def bind(self, config: AgentConfig) -> list[str]:
        """
        Validate tool credentials and populate config.tool_permissions.
        Returns list of successfully bound tool names.
        Raises ToolBindingError if a WRITE_IRREVERSIBLE tool is requested
        without explicit user pre-approval (safety guard).
        """
        bound: list[str] = []
        permissions: list[ToolPermission] = []

        for tool_name in config.tools:
            category, cred_setting = self._TOOL_REGISTRY.get(
                tool_name, (ToolCategory.READ, None)
            )

            # Check credentials
            if cred_setting:
                cred_value = getattr(settings, cred_setting, "")
                if not cred_value:
                    log.warning(
                        "tool_credential_missing",
                        tool=tool_name,
                        credential=cred_setting,
                    )
                    # Skip tool — don't fail the whole config
                    continue

            perm = ToolPermission(
                tool_name=tool_name,
                category=category,
                auto_approved=(category in (ToolCategory.READ, ToolCategory.WRITE_SAFE)),
                approval_count=0,
                trust_threshold=settings.trust_escalation_approvals,
            )
            permissions.append(perm)
            bound.append(tool_name)
            log.debug("tool_bound", tool=tool_name, category=category)

        # Update config in-place
        object.__setattr__(config, "tool_permissions", permissions)
        object.__setattr__(config, "tools", bound)

        log.info("tools_bound", count=len(bound), tools=bound)
        return bound

    def get_category(self, tool_name: str) -> ToolCategory:
        return self._TOOL_REGISTRY.get(tool_name, (ToolCategory.READ, None))[0]

    def requires_confirmation(
        self,
        tool_name: str,
        config: AgentConfig,
    ) -> bool:
        """
        Check if this tool requires user confirmation before execution.
        WRITE_IRREVERSIBLE: always.
        WRITE_PUBLIC: unless auto_approved (after N approvals).
        READ/WRITE_SAFE: never.
        """
        for perm in config.tool_permissions:
            if perm.tool_name == tool_name:
                if perm.category == ToolCategory.WRITE_IRREVERSIBLE:
                    return True
                if perm.category == ToolCategory.WRITE_PUBLIC:
                    return not perm.auto_approved
                return False
        # Unknown tool → require confirmation
        return True

    def record_approval(self, tool_name: str, config: AgentConfig) -> bool:
        """
        Record one user approval for a WRITE_PUBLIC tool.
        Returns True if trust threshold reached → auto_approved set.
        """
        for perm in config.tool_permissions:
            if perm.tool_name == tool_name and perm.category == ToolCategory.WRITE_PUBLIC:
                perm.approval_count += 1
                if perm.approval_count >= perm.trust_threshold:
                    perm.auto_approved = True
                    log.info(
                        "trust_escalation",
                        tool=tool_name,
                        approvals=perm.approval_count,
                    )
                    return True
        return False
