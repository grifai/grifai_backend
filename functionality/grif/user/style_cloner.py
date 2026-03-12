"""
StyleCloner — generates a Style Guide from user-provided text samples.

One Sonnet LLM call (authorised call point — Style Guide generation).
The Style Guide (200-300 tokens) is injected as Layer 4 into every agent
that communicates on behalf of this user.

Output format (stored in UserProfileDB.style_guide):
  - Communication style (formal/casual/technical)
  - Tone (direct/empathetic/enthusiastic)
  - Vocabulary preferences
  - Language (ru/en/etc)
  - Response length preference (brief/detailed)
  - Signature phrases or patterns
"""

from __future__ import annotations

import structlog

from grif.llm.gateway import LLMGateway
from grif.user.profile import UserProfileManager

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a communication analyst. "
    "Analyse the provided text samples and extract the author's writing style. "
    "Generate a concise Style Guide (200-300 tokens max) covering:\n"
    "1. Communication style (formal/casual/technical)\n"
    "2. Tone (direct/empathetic/enthusiastic/etc)\n"
    "3. Vocabulary preferences and signature phrases\n"
    "4. Language and locale (ru/en/etc)\n"
    "5. Preferred response length (brief/detailed)\n\n"
    "Output the Style Guide as a structured text, not JSON. "
    "Write in the same language as the samples."
)


class StyleCloner:
    """
    Generates a Style Guide from the user's writing samples.

    Usage:
        cloner = StyleCloner(gateway, profile_manager)
        guide = await cloner.clone(
            user_id="u1",
            samples=["Hey! Check this out...", "Thanks for your quick reply..."],
        )
        # guide is now stored in UserProfileDB.style_guide
    """

    def __init__(
        self,
        gateway: LLMGateway,
        profile_manager: UserProfileManager,
    ) -> None:
        self._gateway = gateway
        self._profile = profile_manager

    async def clone(
        self,
        user_id: str,
        samples: list[str],
        max_samples: int = 10,
    ) -> str:
        """
        Generate and store a Style Guide from writing samples.
        Makes 1 Sonnet LLM call.
        Falls back to a minimal default guide on error.
        """
        if not samples:
            log.warning("style_cloner_no_samples", user_id=user_id)
            return await self._save_default(user_id)

        # Use at most max_samples, trim each to 500 chars
        sample_text = "\n---\n".join(s[:500] for s in samples[:max_samples])
        user_message = f"Text samples:\n{sample_text}\n\nGenerate the Style Guide."

        try:
            response = await self._gateway.complete(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                purpose="style_cloner",
                temperature=0.3,
                max_tokens=350,
                user_id=user_id,
            )
            guide = response.content.strip()

            if len(guide) < 20:
                # LLM returned near-empty response
                guide = self._default_guide()

            await self._profile.set_style_guide(user_id, guide)
            log.info("style_guide_generated", user_id=user_id, length=len(guide))
            return guide

        except Exception as exc:
            log.warning("style_cloner_llm_failed", user_id=user_id, error=str(exc))
            return await self._save_default(user_id)

    async def _save_default(self, user_id: str) -> str:
        guide = self._default_guide()
        await self._profile.set_style_guide(user_id, guide)
        return guide

    @staticmethod
    def _default_guide() -> str:
        return (
            "Communication style: casual and direct.\n"
            "Tone: friendly, concise.\n"
            "Language: Russian (ru).\n"
            "Response length: brief and to the point.\n"
            "No signature phrases identified."
        )
