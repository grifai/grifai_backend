import json
from datetime import datetime
from pathlib import Path


class JarvisMemory:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {
            "contacts": {},
            "my_profile": None,
            "examples": [],
            "stats": {"approved": 0, "revised": 0, "skipped": 0},
        }
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            n = len(self.data.get("contacts", {}))
            print(f"Loaded: {n} contact profiles, {len(self.data.get('examples', []))} examples")
        else:
            print("Memory empty — will build profiles on startup scan")

    def save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Contacts ──────────────────────────────────────────────────────────────

    def set_contact(self, contact_id: str, name: str, profile: dict):
        self.data["contacts"][contact_id] = {
            "name": name,
            "profile": profile,
            "updated": datetime.now().isoformat(),
        }
        self.save()

    def get_contact(self, contact_id: str) -> dict | None:
        c = self.data.get("contacts", {}).get(contact_id)
        if not c:
            return None
        updated = datetime.fromisoformat(c["updated"])
        if (datetime.now() - updated).total_seconds() > 172800:
            return None
        return c

    def delete_contact(self, contact_id: str):
        self.data["contacts"].pop(contact_id, None)
        self.save()

    def get_contact_ai_mode(self, contact_id: str) -> str:
        """Returns 'auto' (AI enabled) or 'never' (AI disabled for this contact)."""
        return self.data.get("contacts", {}).get(contact_id, {}).get("ai_mode", "auto")

    def set_contact_ai_mode(self, contact_id: str, mode: str):
        if contact_id in self.data.get("contacts", {}):
            self.data["contacts"][contact_id]["ai_mode"] = mode
            self.save()

    # ── My profile ────────────────────────────────────────────────────────────

    def set_my_profile(self, profile_text: str):
        self.data["my_profile"] = {
            "text": profile_text,
            "updated": datetime.now().isoformat(),
        }
        self.save()

    def get_my_profile(self) -> str | None:
        mp = self.data.get("my_profile")
        return mp["text"] if mp else None

    # ── Examples ──────────────────────────────────────────────────────────────

    def add_example(
        self,
        sender: str,
        incoming: str,
        draft: str,
        action: str,
        final: str | None = None,
    ):
        entry = {
            "ts": datetime.now().isoformat(),
            "sender": sender,
            "incoming": incoming,
            "llm_draft": draft,
            "action": action,
        }
        if final:
            entry["final_reply"] = final
        self.data["examples"].append(entry)
        self.data["stats"][action] = self.data["stats"].get(action, 0) + 1
        if len(self.data["examples"]) > 300:
            self.data["examples"] = self.data["examples"][-300:]
        self.save()

    def get_decision_examples(self, sender: str | None = None) -> str:
        exs = self.data.get("examples", [])
        approved_actions = ("approved", "revised")

        contact_exs: list[dict] = []
        if sender:
            contact_exs = [e for e in exs if e["sender"] == sender and e["action"] in approved_actions][-10:]

        general_exs = [e for e in exs if e["action"] in approved_actions][-10:]

        combined = {e["ts"]: e for e in contact_exs + general_exs}
        relevant = sorted(combined.values(), key=lambda x: x["ts"])[-15:]

        if not relevant:
            return ""

        lines = ["\n### My past replies (learn from these):"]
        for e in relevant:
            reply = e.get("final_reply") or e["llm_draft"]
            tag = f" [{e['sender']}]" if e.get("sender") else ""
            lines.append(f'- Incoming{tag}: "{e["incoming"][:100]}" -> I replied: "{reply[:150]}"')
        return "\n".join(lines)

    # ── Per-contact examples (for GhostWriter few-shot learning) ─────────────

    def add_contact_example(self, contact_id: str, incoming: str, reply: str):
        """Save an approved (incoming, reply) pair under the contact. Max 20 kept."""
        c = self.data.get("contacts", {}).get(contact_id)
        if c is None:
            return
        examples = c.setdefault("examples", [])
        examples.append(
            {
                "ts": datetime.now().isoformat(),
                "incoming": incoming,
                "reply": reply,
            }
        )
        if len(examples) > 20:
            c["examples"] = examples[-20:]
        self.save()

    def get_contact_examples(self, contact_id: str, n: int = 5) -> list[dict]:
        """Return the last n approved examples for this contact."""
        c = self.data.get("contacts", {}).get(contact_id, {})
        return c.get("examples", [])[-n:]

    def set_contact_style_profile(self, contact_id: str, profile: dict):
        c = self.data.get("contacts", {}).get(contact_id)
        if c is None:
            return
        c["style_profile"] = profile
        self.save()

    def get_contact_style_profile(self, contact_id: str) -> dict | None:
        c = self.data.get("contacts", {}).get(contact_id, {})
        return c.get("style_profile")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def print_stats(self):
        contacts = len(self.data["contacts"])
        examples = len(self.data["examples"])
        stats = self.data["stats"]
        print(f"Contacts: {contacts}  |  Examples: {examples}  |  {stats}")
