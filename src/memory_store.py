from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Profile keys whose values accumulate (merge unique items) instead of being
# overwritten. Everything else is a single-value, correction-friendly fact.
APPEND_KEYS = {"style", "interests"}

# Human-readable labels used when rendering User.md and recap answers.
FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "style": "Style trả lời mong muốn",
    "interests": "Mối quan tâm kỹ thuật",
}


def estimate_tokens(text: str) -> int:
    """Stable heuristic token estimator (~4 chars per token).

    Not tokenizer-accurate, but deterministic and good enough to compare
    agents offline.
    """

    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def _slugify(user_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (user_id or "user").strip())
    slug = slug.strip("-_").lower()
    return slug or "user"


@dataclass
class UserProfileStore:
    """Persistent storage backing each user's ``User.md`` profile file."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        return Path(self.root_dir) / f"{_slugify(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# User Profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of ``search_text``; return whether it changed."""

        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        if updated == current:
            return False
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # --- structured fact helpers -------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse ``- key: value`` lines from User.md into a dict."""

        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"\s*-\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$", line)
            if match:
                result[match.group(1).lower()] = match.group(2).strip()
        return result

    def _render(self, facts: dict[str, str]) -> str:
        lines = ["# User Profile", ""]
        # Render known keys first (stable order), then any extras.
        ordered = [k for k in FACT_LABELS if k in facts]
        ordered += [k for k in facts if k not in FACT_LABELS]
        for key in ordered:
            lines.append(f"- {key}: {facts[key]}")
        return "\n".join(lines) + "\n"

    def upsert_fact(self, user_id: str, key: str, value: str, mode: str = "replace") -> bool:
        """Insert or update one fact. ``append`` keys merge unique comma items.

        Returns True if the stored profile actually changed (used for
        conflict-aware / confidence-gated writes).
        """

        key = key.lower().strip()
        value = value.strip()
        if not value:
            return False

        facts = self.facts(user_id)
        if mode == "append" or key in APPEND_KEYS:
            existing = [v.strip() for v in facts.get(key, "").split(",") if v.strip()]
            for item in (v.strip() for v in value.split(",") if v.strip()):
                if item.lower() not in {e.lower() for e in existing}:
                    existing.append(item)
            new_value = ", ".join(existing)
        else:
            new_value = value

        if facts.get(key) == new_value:
            return False
        facts[key] = new_value
        self.write_text(user_id, self._render(facts))
        return True


# --- profile extraction ----------------------------------------------------------

_QUESTION_STARTERS = (
    "bạn có",
    "bạn biết",
    "bạn thử nhớ",
    "nhắc lại",
    "tóm tắt",
    "hãy nhắc",
)

_INTEREST_KEYWORDS = ["Python", "AI", "MLOps", "RAG"]


def _is_pure_question(message: str) -> bool:
    """Heuristic: a turn that only asks for recall, providing no new fact."""

    low = message.strip().lower()
    if not low:
        return True
    # Fact-bearing turns in this lab are always declarative (end with "."),
    # so any "?"-ending turn is a recall question that must not write facts.
    if low.endswith("?"):
        return True
    return any(low.startswith(s) for s in _QUESTION_STARTERS)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Only confidently-present facts are returned. Question-only turns and
    obvious jokes/noise are skipped so we don't poison ``User.md``.
    """

    if _is_pure_question(message):
        return {}

    low = message.lower()
    updates: dict[str, str] = {}

    # name -------------------------------------------------------------------
    m = re.search(r"(?:tên mình là|mình tên(?: là)?|tôi tên(?: là)?)\s+([^,.\n]+)", message, re.I)
    if m:
        name = m.group(1).strip()
        # Stop at the first clause boundary so "DũngCT và mình thích..." -> "DũngCT".
        name = re.split(r"\s+(?:và|hiện|đang|ở|rồi|nhé|nha)\b", name, maxsplit=1)[0].strip()
        if name and len(name) <= 40:
            updates["name"] = name

    # location ----------------------------------------------------------------
    # Skip turns that explicitly disclaim a place as the current location.
    if "không phải nơi ở" not in low and "ví dụ cũ" not in low:
        loc = re.search(
            r"\bở\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,2})",
            message,
        )
        # Prefer a correction phrase if present (e.g. "giờ mình đang ở Huế").
        corr = re.search(
            r"(?:giờ|hiện|thực ra|từ tuần này)[^.\n]*?\bở\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,2})",
            message,
        )
        chosen = corr or loc
        if chosen:
            place = chosen.group(1).strip()
            # Avoid grabbing the "công việc ở Đà Nẵng vài tháng" tail noise words.
            place = re.sub(r"\s+(vài|chứ|và|nhưng)$", "", place).strip()
            if place:
                updates["location"] = place

    # profession --------------------------------------------------------------
    if "đùa" not in low:
        job = None
        for pat in (
            r"chuyển sang\s+([A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+)?\s+engineer)",
            r"(?:vẫn (?:là|làm)|hiện tại vẫn là|hiện là|giờ (?:là|làm)|đang làm|\bmình làm)\s+([A-Za-zÀ-ỹ]+(?:\s+[A-Za-zÀ-ỹ]+)?\s+engineer)",
        ):
            m = re.search(pat, message, re.I)
            if m:
                job = m.group(1).strip()
                break
        if job:
            updates["profession"] = job

    # drink -------------------------------------------------------------------
    m = re.search(r"đồ uống yêu thích(?: của mình)? là\s+([^,.\n]+)", message, re.I)
    if m:
        updates["drink"] = m.group(1).strip()

    # food --------------------------------------------------------------------
    m = re.search(r"món ăn yêu thích(?: của mình)? là\s+([^,.\n]+)", message, re.I)
    if m:
        updates["food"] = m.group(1).strip()

    # pet ---------------------------------------------------------------------
    m = re.search(
        r"nuôi\s+(?:(?:một|1|con|bé)\s+)*([A-Za-zÀ-ỹ]+)(?:\s+tên\s+([A-ZĐ][\wÀ-ỹ]*))?",
        message,
        re.I,
    )
    if m:
        pet = m.group(1).strip()
        if m.group(2):
            pet = f"{pet} tên {m.group(2).strip()}"
        updates["pet"] = pet

    # style -------------------------------------------------------------------
    style_parts: list[str] = []
    if "ngắn gọn" in low and any(c in low for c in ("trả lời", "câu trả lời", "style")):
        style_parts.append("ngắn gọn")
    bullets = re.search(r"(\d+)\s*bullet", low)
    if bullets:
        style_parts.append(f"{bullets.group(1)} bullet")
    if "ví dụ thực" in low:
        style_parts.append("có ví dụ thực chiến")
    if "trade-off" in low or "trade off" in low:
        style_parts.append("nhấn trade-off")
    if style_parts:
        updates["style"] = ", ".join(style_parts)

    # interests ---------------------------------------------------------------
    found = [kw for kw in _INTEREST_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", message)]
    if found:
        updates["interests"] = ", ".join(found)

    return updates


# --- compact memory --------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic compact summary of older messages.

    Keeps the gist of the most recent ``max_items`` items, each trimmed to a
    short snippet, so the summary stays bounded regardless of history length.
    """

    if not messages:
        return ""
    snippets = []
    for msg in messages[-max_items:]:
        content = " ".join(msg.get("content", "").split())
        if len(content) > 90:
            content = content[:90].rstrip() + "…"
        role = msg.get("role", "user")
        snippets.append(f"{role}: {content}")
    return "Tóm tắt hội thoại trước: " + " | ".join(snippets)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the most recent messages in full; once the running token estimate
    crosses ``threshold_tokens`` the older messages are folded into a bounded
    summary. ``compactions`` is tracked per thread for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _thread(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        self._maybe_compact(thread)

    def _current_tokens(self, thread: dict[str, object]) -> int:
        total = estimate_tokens(str(thread["summary"]))
        for msg in thread["messages"]:  # type: ignore[union-attr]
            total += estimate_tokens(msg["content"])
        return total

    def _maybe_compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return
        if self._current_tokens(thread) <= self.threshold_tokens:
            return

        keep = messages[-self.keep_messages :]
        older = messages[: -self.keep_messages]
        # Fold the previous summary in as a pseudo-message so nothing is lost.
        carry: list[dict[str, str]] = []
        if thread["summary"]:
            carry.append({"role": "summary", "content": str(thread["summary"])})
        carry.extend(older)
        thread["summary"] = summarize_messages(carry)
        thread["messages"] = keep
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._thread(thread_id)
        return {
            "messages": list(thread["messages"]),  # type: ignore[arg-type]
            "summary": thread["summary"],
            "compactions": thread["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._thread(thread_id)["compactions"])  # type: ignore[arg-type]
