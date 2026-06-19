from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    APPEND_KEYS,
    FACT_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. within-session (short-term) memory
    2. persistent ``User.md`` profile per user
    3. compact memory that summarizes old turns on long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self.force_offline and self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # --- offline deterministic path ---------------------------------------------

    def _persist_profile(self, user_id: str, message: str) -> None:
        """Extract stable facts and write them to User.md with conflict handling.

        Bonus behaviour:
        - confidence gate: only writes facts the extractor is confident about
          (question-only / joke turns return nothing).
        - conflict handling: single-value keys overwrite the old value so a
          correction (e.g. Đà Nẵng -> Huế) never coexists with the stale fact.
        """

        for key, value in extract_profile_updates(message).items():
            mode = "append" if key in APPEND_KEYS else "replace"
            self.profile_store.upsert_fact(user_id, key, value, mode=mode)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        self._persist_profile(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        reply = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", reply)

        out_tokens = estimate_tokens(reply)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + out_tokens

        return {
            "reply": reply,
            "agent_tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: User.md + compact summary + kept turns.

        This is bounded by compact memory, which is what lets the advanced agent
        stay cheap on long threads where the baseline keeps growing.
        """

        ctx = self.compact_memory.context(thread_id)
        parts = [self.profile_store.read_text(user_id), str(ctx["summary"])]
        parts.extend(m["content"] for m in ctx["messages"])  # type: ignore[index]
        return estimate_tokens("\n".join(p for p in parts if p))

    def _profile_recap(self, user_id: str) -> str:
        facts = self.profile_store.facts(user_id)
        if not facts:
            return "Mình chưa có thông tin nào được lưu trong User.md."
        lines = ["Dựa trên User.md đã lưu:"]
        for key in [k for k in FACT_LABELS if k in facts] + [
            k for k in facts if k not in FACT_LABELS
        ]:
            label = FACT_LABELS.get(key, key)
            lines.append(f"- {label}: {facts[key]}")
        return "\n".join(lines)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer grounded in persisted memory.

        Recall questions return the persisted profile (so cross-session recall
        works); fact-providing turns get a short acknowledgement so short
        conversations stay cheap.
        """

        low = message.lower()
        if message.strip().endswith("?") or any(
            cue in low
            for cue in ("nhắc", "tóm tắt", "mô tả", "là ai", "bạn biết")
        ):
            return self._profile_recap(user_id)
        return "Đã ghi nhớ vào User.md và cập nhật bộ nhớ ngắn hạn."

    # --- optional live path ------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Persist profile facts deterministically even on the live path.
        self._persist_profile(user_id, message)
        profile = self.profile_store.read_text(user_id)
        result = self.langchain_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": f"Hồ sơ người dùng:\n{profile}"},
                    {"role": "user", "content": message},
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result["messages"][-1].content
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.compact_memory.append(thread_id, "user", message)
        self.compact_memory.append(thread_id, "assistant", reply)
        out_tokens = estimate_tokens(reply)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + out_tokens
        return {
            "reply": reply,
            "agent_tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with User.md tools + summarization middleware.

        Returns None (offline) when dependencies or credentials are missing.
        """

        try:
            from langchain.agents import create_agent
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
        except Exception:
            return None

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            return None

        store = self.profile_store

        @tool
        def read_user_profile(user_id: str) -> str:
            """Đọc hồ sơ User.md của người dùng."""
            return store.read_text(user_id)

        @tool
        def write_user_fact(user_id: str, key: str, value: str) -> str:
            """Lưu hoặc cập nhật một fact ổn định vào User.md."""
            changed = store.upsert_fact(user_id, key, value)
            return "updated" if changed else "unchanged"

        self.langchain_agent = create_agent(
            model,
            tools=[read_user_profile, write_user_fact],
            checkpointer=InMemorySaver(),
            system_prompt=(
                "Bạn là advanced agent có persistent memory qua User.md và compact "
                "memory cho hội thoại dài. Trả lời ngắn gọn, ưu tiên thông tin đã đính chính."
            ),
        )
        return self.langchain_agent
