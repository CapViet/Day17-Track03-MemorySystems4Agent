from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: naive short-term memory only.

    - remembers within the same thread
    - has no persistent ``User.md``
    - forgets long-term facts across new threads (no cross-session recall)
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self.force_offline and self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        # Baseline naively re-processes the entire thread history every turn.
        context_text = "\n".join(m["content"] for m in session.messages)
        prompt_tokens = estimate_tokens(context_text)
        session.prompt_tokens_processed += prompt_tokens

        reply = self._naive_reply(session, message)
        session.messages.append({"role": "assistant", "content": reply})

        out_tokens = estimate_tokens(reply)
        session.token_usage += out_tokens

        return {
            "reply": reply,
            "agent_tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }

    def _naive_reply(self, session: SessionState, message: str) -> str:
        """Deterministic reply that only knows the current thread.

        It never surfaces long-term facts: when asked a recall question in a
        fresh thread it has nothing stored, which is exactly the baseline's
        weakness we want the benchmark to expose.
        """

        if message.strip().endswith("?"):
            # Only this thread's turns are available — no persistent profile.
            user_turns = [m["content"] for m in session.messages if m["role"] == "user"]
            if len(user_turns) <= 1:
                return (
                    "Trong phiên hiện tại mình chưa được cung cấp thông tin đó, "
                    "nên mình không nhớ được qua phiên mới."
                )
            return (
                "Mình chỉ nhớ những gì vừa trao đổi trong phiên này, "
                "không có bộ nhớ dài hạn giữa các phiên."
            )
        return "Đã ghi nhận trong phiên hiện tại (không lưu dài hạn)."

    def _maybe_build_langchain_agent(self):
        """Optionally wire a live LangGraph agent with an in-thread checkpointer.

        Returns None (offline) when dependencies or credentials are missing.
        """

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except Exception:
            return None

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            return None

        self.langchain_agent = create_agent(
            model,
            tools=[],
            checkpointer=InMemorySaver(),
            system_prompt=(
                "Bạn là baseline agent chỉ có short-term memory trong cùng một thread. "
                "Không có bộ nhớ dài hạn giữa các phiên."
            ),
        )
        return self.langchain_agent

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result["messages"][-1].content
        session = self.sessions.setdefault(thread_id, SessionState())
        prompt_tokens = estimate_tokens(message)
        out_tokens = estimate_tokens(reply)
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += out_tokens
        return {
            "reply": reply,
            "agent_tokens": out_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }
