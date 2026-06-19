"""Live demo: run the agents against a real model (default: local Ollama).

Shows the core memory difference end-to-end with a real LLM:
- Baseline forgets across a NEW thread (short-term memory only).
- Advanced recalls across a NEW thread via persistent User.md.

Usage:
    LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:3b python live_demo.py

Falls back to printing a note if no live model can be built.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def _hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    print(f"Provider: {config.model.provider} | model: {config.model.model_name}")

    # Start each demo from a clean profile directory.
    shutil.rmtree(config.state_dir / "profiles", ignore_errors=True)

    user = "demo_user"

    # --- Baseline (live) ----------------------------------------------------
    _hr("BASELINE (live) — short-term memory only")
    baseline = BaselineAgent(config, force_offline=False)
    if baseline._maybe_build_langchain_agent() is None:
        print("Could not build a live model — check the provider/SDK/credentials.")
        return

    print("[thread t1] user: Chào bạn, mình tên là DũngCT và mình thích cà phê sữa đá.")
    baseline.reply(user, "t1", "Chào bạn, mình tên là DũngCT và mình thích cà phê sữa đá.")
    print("[thread t1] user: Mình tên gì?")
    print("  agent:", baseline.reply(user, "t1", "Mình tên gì?")["reply"].strip())
    print("\n[thread t2 — NEW thread] user: Mình tên gì?")
    print("  agent:", baseline.reply(user, "t2", "Mình tên gì?")["reply"].strip())
    print("  -> Baseline should NOT know the name in a fresh thread.")

    # --- Advanced (live) ----------------------------------------------------
    _hr("ADVANCED (live) — persistent User.md across threads")
    advanced = AdvancedAgent(config, force_offline=False)
    if advanced._maybe_build_langchain_agent() is None:
        print("Could not build a live model — check the provider/SDK/credentials.")
        return

    print("[thread s1] user: Chào bạn, mình tên là DũngCT và mình thích cà phê sữa đá.")
    advanced.reply(user, "s1", "Chào bạn, mình tên là DũngCT và mình thích cà phê sữa đá.")
    print("\nUser.md đã ghi:")
    print("  " + advanced.profile_store.read_text(user).replace("\n", "\n  ").strip())
    print("\n[thread s2 — NEW thread] user: Mình tên gì và thích uống gì?")
    print("  agent:", advanced.reply(user, "s2", "Mình tên gì và thích uống gì?")["reply"].strip())
    print("  -> Advanced recalls via User.md even in a fresh thread.")


if __name__ == "__main__":
    main()
