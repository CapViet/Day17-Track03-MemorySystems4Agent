from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Isolated config: state under tmp_path, low compact threshold for tests."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model = ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=120,
        compact_keep_messages=2,
        model=model,
        judge_model=model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    store = agent.profile_store

    # default profile before anything is written
    assert "User Profile" in store.read_text("u1")
    assert store.file_size("u1") == 0

    store.upsert_fact("u1", "name", "DũngCT")
    assert "DũngCT" in store.read_text("u1")
    assert store.file_size("u1") > 0

    # edit replaces in place
    assert store.edit_text("u1", "DũngCT", "DũngCT Updated") is True
    assert "DũngCT Updated" in store.read_text("u1")
    assert store.edit_text("u1", "not-there", "x") is False


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    thread = "long-thread"
    long_turn = "Đây là một câu rất dài để ép compact memory phải kích hoạt. " * 3
    for _ in range(8):
        agent.reply("u1", thread, long_turn)
    assert agent.compaction_count(thread) >= 1


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    # Session 1: user states their name.
    advanced.reply("u1", "s1", "Chào bạn, mình tên là DũngCT.")
    baseline.reply("u1", "s1", "Chào bạn, mình tên là DũngCT.")

    # Session 2 (fresh thread): ask for the name.
    q = "Mình tên gì?"
    adv_answer = advanced.reply("u1", "s2", q)["reply"]
    base_answer = baseline.reply("u1", "s2", q)["reply"]

    assert "DũngCT" in adv_answer  # advanced recalls via User.md
    assert "DũngCT" not in base_answer  # baseline forgot across sessions


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    long_turn = "Một đoạn nội dung dài lặp lại nhiều lần để kéo dài ngữ cảnh hội thoại. " * 3
    for _ in range(12):
        advanced.reply("u1", "t-adv", long_turn)
        baseline.reply("u1", "t-base", long_turn)

    # Compact memory keeps advanced's per-turn context bounded, so cumulative
    # prompt load stays well below the baseline's ever-growing history.
    assert advanced.compaction_count("t-adv") >= 1
    assert advanced.prompt_token_usage("t-adv") < baseline.prompt_token_usage("t-base")
