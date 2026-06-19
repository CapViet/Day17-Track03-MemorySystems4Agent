from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 1.0 (all facts present), 0.5 (some), or 0.0 (none)."""

    if not expected:
        return 0.0
    low = answer.lower()
    found = sum(1 for fact in expected if fact.lower() in low)
    if found == len(expected):
        return 1.0
    if found > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality score: factual coverage + conciseness."""

    coverage = recall_points(answer, expected)
    # Reward concise answers; penalize very long dumps a little.
    length = len(answer)
    conciseness = 1.0 if length <= 600 else max(0.3, 600 / length)
    return round(0.7 * coverage + 0.3 * conciseness, 3)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config
) -> BenchmarkRow:
    agent_tokens = 0
    prompt_tokens = 0
    compactions = 0
    user_ids: set[str] = set()

    # Phase 1: feed every turn of every conversation (one thread per conversation).
    for conv in conversations:
        user_id = conv["user_id"]
        user_ids.add(user_id)
        thread_id = conv["id"]
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

    # Phase 2: cross-session recall in FRESH threads (baseline should fail here).
    recall_total = 0.0
    quality_total = 0.0
    question_count = 0
    for conv in conversations:
        user_id = conv["user_id"]
        for idx, q in enumerate(conv.get("recall_questions", [])):
            recall_thread = f"recall-{conv['id']}-{idx}"
            result = agent.reply(user_id, recall_thread, q["question"])
            answer = result["reply"]
            expected = q["expected_contains"]
            recall_total += recall_points(answer, expected)
            quality_total += heuristic_quality(answer, expected)
            question_count += 1
            agent_tokens += agent.token_usage(recall_thread)
            prompt_tokens += agent.prompt_token_usage(recall_thread)

    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(uid) for uid in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(recall_total / question_count, 3) if question_count else 0.0,
        response_quality=round(quality_total / question_count, 3) if question_count else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=headers, tablefmt="github")
    except Exception:
        widths = [
            max(len(str(h)), *(len(str(row[i])) for row in table))
            for i, h in enumerate(headers)
        ]
        sep = " | "
        lines = [sep.join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
        lines.append("-+-".join("-" * w for w in widths))
        for row in table:
            lines.append(sep.join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
        return "\n".join(lines)


def _run_suite(title: str, dataset: Path, config) -> None:
    conversations = load_conversations(dataset)
    rows = [
        run_agent_benchmark(
            "Baseline", BaselineAgent(config, force_offline=True), conversations, config
        ),
        run_agent_benchmark(
            "Advanced", AdvancedAgent(config, force_offline=True), conversations, config
        ),
    ]
    print(f"\n=== {title} ===")
    print(f"dataset: {dataset.name}, conversations: {len(conversations)}")
    print(format_rows(rows))


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    _run_suite(
        "Standard Benchmark",
        config.data_dir / "conversations.json",
        config,
    )
    _run_suite(
        "Long-Context Stress Benchmark",
        config.data_dir / "advanced_long_context.json",
        config,
    )


if __name__ == "__main__":
    main()
