#!/usr/bin/env python3

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import TEST_RUNS_DIR, bootstrap, load_trace_records, preview, single_trace_path


TMP_DIR = TEST_RUNS_DIR / "agent_extension_checks"


@dataclass
class AgentExtensionResult:
    status: str
    detail: str
    output_preview: str


def main() -> int:
    bootstrap()

    from agent_base import agent_role
    from agent_base.prompt import SYSTEM_PROMPT
    from agent_base.react_agent import MultiTurnReactAgent, resolve_agent_class_for_role_prompt_files
    from benchmarks.ResearchClawBench.adapter import ResearchClawBenchAgent

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    trace_dir = TMP_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for existing_trace in trace_dir.glob("*.jsonl"):
        existing_trace.unlink()

    @agent_role(
        name="judge",
        role_prompt="You are the Judge agent. Evaluate the provided artifact carefully and return JSON only.",
        function_list=[],
    )
    class DummyJudgeAgent(MultiTurnReactAgent):
        def __init__(self):
            super().__init__(
                llm={
                    "model": "fake-model",
                    "generate_cfg": {
                        "max_input_tokens": 10000,
                        "max_retries": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "presence_penalty": 0.0,
                    },
                },
                trace_dir=str(trace_dir),
            )
            self.seen_messages = []

        def call_llm_api(self, msgs, max_tries=10, runtime_deadline=None):
            self.seen_messages = msgs
            return {
                "status": "ok",
                "finish_reason": "stop",
                "content": '{"overall_score": 8.5, "verdict": "good"}',
                "tool_calls": [],
            }

    agent = DummyJudgeAgent()
    session = agent._run_session("Review this artifact.", workspace_root=str(TMP_DIR))
    trace_path = single_trace_path(trace_dir)
    if trace_path is None:
        raise RuntimeError(f"Expected exactly one trace file in {trace_dir}")
    rows = load_trace_records(trace_path)

    system_message = agent.seen_messages[0]["content"] if agent.seen_messages else ""
    rcb_prompt_path = ROOT / "benchmarks" / "ResearchClawBench" / "role_prompt.md"
    resolved_default_cls = resolve_agent_class_for_role_prompt_files([])
    resolved_rcb_cls = resolve_agent_class_for_role_prompt_files([str(rcb_prompt_path)])
    preview_text = preview(
        json.dumps(
            {
                "termination": session.get("termination"),
                "result_text": session.get("result_text"),
                "tool_names": agent.tool_names,
                "trace_roles": [row.get("role") for row in rows],
                "default_agent_class": resolved_default_cls.__name__,
                "rcb_agent_class": resolved_rcb_cls.__name__,
                "system_prompt_tail": system_message[-300:],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    ok = (
        agent.tool_names == []
        and agent._native_tools == []
        and SYSTEM_PROMPT.splitlines()[0] in system_message
        and "You are the Judge agent." in system_message
        and session.get("result_text") == '{"overall_score": 8.5, "verdict": "good"}'
        and not any(row.get("role") == "tool" for row in rows)
        and any(row.get("termination") == "result" for row in rows)
        and resolved_default_cls is MultiTurnReactAgent
        and resolved_rcb_cls is ResearchClawBenchAgent
    )

    result = AgentExtensionResult(
        status="PASS" if ok else "FAIL",
        detail="Role-prompt composition and no-tool agent defaults are working."
        if ok
        else "Agent extension checks failed.",
        output_preview=preview_text,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
