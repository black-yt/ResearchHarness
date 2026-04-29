# ResearchClawBench

This directory contains the tracked files needed to document how `ResearchHarness`
should be integrated into `ResearchClawBench`.

ResearchHarness is intended to serve here as a **general and fair execution
substrate** for tool-using LLM evaluation, while `ResearchClawBench` remains in
charge of task construction, hidden-answer isolation, and scoring.

## Recommended `agents.json` Entry

Use a single direct command that launches the thin top-level ResearchHarness
entrypoint.

```json
{
  "researchharness": {
    "label": "ResearchHarness",
    "icon": "H",
    "logo": "/static/logos/rh.svg",
    "cmd": "python3 /abs/path/to/ResearchHarness/run_agent.py <PROMPT> --workspace-root <WORKSPACE> --role-prompt-file /abs/path/to/ResearchHarness/benchmarks/ResearchClawBench/role_prompt.md --trace-dir <WORKSPACE>"
  }
}
```

## Why This Shape

- `ResearchClawBench` already prepares the workspace, writes `INSTRUCTIONS.md`,
  and isolates hidden checklist data.
- `ResearchHarness` should only execute the agent through a stable harness
  interface.
- The command stays unchanged. The entrypoint automatically selects the
  lightweight adapter in `benchmarks/ResearchClawBench/adapter.py` when this
  benchmark role prompt is used.

## Notes

- Replace `/abs/path/to/ResearchHarness/` with the real local checkout path.
- The command should stay one-line and non-interactive.
- The adapter prevents premature termination on long tasks by refusing to accept
  plain-text completion before `report/report.md` exists in the workspace.
- The adapter excludes `AskUser`; RCB runs must remain fully non-interactive.
- Any local batch helpers or ad hoc benchmark scripts should remain untracked
  and live outside the formal integration contract.
