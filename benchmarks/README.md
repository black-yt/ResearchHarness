# Benchmarks

This folder records benchmark-specific integration contracts that live
**outside** `agent_base` so the core harness stays generic, lightweight, and
fair across different evaluations.

| Benchmark | Directory | Tracked contract |
| --- | --- | --- |
| [ResearchClawBench](https://github.com/InternScience/ResearchClawBench) | `benchmarks/ResearchClawBench/` | `README.md` + `role_prompt.md` + `adapter.py` |
| QA / VQA-style benchmarks | `benchmarks/QA/` | `README.md` + `role_prompt.md` |

## Notes

- `agent_base/` stays focused on the reusable harness runtime.
- Benchmark-specific prompts, adapters, and integration notes should live under
  their own benchmark subdirectory.
- Local benchmark helpers may exist for private experimentation, but they do
  not define the formal external integration contract.
