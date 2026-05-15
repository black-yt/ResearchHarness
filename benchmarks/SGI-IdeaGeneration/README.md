# SGI-IdeaGeneration

This directory contains the ResearchHarness benchmark role overlay for
[SGI-IdeaGeneration](https://huggingface.co/datasets/InternScience/SGI-IdeaGeneration).

SGI-IdeaGeneration is a scientific proposal-generation benchmark. It is not a
generic summarization task, so the generic QA wrappers are not recommended.

## Recommended Server Command

```bash
python3 /abs/path/to/ResearchHarness/run_server.py \
  --api-runs-dir ./api_runs \
  --host 127.0.0.1 \
  --port 8686 \
  --role-prompt-file /abs/path/to/ResearchHarness/benchmarks/SGI-IdeaGeneration/role_prompt.md \
  --no-input-wrapper \
  --no-output-wrapper
```

## Rationale

- The benchmark parses a structured proposal with fixed fields.
- The scoring emphasizes effectiveness, novelty, detailedness, and feasibility.
- Input rewriting can weaken domain context or constraints.
- Output rewriting can break JSON or convert a proposal into a generic summary.
- The benchmark-specific role prompt should guide the agent to inspect recent
  literature, identify limitations, and propose a concrete new method directly.

Use `RH` or `RH--<llm-model-name>` in the OpenAI SDK request as usual.
