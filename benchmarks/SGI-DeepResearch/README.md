# SGI-DeepResearch

This directory contains the ResearchHarness benchmark role overlay for
[SGI-DeepResearch](https://huggingface.co/datasets/InternScience/SGI-DeepResearch).

SGI-DeepResearch is a scientific deep-research QA benchmark. It is not a
generic QA task, so the generic QA wrappers are not recommended.

## Recommended Server Command

```bash
python3 /abs/path/to/ResearchHarness/run_server.py \
  --api-runs-dir ./api_runs \
  --host 127.0.0.1 \
  --port 8686 \
  --role-prompt-file /abs/path/to/ResearchHarness/benchmarks/SGI-DeepResearch/role_prompt.md \
  --no-input-wrapper \
  --no-output-wrapper
```

## Rationale

- The benchmark evaluates both the reasoning process and the final answer.
- The official evaluation expects the final answer to be recoverable from an
  `<answer>...</answer>` span.
- Input/output rewriting can remove cited evidence, intermediate calculations,
  or the final answer span.
- The benchmark-specific role prompt should guide the agent to search relevant
  literature, reason, calculate when needed, and then return a complete answer
  directly.

Use `RH` or `RH--<llm-model-name>` in the OpenAI SDK request as usual.
