# SGI-Reasoning

This directory contains the ResearchHarness benchmark role overlay for
[SGI-Reasoning](https://huggingface.co/datasets/InternScience/SGI-Reasoning).

SGI-Reasoning is a multimodal scientific reasoning benchmark. It is not a
generic VQA task, so the generic QA wrappers are not recommended.

## Recommended Server Command

```bash
python3 /abs/path/to/ResearchHarness/run_server.py \
  --api-runs-dir ./api_runs \
  --host 127.0.0.1 \
  --port 8686 \
  --role-prompt-file /abs/path/to/ResearchHarness/benchmarks/SGI-Reasoning/role_prompt.md \
  --no-input-wrapper \
  --no-output-wrapper
```

## Rationale

- The benchmark evaluates both the multiple-choice answer and the reasoning
  validity.
- The official answer extraction expects the final option letter in
  `\boxed{...}` format.
- Input/output rewriting can drop images, options, or the exact boxed answer.
- The benchmark-specific role prompt should guide the agent to inspect images,
  use image processing or literature search when useful, and return complete
  reasoning plus a boxed option.

Use `RH` or `RH--<llm-model-name>` in the OpenAI SDK request as usual.
