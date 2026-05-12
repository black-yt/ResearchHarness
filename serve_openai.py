"""Run ResearchHarness as a minimal OpenAI-compatible synchronous API server."""

from __future__ import annotations

import argparse
import sys

from agent_base.utils import PROJECT_ROOT, MissingRequiredEnvError, load_dotenv, require_required_env
from api.openai_server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve ResearchHarness through /v1/chat/completions.")
    parser.add_argument(
        "--workspace-root",
        required=True,
        help="Directory where the server creates one isolated subdirectory per request.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind. Defaults to 8000.")
    parser.add_argument(
        "--role-prompt-file",
        action="append",
        default=[],
        dest="role_prompt_files",
        help="Optional role prompt file appended to the base ResearchHarness prompt.",
    )
    parser.add_argument(
        "--input-wrapper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the input LLM wrapper. Enabled by default.",
    )
    parser.add_argument(
        "--output-wrapper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the output LLM wrapper. Enabled by default.",
    )
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        require_required_env("ResearchHarness API server")
        serve(
            workspace_root=args.workspace_root,
            host=args.host,
            port=args.port,
            role_prompt_files=list(args.role_prompt_files),
            input_wrapper=args.input_wrapper,
            output_wrapper=args.output_wrapper,
        )
    except (MissingRequiredEnvError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
