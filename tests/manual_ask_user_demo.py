#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_base.tools.tool_user import AskUser


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual interactive AskUser smoke test.")
    parser.add_argument(
        "--mock-answer",
        help="Optional non-interactive answer for automated smoke checks.",
    )
    args = parser.parse_args()

    tool = AskUser()
    kwargs = {}
    if args.mock_answer is not None:
        kwargs["input_stream"] = io.StringIO(args.mock_answer.rstrip("\n") + "\n")
        kwargs["output_stream"] = sys.stdout

    result = tool.call(
        {
            "question": "Please type a short answer to confirm AskUser can read from this terminal.",
            "context": "This is a manual ResearchHarness AskUser smoke test.",
        },
        **kwargs,
    )
    print("\n--- AskUser returned ---")
    print(result)
    return 0 if "User answer" in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
