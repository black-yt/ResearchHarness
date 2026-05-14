"""Launch the local ResearchHarness browser UI."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

import uvicorn

from agent_base.utils import read_role_prompt_files
from frontend.local_server import app, configure_frontend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local ResearchHarness frontend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind. Default: 8765")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument("--trace-dir", help="Optional directory where frontend agent traces are written.")
    parser.add_argument(
        "--role-prompt-file",
        action="append",
        default=[],
        dest="role_prompt_files",
        metavar="PATH",
        help="Append one role-specific prompt file to the frontend agent. May be passed multiple times.",
    )
    parser.add_argument(
        "--extra-tool",
        action="append",
        default=[],
        dest="extra_tools",
        metavar="NAME",
        help="Enable one optional extra tool in frontend runs. Currently supported: str_replace_editor. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    try:
        role_prompt = read_role_prompt_files(args.role_prompt_files)
        configure_frontend(role_prompt=role_prompt, trace_dir=args.trace_dir, extra_tools=list(args.extra_tools))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"ResearchHarness frontend: {url}")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
