import argparse
import json
from pathlib import Path
from typing import Any, Optional


class ConsoleEventPrinter:
    def __init__(self, *, model_name: str, workspace_root: Path, user_request: str):
        self.model_name = model_name
        self.workspace_root = workspace_root
        self.user_request = user_request.strip()
        self._last_round_printed: Optional[int] = None

    def print_header(self) -> None:
        print(f"Model: {self.model_name}")
        print(f"Workspace: {self.workspace_root}")
        print("User Request:")
        print(self.user_request)

    def _print_round_header(self, turn_index: int) -> None:
        if turn_index <= 0 or self._last_round_printed == turn_index:
            return
        self._last_round_printed = turn_index
        print(f"\n== Round {turn_index} ==")

    def _format_tool_call(self, tool_name: str, tool_args: Any) -> str:
        try:
            tool_args_text = json.dumps(tool_args, ensure_ascii=False, indent=2)
        except TypeError:
            tool_args_text = str(tool_args)
        return f"- {tool_name}\n{tool_args_text}"

    def handle_event(self, row: dict[str, Any]) -> None:
        role = str(row.get("role", ""))
        turn_index = int(row.get("turn_index", 0) or 0)
        text = str(row.get("text", ""))
        tool_names = row.get("tool_names") if isinstance(row.get("tool_names"), list) else []
        tool_arguments = row.get("tool_arguments") if isinstance(row.get("tool_arguments"), list) else []
        finish_reason = str(row.get("finish_reason", ""))
        error = str(row.get("error", ""))

        if role == "system":
            return

        if role == "user":
            if turn_index == 0:
                return
            self._print_round_header(turn_index)
            print("[Runtime Message]")
            print(text)
            return

        if role == "assistant":
            self._print_round_header(turn_index)
            print("[Assistant]")
            if tool_names:
                if text.strip():
                    print(text)
                else:
                    suffix = f" finish_reason={finish_reason}" if finish_reason else ""
                    print(f"(no text; native tool-calls only.{suffix})")
                print("[Assistant Tool Calls]")
                for idx, tool_name in enumerate(tool_names):
                    tool_args = tool_arguments[idx] if idx < len(tool_arguments) else {}
                    print(self._format_tool_call(str(tool_name), tool_args))
            elif text.strip():
                print(text)
            else:
                suffix = f" finish_reason={finish_reason}" if finish_reason else ""
                print(f"(empty assistant output.{suffix})")
            if error:
                print(f"[Assistant Error] {error}")
            return

        if role == "tool":
            self._print_round_header(turn_index)
            tool_name = str(tool_names[0]) if tool_names else "Tool"
            print(f"[{tool_name} Result]")
            print(text)
            if error:
                print(f"[{tool_name} Error] {error}")
            return

        if role == "runtime":
            print("[Runtime]")
            print(text)
            if error:
                print(f"[Runtime Error] {error}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show a minimal example of the CLI console event formatter.")
    parser.parse_args(argv)
    printer = ConsoleEventPrinter(model_name="demo-model", workspace_root=Path("."), user_request="demo question")
    printer.print_header()
    printer.handle_event(
        {
            "role": "assistant",
            "turn_index": 1,
            "text": "",
            "tool_names": ["Read"],
            "tool_arguments": [{"path": "demo.txt"}],
            "termination": "",
            "error": "",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
