from __future__ import annotations

import os
import sys
from typing import Any, TextIO, Union

from agent_base.tools.tooling import ToolBase


class AskUser(ToolBase):
    name = "AskUser"
    description = (
        "Ask the human user a concise clarification question when progress depends on "
        "information, preference, or approval that cannot be determined from the workspace or other tools."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The concise question to ask the user.",
            },
            "context": {
                "type": "string",
                "description": "Optional brief context explaining why the question is necessary.",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }

    def call(self, params: Union[str, dict], **kwargs: Any) -> str:
        try:
            parsed = self.parse_json_args(params)
        except ValueError as exc:
            return f"[AskUser] {exc}"

        question = str(parsed.get("question", "")).strip()
        context = str(parsed.get("context", "") or "").strip()
        if not question:
            return "[AskUser] question must be a non-empty string."

        input_stream = kwargs.get("input_stream")
        output_stream = kwargs.get("output_stream")
        close_stream = False
        if input_stream is None or output_stream is None:
            input_stream, output_stream, close_stream = _resolve_interactive_streams()
        if input_stream is None or output_stream is None:
            return (
                "[AskUser] Cannot ask the user because no interactive terminal is available. "
                "Continue with available evidence, or state the blocker if the answer is essential."
            )

        try:
            _write_question(output_stream, question=question, context=context)
            answer = input_stream.readline()
        except OSError as exc:
            return f"[AskUser] Failed to read user input: {exc}"
        finally:
            if close_stream:
                try:
                    input_stream.close()
                except OSError:
                    pass

        answer = str(answer or "").strip()
        if not answer:
            return "[AskUser] User answer was empty."
        return f"[AskUser] User answer:\n{answer}"


def _resolve_interactive_streams() -> tuple[TextIO | None, TextIO | None, bool]:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return sys.stdin, sys.stdout, False
    if os.name == "nt":
        return None, None, False
    try:
        tty = open("/dev/tty", "r+", encoding="utf-8")
    except OSError:
        return None, None, False
    return tty, tty, True


def _write_question(output_stream: TextIO, *, question: str, context: str = "") -> None:
    output_stream.write("\n[AskUser]\n")
    if context:
        output_stream.write(f"Context: {context}\n")
    output_stream.write(f"Question: {question}\n> ")
    output_stream.flush()
