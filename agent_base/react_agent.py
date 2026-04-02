import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
import tiktoken
from agent_base.console_utils import ConsoleEventPrinter
from agent_base.prompt import SYSTEM_PROMPT
from agent_base.trace_utils import FlatTraceWriter
from agent_base.tools.tooling import normalize_workspace_root
from agent_base.tools.tool_file import Edit, Glob, Grep, Read, ReadImage, ReadPDF, Write
from agent_base.tools.tool_runtime import Bash, TerminalInterrupt, TerminalKill, TerminalRead, TerminalStart, TerminalWrite
from agent_base.tools.tool_web import ScholarSearch, WebFetch, WebSearch
from agent_base.utils import PROJECT_ROOT, env_flag, load_dotenv, safe_jsonable

import datetime
import random
import time

AVAILABLE_TOOLS = [
    Glob(),
    Grep(),
    Read(),
    ReadPDF(),
    ReadImage(),
    Write(),
    Edit(),
    Bash(),
    WebSearch(),
    ScholarSearch(),
    WebFetch(),
    TerminalStart(),
    TerminalWrite(),
    TerminalRead(),
    TerminalInterrupt(),
    TerminalKill(),
]
AVAILABLE_TOOL_MAP = {tool.name: tool for tool in AVAILABLE_TOOLS}
DEFAULT_IMAGE_TOKEN_ESTIMATE = 1536


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def max_llm_calls_per_run() -> int:
    return int(os.getenv("MAX_LLM_CALL_PER_RUN", "100"))


def max_agent_runtime_seconds() -> int:
    return int(os.getenv("MAX_AGENT_RUNTIME_SECONDS", str(150 * 60)))


def llm_max_output_tokens() -> int:
    return int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "10000"))


def debug_enabled() -> bool:
    return env_flag("DEBUG_AGENT")


def assistant_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)


def assistant_has_meaningful_text(content: Any) -> bool:
    return bool(assistant_text_content(content).strip())


def legacy_protocol_error(content: str) -> Optional[str]:
    stripped = content.lstrip()
    if stripped.startswith("<tool_call>"):
        return "assistant emitted deprecated text <tool_call> protocol"
    if stripped.startswith("<tool_response>"):
        return "assistant emitted deprecated text <tool_response> protocol"
    if stripped.startswith("<think>"):
        return "assistant emitted deprecated text <think> protocol"
    if stripped.startswith("<answer>"):
        return "assistant emitted deprecated text <answer> protocol"
    return None


def tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def normalized_tool_call(tool_call: Any) -> dict[str, Any]:
    return {
        "id": getattr(tool_call, "id", ""),
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def tool_result_message_content(result: Any) -> str:
    if isinstance(result, dict) and result.get("kind") == "image_tool_result":
        return str(result.get("text", "")).strip() or "ReadImage returned no metadata."
    if isinstance(result, (dict, list)):
        return json.dumps(safe_jsonable(result), ensure_ascii=False)
    return str(result)


def image_context_message(result: Any) -> Optional[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("kind") != "image_tool_result":
        return None
    image_url = str(result.get("image_url", "")).strip()
    if not image_url:
        return None
    metadata_text = str(result.get("text", "")).strip()
    text = (
        "Runtime image context from ReadImage.\n"
        "Use the attached image as evidence produced by that tool call when deciding the next step or final result.\n"
        "Do not assume that all required tool work is complete merely because an image is attached."
    )
    if metadata_text:
        text += "\n\nReadImage metadata:\n" + metadata_text
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
        ],
    }


def api_tool_message(tool_call_id: str, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": tool_result_message_content(result),
    }


def parse_tool_arguments_list(tool_calls: list[dict[str, Any]]) -> list[Any]:
    parsed_arguments: list[Any] = []
    for tool_call in tool_calls:
        function_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        tool_arguments_raw = function_block.get("arguments", {})
        try:
            parsed = json.loads(tool_arguments_raw) if isinstance(tool_arguments_raw, str) else tool_arguments_raw
        except (TypeError, ValueError):
            parsed = tool_arguments_raw
        parsed_arguments.append(safe_jsonable(parsed))
    return parsed_arguments


def image_trace_paths(result: Any) -> list[str]:
    if not isinstance(result, dict) or result.get("kind") != "image_tool_result":
        return []
    path = str(result.get("path", "")).strip()
    return [path] if path else []


def image_context_trace_text(result: Any) -> str:
    if not isinstance(result, dict) or result.get("kind") != "image_tool_result":
        return ""
    metadata_text = str(result.get("text", "")).strip()
    text = (
        "Runtime image context from ReadImage.\n"
        "Use the attached image as evidence produced by that tool call when deciding the next step or final result.\n"
        "Do not assume that all required tool work is complete merely because an image is attached."
    )
    if metadata_text:
        text += "\n\nReadImage metadata:\n" + metadata_text
    return text


def default_llm_config() -> dict:
    model_name = os.environ.get("MODEL_NAME", os.environ.get("SUMMARY_MODEL_NAME", "gpt-5.4"))
    return {
        "model": model_name,
        "generate_cfg": {
            "max_input_tokens": int(os.environ.get("MAX_INPUT_TOKENS", "320000")),
            "max_retries": int(os.environ.get("LLM_MAX_RETRIES", "10")),
            "temperature": float(os.environ.get("TEMPERATURE", "0.6")),
            "top_p": float(os.environ.get("TOP_P", "0.95")),
            "presence_penalty": float(os.environ.get("PRESENCE_PENALTY", "1.1")),
        },
    }


def build_default_agent(*, trace_path: Optional[str] = None, function_list: Optional[List[str]] = None) -> "MultiTurnReactAgent":
    return MultiTurnReactAgent(
        llm=default_llm_config(),
        trace_path=trace_path,
        function_list=function_list or list(AVAILABLE_TOOL_MAP.keys()),
    )

class MultiTurnReactAgent:
    def __init__(
        self,
        function_list: Optional[List[str]] = None,
        llm: Optional[Dict] = None,
        trace_path: Optional[str] = None,
    ):
        if not isinstance(llm, dict):
            raise ValueError("llm must be a dict configuration.")
        requested_tools = function_list or list(AVAILABLE_TOOL_MAP.keys())
        unknown_tools = [tool for tool in requested_tools if tool not in AVAILABLE_TOOL_MAP]
        if unknown_tools:
            raise ValueError(f"Unknown tools requested: {unknown_tools}")
        if "model" not in llm or not str(llm["model"]).strip():
            raise ValueError('llm["model"] must be a non-empty string.')
        if "generate_cfg" not in llm or not isinstance(llm["generate_cfg"], dict):
            raise ValueError('llm["generate_cfg"] must be a dict.')

        self.tool_map = {tool_name: AVAILABLE_TOOL_MAP[tool_name] for tool_name in requested_tools}
        self.tool_names = list(self.tool_map.keys())
        self.model = str(llm["model"])
        self.llm_generate_cfg = llm["generate_cfg"]
        self.trace_path = Path(trace_path) if trace_path else None
        self._native_tools = [tool_schema(self.tool_map[tool_name]) for tool_name in self.tool_names]
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._native_tools_token_estimate = len(
            self._encoding.encode(json.dumps(self._native_tools, ensure_ascii=False))
        )
        self._llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))
        self._llm_api_key = os.environ.get("API_KEY", "EMPTY")
        self._llm_api_base = os.environ.get("API_BASE")
        self._llm_client = (
            OpenAI(
                api_key=self._llm_api_key,
                base_url=self._llm_api_base,
                timeout=self._llm_timeout_seconds,
            )
            if self._llm_api_base
            else None
        )

    def call_llm_api(self, msgs, max_tries=10) -> dict[str, Any]:
        max_tries = int(self.llm_generate_cfg.get("max_retries", max_tries))
        if self._llm_client is None or not self._llm_api_base:
            return {"status": "error", "error": "llm api error: API_BASE is not set."}

        base_sleep_time = 1
        last_error = "unknown llm error"
        for attempt in range(max_tries):
            try:
                if debug_enabled():
                    print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                request_kwargs = dict(
                    model=self.model,
                    messages=msgs,
                    tools=self._native_tools,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=llm_max_output_tokens(),
                )
                request_kwargs["presence_penalty"] = self.llm_generate_cfg.get('presence_penalty', 1.1)
                chat_response = self._llm_client.chat.completions.create(**request_kwargs)
                choice = chat_response.choices[0]
                message = choice.message
                content = message.content
                tool_calls = [normalized_tool_call(tool_call) for tool_call in (message.tool_calls or [])]

                if assistant_has_meaningful_text(content) or tool_calls:
                    if debug_enabled():
                        print("--- Service call successful, received a valid response ---")
                    return {
                        "status": "ok",
                        "finish_reason": choice.finish_reason,
                        "content": content,
                        "tool_calls": tool_calls,
                    }
                else:
                    last_error = "empty response from llm api"
                    if debug_enabled():
                        print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                last_error = str(e)
                if debug_enabled():
                    print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                if debug_enabled():
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                if debug_enabled():
                    print("Error: All retry attempts have been exhausted. The call has failed.")

        return {"status": "error", "error": f"llm api error: {last_error}"}

    def count_tokens(self, messages):
        image_token_estimate = int(os.getenv("IMAGE_PART_TOKEN_ESTIMATE", str(DEFAULT_IMAGE_TOKEN_ESTIMATE)))
        token_count = self._native_tools_token_estimate
        for message in messages:
            token_count += len(self._encoding.encode(message.get("role", "")))
            content = message.get("content", "")
            if isinstance(content, str):
                token_count += len(self._encoding.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        token_count += len(self._encoding.encode(str(part)))
                        continue
                    if part.get("type") == "text":
                        token_count += len(self._encoding.encode(str(part.get("text", ""))))
                    elif part.get("type") == "image_url":
                        token_count += image_token_estimate
                    else:
                        token_count += len(self._encoding.encode(str(part)))
            else:
                token_count += len(self._encoding.encode(str(content)))
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                token_count += len(self._encoding.encode(json.dumps(tool_calls, ensure_ascii=False)))
        return token_count

    def run(self, user_input: str, workspace_dir: Optional[str] = None) -> str:
        """Run the agent on one user input and return only the final result text."""
        return self._run_session(user_input, workspace_dir=workspace_dir)["result_text"]

    def _run_session(
        self,
        user_input: str,
        workspace_dir: Optional[str] = None,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict:
        """Internal execution path with trace data for tests and debugging."""
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")

        user_request = user_input.strip()
        workspace_root = normalize_workspace_root(workspace_dir)
        start_time = time.time()
        trace_path = self.trace_path
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        user_content = (
            f"Current workspace directory: {workspace_root}\n"
            "Relative local file paths resolve from the workspace directory.\n\n"
            f"User request:\n{user_request}"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        max_llm_calls = max_llm_calls_per_run()
        agent_runtime_limit = max_agent_runtime_seconds()
        num_llm_calls_available = max_llm_calls
        round_index = 0
        trace_writer = FlatTraceWriter(
            path=trace_path,
            model_name=self.model,
            workspace_root=workspace_root,
            on_event=event_callback,
        )

        def finalize(result_text: str, termination: str, *, role: str = "runtime", error: str = "") -> dict[str, Any]:
            trace_writer.append(
                role=role,
                text=result_text,
                turn_index=round_index,
                termination=termination,
                error=error,
            )
            return {
                "user_request": user_request,
                "messages": messages,
                "result_text": result_text,
                "termination": termination,
            }

        trace_writer.append(role="system", text=system_prompt, turn_index=0)
        trace_writer.append(role="user", text=user_content, turn_index=0)

        while num_llm_calls_available > 0:
            if time.time() - start_time > agent_runtime_limit:
                result_text = "No result found before the maximum agent runtime limit."
                termination = f"agent runtime limit reached: {agent_runtime_limit}s"
                return finalize(result_text, termination, error=termination)
            round_index += 1
            num_llm_calls_available -= 1
            llm_reply = self.call_llm_api(messages)
            assistant_content = llm_reply.get("content") if isinstance(llm_reply, dict) else None
            assistant_tool_calls = llm_reply.get("tool_calls", []) if isinstance(llm_reply, dict) else []
            assistant_text = assistant_text_content(assistant_content)
            finish_reason = llm_reply.get("finish_reason") if isinstance(llm_reply, dict) else None
            assistant_tool_arguments = parse_tool_arguments_list(assistant_tool_calls)
            assistant_tool_call_ids = [str(tool_call.get("id", "")) for tool_call in assistant_tool_calls]
            assistant_tool_names = [
                str((tool_call.get("function", {}) if isinstance(tool_call, dict) else {}).get("name", ""))
                for tool_call in assistant_tool_calls
            ]
            if debug_enabled():
                if assistant_tool_calls:
                    print(f"Round {round_index}: tool_calls={json.dumps(assistant_tool_calls, ensure_ascii=False)}")
                    if assistant_text.strip():
                        print(f"Round {round_index} content: {assistant_text}")
                else:
                    print(f"Round {round_index}: {assistant_text}")
            if not isinstance(llm_reply, dict) or llm_reply.get("status") == "error":
                result_text = llm_reply.get("error", "llm api error: unknown error") if isinstance(llm_reply, dict) else str(llm_reply)
                termination = "llm api error"
                return finalize(result_text, termination, error=result_text)

            deprecated_protocol = legacy_protocol_error(assistant_text)
            if deprecated_protocol is not None:
                trace_writer.append(
                    role="assistant",
                    text=assistant_text.strip(),
                    turn_index=round_index,
                    tool_call_ids=assistant_tool_call_ids,
                    tool_names=assistant_tool_names,
                    tool_arguments=assistant_tool_arguments,
                    finish_reason=finish_reason,
                    error=deprecated_protocol,
                )
                correction_text = (
                    "Error: The previous assistant turn used the deprecated text-tag protocol. "
                    "Do not emit <tool_call>, <tool_response>, <think>, or <answer> in plain text. "
                    "Use only the native tool calling interface when tools are needed, or plain final result text when no more tools are needed."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": correction_text,
                    }
                )
                trace_writer.append(role="user", text=correction_text, turn_index=round_index)
                continue

            if assistant_tool_calls and assistant_has_meaningful_text(assistant_content):
                protocol_error = "assistant mixed native tool calls and plain result text"
                trace_writer.append(
                    role="assistant",
                    text=assistant_text.strip(),
                    turn_index=round_index,
                    tool_call_ids=assistant_tool_call_ids,
                    tool_names=assistant_tool_names,
                    tool_arguments=assistant_tool_arguments,
                    finish_reason=finish_reason,
                    error=protocol_error,
                )
                correction_text = (
                    "Error: The previous assistant turn was invalid because it mixed native tool calls and plain result text. "
                    "A tool-using assistant turn must contain only tool calls and no free-form result text. "
                    "No tools from that invalid turn were executed. Discard the guessed result text from that turn; it may be wrong and was not accepted. "
                    "If you still need tools, re-emit only the required tool calls. If no more tools are needed, return only the final result text."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": correction_text,
                    }
                )
                trace_writer.append(role="user", text=correction_text, turn_index=round_index)
                continue

            if assistant_tool_calls:
                trace_writer.append(
                    role="assistant",
                    text=assistant_text.strip(),
                    turn_index=round_index,
                    tool_call_ids=assistant_tool_call_ids,
                    tool_names=assistant_tool_names,
                    tool_arguments=assistant_tool_arguments,
                    finish_reason=finish_reason,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": assistant_tool_calls,
                }
                messages.append(assistant_message)
                for tool_call, tool_arguments in zip(assistant_tool_calls, assistant_tool_arguments):
                    tool_call_id = str(tool_call.get("id", ""))
                    function_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function_block.get("name", ""))
                    result = self.custom_call_tool(tool_name, tool_arguments, workspace_root=workspace_root)
                    tool_result_text = tool_result_message_content(result)
                    messages.append(api_tool_message(tool_call_id, result))
                    trace_writer.append(
                        role="tool",
                        text=tool_result_text,
                        turn_index=round_index,
                        tool_call_ids=[tool_call_id],
                        tool_names=[tool_name],
                        tool_arguments=[tool_arguments],
                    )
                    extra_image_context = image_context_message(result)
                    if extra_image_context is not None:
                        messages.append(extra_image_context)
                        trace_writer.append(
                            role="user",
                            text=image_context_trace_text(result),
                            turn_index=round_index,
                            tool_call_ids=[tool_call_id],
                            tool_names=[tool_name],
                            tool_arguments=[tool_arguments],
                            image_paths=image_trace_paths(result),
                        )
            elif assistant_has_meaningful_text(assistant_content):
                current_result_text = assistant_text.strip()
                current_termination = "result"
                messages.append({"role": "assistant", "content": current_result_text})
                trace_writer.append(
                    role="assistant",
                    text=current_result_text,
                    turn_index=round_index,
                    finish_reason=finish_reason,
                    termination=current_termination,
                )
                return {
                    "user_request": user_request,
                    "messages": messages,
                    "result_text": current_result_text,
                    "termination": current_termination,
                }
            else:
                protocol_error = "assistant emitted empty response"
                trace_writer.append(
                    role="assistant",
                    text="",
                    turn_index=round_index,
                    finish_reason=finish_reason,
                    error=protocol_error,
                )
                correction_text = (
                    "Error: The previous assistant turn was empty. "
                    "If tools are needed, use native tool calling. Otherwise return the final result text."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": correction_text,
                    }
                )
                trace_writer.append(role="user", text=correction_text, turn_index=round_index)
                continue

            max_tokens = int(self.llm_generate_cfg.get("max_input_tokens", 110 * 1024))
            token_count = self.count_tokens(messages)
            if debug_enabled():
                print(f"round: {round_index}, token count: {token_count}")

            if token_count > max_tokens:
                if debug_enabled():
                    print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                result_text = "No result found before the maximum input token limit."
                termination = f"input token limit reached: {token_count} > {max_tokens}"
                return finalize(result_text, termination, error=termination)

        result_text = 'No result found.'
        termination = 'result not found'
        if num_llm_calls_available == 0:
            termination = 'exceed available llm calls'
        return finalize(result_text, termination, error=termination)

    def custom_call_tool(self, tool_name: str, tool_args: Any, **kwargs):
        if tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            if tool_name == "ReadImage" and hasattr(tool, "call_for_llm"):
                return tool.call_for_llm(tool_args, **kwargs)
            return tool.call(tool_args, **kwargs)
        return f"Error: Tool {tool_name} not found"


def _parse_cli_args(argv: list[str]) -> tuple[str, Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description="Run the local agent directly from agent_base.react_agent.")
    parser.add_argument("question", nargs="*", help="User request text.")
    parser.add_argument("--question-file", help="Optional UTF-8 text file containing the user request.")
    parser.add_argument("--save-path", help="Optional JSONL trace output path.")
    parser.add_argument("--workspace-dir", help="Optional workspace directory for Bash and TerminalStart.")
    args = parser.parse_args(argv)

    user_request = ""
    if args.question_file:
        user_request = Path(args.question_file).read_text(encoding="utf-8").strip()
    elif args.question:
        user_request = " ".join(args.question).strip()

    if not user_request:
        raise ValueError("A non-empty question is required via positional args or --question-file.")
    return user_request, args.save_path, args.workspace_dir


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        user_request, save_path, workspace_dir = _parse_cli_args(argv or sys.argv[1:])
        agent = build_default_agent(trace_path=save_path)
        workspace_root = normalize_workspace_root(workspace_dir)
        printer = ConsoleEventPrinter(
            model_name=agent.model,
            workspace_root=workspace_root,
            user_request=user_request,
        )
        printer.print_header()
        agent._run_session(user_request, workspace_dir=workspace_dir, event_callback=printer.handle_event)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
