import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type

from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
import tiktoken
from agent_base.base import BaseAgent
from agent_base.console_utils import ConsoleEventPrinter
from agent_base.prompt import composed_system_prompt
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
DEFAULT_MODEL_NAME = "gpt-5.4"
DEFAULT_MAX_LLM_CALLS = 100
DEFAULT_MAX_ROUNDS = 100
DEFAULT_MAX_RUNTIME_SECONDS = 150 * 60
DEFAULT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_MAX_INPUT_TOKENS = 320000
DEFAULT_MAX_RETRIES = 10
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_PRESENCE_PENALTY = 1.1
DEFAULT_LLM_TIMEOUT_SECONDS = 600.0


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def max_llm_calls_per_run() -> int:
    return int(os.getenv("MAX_LLM_CALL_PER_RUN", str(DEFAULT_MAX_LLM_CALLS)))


def max_agent_rounds() -> int:
    return int(os.getenv("MAX_AGENT_ROUNDS", str(DEFAULT_MAX_ROUNDS)))


def max_agent_runtime_seconds() -> int:
    return int(os.getenv("MAX_AGENT_RUNTIME_SECONDS", str(DEFAULT_MAX_RUNTIME_SECONDS)))


def llm_max_output_tokens() -> int:
    return int(os.getenv("LLM_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)))


def remaining_runtime_seconds(runtime_deadline: Optional[float]) -> Optional[float]:
    if runtime_deadline is None:
        return None
    return runtime_deadline - time.time()


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


def resolved_tool_names(function_list: Optional[Sequence[str]]) -> list[str]:
    if function_list is None:
        return list(AVAILABLE_TOOL_MAP.keys())
    resolved: list[str] = []
    for raw_name in function_list:
        name = str(raw_name).strip()
        if name:
            resolved.append(name)
    return resolved


def available_tool_schemas(function_list: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    return [tool_schema(AVAILABLE_TOOL_MAP[name]) for name in resolved_tool_names(function_list)]


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
    model_name = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    return {
        "model": model_name,
        "api_key": os.environ.get("API_KEY", "EMPTY"),
        "api_base": os.environ.get("API_BASE"),
        "timeout_seconds": float(os.environ.get("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS))),
        "generate_cfg": {
            "max_input_tokens": int(os.environ.get("MAX_INPUT_TOKENS", str(DEFAULT_MAX_INPUT_TOKENS))),
            "max_output_tokens": int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
            "max_retries": int(os.environ.get("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
            "temperature": float(os.environ.get("TEMPERATURE", str(DEFAULT_TEMPERATURE))),
            "top_p": float(os.environ.get("TOP_P", str(DEFAULT_TOP_P))),
            "presence_penalty": float(os.environ.get("PRESENCE_PENALTY", str(DEFAULT_PRESENCE_PENALTY))),
        },
    }


def execute_tool_by_name(tool_map: dict[str, Any], tool_name: str, tool_args: Any, **kwargs):
    if tool_name not in tool_map:
        return f"Error: Tool {tool_name} not found"
    tool = tool_map[tool_name]
    if tool_name == "ReadImage" and hasattr(tool, "call_for_llm"):
        return tool.call_for_llm(tool_args, **kwargs)
    return tool.call(tool_args, **kwargs)


def model_rejects_sampling_params(model_name: str) -> bool:
    return str(model_name or "").strip().lower().startswith("claude")


class MultiTurnReactAgent(BaseAgent):
    def __init__(
        self,
        function_list: Optional[List[str]] = None,
        llm: Optional[Dict] = None,
        trace_dir: Optional[str] = None,
        role_prompt: Optional[str] = None,
        max_llm_calls: Optional[int] = None,
        max_rounds: Optional[int] = None,
        max_runtime_seconds: Optional[int] = None,
    ):
        if not isinstance(llm, dict):
            raise ValueError("llm must be a dict configuration.")
        requested_tools = self.resolve_function_list(function_list)
        if requested_tools is None:
            requested_tools = list(AVAILABLE_TOOL_MAP.keys())
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
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.trace_path: Optional[Path] = None
        self.role_prompt = self.resolve_role_prompt(role_prompt)
        self.max_llm_calls = int(max_llm_calls) if max_llm_calls is not None else max_llm_calls_per_run()
        self.max_rounds = int(max_rounds) if max_rounds is not None else max_agent_rounds()
        self.max_runtime_seconds = (
            int(max_runtime_seconds) if max_runtime_seconds is not None else max_agent_runtime_seconds()
        )
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be > 0.")
        self._native_tools = [tool_schema(self.tool_map[tool_name]) for tool_name in self.tool_names]
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._native_tools_token_estimate = len(
            self._encoding.encode(json.dumps(self._native_tools, ensure_ascii=False))
        )
        self._llm_timeout_seconds = float(
            llm.get("timeout_seconds", os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS)))
        )
        self._llm_api_key = str(llm.get("api_key") or os.environ.get("API_KEY", "EMPTY"))
        api_base = str(llm.get("api_base") or os.environ.get("API_BASE", "")).strip()
        self._llm_api_base = api_base or None
        self._llm_client = (
            OpenAI(
                api_key=self._llm_api_key,
                base_url=self._llm_api_base,
                timeout=self._llm_timeout_seconds,
            )
            if self._llm_api_base
            else None
        )

    def call_llm_api(self, msgs, max_tries=10, runtime_deadline: Optional[float] = None) -> dict[str, Any]:
        max_tries = int(self.llm_generate_cfg.get("max_retries", max_tries))
        if self._llm_client is None or not self._llm_api_base:
            return {"status": "error", "error": "llm api error: API_BASE is not set."}

        base_sleep_time = 1
        last_error = "unknown llm error"
        for attempt in range(max_tries):
            remaining = remaining_runtime_seconds(runtime_deadline)
            if remaining is not None and remaining <= 0:
                last_error = "agent runtime limit reached before llm call could complete"
                break
            try:
                if debug_enabled():
                    print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                request_client = (
                    self._llm_client.with_options(timeout=min(self._llm_timeout_seconds, max(remaining, 0.001)))
                    if remaining is not None
                    else self._llm_client
                )
                request_kwargs = dict(
                    model=self.model,
                    messages=msgs,
                    max_tokens=int(self.llm_generate_cfg.get("max_output_tokens", llm_max_output_tokens())),
                )
                if not model_rejects_sampling_params(self.model):
                    request_kwargs["temperature"] = self.llm_generate_cfg.get("temperature", 0.6)
                    request_kwargs["top_p"] = self.llm_generate_cfg.get("top_p", 0.95)
                if self._native_tools:
                    request_kwargs["tools"] = self._native_tools
                    request_kwargs["tool_choice"] = "auto"
                    request_kwargs["parallel_tool_calls"] = True
                request_kwargs["presence_penalty"] = self.llm_generate_cfg.get('presence_penalty', 1.1)
                chat_response = request_client.chat.completions.create(**request_kwargs)
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
                remaining = remaining_runtime_seconds(runtime_deadline)
                if remaining is not None:
                    if remaining <= 0:
                        last_error = "agent runtime limit reached before llm retry could complete"
                        break
                    sleep_time = min(sleep_time, remaining)
                if debug_enabled():
                    print(f"Retrying in {sleep_time:.2f} seconds...")
                if sleep_time > 0:
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

    def run(self, prompt: str, workspace_root: Optional[str] = None) -> str:
        """Run the agent on one prompt and return only the final result text."""
        return self._run_session(prompt, workspace_root=workspace_root)["result_text"]

    def _run_session(
        self,
        prompt: str,
        workspace_root: Optional[str] = None,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict:
        """Internal execution path with trace data for tests and debugging."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        prompt_text = prompt.strip()
        resolved_workspace_root = normalize_workspace_root(workspace_root)
        start_time = time.time()
        trace_dir = self.trace_dir
        cur_date = today_date()
        extra_blocks = [self.role_prompt] if self.role_prompt else None
        system_prompt = composed_system_prompt(current_date=str(cur_date), extra_blocks=extra_blocks)
        user_content = (
            f"Current workspace root: {resolved_workspace_root}\n"
            "Relative local file paths resolve from the workspace root.\n\n"
            f"Prompt:\n{prompt_text}"
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        max_llm_calls = self.max_llm_calls
        agent_runtime_limit = self.max_runtime_seconds
        runtime_deadline = start_time + agent_runtime_limit
        num_llm_calls_available = max_llm_calls
        round_index = 0
        trace_writer = FlatTraceWriter(
            trace_dir=trace_dir,
            model_name=self.model,
            workspace_root=resolved_workspace_root,
            on_event=event_callback,
        )
        self.trace_path = trace_writer.path

        def finalize(result_text: str, termination: str, *, role: str = "runtime", error: str = "") -> dict[str, Any]:
            trace_writer.append(
                role=role,
                text=result_text,
                turn_index=round_index,
                termination=termination,
                error=error,
            )
            return {
                "prompt": prompt_text,
                "messages": messages,
                "result_text": result_text,
                "termination": termination,
            }

        trace_writer.append(role="system", text=system_prompt, turn_index=0)
        trace_writer.append(role="user", text=user_content, turn_index=0)

        while num_llm_calls_available > 0 and round_index < self.max_rounds:
            if remaining_runtime_seconds(runtime_deadline) is not None and remaining_runtime_seconds(runtime_deadline) <= 0:
                result_text = "No result found before the maximum agent runtime limit."
                termination = f"agent runtime limit reached: {agent_runtime_limit}s"
                return finalize(result_text, termination, error=termination)
            round_index += 1
            num_llm_calls_available -= 1
            llm_reply = self.call_llm_api(messages, runtime_deadline=runtime_deadline)
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
                deferred_image_contexts: list[tuple[str, str, Any, Any, dict[str, Any]]] = []
                for tool_call, tool_arguments in zip(assistant_tool_calls, assistant_tool_arguments):
                    if remaining_runtime_seconds(runtime_deadline) is not None and remaining_runtime_seconds(runtime_deadline) <= 0:
                        result_text = "No result found before the maximum agent runtime limit."
                        termination = f"agent runtime limit reached: {agent_runtime_limit}s"
                        return finalize(result_text, termination, error=termination)
                    tool_call_id = str(tool_call.get("id", ""))
                    function_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                    tool_name = str(function_block.get("name", ""))
                    result = self.custom_call_tool(
                        tool_name,
                        tool_arguments,
                        workspace_root=resolved_workspace_root,
                        runtime_deadline=runtime_deadline,
                    )
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
                        deferred_image_contexts.append((tool_call_id, tool_name, tool_arguments, result, extra_image_context))
                for tool_call_id, tool_name, tool_arguments, result, extra_image_context in deferred_image_contexts:
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
                    if remaining_runtime_seconds(runtime_deadline) is not None and remaining_runtime_seconds(runtime_deadline) <= 0:
                        result_text = "No result found before the maximum agent runtime limit."
                        termination = f"agent runtime limit reached: {agent_runtime_limit}s"
                        return finalize(result_text, termination, error=termination)
            elif assistant_has_meaningful_text(assistant_content):
                current_result_text = assistant_text.strip()
                messages.append({"role": "assistant", "content": current_result_text})
                should_accept_result = self.should_accept_plaintext_result(
                    result_text=current_result_text,
                    workspace_root=resolved_workspace_root,
                    messages=messages,
                )
                if should_accept_result:
                    current_termination = "result"
                    trace_writer.append(
                        role="assistant",
                        text=current_result_text,
                        turn_index=round_index,
                        finish_reason=finish_reason,
                        termination=current_termination,
                    )
                    return {
                        "prompt": prompt_text,
                        "messages": messages,
                        "result_text": current_result_text,
                        "termination": current_termination,
                    }
                protocol_error = "plain result rejected by additional stop condition"
                trace_writer.append(
                    role="assistant",
                    text=current_result_text,
                    turn_index=round_index,
                    finish_reason=finish_reason,
                    error=protocol_error,
                )
                correction_text = self.rejected_plaintext_result_message(
                    result_text=current_result_text,
                    workspace_root=resolved_workspace_root,
                    messages=messages,
                ).strip()
                if not correction_text:
                    correction_text = (
                        "The previous assistant turn was not accepted as the final result because the additional stop condition returned false. "
                        "Continue working. If the task is incomplete, use tool calls to produce the required artifacts before finishing."
                    )
                messages.append({"role": "user", "content": correction_text})
                trace_writer.append(role="user", text=correction_text, turn_index=round_index)
                continue
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
        if round_index >= self.max_rounds:
            termination = 'exceed available rounds'
        elif num_llm_calls_available == 0:
            termination = 'exceed available llm calls'
        return finalize(result_text, termination, error=termination)

    def custom_call_tool(self, tool_name: str, tool_args: Any, **kwargs):
        return execute_tool_by_name(self.tool_map, tool_name, tool_args, **kwargs)


def _read_role_prompt_files(paths: Iterable[str]) -> str:
    blocks: list[str] = []
    for raw_path in paths:
        path = Path(str(raw_path)).expanduser()
        blocks.append(path.read_text(encoding="utf-8").strip())
    return "\n\n".join(block for block in blocks if block.strip())


def _path_has_suffix(path: Path, suffix_parts: Sequence[str]) -> bool:
    normalized_parts = tuple(part.casefold() for part in path.parts)
    normalized_suffix = tuple(part.casefold() for part in suffix_parts)
    if len(normalized_parts) < len(normalized_suffix):
        return False
    return normalized_parts[-len(normalized_suffix) :] == normalized_suffix


def resolve_agent_class_for_role_prompt_files(role_prompt_files: Sequence[str]) -> Type[MultiTurnReactAgent]:
    for raw_path in role_prompt_files:
        path_text = str(raw_path).strip()
        if not path_text:
            continue
        path = Path(path_text).expanduser().resolve(strict=False)
        if _path_has_suffix(path, ("benchmarks", "ResearchClawBench", "role_prompt.md")):
            from benchmarks.ResearchClawBench.adapter import ResearchClawBenchAgent

            return ResearchClawBenchAgent
    return MultiTurnReactAgent


def _parse_cli_args(argv: list[str]) -> tuple[str, Optional[str], Optional[str], str, list[str]]:
    parser = argparse.ArgumentParser(description="Run the local agent directly from agent_base.react_agent.")
    parser.add_argument("prompt", nargs="*", help="Prompt text.")
    parser.add_argument("--prompt-file", help="Optional UTF-8 text file containing the prompt.")
    parser.add_argument("--trace-dir", help="Optional directory where the run trace JSONL should be created.")
    parser.add_argument(
        "--workspace-root",
        help="Optional workspace root for local file tools, Bash, and TerminalStart.",
    )
    parser.add_argument(
        "--role-prompt-file",
        action="append",
        default=[],
        dest="role_prompt_files",
        metavar="PATH",
        help="Append one role-specific prompt file to the base system prompt. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    prompt_text = ""
    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt_text = " ".join(args.prompt).strip()

    if not prompt_text:
        raise ValueError("A non-empty prompt is required via positional args or --prompt-file.")
    role_prompt = _read_role_prompt_files(args.role_prompt_files)
    return (
        prompt_text,
        args.trace_dir,
        args.workspace_root,
        role_prompt,
        list(args.role_prompt_files),
    )


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        prompt_text, trace_dir, workspace_root, role_prompt, role_prompt_files = _parse_cli_args(argv or sys.argv[1:])
        agent_cls = resolve_agent_class_for_role_prompt_files(role_prompt_files)
        agent = agent_cls(
            llm=default_llm_config(),
            trace_dir=trace_dir,
            role_prompt=role_prompt or None,
        )
        resolved_workspace_root = normalize_workspace_root(workspace_root)
        printer = ConsoleEventPrinter(
            model_name=agent.model,
            workspace_root=resolved_workspace_root,
            prompt=prompt_text,
        )
        printer.print_header()
        agent._run_session(prompt_text, workspace_root=workspace_root, event_callback=printer.handle_event)
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
