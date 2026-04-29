from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from agent_base.react_agent import AVAILABLE_TOOL_MAP, MultiTurnReactAgent
from agent_base.tools.tooling import normalize_workspace_root


class ResearchClawBenchAgent(MultiTurnReactAgent):
    """
    Lightweight benchmark adapter for ResearchClawBench.

    The benchmark task is not complete until the run workspace contains the
    canonical final report at report/report.md. Pure planning text without that
    artifact should not terminate the agent loop.
    """

    required_report_relpath = Path("report") / "report.md"
    forbidden_tool_names = {"AskUser"}

    def __init__(self, function_list: Optional[Sequence[str]] = None, *args: Any, **kwargs: Any):
        if function_list is None:
            function_list = [
                tool_name
                for tool_name in AVAILABLE_TOOL_MAP
                if tool_name not in self.forbidden_tool_names
            ]
        else:
            function_list = [str(tool_name).strip() for tool_name in function_list if str(tool_name).strip()]
            forbidden = sorted(set(function_list) & self.forbidden_tool_names)
            if forbidden:
                raise ValueError(f"Tools are not allowed in ResearchClawBench runs: {forbidden}")
        super().__init__(function_list=list(function_list), *args, **kwargs)

    def _required_report_path(self, workspace_root: Optional[str]) -> Path:
        workspace = Path(normalize_workspace_root(workspace_root))
        return workspace / self.required_report_relpath

    def should_accept_plaintext_result(
        self,
        *,
        result_text: str,
        workspace_root: Optional[str],
        messages: Sequence[dict[str, Any]],
    ) -> bool:
        if not self._required_report_path(workspace_root).exists():
            return False
        return super().should_accept_plaintext_result(
            result_text=result_text,
            workspace_root=workspace_root,
            messages=messages,
        )

    def rejected_plaintext_result_message(
        self,
        *,
        result_text: str,
        workspace_root: Optional[str],
        messages: Sequence[dict[str, Any]],
    ) -> str:
        if not self._required_report_path(workspace_root).exists():
            return (
                "The previous assistant turn was not accepted as the final result because "
                "ResearchClawBench requires report/report.md and that file is still missing. "
                "Continue working and use tool calls to produce or verify report/report.md before finishing."
            )
        return super().rejected_plaintext_result_message(
            result_text=result_text,
            workspace_root=workspace_root,
            messages=messages,
        )

    def should_accept_terminal_error(
        self,
        *,
        error_text: str,
        workspace_root: Optional[str],
        messages: Sequence[dict[str, Any]],
    ) -> bool:
        return self._required_report_path(workspace_root).exists()

    def accepted_terminal_error_result_text(
        self,
        *,
        error_text: str,
        workspace_root: Optional[str],
        messages: Sequence[dict[str, Any]],
    ) -> str:
        return (
            "ResearchClawBench completion recovered after a terminal LLM/runtime error because "
            "report/report.md already exists and the required final artifact has been produced."
        )
