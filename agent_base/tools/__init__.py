from importlib import import_module

__all__ = [
    "Bash",
    "AskUser",
    "Edit",
    "Glob",
    "Grep",
    "Read",
    "ReadImage",
    "ReadPDF",
    "ScholarSearch",
    "TerminalInterrupt",
    "TerminalKill",
    "TerminalRead",
    "TerminalStart",
    "TerminalWrite",
    "WebFetch",
    "WebSearch",
    "Write",
]

_EXPORT_TO_MODULE = {
    "Bash": "agent_base.tools.tool_runtime",
    "AskUser": "agent_base.tools.tool_user",
    "Edit": "agent_base.tools.tool_file",
    "Glob": "agent_base.tools.tool_file",
    "Grep": "agent_base.tools.tool_file",
    "Read": "agent_base.tools.tool_file",
    "ReadImage": "agent_base.tools.tool_file",
    "ReadPDF": "agent_base.tools.tool_file",
    "ScholarSearch": "agent_base.tools.tool_web",
    "TerminalInterrupt": "agent_base.tools.tool_runtime",
    "TerminalKill": "agent_base.tools.tool_runtime",
    "TerminalRead": "agent_base.tools.tool_runtime",
    "TerminalStart": "agent_base.tools.tool_runtime",
    "TerminalWrite": "agent_base.tools.tool_runtime",
    "WebFetch": "agent_base.tools.tool_web",
    "WebSearch": "agent_base.tools.tool_web",
    "Write": "agent_base.tools.tool_file",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
