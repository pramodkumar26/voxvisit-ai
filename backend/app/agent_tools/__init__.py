"""The agent tool layer: the only surface the conversational agent will be allowed to call.

The agent is an untrusted caller. It supplies tool names and the documented arguments and nothing
else. Everything it must not control comes from the trusted runtime: the verified session, the
request id, the idempotency key, and the clock. A fixed allowlist decides which tools exist, so the
appointment confirmation operation and the commit service are unreachable from here by
construction (docs/agent-tool-contracts.md, "Confirmation Boundary").

Every result, success or failure, is returned in the approved envelope:
`{"ok", "request_id", "data", "error"}`.
"""

from app.agent_tools.registry import ALLOWED_TOOLS, ToolRegistry, build_default_registry
from app.agent_tools.runtime import ToolCall, ToolRuntime

__all__ = [
    "ALLOWED_TOOLS",
    "ToolCall",
    "ToolRegistry",
    "ToolRuntime",
    "build_default_registry",
]
