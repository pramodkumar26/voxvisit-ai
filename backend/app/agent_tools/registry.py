"""The tool registry and its strict runtime allowlist.

Exactly four tools exist. A tool can be registered only if its name is on ALLOWED_TOOLS and its
handler lives in this package, so registering the confirmation operation, the commit service, or any
other callable is refused. Adding a tool is a deliberate edit to ALLOWED_TOOLS, reviewed as such.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, TypeAdapter

from app.agent_tools import tools
from app.agent_tools.schemas import (
    AvailabilityArgs,
    DirectoryArgs,
    MyAppointmentsArgs,
    PrepareRequest,
)

ALLOWED_TOOLS = frozenset(
    {
        "search_clinic_directory",
        "search_appointment_availability",
        "get_my_appointments",
        "prepare_appointment_action",
    }
)
# Names that denote committing or confirming are never registrable, even if someone adds them to
# ALLOWED_TOOLS by mistake.
FORBIDDEN_NAME_FRAGMENTS = ("confirm", "commit", "execute", "consume", "approve")
HANDLER_PACKAGE = "app.agent_tools."


class Access(StrEnum):
    PUBLIC_OR_PATIENT = "PUBLIC_OR_PATIENT"  # Anonymous callers and verified patients; not staff.
    PATIENT = "PATIENT"  # A verified patient session only.


class ToolRegistryError(ValueError):
    """A registration that would widen what the agent can reach."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    access: Access
    arguments: type[BaseModel] | Any
    handler: Callable[..., dict[str, Any]]
    adapter: TypeAdapter = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter", TypeAdapter(self.arguments))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name not in ALLOWED_TOOLS:
            raise ToolRegistryError(f"{spec.name!r} is not on the tool allowlist")
        if any(fragment in spec.name.lower() for fragment in FORBIDDEN_NAME_FRAGMENTS):
            raise ToolRegistryError(f"{spec.name!r} names a commit or confirmation operation")
        if not getattr(spec.handler, "__module__", "").startswith(HANDLER_PACKAGE):
            raise ToolRegistryError(f"{spec.name!r} has a handler outside the tool package")
        if spec.name in self._tools:
            raise ToolRegistryError(f"{spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def get(self, name: object) -> ToolSpec | None:
        """The spec for an allowlisted, registered tool; anything else is simply unknown."""
        return self._tools.get(name) if isinstance(name, str) else None

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        """What a model is told about each tool: name, purpose, and argument schema. Nothing about
        identity, sessions, or idempotency appears because those arguments do not exist."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.adapter.json_schema(),
            }
            for spec in sorted(self._tools.values(), key=lambda item: item.name)
        ]


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="search_clinic_directory",
            description=(
                "Find fictional clinics and doctors by clinic name, doctor name, specialty, or "
                "clinic id, or browse the directory. Returns at most 20 results per page."
            ),
            access=Access.PUBLIC_OR_PATIENT,
            arguments=DirectoryArgs,
            handler=tools.search_clinic_directory,
        )
    )
    registry.register(
        ToolSpec(
            name="search_appointment_availability",
            description=(
                "List currently bookable appointment times at one clinic for a date range of at "
                "most 31 days. A result is not a reservation."
            ),
            access=Access.PUBLIC_OR_PATIENT,
            arguments=AvailabilityArgs,
            handler=tools.search_appointment_availability,
        )
    )
    registry.register(
        ToolSpec(
            name="get_my_appointments",
            description="List the verified patient's own appointments, confirmed or cancelled.",
            access=Access.PATIENT,
            arguments=MyAppointmentsArgs,
            handler=tools.get_my_appointments,
        )
    )
    registry.register(
        ToolSpec(
            name="prepare_appointment_action",
            description=(
                "Prepare booking, rescheduling, or cancelling an appointment for the patient to "
                "confirm in the interface. This only prepares; it changes nothing and cannot "
                "confirm."
            ),
            access=Access.PATIENT,
            arguments=PrepareRequest,
            handler=tools.prepare_appointment_action_tool,
        )
    )
    return registry
