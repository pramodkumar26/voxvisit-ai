"""The tool handlers: thin wrappers over the existing trusted services.

A handler receives validated arguments and the trusted context and returns the response data, or
raises a backend error that the runtime turns into the envelope. Handlers import no commit or
confirmation code; the only write-capable service reachable from here is preparation, which stores
a pending action and changes no appointment or slot.
"""

from typing import Any

from app.agent_tools.context import ToolContext
from app.agent_tools.envelope import ToolError
from app.agent_tools.schemas import AvailabilityArgs, DirectoryArgs, MyAppointmentsArgs
from app.models.appointment_actions import PrepareBook, PrepareCancel, PrepareReschedule
from app.services import appointments, availability, directory
from app.services.pagination import MAX_PAGE_SIZE
from app.services.pending_actions import prepare_appointment_action
from app.services.read_access import read_only_connection
from app.services.search_text import search_tokens

MAX_CHOICES = 5


class ToolFailure(Exception):
    """An outcome the contract defines as an error but the REST services treat as a normal result,
    such as an empty search (NO_RESULTS) or an ambiguous name (AMBIGUOUS_MATCH)."""

    def __init__(self, error: ToolError) -> None:
        super().__init__(error.code)
        self.error = error


def _no_results() -> ToolFailure:
    return ToolFailure(
        ToolError(
            code="NO_RESULTS",
            message="No matching clinic or doctor was found.",
            retryable=False,
        )
    )


def _no_availability() -> ToolFailure:
    return ToolFailure(
        ToolError(
            code="NO_AVAILABILITY",
            message="No matching times are currently available for that date range.",
            retryable=False,
        )
    )


def _ambiguous(kind: str, choices: list[dict[str, Any]]) -> ToolFailure:
    return ToolFailure(
        ToolError(
            code="AMBIGUOUS_MATCH",
            message=f"More than one {kind} matches. Ask the user which one they mean.",
            retryable=False,
            choices=choices[:MAX_CHOICES],
        )
    )


def _exact(query: str, names: list[str]) -> bool:
    """Whether exactly one candidate's name is the query itself, which settles the ambiguity."""
    wanted = search_tokens(query, drop_titles=True)
    return sum(search_tokens(name, drop_titles=True) == wanted for name in names) == 1


def _clinic_ambiguity(args: DirectoryArgs, page: Any) -> ToolFailure | None:
    names = [clinic.name for clinic in page.clinics]
    if not args.clinic_name or args.clinic_id or len(names) < 2 or _exact(args.clinic_name, names):
        return None
    return _ambiguous(
        "clinic",
        [{"clinic_id": str(c.clinic_id), "name": c.name, "city": c.city} for c in page.clinics],
    )


def _doctor_ambiguity(args: DirectoryArgs, page: Any) -> ToolFailure | None:
    names = [doctor.display_name for doctor in page.doctors]
    if not args.doctor_name or len(names) < 2 or _exact(args.doctor_name, names):
        return None
    return _ambiguous(
        "doctor",
        [
            {
                "doctor_id": str(d.doctor_id),
                "display_name": d.display_name,
                "specialty": d.specialty,
            }
            for d in page.doctors
        ],
    )


def search_clinic_directory(args: DirectoryArgs, context: ToolContext) -> dict[str, Any]:
    with read_only_connection(context.engine) as connection:
        page = directory.search_directory(connection, **args.model_dump(), limit=MAX_PAGE_SIZE)
    if args.cursor is None:  # Continuing a page the agent already chose is never re-judged.
        if not page.clinics and not page.doctors:
            raise _no_results()
        # A name that matches several clinics or doctors is ambiguous: the agent must ask the user.
        ambiguity = _clinic_ambiguity(args, page) or _doctor_ambiguity(args, page)
        if ambiguity is not None:
            raise ambiguity
    return page.model_dump(mode="json")


def search_appointment_availability(args: AvailabilityArgs, context: ToolContext) -> dict[str, Any]:
    with read_only_connection(context.engine) as connection:
        page = availability.search_availability(
            connection, **args.model_dump(), limit=MAX_PAGE_SIZE, now=context.now
        )
    if not page.slots and args.cursor is None:
        raise _no_availability()
    return page.model_dump(mode="json")


def get_my_appointments(args: MyAppointmentsArgs, context: ToolContext) -> dict[str, Any]:
    # The patient is always the verified session's patient; the arguments cannot name one.
    with read_only_connection(context.engine) as connection:
        page = appointments.list_my_appointments(
            connection,
            patient_id=context.principal.patient_id,
            **args.model_dump(),
            limit=MAX_PAGE_SIZE,
        )
    return page.model_dump(mode="json")


def prepare_appointment_action_tool(
    args: PrepareBook | PrepareReschedule | PrepareCancel, context: ToolContext
) -> dict[str, Any]:
    view, _replayed = prepare_appointment_action(
        context.engine,
        principal=context.principal,
        request=args,
        idempotency_key=context.idempotency_key,
        request_id=context.request_id,
        now=context.now,
    )
    return view.model_dump(mode="json")
