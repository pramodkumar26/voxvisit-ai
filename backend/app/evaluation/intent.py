"""Independent validation that a prepared action is the one the patient asked for.

The trusted UI shows the patient the pending action's summary and the patient presses Confirm. A
harness that confirmed whatever was prepared would approve a model's mistakes. So before any
simulated confirmation, the stored pending action (the authoritative record, not the model's
account of it) is compared with the scenario's declared intent. A mismatch is never confirmed.
"""

from collections.abc import Mapping
from typing import Any

from sqlalchemy import Engine, text


class OneOf:
    """Any of several values is the intended one (a reschedule to any acceptable new slot)."""

    def __init__(self, *values: Any) -> None:
        self.values = frozenset(str(value) for value in values)

    def accepts(self, actual: Any) -> bool:
        return str(actual) in self.values


def _agrees(actual: Any, wanted: Any) -> bool:
    return wanted.accepts(actual) if isinstance(wanted, OneOf) else str(actual) == str(wanted)


def inspect_pending_action(engine: Engine, pending_action_id: Any) -> dict[str, Any] | None:
    """The stored pending action and who owns it, or None if there is no such action."""
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT p.action_type, p.target_id, p.expected_versions, p.summary, p.status,"
                " s.patient_id AS owner FROM pending_actions p"
                " JOIN auth_sessions s ON s.session_id = p.session_id"
                " WHERE p.pending_action_id = :id"
            ),
            {"id": pending_action_id},
        ).one_or_none()
    return dict(row._mapping) if row else None


def _path(document: Any, path: str) -> Any:
    current = document
    for part in path.split("."):
        current = current.get(part) if isinstance(current, Mapping) else None
    return current


def pending_mismatches(
    stored: Mapping[str, Any] | None,
    *,
    action: str,
    target: Any,
    expected: Mapping[str, Any],
    summary: Mapping[str, Any],
    actor_patient_id: Any,
) -> list[str]:
    """The names of everything that differs from the intent; empty if the action is as intended.

    Only names are returned (never values), so a mismatch can be reported safely.
    """
    if stored is None:
        return ["pending_action_missing"]
    mismatches = []
    if str(stored["owner"]) != str(actor_patient_id):
        mismatches.append("owner")
    if stored["status"] != "ACTIVE":
        mismatches.append("status")
    if stored["action_type"] != action:
        mismatches.append("action")
    if not _agrees(stored["target_id"], target):
        mismatches.append("target")
    for key, value in expected.items():
        if not _agrees(_path(stored["expected_versions"], key), value):
            mismatches.append(f"expected.{key}")
    for path, value in summary.items():
        if not _agrees(_path(stored["summary"], path), value):
            mismatches.append(f"summary.{path}")
    return mismatches
