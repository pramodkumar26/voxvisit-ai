"""A local demonstration of how the synthetic FHIR-like resources reference one another.

    uv run python -m app.fhir_like.demo

Reads the local development database (fictional data only) through the adapter, in a READ ONLY
transaction, and prints the reference graph from one appointment, then what each kind of caller is
and is not allowed to see. The access contexts are built here by hand because there is no HTTP
session; the HTTP routes always derive them from a verified session. Nothing is written.
"""

import json
from datetime import UTC, datetime

from sqlalchemy import select

from app.core.config import get_settings
from app.core.errors import ApplicationError
from app.db.models import Appointment, Patient
from app.db.session import create_database_engine
from app.fhir_like.access import AccessContext
from app.fhir_like.adapter import FhirLikeAdapter, open_adapter
from app.fhir_like.mappers import CLINIC_ID_SYSTEM
from app.services.read_access import read_only_connection


def literal_references(value) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        if isinstance(value.get("reference"), str):
            found.append(value["reference"])
        for item in value.values():
            found.extend(literal_references(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(literal_references(item))
    return found


def summary(resource: dict) -> str:
    kind = resource["resourceType"]
    extras = {
        "Patient": lambda: f'"{resource["name"][0]["text"]}"',
        "Practitioner": lambda: f'"{resource["name"][0]["text"]}"',
        "Schedule": lambda: f"active={resource['active']}",
        "Slot": lambda: f"status={resource['status']}  {resource['start']}",
        "Appointment": lambda: f"status={resource['status']}  {resource['start']}",
    }
    return f"{kind}/{resource['id'][:13]}...  {extras[kind]()}"


def walk(adapter: FhirLikeAdapter, reference: str, depth: int, seen: set[str]) -> None:
    resource = json.loads(adapter.resolve(reference).model_dump_json(exclude_none=True))
    print("    " * depth + ("└─ " if depth else "") + summary(resource))
    if reference in seen:
        return
    seen.add(reference)
    for item in resource.get("participant", []) + resource.get("actor", []):
        actor = item.get("actor", item)
        if "identifier" in actor and actor["identifier"]["system"] == CLINIC_ID_SYSTEM:
            print(
                "    " * (depth + 1)
                + f'└─ (logical) Location "{actor["display"]}" — not a served resource'
            )
    for child in literal_references(resource):
        if child not in seen:
            walk(adapter, child, depth + 1, seen)


def attempt(label: str, adapter: FhirLikeAdapter, reference: str) -> None:
    try:
        adapter.resolve(reference)
        print(f"  {label}: allowed")
    except ApplicationError as error:
        print(f"  {label}: refused ({error.status_code} {error.code})")


def main() -> None:
    engine = create_database_engine(get_settings())
    now = datetime.now(UTC)
    print("Synthetic FHIR-like resources: simplified and fictional, NOT HL7 FHIR conformant.\n")

    with read_only_connection(engine) as connection:
        confirmed = (
            select(Appointment.appointment_id, Appointment.patient_id, Patient.patient_code)
            .join(Patient, Patient.patient_id == Appointment.patient_id)
            .where(Appointment.status == "CONFIRMED")
            .order_by(Patient.patient_code, Appointment.appointment_id)
        )
        first = connection.execute(confirmed.limit(1)).one()
        second = connection.execute(
            confirmed.where(Appointment.patient_id != first.patient_id).limit(1)
        ).one()
    owner = AccessContext.patient(first.patient_id)

    print(
        f"Reference graph from one confirmed appointment, read as its owner ({first.patient_code}):"
    )
    with open_adapter(engine, owner, now) as adapter:
        walk(adapter, f"Appointment/{first.appointment_id}", 0, set())

    print("\nWhat each kind of caller may reach:")
    for label, access in (
        ("public", AccessContext.public()),
        (f"patient {first.patient_code}", owner),
        ("demo staff", AccessContext.staff()),
    ):
        print(f" {label}")
        with open_adapter(engine, access, now) as adapter:
            attempt("own appointment", adapter, f"Appointment/{first.appointment_id}")
            attempt("own patient record", adapter, f"Patient/{first.patient_id}")
            attempt(
                "another patient's appointment", adapter, f"Appointment/{second.appointment_id}"
            )
            attempt("another patient's record", adapter, f"Patient/{second.patient_id}")
    engine.dispose()


if __name__ == "__main__":
    main()
