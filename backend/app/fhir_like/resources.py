"""Simplified FHIR-like resource shapes. Field names follow FHIR where a field exists; the models
are intentionally small, forbid unknown fields, and are not HL7 FHIR profiles."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

MODEL_SYSTEM = "urn:voxvisit:synthetic:resource-model"
MODEL_CODE = "synthetic-fhir-like"
MODEL_DISPLAY = "Simplified synthetic FHIR-like resource; not HL7 FHIR conformant"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Coding(Model):
    system: str
    code: str
    display: str | None = None


class Text(Model):
    text: str


class Identifier(Model):
    system: str
    value: str


class Reference(Model):
    """A literal reference ("Slot/<id>") or, for a clinic, a logical one (identifier + display)."""

    reference: str | None = None
    type: str | None = None
    identifier: Identifier | None = None
    display: str | None = None


class Extension(Model):
    url: str
    valueCode: str | None = None
    valueDateTime: datetime | None = None


class Meta(Model):
    versionId: str | None = None
    lastUpdated: datetime | None = None
    tag: list[Coding]


class HumanName(Model):
    text: str


class Resource(Model):
    id: str
    meta: Meta


class Patient(Resource):
    resourceType: Literal["Patient"] = "Patient"
    identifier: list[Identifier]
    active: bool
    name: list[HumanName]


class Practitioner(Resource):
    resourceType: Literal["Practitioner"] = "Practitioner"
    active: bool
    name: list[HumanName]
    qualification: list[dict[str, Text]]


class Schedule(Resource):
    resourceType: Literal["Schedule"] = "Schedule"
    active: bool
    actor: list[Reference]
    specialty: list[Text]
    extension: list[Extension]


class Slot(Resource):
    resourceType: Literal["Slot"] = "Slot"
    status: Literal["free", "busy", "busy-unavailable", "busy-tentative"]
    schedule: Reference
    specialty: list[Text]
    start: datetime
    end: datetime
    extension: list[Extension]


class AppointmentParticipant(Model):
    actor: Reference
    status: Literal["accepted"] = "accepted"


class Appointment(Resource):
    resourceType: Literal["Appointment"] = "Appointment"
    status: Literal["booked", "cancelled"]
    slot: list[Reference]
    specialty: list[Text]
    start: datetime
    end: datetime
    participant: list[AppointmentParticipant]
    extension: list[Extension]


AnyResource = Annotated[
    Patient | Practitioner | Schedule | Slot | Appointment, Field(discriminator="resourceType")
]


class BundleEntry(Model):
    fullUrl: str
    resource: AnyResource


class BundleLink(Model):
    relation: Literal["next"]
    url: str


class Bundle(Model):
    resourceType: Literal["Bundle"] = "Bundle"
    type: Literal["searchset"] = "searchset"
    meta: Meta
    entry: list[BundleEntry]
    link: list[BundleLink] = []
