"""A generic stand-in for the model, used only to check that the acceptance cases are executable
and that their expected end states can be reached by a correct agent. It reads the kind of request,
the clinic, the doctor and a position in a list; every date and time is left to the controller.

It is not a model, it never sees the real model's behavior, and its results are not the
acceptance result.
"""

import re

from app.evaluation.hybrid import PatientIntent

_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


class AcceptanceReader:
    def __init__(self, *, clinics: list[str], doctors: list[str]) -> None:
        self.clinics, self.doctors = clinics, doctors

    def salvage(self, *_args, **_kwargs):
        return None

    def extract(self, request, *, context="", today=None, anchor=None, repair=False, **_):
        lower = request.casefold()
        if re.search(r"never\s*mind|forget it", lower):
            return PatientIntent(operation="UNKNOWN", message_type="ABANDON")
        if re.search(r"\bcancel\b|can'?t make", lower):
            operation = "CANCEL"
        elif re.search(r"\b(move|reschedul|push)\w*", lower):
            operation = "RESCHEDULE"
        elif re.search(r"\b(book|appointment|visit|see|openings|get in|fit me|need|like)\b", lower):
            operation = "BOOK"
        else:
            operation = "UNKNOWN"
        clinic = max((c for c in self.clinics if c.casefold() in lower), key=len, default=None)
        doctor = max((d for d in self.doctors if d.casefold() in lower), key=len, default=None)
        ordinal = re.search(r"\b(first|second|third|fourth|fifth) one\b", lower)
        selection = (
            "EARLIEST" if re.search(r"\b(earliest|first available)\b", lower) else "UNSPECIFIED"
        )
        appointment_time = None
        found = re.search(r"\bthe (\d{1,2} ?[ap]m) one\b", lower)
        if found:
            appointment_time = found.group(1)
        return PatientIntent(
            operation=operation,
            clinic_name=clinic,
            doctor_name=doctor,
            option_number=_ORDINAL[ordinal.group(1)] if ordinal else None,
            selection=selection,
            appointment_time=appointment_time,
        )
