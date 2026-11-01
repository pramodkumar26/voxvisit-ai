"""A stand-in for the model's reading of a message, for testing the conversation layer.

It reads the way a careful model would: from the words of the message and the names it is told
exist, and from nothing else. It is not a model and not a candidate, and it is deliberately
separate from the production fallback in the controller, so a test of recovery means something.
The real model is exercised by the separate smoke run, not by these tests.
"""

import re

from app.evaluation.hybrid import PatientIntent, date_mentions, minutes_of

_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?m\b", re.IGNORECASE)
_ORDINAL = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


class StandInReader:
    def __init__(self, *, clinics: list[str], doctors: list[str]) -> None:
        self.clinics, self.doctors = clinics, doctors
        self.calls = 0
        self.contexts: list[str] = []

    def salvage(self, *_args, **_kwargs):
        return None

    def extract(self, request, *, context="", today=None, anchor=None, repair=False):
        self.calls += 1
        self.contexts.append(context)
        text, lower = request, request.casefold()
        update: dict = {}
        if re.search(r"never\s*mind|forget it", lower):
            return PatientIntent(operation="UNKNOWN", message_type="ABANDON")
        if "pharmacy" in lower:
            return PatientIntent(operation="UNKNOWN", message_type="OFF_TOPIC")
        if re.search(r"\bcancel\b", lower):
            operation = "CANCEL"
        elif re.search(r"\b(move|reschedule)\b", lower):
            operation = "RESCHEDULE"
        elif re.search(r"\b(book|appointment|need|want|schedule)\b", lower):
            operation = "BOOK"
        else:
            operation = "UNKNOWN"
        update["operation"] = operation
        clinic = max((c for c in self.clinics if c.casefold() in lower), key=len, default=None)
        doctor = max((d for d in self.doctors if d.casefold() in lower), key=len, default=None)
        update["clinic_name"], update["doctor_name"] = clinic, doctor
        mentions = [m for m in date_mentions(text, today, anchor) if m.iso]
        if operation == "RESCHEDULE" and len(mentions) >= 2:
            update["start_date"] = update["end_date"] = mentions[0].iso
            update["new_date"] = mentions[1].iso
        elif (
            operation == "RESCHEDULE"
            and len(mentions) == 1
            and " to " in lower[: mentions[0].position + 1]
        ):
            update["new_date"] = mentions[0].iso
        elif mentions:
            update["start_date"] = update["end_date"] = mentions[0].iso
        clock = _CLOCK.search(text)
        if clock:
            identifies = re.search(r"\bone at\b|\bthe\s+\d[^,]*?\bone\b", lower)
            if identifies:
                update["appointment_time"] = clock.group(0)
            else:
                update["time"] = clock.group(0)
                update["selection"] = "SPECIFIC_TIME"
        if "morning" in lower:
            update["time_preference"] = "MORNING"
        if re.search(r"\b(first appointment|earliest|soonest|next available)\b", lower):
            update["selection"] = "EARLIEST"
        ordinal = re.search(r"\b(first|second|third|fourth|fifth) one\b", lower)
        if ordinal:
            update["option_number"] = _ORDINAL[ordinal.group(1)]
        if minutes_of(update.get("time")) is None:
            update.pop("time", None)
            if update.get("selection") == "SPECIFIC_TIME":
                update.pop("selection")
        return PatientIntent(**{k: v for k, v in update.items() if v is not None})


class MinimalReader:
    """A reading that finds only the kind of request and the clinic, and leaves every date and time
    to the controller. It shows the spoken-language layer works without a model's help."""

    def __init__(self, *, clinics: list[str]) -> None:
        self.clinics = clinics
        self.calls = 0

    def salvage(self, *_args, **_kwargs):
        return None

    def extract(self, request, *, context="", today=None, anchor=None, repair=False):
        self.calls += 1
        lower = request.casefold()
        operation = "BOOK" if re.search(r"\b(book|appointment|need|want)\b", lower) else "UNKNOWN"
        if re.search(r"\bcancel\b", lower):
            operation = "CANCEL"
        clinic = max((c for c in self.clinics if c.casefold() in lower), key=len, default=None)
        ordinal = re.search(r"\b(first|second|third|fourth|fifth) one\b", lower)
        return PatientIntent(
            operation=operation,
            clinic_name=clinic,
            option_number=_ORDINAL[ordinal.group(1)] if ordinal else None,
        )
