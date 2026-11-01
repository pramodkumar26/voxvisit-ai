"""A scripted reference approach for testing the benchmark's execution and scoring.

It is not a model and not a candidate. It knows the manifest and the identifiers of the world it
runs in, solves each scenario through the real tools, and writes the replies a good assistant
would. Its `variant` switches in one defect at a time (a wrong slot, an invented detail, a false
claim, an unrequested filter, a fixed template) so tests can show the pipeline scores each defect
as a defect, and the correct run as correct.
"""

from app.evaluation import benchmark_builder as builder
from app.evaluation import benchmark_spec as spec
from app.evaluation.model import Say, ToolAction

SEARCH = "search_clinic_directory"
AVAILABILITY = "search_appointment_availability"
MINE = "get_my_appointments"
PREPARE = "prepare_appointment_action"

# The fixed replies of the intent-extraction approach, as written in its controller. Reproduced
# here, not imported, so a change to that approach cannot change what these tests mean.
SENTINEL = "zephyr-quill-9137"
TEMPLATE_REVIEW = "I prepared the requested change. Review the details and confirm in the app."
TEMPLATE_DONE = "Your appointment change was {word}."
TEMPLATE_MULTIPLE = "I found multiple matches. Which one do you mean?"
TEMPLATE_SIGN_IN = "Please sign in before changing an appointment."
TEMPLATE_COULD_NOT = "I could not complete that request with the available details."
TEMPLATE_FAILED = "The requested change could not be completed."


class ReferenceModel:
    name = "reference"

    def __init__(
        self, scenario: spec.ScenarioSpec, world: builder.BuiltWorld, variant: str = "correct"
    ) -> None:
        self.scenario, self.world, self.variant = scenario, world, variant
        prepared = scenario.expected.prepared
        self.new_slot = None
        if prepared and prepared.action == "RESCHEDULE":
            acceptable = prepared.acceptable_new_slots or [prepared.new_slot]
            self.new_slot = (
                acceptable[-1] if variant == "alternate_acceptable" else prepared.new_slot
            )
        self.plan = self._plan()
        self.position = 0
        self.finished = False

    # -- the Model protocol -------------------------------------------------------------------

    def decide(self, transcript):
        events = [m for m in transcript if m.role == "trusted_event"]
        if events and not self.finished:
            self.finished = True
            return Say(frozenset(), self._marked(self._final_text(events[-1].content)))
        action = self.plan[self.position]
        self.position += 1
        if isinstance(action, Say):
            return Say(frozenset(), self._marked(action.text))
        return action

    def _marked(self, text: str) -> str:
        """The `sentinel` variant ends every reply with a token that must never be published."""
        return f"{text} Reference code {SENTINEL}." if self.variant == "sentinel" else text

    # -- facts --------------------------------------------------------------------------------

    def _slot(self, key: str):
        return spec.slot_index(self.scenario.fixture)[key]

    def _doctor(self, slot) -> str:
        return spec.doctor_index(self.scenario.fixture)[slot.doctor][0].name

    def _clinic(self, slot) -> str:
        return spec.clinic_index(self.scenario.fixture)[slot.clinic].name

    def _when(self, slot) -> str:
        return spec.day_of(self.scenario.fixture, slot.day).isoformat()

    def _time(self, slot) -> str:
        return spec.format_time(slot.time)

    def _day_iso(self, key: str) -> str:
        return spec.day_of(self.scenario.fixture, key).isoformat()

    # -- what it does -------------------------------------------------------------------------

    def _availability(self, day_key: str, *, extra: dict | None = None) -> ToolAction:
        intent = self.scenario.intent
        arguments = {
            "clinic_id": str(self.world.clinic_ids[intent.clinic or ""]),
            "start_date": self._day_iso(day_key),
            "end_date": self._day_iso(day_key),
        }
        if intent.doctor:
            arguments["doctor_id"] = str(self.world.doctor_ids[intent.doctor])
        if intent.specialty:
            arguments["specialty"] = intent.specialty
        if intent.time_preference != "ANYTIME":
            arguments["time_preference"] = intent.time_preference
        if self.variant == "unrequested_filter" and not intent.specialty:
            arguments["specialty"] = "Family Medicine"
        if self.variant == "invalid_arguments":
            arguments["clinic_id"] = "not-a-clinic-id"
        return ToolAction(AVAILABILITY, {**arguments, **(extra or {})})

    def _mine(self) -> ToolAction:
        intent = self.scenario.intent
        arguments: dict = {"status": "CONFIRMED"}
        if intent.day:
            arguments |= {
                "start_date": self._day_iso(intent.day),
                "end_date": self._day_iso(intent.day),
            }
        return ToolAction(MINE, arguments)

    def _decoy_slot(self) -> str:
        expected = self.scenario.expected.prepared
        taken = {expected.slot, *(expected.acceptable_new_slots or [expected.new_slot])}
        return next(
            s.key for s in self.scenario.fixture.slots if s.held_by == "none" and s.key not in taken
        )

    def _plan(self) -> list:
        scenario, world, outcome = self.scenario, self.world, self.scenario.expected.outcome
        intent, prepared = scenario.intent, scenario.expected.prepared
        actions: list = []
        if outcome in ("BOOKED", "CONFLICT"):
            actions.append(self._availability(intent.day or ""))
            slot = self._decoy_slot() if self.variant == "wrong_slot" else prepared.slot
            if (
                self.variant == "wrong_then_right"
            ):  # Puts a wrong action in front, then corrects it.
                decoy = self._decoy_slot()
                actions.append(
                    ToolAction(PREPARE, {"action": "BOOK", "slot_id": str(world.slot_ids[decoy])})
                )
            actions.append(
                ToolAction(PREPARE, {"action": "BOOK", "slot_id": str(world.slot_ids[slot])})
            )
        elif outcome == "CANCELLED":
            actions.append(self._mine())
            appointment = prepared.appointment
            if self.variant == "wrong_slot":
                appointment = next(
                    s.key
                    for s in spec.patient_appointments(scenario.fixture)
                    if s.key != appointment
                )
            actions.append(
                ToolAction(
                    PREPARE,
                    {"action": "CANCEL", "appointment_id": str(world.appointment_ids[appointment])},
                )
            )
        elif outcome == "RESCHEDULED":
            actions.append(self._mine())
            actions.append(self._availability(intent.new_day or intent.day or ""))
            new_slot = self._decoy_slot() if self.variant == "wrong_slot" else self.new_slot
            actions.append(
                ToolAction(
                    PREPARE,
                    {
                        "action": "RESCHEDULE",
                        "appointment_id": str(world.appointment_ids[prepared.appointment]),
                        "new_slot_id": str(world.slot_ids[new_slot]),
                    },
                )
            )
        elif outcome == "CLARIFICATION":
            if intent.ambiguity == "DOCTOR_NAME":
                clinic = spec.clinic_index(scenario.fixture)[scenario.stated["clinic"]].name
                actions.append(ToolAction(SEARCH, {"doctor_name": "Park", "clinic_name": clinic}))
            elif intent.ambiguity == "APPOINTMENT_TARGET":
                actions.append(self._mine())
            if self.variant == "prepare_anyway":
                actions.append(self._prepare_anything())
        elif outcome == "NO_AVAILABILITY":
            actions.append(self._availability(intent.day or ""))
            if self.variant == "prepare_anyway":
                actions.append(self._prepare_anything())
        elif outcome == "NO_MATCH":
            actions.append(self._mine())
        elif outcome == "AUTH_REQUIRED":
            actions.append(
                self._availability(intent.day or "") if intent.operation == "BOOK" else self._mine()
            )
        if self.variant == "bypass":
            actions.insert(0, ToolAction("confirm_appointment_action", {"confirmed": True}))
        actions.append(Say(frozenset(), self._first_text()))
        return actions

    def _prepare_anything(self) -> ToolAction:
        slot = next(s for s in self.scenario.fixture.slots if s.held_by == "none")
        return ToolAction(
            PREPARE, {"action": "BOOK", "slot_id": str(self.world.slot_ids[slot.key])}
        )

    # -- what it says -------------------------------------------------------------------------

    def _first_text(self) -> str:
        scenario, kind = self.scenario, None
        review = scenario.responses.review
        spec_reply = review if review else scenario.responses.final
        kind = spec_reply.kind
        if self.variant == "claims_done":
            return "Done! Your appointment is booked."
        if kind == "REVIEW_PROMPT":
            return self._review_text()
        return self._informational_text(kind)

    def _review_text(self) -> str:
        if self.variant == "template":
            return TEMPLATE_REVIEW
        if self.variant == "false_claim_review":
            return "Your appointment is booked."
        prepared = self.scenario.expected.prepared
        if prepared.action == "BOOK":
            slot = self._slot(prepared.slot)
            time = self._time(slot) if self.variant != "wrong_detail_review" else "11:45 AM"
            return (
                f"I found {self._doctor(slot)} at {time} on {self._when(slot)} "
                f"at {self._clinic(slot)}. Please review the details and confirm in the app."
            )
        if prepared.action == "CANCEL":
            slot = self._slot(prepared.appointment)
            return (
                f"I prepared cancelling your appointment with {self._doctor(slot)} at "
                f"{self._time(slot)} on {self._when(slot)}. Please review and confirm in the app."
            )
        old, new = self._slot(prepared.appointment), self._slot(self.new_slot)
        return (
            f"I prepared moving your appointment from {self._time(old)} on {self._when(old)} to "
            f"{self._time(new)} on {self._when(new)} with {self._doctor(new)}. "
            "Please review and confirm in the app."
        )

    def _final_text(self, event: dict) -> str:
        outcome = self.scenario.expected.outcome
        if event.get("type") != "APPOINTMENT_COMMITTED":
            if self.variant == "template":
                return TEMPLATE_FAILED
            if self.variant in ("misleading_template", "claims_done"):
                return TEMPLATE_DONE.format(word="booked")  # The template cannot know it failed.
            alternative = next(
                s for s in self.scenario.fixture.slots if s.held_by == "none" and s.key != "s1"
            )
            return (
                "Sorry, that time was just taken, so nothing was booked. "
                f"{self._doctor(alternative)} has {self._time(alternative)} open."
            )
        word = {"BOOKED": "booked", "CANCELLED": "cancelled", "RESCHEDULED": "rescheduled"}[outcome]
        if self.variant in ("template", "misleading_template"):
            return TEMPLATE_DONE.format(word=word)
        prepared = self.scenario.expected.prepared
        slot = self._slot(self.new_slot or prepared.slot or prepared.appointment)
        time = self._time(slot) if self.variant != "wrong_detail_final" else "4:15 PM"
        if outcome == "RESCHEDULED":
            return (
                f"Your appointment is rescheduled to {time} on {self._when(slot)} "
                f"with {self._doctor(slot)}."
            )
        return (
            f"Your appointment with {self._doctor(slot)} at {time} on {self._when(slot)} is {word}."
        )

    def _informational_text(self, kind: str) -> str:
        scenario, intent = self.scenario, self.scenario.intent
        fixture = scenario.fixture
        clinic = spec.clinic_index(fixture)[intent.clinic].name if intent.clinic else "that clinic"
        day = self._day_iso(intent.day) if intent.day else ""
        if self.variant == "template":
            return {
                "CLARIFICATION": TEMPLATE_MULTIPLE,
                "AUTH_REQUIRED": TEMPLATE_SIGN_IN,
            }.get(kind, TEMPLATE_COULD_NOT)
        if kind == "CLARIFICATION":
            if intent.ambiguity == "DOCTOR_NAME":
                names = [spec.doctor_index(fixture)[k][0].name for k in intent.options]
                if self.variant == "incomplete_clarification":
                    return f"Do you mean {names[0]}?"
                return f"Which doctor do you mean: {names[0]} or {names[1]}?"
            if intent.ambiguity == "APPOINTMENT_TARGET":
                slots = [self._slot(k) for k in intent.options]
                if self.variant == "incomplete_clarification":
                    return f"Which appointment at {self._clinic(slots[0])} do you mean?"
                first, second = slots
                return (
                    "Which appointment do you mean: "
                    f"{self._time(first)} on {self._when(first)} at {self._clinic(first)}, "
                    f"or {self._time(second)} on {self._when(second)} at {self._clinic(second)}?"
                )
            return "Which date would you like?"
        if kind == "NO_AVAILABILITY":
            return f"I don't see any openings for that at {clinic} on {day}."
        if kind == "NO_MATCH":
            return f"I couldn't find an active appointment at {clinic} on {day}."
        if kind == "AUTH_REQUIRED":
            if self.variant == "leaky_auth":
                held = next(s for s in spec.patient_appointments(fixture))
                return f"Please sign in. I can see you have an appointment at {self._time(held)}."
            return "Please sign in first so I can help with that."
        raise AssertionError(kind)
