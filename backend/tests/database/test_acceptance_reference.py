"""The acceptance cases are executable and their expected end states are reachable.

A generic stand-in reader (never the real model) plays each of the 25 conversations through the real
tools, the trusted-UI confirmation and the disposable database. This proves the cases are well
formed: the worlds exist, the turns run, and a correct agent can reach the expected state. It says
nothing about the real model.
"""

import re
import uuid

import pytest
from acceptance_support import AcceptanceReader
from test_conversation import evaluation  # noqa: F401  (the shared fixture)

from app.evaluation.acceptance_cases import CASES
from app.evaluation.acceptance_score import score
from app.evaluation.conversation_eval import run_conversations
from app.evaluation.scenario import Said


def reader_for(spec, resolve):
    names = {}
    for term in spec.request_terms:
        if term.startswith("clinic"):
            names[term] = str(resolve(Said(term)))
    clinics = list(names.values())
    # Doctors named in the patient's own words, taken from the request and the later turns.
    text = " ".join([spec.request, *[getattr(t, "text", "") for t in spec.timeline]])
    doctors = sorted(set(re.findall(r"Dr\. ([A-Z][a-z]+(?: [A-Z][a-z]+)?)", text)), key=len)
    return AcceptanceReader(clinics=clinics, doctors=doctors)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_each_case_can_be_completed_by_a_correct_agent(evaluation, case) -> None:  # noqa: F811
    [run] = run_conversations(
        evaluation,
        reader_for,
        [case.scenario],
        faults={case.id: case.fault} if case.fault else {},
        suffix_base=uuid.uuid4().hex[:6],
    )

    scored = score(case, run.result, run.model)

    assert scored.violations == [], scored.violations
    assert scored.success, (scored.reasons, run.result.failures)


def test_a_long_run_stays_under_the_applications_sign_in_limit(evaluation) -> None:  # noqa: F811
    """Twenty-five conversations sign in from one source; the application allows 20 in 15 minutes.

    The first attempt at the real run stopped on the 21st sign-in; the run now moves the clock on.
    """
    from datetime import timedelta

    started = evaluation.clock.now
    scenarios = [case.scenario for case in CASES[4:8]]  # Four short conversations.

    runs = run_conversations(
        evaluation, reader_for, scenarios, suffix_base=uuid.uuid4().hex[:6], pause_every=2
    )

    assert len(runs) == 4
    assert evaluation.clock.now == started + timedelta(minutes=32)
    evaluation.clock.now = started
