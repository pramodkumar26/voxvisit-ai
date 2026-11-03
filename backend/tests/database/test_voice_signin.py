"""The voice agent signs in through the real login service, with synthetic credentials."""

import pytest

from app.auth.service import AuthService
from app.evaluation.conversation import TextSession
from app.voice.stt import SpeechError
from app.voice.voice_agent import sign_in

PATIENT_A = "PT-1001"  # A synthetic seeded patient; the PIN is the test database's own.


def test_a_synthetic_patient_signs_in_and_gets_a_verified_principal(auth_env) -> None:
    issued = sign_in(auth_env.service, PATIENT_A, auth_env.pin)

    assert issued.principal.actor_type == "PATIENT" and issued.principal.patient_id
    assert auth_env.service.authenticate(issued.token).session_id == issued.principal.session_id


def test_the_session_it_gets_works_with_the_text_agent_tools(auth_env) -> None:
    from app.agent_tools import ToolRuntime

    issued = sign_in(auth_env.service, PATIENT_A, auth_env.pin)
    runtime = ToolRuntime(auth_env.app.state.db_engine, clock=auth_env.clock)

    mine = runtime.invoke("get_my_appointments", {}, principal=issued.principal)

    assert mine.ok  # A protected tool accepts the voice agent's verified session.
    assert isinstance(
        TextSession(object(), runtime, issued.principal).principal, type(issued.principal)
    )


def test_a_wrong_pin_is_refused_with_a_message_that_repeats_nothing(auth_env) -> None:
    with pytest.raises(SpeechError, match="not accepted") as error:
        sign_in(auth_env.service, PATIENT_A, "000000-wrong")

    assert "000000" not in str(error.value) and PATIENT_A not in str(error.value)


def test_an_unknown_patient_code_is_refused_the_same_way(auth_env) -> None:
    with pytest.raises(SpeechError, match="not accepted"):
        sign_in(auth_env.service, "PT-NOBODY", auth_env.pin)


def test_the_service_type_used_is_the_real_one(auth_env) -> None:
    assert isinstance(auth_env.service, AuthService)
