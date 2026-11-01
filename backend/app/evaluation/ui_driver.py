"""The trusted UI, for tests of a flow that needs the patient to press Confirm.

This is what the browser does: it holds the session cookie and CSRF token, and calls the
confirmation endpoint over HTTP with an allowed Origin and an Idempotency-Key. It is deliberately
separate from the agent: the model never receives the cookie, the token, or any handle to this
module, and the confirmation route is not in the tool registry. The commit service is not imported
here; the real endpoint is what runs.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from app.evaluation.environment import EvalEnvironment


@dataclass(frozen=True)
class UiSession:
    token: str  # Session cookie value. Held by the UI only.
    csrf: str


@dataclass(frozen=True)
class UiOutcome:
    committed: bool
    status_code: int
    error_code: str | None
    reason: str | None
    result: str | None  # BOOKED, RESCHEDULED, or CANCELLED, from the committed response.


class UiDriver:
    def __init__(self, environment: EvalEnvironment) -> None:
        self._env = environment

    def sign_in(self, patient_code: str) -> UiSession:
        """The private verification form: the PIN goes to the login endpoint, never to the agent."""
        response = self._env.client.post(
            "/auth/patient/login",
            json={"patient_code": patient_code, "pin": self._env.pin},
            headers={"Origin": self._env.origin},
        )
        self._env.client.cookies.clear()
        if response.status_code != 200:
            raise RuntimeError(f"UI sign-in failed with HTTP {response.status_code}")
        token = response.headers["set-cookie"].split(";")[0].split("=", 1)[1]
        return UiSession(token=token, csrf=response.json()["csrf_token"])

    def confirm(self, session: UiSession, pending_action_id: Any) -> UiOutcome:
        """Press Confirm: the explicit, CSRF-protected request the agent can never make."""
        response = self._env.client.post(
            f"/me/appointment-actions/{pending_action_id}/confirm",
            json={"confirmed": True},
            headers={
                "Cookie": f"{self._env.cookie_name}={session.token}",
                "Origin": self._env.origin,
                "X-CSRF-Token": session.csrf,
                "Idempotency-Key": f"ui.{uuid.uuid4().hex}",
            },
        )
        self._env.client.cookies.clear()
        body = response.json()
        if response.status_code == 200:
            return UiOutcome(True, 200, None, None, body.get("result"))
        error = body.get("error", {})
        return UiOutcome(False, response.status_code, error.get("code"), error.get("reason"), None)
