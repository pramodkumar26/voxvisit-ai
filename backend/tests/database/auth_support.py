from types import SimpleNamespace
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Request
from pydantic import SecretStr
from sqlalchemy import Engine

from app.auth.dependencies import (
    CSRF_HEADER,
    OptionalPrincipal,
    PatientPrincipal,
    ensure_owns_patient,
    require_permission,
)
from app.auth.permissions import Permission
from app.core.config import Settings
from app.main import create_app

ORIGIN = "http://localhost:3000"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
COOKIE = "voxvisit_session"

# Routes that exist only in tests, to prove the reusable authorization dependencies without
# implementing any business API.
probe_router = APIRouter(prefix="/_probe")


@probe_router.get("/public")
def public_probe(principal: OptionalPrincipal) -> dict[str, Any]:
    return {"actor": principal.actor_type if principal else None}


@probe_router.get("/patients/{patient_id}/profile")
def patient_profile(patient_id: UUID, principal: PatientPrincipal) -> dict[str, Any]:
    ensure_owns_patient(principal, patient_id)
    return {"display_name": principal.display_name}


@probe_router.get("/me")
def my_identity(principal: PatientPrincipal) -> dict[str, Any]:
    return {"patient_id": str(principal.patient_id)}


@probe_router.post("/patient/action")
def patient_action(principal: PatientPrincipal) -> dict[str, bool]:
    return {"ok": True}


@probe_router.get(
    "/staff/schedule", dependencies=[Depends(require_permission(Permission.VIEW_CLINIC_SCHEDULES))]
)
def staff_schedule() -> dict[str, bool]:
    return {"ok": True}


@probe_router.post(
    "/staff/availability",
    dependencies=[Depends(require_permission(Permission.MANAGE_DOCTOR_AVAILABILITY))],
)
def staff_availability() -> dict[str, bool]:
    return {"ok": True}


@probe_router.post(
    "/appointments/cancel",
    dependencies=[Depends(require_permission(Permission.MANAGE_OWN_APPOINTMENTS))],
)
def cancel_appointment(request: Request) -> dict[str, bool]:
    return {"ok": True}


@probe_router.post(
    "/waitlist/join", dependencies=[Depends(require_permission(Permission.MANAGE_OWN_WAITLIST))]
)
def join_waitlist() -> dict[str, bool]:
    return {"ok": True}


def make_settings(engine: Engine, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": SecretStr(engine.url.render_as_string(hide_password=False)),
        "environment": "test",
        "auth_secret_key": SecretStr("test-only-auth-secret-key-0123456789"),
        "auth_trust_forwarded_for": True,
        **overrides,
    }
    return Settings(_env_file=None, **values)


def build_app(engine: Engine, clock, hasher, **settings: Any) -> FastAPI:
    app = create_app(make_settings(engine, **settings), clock=clock, hasher=hasher)
    app.include_router(probe_router)
    return app


def login(
    env: SimpleNamespace,
    kind: str,
    identifier: str,
    secret: str,
    *,
    origin: str | None = ORIGIN,
    address: str | None = None,
):
    field, secret_field = ("patient_code", "pin") if kind == "patient" else ("username", "password")
    headers = {}
    if origin:
        headers["Origin"] = origin
    if address:
        headers["X-Forwarded-For"] = address
    response = env.client.post(
        f"/auth/{kind}/login", json={field: identifier, secret_field: secret}, headers=headers
    )
    env.client.cookies.clear()
    return response


def sign_in(env: SimpleNamespace, kind: str, identifier: str, secret: str, **kwargs):
    response = login(env, kind, identifier, secret, **kwargs)
    assert response.status_code == 200, response.text
    token = response.headers["set-cookie"].split(";")[0].split("=", 1)[1]
    return SimpleNamespace(token=token, csrf=response.json()["csrf_token"], response=response)


def call(
    env: SimpleNamespace,
    method: str,
    path: str,
    session: SimpleNamespace | None = None,
    *,
    token: str | None = None,
    csrf: str | None = "session",
    origin: str | None = ORIGIN,
    **kwargs: Any,
):
    headers = dict(kwargs.pop("headers", {}))
    cookie = token or (session.token if session else None)
    if cookie:
        headers["Cookie"] = f"{COOKIE}={cookie}"
    if method.upper() not in SAFE_METHODS:
        if origin:
            headers["Origin"] = origin
        if csrf == "session" and session:
            headers[CSRF_HEADER] = session.csrf
        elif csrf and csrf != "session":
            headers[CSRF_HEADER] = csrf
    response = env.client.request(method, path, headers=headers, **kwargs)
    env.client.cookies.clear()
    return response
