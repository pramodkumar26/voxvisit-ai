"""Approved session and login-protection values from docs/authentication-and-sessions.md."""

from dataclasses import dataclass
from datetime import timedelta


@dataclass(frozen=True)
class SessionPolicy:
    actor_type: str
    absolute_lifetime: timedelta
    idle_timeout: timedelta
    max_active_sessions: int
    lockout_duration: timedelta


PATIENT_POLICY = SessionPolicy(
    actor_type="PATIENT",
    absolute_lifetime=timedelta(minutes=30),
    idle_timeout=timedelta(minutes=15),
    max_active_sessions=3,
    lockout_duration=timedelta(minutes=15),
)
STAFF_POLICY = SessionPolicy(
    actor_type="STAFF",
    absolute_lifetime=timedelta(minutes=20),
    idle_timeout=timedelta(minutes=10),
    max_active_sessions=1,
    lockout_duration=timedelta(minutes=30),
)
POLICIES = {policy.actor_type: policy for policy in (PATIENT_POLICY, STAFF_POLICY)}

FAILED_ATTEMPT_LIMIT = 5
FAILED_ATTEMPT_WINDOW = timedelta(minutes=15)
NETWORK_ATTEMPT_LIMIT = 20
NETWORK_ATTEMPT_WINDOW = timedelta(minutes=15)
