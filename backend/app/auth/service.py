"""Trusted authentication and session logic.

Every operation runs in one PostgreSQL transaction. Login follows the approved lock order:
network-source advisory lock, credential row, auth_sessions, then pending_actions.
Failures are returned as values, not raised, so their counters and audit events still commit.
"""

import logging
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from argon2 import PasswordHasher
from sqlalchemy import Connection, Engine, func, insert, select, text, update

from app.auth.passwords import verify_secret
from app.auth.policy import (
    FAILED_ATTEMPT_LIMIT,
    FAILED_ATTEMPT_WINDOW,
    NETWORK_ATTEMPT_LIMIT,
    NETWORK_ATTEMPT_WINDOW,
    POLICIES,
    SessionPolicy,
)
from app.auth.tokens import csrf_token_for, hash_token, network_fingerprint, new_session_token
from app.core.errors import AuthenticationRequiredError, SessionExpiredError
from app.db.models import (
    AuthSession,
    Patient,
    PatientCredential,
    PendingAction,
    SecurityAuditEvent,
    StaffCredential,
    StaffUser,
)

logger = logging.getLogger(__name__)
Clock = Callable[[], datetime]

PATIENT = "PATIENT"
STAFF = "STAFF"
COUNTED_LOGIN_EVENTS = ("LOGIN_SUCCESS", "LOGIN_FAILURE")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Principal:
    """The verified identity of a request. Holds no token, PIN, password, or hash."""

    session_id: uuid.UUID
    actor_type: str
    login: str
    display_name: str
    patient_id: uuid.UUID | None
    staff_id: uuid.UUID | None
    absolute_expires_at: datetime
    idle_expires_at: datetime
    csrf_token: str


@dataclass(frozen=True)
class IssuedSession:
    token: str
    principal: Principal
    lifetime_seconds: int


@dataclass(frozen=True)
class LoginResult:
    issued: IssuedSession | None = None
    rate_limited: bool = False
    retry_after_seconds: int = 0


@dataclass(frozen=True)
class _Actor:
    policy: SessionPolicy
    account: type
    credential: type
    id_name: str
    login_name: str
    hash_name: str
    blocked_name: str

    def column(self, model: type, name: str) -> Any:
        return getattr(model, name)

    @property
    def session_fk(self) -> Any:
        return getattr(AuthSession, self.id_name)


ACTORS = {
    PATIENT: _Actor(
        POLICIES[PATIENT],
        Patient,
        PatientCredential,
        "patient_id",
        "patient_code",
        "pin_hash",
        "disabled",
    ),
    STAFF: _Actor(
        POLICIES[STAFF],
        StaffUser,
        StaffCredential,
        "staff_id",
        "username",
        "password_hash",
        "reset_required",
    ),
}


class AuthService:
    def __init__(
        self,
        engine: Engine,
        *,
        secret_key: bytes,
        hasher: PasswordHasher,
        clock: Clock = utc_now,
    ) -> None:
        self._engine = engine
        self._key = secret_key
        self._hasher = hasher
        self._clock = clock
        # Verified when an account does not exist, so unknown accounts cost the same time.
        self._dummy_hash = hasher.hash("voxvisit-unknown-account-placeholder")

    # --- login ---------------------------------------------------------------------------

    def login(
        self,
        actor_type: str,
        identifier: str,
        secret: str,
        *,
        network_address: str | None,
        request_id: uuid.UUID,
    ) -> LoginResult:
        actor = ACTORS[actor_type]
        now = self._clock()
        source = network_fingerprint(self._key, network_address)
        audit = {"request_id": request_id, "now": now, "network_source": source}

        with self._engine.begin() as conn:
            conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": f"login-source:{source}"},
            )
            retry_after = self._source_retry_after(conn, source, now)
            if retry_after:
                self._audit(conn, "LOGIN_RATE_LIMITED", "BLOCKED", actor_type=actor_type, **audit)
                return LoginResult(rate_limited=True, retry_after_seconds=retry_after)

            account = self._find_account(conn, actor, identifier.strip())
            stored_hash = account.secret_hash if account else self._dummy_hash
            verified = verify_secret(self._hasher, stored_hash, secret)
            rejection = self._rejection_reason(account, now)

            if account is not None and rejection is None and verified:
                return LoginResult(
                    issued=self._complete_login(conn, actor, account, secret, **audit)
                )

            self._record_failure(conn, actor, account, rejection, **audit)
            return LoginResult()

    def _source_retry_after(self, conn: Connection, source: str, now: datetime) -> int:
        count, oldest = conn.execute(
            select(func.count(), func.min(SecurityAuditEvent.occurred_at)).where(
                SecurityAuditEvent.network_source == source,
                SecurityAuditEvent.event_type.in_(COUNTED_LOGIN_EVENTS),
                SecurityAuditEvent.occurred_at > now - NETWORK_ATTEMPT_WINDOW,
            )
        ).one()
        if count < NETWORK_ATTEMPT_LIMIT:
            return 0
        return max(1, math.ceil((oldest + NETWORK_ATTEMPT_WINDOW - now).total_seconds()))

    def _find_account(self, conn: Connection, actor: _Actor, identifier: str) -> Any:
        account, credential = actor.account, actor.credential
        login_column = actor.column(account, actor.login_name)
        matches = (
            login_column == identifier
            if actor.policy.actor_type == PATIENT
            else func.lower(login_column) == identifier.lower()
        )
        statement = (
            select(
                actor.column(account, actor.id_name).label("account_id"),
                account.display_name,
                login_column.label("login"),
                account.active,
                actor.column(credential, actor.hash_name).label("secret_hash"),
                actor.column(credential, actor.blocked_name).label("blocked"),
                credential.failed_attempt_count,
                credential.failed_attempt_window_started_at,
                credential.locked_until,
            )
            .join(
                credential,
                actor.column(credential, actor.id_name) == actor.column(account, actor.id_name),
            )
            .where(matches)
            .with_for_update(of=credential)
        )
        return conn.execute(statement).one_or_none()

    @staticmethod
    def _rejection_reason(account: Any, now: datetime) -> str | None:
        if account is None:
            return "unknown_account"
        if not account.active:
            return "inactive_account"
        if account.blocked:
            return "credential_unavailable"
        if account.locked_until is not None and account.locked_until > now:
            return "account_locked"
        return None

    def _record_failure(
        self,
        conn: Connection,
        actor: _Actor,
        account: Any,
        rejection: str | None,
        **audit: Any,
    ) -> None:
        now = audit["now"]
        actor_id = account.account_id if account is not None else None
        locked_until = None
        if account is not None and rejection is None:
            window = account.failed_attempt_window_started_at
            in_window = window is not None and now - window < FAILED_ATTEMPT_WINDOW
            failures = account.failed_attempt_count + 1 if in_window else 1
            if failures >= FAILED_ATTEMPT_LIMIT:
                locked_until = now + actor.policy.lockout_duration
            conn.execute(
                update(actor.credential)
                .where(actor.column(actor.credential, actor.id_name) == actor_id)
                .values(
                    failed_attempt_count=failures,
                    failed_attempt_window_started_at=window if in_window else now,
                    locked_until=locked_until,
                    updated_at=now,
                )
            )
        self._audit(
            conn,
            "LOGIN_FAILURE",
            "FAILURE",
            actor_type=actor.policy.actor_type,
            actor_id=actor_id,
            details={"reason": rejection or "bad_credential"},
            **audit,
        )
        if locked_until is not None:
            self._audit(
                conn,
                "ACCOUNT_LOCKED",
                "BLOCKED",
                actor_type=actor.policy.actor_type,
                actor_id=actor_id,
                details={"locked_until": locked_until.isoformat()},
                **audit,
            )

    def _complete_login(
        self, conn: Connection, actor: _Actor, account: Any, secret: str, **audit: Any
    ) -> IssuedSession:
        now = audit["now"]
        values: dict[str, Any] = {
            "failed_attempt_count": 0,
            "failed_attempt_window_started_at": None,
            "locked_until": None,
            "updated_at": now,
        }
        if self._hasher.check_needs_rehash(account.secret_hash):
            values[actor.hash_name] = self._hasher.hash(secret)
        conn.execute(
            update(actor.credential)
            .where(actor.column(actor.credential, actor.id_name) == account.account_id)
            .values(**values)
        )
        issued = self._create_session(conn, actor, account, **audit)
        self._audit(
            conn,
            "LOGIN_SUCCESS",
            "SUCCESS",
            actor_type=actor.policy.actor_type,
            actor_id=account.account_id,
            session_id=issued.principal.session_id,
            **audit,
        )
        return issued

    def _create_session(
        self, conn: Connection, actor: _Actor, account: Any, **audit: Any
    ) -> IssuedSession:
        now, policy = audit["now"], actor.policy
        active = conn.execute(
            select(
                AuthSession.session_id,
                AuthSession.idle_expires_at,
                AuthSession.absolute_expires_at,
            )
            .where(actor.session_fk == account.account_id, AuthSession.status == "ACTIVE")
            .order_by(AuthSession.created_at, AuthSession.session_id)
            .with_for_update()
        ).all()
        stale = [
            r.session_id for r in active if now >= r.idle_expires_at or now >= r.absolute_expires_at
        ]
        live = [r.session_id for r in active if r.session_id not in stale]
        self._terminate(conn, stale, "EXPIRED", "EXPIRED", now)
        for session_id in stale:
            self._audit(
                conn,
                "SESSION_EXPIRED",
                "SUCCESS",
                actor_type=policy.actor_type,
                actor_id=account.account_id,
                session_id=session_id,
                details={"reason": "EXPIRED_BEFORE_LOGIN"},
                **audit,
            )

        revoked = live[: max(0, len(live) - policy.max_active_sessions + 1)]
        reason = "SESSION_LIMIT" if policy.actor_type == PATIENT else "REPLACED_BY_LOGIN"
        self._terminate(conn, revoked, "REVOKED", reason, now)
        for session_id in revoked:
            self._audit(
                conn,
                "SESSION_REVOKED",
                "SUCCESS",
                actor_type=policy.actor_type,
                actor_id=account.account_id,
                session_id=session_id,
                details={"reason": reason},
                **audit,
            )

        token = new_session_token()
        token_hash = hash_token(token)
        session_id = conn.execute(
            insert(AuthSession)
            .values(
                actor_type=policy.actor_type,
                role=policy.actor_type,
                token_hash=token_hash,
                status="ACTIVE",
                created_at=now,
                last_activity_at=now,
                idle_expires_at=now + policy.idle_timeout,
                absolute_expires_at=now + policy.absolute_lifetime,
                **{actor.id_name: account.account_id},
            )
            .returning(AuthSession.session_id)
        ).scalar_one()
        principal = Principal(
            session_id=session_id,
            actor_type=policy.actor_type,
            login=account.login,
            display_name=account.display_name,
            patient_id=account.account_id if policy.actor_type == PATIENT else None,
            staff_id=account.account_id if policy.actor_type == STAFF else None,
            absolute_expires_at=now + policy.absolute_lifetime,
            idle_expires_at=now + policy.idle_timeout,
            csrf_token=csrf_token_for(self._key, token_hash),
        )
        return IssuedSession(
            token=token,
            principal=principal,
            lifetime_seconds=int(policy.absolute_lifetime.total_seconds()),
        )

    # --- authenticated requests ----------------------------------------------------------

    def authenticate(self, token: str | None) -> Principal:
        """Validate a session cookie and refresh its idle deadline. Raises on any rejection."""
        if not token:
            raise AuthenticationRequiredError()
        outcome = self._authenticate(hash_token(token), self._clock())
        if isinstance(outcome, Principal):
            return outcome
        raise (
            SessionExpiredError() if outcome == "SESSION_EXPIRED" else AuthenticationRequiredError()
        )

    def _authenticate(self, token_hash: bytes, now: datetime) -> Principal | str:
        with self._engine.begin() as conn:
            session = conn.execute(
                select(AuthSession).where(AuthSession.token_hash == token_hash).with_for_update()
            ).one_or_none()
            if session is None:
                return "AUTH_REQUIRED"
            if session.status != "ACTIVE":
                return "SESSION_EXPIRED" if session.status == "EXPIRED" else "AUTH_REQUIRED"

            actor = ACTORS[session.actor_type]
            account_id = getattr(session, actor.id_name)
            audit = {"request_id": uuid.uuid4(), "now": now, "network_source": None}
            expiry = None
            if now >= session.absolute_expires_at:
                expiry = "ABSOLUTE_TIMEOUT"
            elif now >= session.idle_expires_at:
                expiry = "IDLE_TIMEOUT"
            if expiry:
                self._terminate(conn, [session.session_id], "EXPIRED", expiry, now)
                self._audit(
                    conn,
                    "SESSION_EXPIRED",
                    "SUCCESS",
                    actor_type=session.actor_type,
                    actor_id=account_id,
                    session_id=session.session_id,
                    details={"reason": expiry},
                    **audit,
                )
                return "SESSION_EXPIRED"

            account = conn.execute(
                select(
                    actor.account.display_name,
                    actor.column(actor.account, actor.login_name).label("login"),
                    actor.account.active,
                    actor.column(actor.credential, actor.blocked_name).label("blocked"),
                )
                .join(
                    actor.credential,
                    actor.column(actor.credential, actor.id_name)
                    == actor.column(actor.account, actor.id_name),
                )
                .where(actor.column(actor.account, actor.id_name) == account_id)
            ).one_or_none()
            if account is None or not account.active or account.blocked:
                self._terminate(conn, [session.session_id], "REVOKED", "ACCOUNT_UNAVAILABLE", now)
                self._audit(
                    conn,
                    "SESSION_REVOKED",
                    "SUCCESS",
                    actor_type=session.actor_type,
                    actor_id=account_id,
                    session_id=session.session_id,
                    details={"reason": "ACCOUNT_UNAVAILABLE"},
                    **audit,
                )
                return "AUTH_REQUIRED"

            # Application instances do not share a perfectly synchronized clock. Activity never
            # moves backwards, so a request served by an instance whose clock is slightly behind
            # cannot violate the stored ordering (created <= activity), and the idle deadline
            # can only move forward.
            activity_at = max(now, session.last_activity_at)
            idle_expires_at = min(
                activity_at + actor.policy.idle_timeout, session.absolute_expires_at
            )
            conn.execute(
                update(AuthSession)
                .where(AuthSession.session_id == session.session_id)
                .values(last_activity_at=activity_at, idle_expires_at=idle_expires_at)
            )
            return Principal(
                session_id=session.session_id,
                actor_type=session.actor_type,
                login=account.login,
                display_name=account.display_name,
                patient_id=session.patient_id,
                staff_id=session.staff_id,
                absolute_expires_at=session.absolute_expires_at,
                idle_expires_at=idle_expires_at,
                csrf_token=csrf_token_for(self._key, token_hash),
            )

    def logout(self, principal: Principal, *, request_id: uuid.UUID) -> None:
        now = self._clock()
        with self._engine.begin() as conn:
            status = conn.execute(
                select(AuthSession.status)
                .where(AuthSession.session_id == principal.session_id)
                .with_for_update()
            ).scalar_one_or_none()
            if status != "ACTIVE":
                return
            self._terminate(conn, [principal.session_id], "LOGGED_OUT", "LOGOUT", now)
            self._audit(
                conn,
                "LOGOUT",
                "SUCCESS",
                request_id=request_id,
                now=now,
                actor_type=principal.actor_type,
                actor_id=principal.patient_id or principal.staff_id,
                session_id=principal.session_id,
            )

    # --- shared helpers ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        result: str,
        *,
        request_id: uuid.UUID,
        network_address: str | None = None,
        principal: Principal | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            self._audit(
                conn,
                event_type,
                result,
                request_id=request_id,
                now=self._clock(),
                network_source=network_fingerprint(self._key, network_address),
                actor_type=principal.actor_type if principal else None,
                actor_id=(principal.patient_id or principal.staff_id) if principal else None,
                session_id=principal.session_id if principal else None,
                details=details,
            )

    @staticmethod
    def _terminate(
        conn: Connection,
        session_ids: list[uuid.UUID],
        status: str,
        reason: str,
        now: datetime,
    ) -> None:
        if not session_ids:
            return
        conn.execute(
            update(AuthSession)
            .where(AuthSession.session_id.in_(session_ids))
            .values(status=status, terminated_at=now, termination_reason=reason)
        )
        conn.execute(
            update(PendingAction)
            .where(PendingAction.session_id.in_(session_ids), PendingAction.status == "ACTIVE")
            .values(status="INVALIDATED", updated_at=now)
        )

    @staticmethod
    def _audit(
        conn: Connection,
        event_type: str,
        result: str,
        *,
        request_id: uuid.UUID,
        now: datetime,
        network_source: str | None = None,
        actor_type: str | None = None,
        actor_id: uuid.UUID | None = None,
        session_id: uuid.UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            insert(SecurityAuditEvent).values(
                event_type=event_type,
                result=result,
                actor_type=actor_type,
                actor_id=actor_id,
                session_id=session_id,
                request_id=request_id,
                occurred_at=now,
                network_source=network_source,
                details=details or {},
            )
        )
