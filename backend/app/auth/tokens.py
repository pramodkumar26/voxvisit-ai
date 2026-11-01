import base64
import hashlib
import hmac
import secrets


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def csrf_token_for(secret_key: bytes, token_hash: bytes) -> str:
    """A CSRF token bound to one session: only the server can derive it from the session."""
    digest = hmac.new(secret_key, b"csrf:" + token_hash, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def tokens_match(expected: str, presented: str | None) -> bool:
    return presented is not None and hmac.compare_digest(expected.encode(), presented.encode())


def network_fingerprint(secret_key: bytes, address: str | None) -> str:
    """A keyed hash so audit records never store raw network addresses."""
    keyed = hmac.new(secret_key, b"network:" + (address or "unknown").encode(), hashlib.sha256)
    return keyed.hexdigest()[:32]
