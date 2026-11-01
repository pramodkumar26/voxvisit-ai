import unicodedata

from sqlalchemy import ColumnElement, func

from app.core.errors import InvalidArgumentError

TITLES = frozenset({"dr", "doctor"})


def search_tokens(value: str, *, drop_titles: bool = False) -> list[str]:
    """Normalize a name query into lowercase tokens. Every token must appear in a match."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    tokens = normalized.replace(".", " ").replace(",", " ").split()
    if drop_titles:
        tokens = [token for token in tokens if token not in TITLES]
    if not tokens:
        raise InvalidArgumentError("Search text must contain at least one letter or digit.")
    return tokens


def contains_all(column: ColumnElement, tokens: list[str]) -> list[ColumnElement[bool]]:
    """Case-insensitive 'contains' tests. LIKE wildcards in user input are matched literally."""
    return [func.lower(column).like(f"%{_escape(token)}%", escape="\\") for token in tokens]


def normalize_specialty(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _escape(token: str) -> str:
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
