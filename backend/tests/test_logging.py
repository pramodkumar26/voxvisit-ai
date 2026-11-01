import logging

from app.core.logging import REDACTED, configure_logging


def test_logging_redacts_sensitive_values(capsys) -> None:
    configure_logging("INFO")
    logger = logging.getLogger("test.redaction")

    logger.warning(
        "database_url=postgresql+psycopg://user:db-secret@localhost/db "
        "password=account-secret token=session-secret"
    )

    captured = capsys.readouterr().err
    assert "db-secret" not in captured
    assert "account-secret" not in captured
    assert "session-secret" not in captured
    assert REDACTED in captured


def test_logging_redacts_percent_style_arguments(capsys) -> None:
    configure_logging("INFO")

    logging.getLogger("test.redaction").warning("connecting with %s", "password=arg-secret")

    captured = capsys.readouterr().err
    assert "arg-secret" not in captured
    assert REDACTED in captured
