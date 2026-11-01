"""Keep the CI workflow honest. It cannot run here, so these tests pin what it must contain."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
RUN_COMMANDS = [
    "uv lock --check",
    "uv sync --locked --group dev",
    "uv run ruff check .",
    "uv run ruff format --check .",
    "alembic upgrade head",
    "alembic check",
    "alembic downgrade base",
    "uv run pytest --strict-markers",
    "check_test_report.py",
    "no-leaked-test-databases",
]


def test_the_workflow_runs_every_required_check() -> None:
    for command in RUN_COMMANDS:
        assert command in WORKFLOW, command
    assert 'RUN_DATABASE_TESTS: "1"' in WORKFLOW


def test_the_database_is_a_disposable_postgres_17_service_with_btree_gist_checks() -> None:
    assert "image: postgres:17" in WORKFLOW
    assert "--health-cmd" in WORKFLOW
    assert "check_postgres.py preflight" in WORKFLOW
    compose = (ROOT / "compose.yaml").read_text()
    assert re.search(r"image: postgres:17", compose)  # CI matches local development.


def test_the_workflow_uses_python_311_and_uv_from_the_repository_pins() -> None:
    assert "python-version-file: .python-version" in WORKFLOW
    assert (ROOT / ".python-version").read_text().strip() == "3.11"
    assert re.search(r'version: "\d+\.\d+\.\d+"', WORKFLOW)  # uv is pinned, not "latest"


def test_actions_are_pinned_to_commit_shas_with_read_only_permissions() -> None:
    uses = re.findall(r"^\s*uses:\s*(\S+)", WORKFLOW, re.MULTILINE)
    assert uses
    for reference in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", reference), reference
    assert re.search(r"^permissions:\n  contents: read$", WORKFLOW, re.MULTILINE)
    assert "persist-credentials: false" in WORKFLOW


def test_the_workflow_needs_no_secrets_or_external_services() -> None:
    assert "secrets." not in WORKFLOW
    assert "POSTGRES_PASSWORD" not in WORKFLOW
    assert "POSTGRES_HOST_AUTH_METHOD: trust" in WORKFLOW
    assert not re.search(r"(?i)api[_-]?key|token:|password:", WORKFLOW)
    # The database URL carries no password.
    url = re.search(r"DATABASE_URL: (\S+)", WORKFLOW).group(1)
    assert re.fullmatch(r"postgresql\+psycopg://\w+@127\.0\.0\.1:\d+/\w+", url)


def test_ci_helper_scripts_exist() -> None:
    for name in ("check_postgres.py", "check_test_report.py"):
        assert (ROOT / ".github" / "scripts" / name).is_file()
