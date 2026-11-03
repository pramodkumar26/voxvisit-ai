from app.db.base import Base

POSTGRESQL_IDENTIFIER_LIMIT = 63


def test_model_metadata_defines_the_approved_tables() -> None:
    import app.db.models  # noqa: F401

    # The 24 approved tables, plus review_handoffs (migration 0004, Phase 3.3).
    assert len(Base.metadata.tables) == 25


def test_constraint_and_index_names_fit_postgresql_identifier_limit() -> None:
    import app.db.models  # noqa: F401

    names = [
        item.name
        for table in Base.metadata.tables.values()
        for item in (*table.constraints, *table.indexes)
        if item.name
    ]

    assert [name for name in names if len(name) > POSTGRESQL_IDENTIFIER_LIMIT] == []
