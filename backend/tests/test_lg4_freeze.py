"""The LG-4 holdout is frozen: written before any evaluation of it, and never changed after."""

from app.evaluation.lg4_scenarios import (
    CATEGORIES,
    DEVELOPMENT,
    HOLDOUT,
    HOLDOUT_FINGERPRINT,
    fingerprint,
)


def test_the_holdout_is_exactly_what_was_frozen() -> None:
    assert fingerprint(HOLDOUT) == HOLDOUT_FINGERPRINT, (
        "The LG-4 holdout changed. It must stay as it was frozen; put new cases in a new set."
    )


def test_the_two_sets_are_separate_and_cover_every_category() -> None:
    development, held_out = {s.id for s in DEVELOPMENT}, {s.id for s in HOLDOUT}
    assert not development & held_out
    assert not {s.turns for s in DEVELOPMENT} & {s.turns for s in HOLDOUT}
    assert 40 <= len(development) + len(held_out) <= 60
    for category in CATEGORIES:
        assert any(s.category == category for s in (*DEVELOPMENT, *HOLDOUT)), category
    assert {s.category for s in (*DEVELOPMENT, *HOLDOUT)} <= set(CATEGORIES)


def test_every_conversation_has_turns_and_unique_ids() -> None:
    everything = (*DEVELOPMENT, *HOLDOUT)
    assert len({s.id for s in everything}) == len(everything)
    assert all(s.turns for s in everything)
