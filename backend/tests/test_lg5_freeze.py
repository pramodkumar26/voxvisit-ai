"""The LG-5 holdout is frozen: written before any LG-5 implementation change, and never changed."""

from app.evaluation import lg4_scenarios, lg5_scenarios


def test_the_lg5_holdout_is_exactly_what_was_frozen() -> None:
    assert lg5_scenarios.fingerprint(lg5_scenarios.HOLDOUT) == lg5_scenarios.HOLDOUT_FINGERPRINT, (
        "The LG-5 holdout changed. It must stay as it was frozen; put new cases in a new set."
    )


def test_the_lg4_holdout_is_still_frozen_as_a_spent_benchmark() -> None:
    assert lg4_scenarios.fingerprint(lg4_scenarios.HOLDOUT) == lg4_scenarios.HOLDOUT_FINGERPRINT


def test_the_lg5_sets_are_fresh_separate_and_large_enough() -> None:
    development, held_out = lg5_scenarios.DEVELOPMENT, lg5_scenarios.HOLDOUT
    assert len(held_out) >= 40 and 60 <= len(development) + len(held_out) <= 100
    older = {s.turns for s in (*lg4_scenarios.DEVELOPMENT, *lg4_scenarios.HOLDOUT)}
    assert not older & {s.turns for s in (*development, *held_out)}
    assert not {s.turns for s in development} & {s.turns for s in held_out}
    assert not {s.id for s in development} & {s.id for s in held_out}
    everything = (*development, *held_out)
    assert len({s.id for s in everything}) == len(everything)
    assert {s.category for s in everything} <= set(lg5_scenarios.CATEGORIES)
    assert all(s.turns for s in everything)
