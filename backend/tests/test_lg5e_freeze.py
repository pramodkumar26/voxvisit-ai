"""The LG-5E holdout is frozen: written before any LG-5E implementation change; never changed."""

from app.evaluation import lg4_scenarios, lg5_scenarios, lg5d_scenarios, lg5e_scenarios


def test_the_lg5e_holdout_is_exactly_what_was_frozen() -> None:
    assert (
        lg5e_scenarios.fingerprint(lg5e_scenarios.HOLDOUT) == lg5e_scenarios.HOLDOUT_FINGERPRINT
    ), "The LG-5E holdout changed. It must stay as it was frozen; put new cases in a new set."


def test_the_earlier_holdouts_are_still_frozen_as_spent_benchmarks() -> None:
    assert lg4_scenarios.fingerprint(lg4_scenarios.HOLDOUT) == lg4_scenarios.HOLDOUT_FINGERPRINT
    assert lg5_scenarios.fingerprint(lg5_scenarios.HOLDOUT) == lg5_scenarios.HOLDOUT_FINGERPRINT
    assert lg5d_scenarios.fingerprint(lg5d_scenarios.HOLDOUT) == lg5d_scenarios.HOLDOUT_FINGERPRINT


def test_the_lg5e_sets_are_fresh_separate_and_sized_as_required() -> None:
    development, held_out = lg5e_scenarios.DEVELOPMENT, lg5e_scenarios.HOLDOUT
    assert len(held_out) == 30 and len(development) >= 30
    older = {
        s.turns
        for s in (
            *lg4_scenarios.DEVELOPMENT,
            *lg4_scenarios.HOLDOUT,
            *lg5_scenarios.DEVELOPMENT,
            *lg5_scenarios.HOLDOUT,
            *lg5d_scenarios.DEVELOPMENT,
            *lg5d_scenarios.HOLDOUT,
        )
    }
    assert not older & {s.turns for s in (*development, *held_out)}
    assert not {s.turns for s in development} & {s.turns for s in held_out}
    assert not {s.id for s in development} & {s.id for s in held_out}
    assert {s.category for s in (*development, *held_out)} <= set(lg5e_scenarios.CATEGORIES)
