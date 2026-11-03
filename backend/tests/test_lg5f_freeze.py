"""The LG-5F holdout is frozen: written before any LG-5F implementation change; never changed."""

from app.evaluation import lg4_scenarios, lg5_scenarios, lg5d_scenarios, lg5e_scenarios
from app.evaluation import lg5f_scenarios as f


def test_the_lg5f_holdout_is_exactly_what_was_frozen() -> None:
    assert f.fingerprint(f.HOLDOUT) == f.HOLDOUT_FINGERPRINT, (
        "The LG-5F holdout changed. It must stay as it was frozen; put new cases in a new set."
    )


def test_the_earlier_holdouts_are_still_frozen_as_spent_benchmarks() -> None:
    for module in (lg4_scenarios, lg5_scenarios, lg5d_scenarios, lg5e_scenarios):
        assert module.fingerprint(module.HOLDOUT) == module.HOLDOUT_FINGERPRINT


def test_the_lg5f_sets_are_fresh_separate_and_sized_as_required() -> None:
    assert len(f.HOLDOUT) == 30
    older = {
        s.turns
        for module in (lg4_scenarios, lg5_scenarios, lg5d_scenarios, lg5e_scenarios)
        for s in (*module.DEVELOPMENT, *module.HOLDOUT)
    }
    mine = [*f.DEVELOPMENT, *f.HOLDOUT]
    assert not older & {s.turns for s in mine}
    assert not {s.turns for s in f.DEVELOPMENT} & {s.turns for s in f.HOLDOUT}
    assert len({s.id for s in mine}) == len(mine)
    assert {s.category for s in mine} <= set(f.CATEGORIES)
