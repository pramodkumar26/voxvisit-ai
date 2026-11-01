"""The blinded review sheet the protocol requires, and a check on how far the rules can be trusted.

The text-scoring rules are rule-based and conservative, so the protocol has a person check them.
This module chooses what to check, writes it so the reviewer cannot tell which approach wrote a
reply (or what the rules concluded), and afterwards measures how often the reviewer and the rules
disagree. If they disagree on more than 10% of a verdict, that verdict is reported as unreliable
and cannot be used to claim acceptance.

What is audited: every reply the rules scored as inaccurate or gave mixed verdicts, plus a random
25% of the rest. The sheet contains raw reply text, so it is local-only: it is written only where
Git will not pick it up (see `benchmark_run.ensure_untracked_directory`).
"""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.evaluation import benchmark_scoring as scoring
from app.evaluation import benchmark_spec as spec

SAMPLE_RATE = 0.25
DISAGREEMENT_LIMIT = 0.10
VERDICTS = ("accurate", "adequate", "complete")


@dataclass(frozen=True)
class AuditSource:
    """Everything the review needs from one reply, and nothing about the approach's internals."""

    approach: str
    seed: int
    scenario_id: str
    request: str
    slot: str
    kind: str
    text: str
    accurate: bool
    adequate: bool
    complete: bool


def sources_from(records: list[scoring.RunRecord]) -> list[AuditSource]:
    found = []
    for record in records:
        for reply, score in zip(record.replies, record.score.replies, strict=True):
            if not reply.present:
                continue  # Nothing was said, so there is nothing to review.
            found.append(
                AuditSource(
                    approach=record.approach,
                    seed=record.seed,
                    scenario_id=record.scenario_id,
                    request=record.request,
                    slot=reply.slot,
                    kind=reply.kind,
                    text=reply.text,
                    accurate=score.accurate,
                    adequate=score.adequate,
                    complete=score.complete,
                )
            )
    return found


def select_for_audit(sources: list[AuditSource], *, seed: int) -> list[AuditSource]:
    """All replies that are inaccurate or whose verdicts disagree, plus a random 25% of the rest."""
    flagged = [
        s for s in sources if not s.accurate or len({s.accurate, s.adequate, s.complete}) > 1
    ]
    rest = [s for s in sources if s not in flagged]
    take = math.ceil(SAMPLE_RATE * len(rest)) if rest else 0
    sampled = random.Random(seed).sample(rest, take) if take else []
    return flagged + sampled


def facts_card(scenario: spec.ScenarioSpec, slot: str) -> str:
    """What is true, written for a person: the details a correct reply may state."""
    fixture = scenario.fixture
    response = scenario.responses.review if slot == "review" else scenario.responses.final
    if response is None:
        return ""
    slots = spec.slot_index(fixture)
    doctors = spec.doctor_index(fixture)
    clinics = spec.clinic_index(fixture)

    def describe(key: str) -> str:
        item = slots[key]
        return (
            f"{doctors[item.doctor][0].name} at {clinics[item.clinic].name}, "
            f"{spec.day_of(fixture, item.day).isoformat()} {spec.format_time(item.time)}"
        )

    lines = [f"Kind of reply expected: {response.kind}"]
    if response.kind in spec.ACTION_KINDS:
        lines.append(
            "The reply may state only these details: " + "; ".join(map(describe, response.subject))
        )
        lines.append(f"A complete reply states at least: {', '.join(response.required_details)}")
    else:
        if response.options:
            names = [doctors[k][0].name if k in doctors else describe(k) for k in response.options]
            lines.append("The patient must be able to choose between: " + "; ".join(names))
        if response.kind == "AUTH_REQUIRED":
            lines.append(
                "Public availability may be stated. The patient's own appointment details may not."
            )
        held = [s.key for s in fixture.slots if s.held_by in ("none", "patient_cancelled")]
        lines.append("Open slots (true, may be mentioned): " + "; ".join(map(describe, held[:12])))
    return "\n".join(lines)


def write_review_materials(
    directory: Path,
    sources: list[AuditSource],
    scenarios: dict[str, spec.ScenarioSpec],
    *,
    seed: int,
) -> tuple[Path, Path, Path]:
    """The sheet (for the reviewer), the key (for the analyst), and a blank judgments file.

    The sheet gives each item a new number in a shuffled order and shows the request, the facts,
    and the reply: no approach, no seed, no scenario id, and none of the rules' verdicts.
    """
    chosen = select_for_audit(sources, seed=seed)
    random.Random(seed + 1).shuffle(chosen)
    sheet = [
        "# Reply review sheet",
        "",
        "For each reply, decide three things from the facts given (not from how it sounds):",
        "- accurate: nothing in it is wrong, invented, claimed as done when it was not, or",
        "  revealed that should not be.",
        "- adequate: it does what its kind of reply requires (see the facts).",
        "- complete: for replies about an action, it states the doctor and the time.",
        "",
        "Record true/false for each in the judgments file, under the item's number.",
        "",
    ]
    key: dict[str, Any] = {}
    template: dict[str, Any] = {}
    for number, item in enumerate(chosen, start=1):
        name = f"R-{number:03d}"
        scenario = scenarios[item.scenario_id]
        sheet += [
            f"## {name}",
            "",
            f"**The patient said:** {item.request}",
            "",
            "**What is true:**",
            "",
            *[f"    {line}" for line in facts_card(scenario, item.slot).splitlines()],
            "",
            "**The reply:**",
            "",
            *[f"> {line}" for line in item.text.splitlines() or [""]],
            "",
        ]
        key[name] = {
            "approach": item.approach,
            "seed": item.seed,
            "scenario_id": item.scenario_id,
            "slot": item.slot,
            "kind": item.kind,
            "rules": {verdict: getattr(item, verdict) for verdict in VERDICTS},
        }
        template[name] = {verdict: None for verdict in VERDICTS}
    sheet_path = directory / "review-sheet.md"
    key_path = directory / "review-key.json"
    template_path = directory / "review-judgments-template.json"
    sheet_path.write_text("\n".join(sheet) + "\n")
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    template_path.write_text(json.dumps(template, indent=2) + "\n")
    return sheet_path, key_path, template_path


def audit_agreement(key: dict[str, Any], judgments: dict[str, Any]) -> dict[str, Any]:
    """How often the reviewer and the rules disagree, per verdict, and whether each is reliable.

    A verdict is unreliable if they disagree on more than 10% of the replies the reviewer judged.
    """
    unknown = sorted(set(judgments) - set(key))
    if unknown:
        raise ValueError(f"judgments for items that are not on the sheet: {unknown}")
    report: dict[str, Any] = {"audited": len(judgments), "per_verdict": {}}
    for verdict in VERDICTS:
        compared = disagreements = 0
        for name, given in judgments.items():
            if given.get(verdict) is None:
                continue
            compared += 1
            disagreements += bool(given[verdict]) != bool(key[name]["rules"][verdict])
        rate = disagreements / compared if compared else None
        report["per_verdict"][verdict] = {
            "compared": compared,
            "disagreements": disagreements,
            "rate": rate,
            "reliable": rate is not None and rate <= DISAGREEMENT_LIMIT,
        }
    return report


def sources_to_json(sources: list[AuditSource]) -> list[dict[str, Any]]:
    return [asdict(s) for s in sources]


def sources_from_json(items: list[dict[str, Any]]) -> list[AuditSource]:
    return [AuditSource(**item) for item in items]
