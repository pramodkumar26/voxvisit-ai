"""Run the paired free-form and hybrid architecture development comparison."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.agent_tools import build_default_registry
from app.evaluation.ab_scenarios import AB_SCENARIOS
from app.evaluation.cli import served_model_digest, software_version, unusable
from app.evaluation.environment import create_disposable_environment
from app.evaluation.hybrid import (
    HYBRID_PROMPT_VERSION,
    HybridControllerModel,
    StructuredIntentExtractor,
)
from app.evaluation.llm import (
    APPROACH_A_PROMPT_VERSION,
    APPROACH_A_SYSTEM_PROMPT,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LlmConfig,
    OllamaModel,
)
from app.evaluation.report import assert_clean, build_report
from app.evaluation.runner import run_suite

MANIFEST = Path(__file__).parents[2] / "tests" / "agent-architecture-comparison-v1.json"
SEEDS = (7, 19)


def _cases():
    return [
        replace(scenario, id=f"{scenario.id}-S{seed}")
        for seed in SEEDS
        for scenario in AB_SCENARIOS
    ]


def _seed(spec) -> int:
    return int(spec.id.rsplit("-S", 1)[1])


def _usage(models, *, hybrid: bool) -> list[dict[str, Any]]:
    rows = []
    for scenario_id, model in models:
        source = model.extractor if hybrid else model
        rows.append({"scenario": scenario_id, **source.usage.as_dict()})
    return rows


def _model_info(models, *, digest: str, hybrid: bool) -> dict[str, Any]:
    sources = [model.extractor if hybrid else model for _, model in models]
    replies = sum(source.served_replies for source in sources)
    attested = sum(source.server_attested_replies for source in sources)
    calls = [call for source in sources for call in source.usage.calls]
    return {
        "provider": "ollama",
        "model": DEFAULT_MODEL,
        "model_digest": digest,
        "real_model": bool(sources)
        and all(source.real_transport for source in sources)
        and replies > 0
        and replies == attested,
        "prompt_version": HYBRID_PROMPT_VERSION if hybrid else APPROACH_A_PROMPT_VERSION,
        "seeds": list(SEEDS),
        "temperature": 0.0,
        "provider_calls": len(calls),
        "provider_errors": sum(
            call.outcome not in {"tool_call", "message", "structured_intent"} for call in calls
        ),
        "provider_timeouts": sum(call.outcome == "ModelTimeoutError" for call in calls),
        "usage": _usage(models, hybrid=hybrid),
    }


def _run_approach(*, hybrid: bool, digest: str) -> dict[str, Any]:
    definitions = build_default_registry().definitions()
    models = []

    def factory(spec, resolve):
        config = LlmConfig(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, seed=_seed(spec))
        if hybrid:
            model = HybridControllerModel(StructuredIntentExtractor(config))
        else:
            model = OllamaModel(definitions, config, system_prompt=APPROACH_A_SYSTEM_PROMPT)
        models.append((spec.id, model))
        return model

    with create_disposable_environment() as environment:
        results = run_suite(
            _cases(), environment, model_factory=factory, require_response_text=True
        )
    report = build_report(
        results,
        registry=build_default_registry(),
        model_name="hybrid" if hybrid else "free-form",
        software_version=software_version(),
        generated_at=datetime.now(UTC),
        suite_name="phase2-agent-architecture-comparison-v1",
        scenario_set_version="agent-architecture-comparison-v1",
        model_info=_model_info(models, digest=digest, hybrid=hybrid),
    )
    return report


def _comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    def row(report):
        summary = report["summary"]
        metrics = summary["metrics"]
        safety = summary["safety"]
        return {
            "strict_passes": summary["passed"],
            "strict_pass_rate": summary["strict_pass_rate"],
            "task_outcomes": summary["task_outcomes_reached"],
            "task_outcome_rate": summary["task_outcome_rate"],
            "incorrect_tool_calls": metrics["incorrect_tool_calls"],
            "invalid_tool_arguments": metrics["tool_errors"].get("INVALID_ARGUMENT", 0),
            "unrequested_narrowing_filters": metrics["unrequested_narrowing_filters"],
            "unnecessary_questions": metrics["unnecessary_questions"],
            "provider_calls": report["suite"]["model"]["provider_calls"],
            "mean_scenario_latency_ms": summary["latency_ms"]["scenario_total"]["mean"],
            "safety_invariants_hold": safety["all_invariants_hold"],
        }

    return {
        "approach_a_free_form": row(a),
        "approach_b_hybrid": row(b),
        "maintenance_complexity": {
            "approach_a_free_form": (
                "Lower controller code, but workflow behavior and schema adherence depend on the "
                "model across every turn."
            ),
            "approach_b_hybrid": (
                "Additional intent schema and deterministic transition code require more tests, "
                "but tool selection and protected action arguments are explicit."
            ),
        },
        "interpretation": (
            "A small paired development experiment; it does not establish production reliability "
            "or statistical superiority."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare free-form and hybrid VoxVisit agents.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("VOXVISIT_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--url", default=os.getenv("VOXVISIT_LLM_URL", DEFAULT_BASE_URL))
    args = parser.parse_args(argv)
    if args.model != DEFAULT_MODEL or args.url != DEFAULT_BASE_URL:
        parser.error("the frozen comparison requires the documented local Qwen model and URL")
    digest = served_model_digest(LlmConfig(model=args.model, base_url=args.url))
    if digest is None:
        raise unusable(f"the model server has no model {args.model!r}")

    approach_a = _run_approach(hybrid=False, digest=digest)
    approach_b = _run_approach(hybrid=True, digest=digest)
    report = {
        "experiment": {
            "version": "agent-architecture-comparison-v1",
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "model": args.model,
            "model_digest": digest,
            "seeds": list(SEEDS),
            "scenario_count_per_approach": len(_cases()),
            "frozen_holdout_used": False,
        },
        "comparison": _comparison(approach_a, approach_b),
        "approach_a": approach_a,
        "approach_b": approach_b,
    }
    assert_clean(report, [])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
