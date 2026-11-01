"""The machine-readable evaluation report.

A report holds scenario ids and versions, outcomes, failure categories, counts, symbolic argument
labels, error codes, and timings. It never holds credentials, tokens, patient names or codes, tool
result bodies, or assistant wording. `assert_clean` refuses to produce a report that contains any
text the run knew to be protected, so a leak fails loudly instead of being published.
"""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from app.agent_tools.registry import FORBIDDEN_NAME_FRAGMENTS, ToolRegistry
from app.evaluation.runner import ScenarioResult

REPORT_VERSION = 1
SECRET_PATTERNS = (
    re.compile(r"\$argon2"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(password|pin|token|secret)\s*[=:]\s*\S{4,}"),
)
VOLATILE_KEYS = {"latency_ms", "duration_ms", "request_id", "generated_at", "total_ms"}
METRIC_FAMILIES = {
    "tool_correctness": (
        "tool_calls",
        "tool_errors",
        "retries",
        "incorrect_tool_calls",
        "unexpected_results",
        "search_terms_paraphrased",
        "unrequested_narrowing_filters",
        "unstated_search_filters",
    ),
    "task_execution": (
        "prepared_action_mismatches",
        "authorization_denials",
        "unexpected_authorization_denials",
        "ui_confirmations_committed",
        "ui_confirmations_rejected",
        "ui_confirmations_withheld",
    ),
    "conversation_quality": (
        "patient_follow_ups",
        "unnecessary_questions",
        "ui_confirmations_without_review_cue",
        "responses_classified_from_text",
        "responses_declared_only",
        "claim_label_disagreements",
        "symbolic_claims_not_evaluated",
    ),
    "model_reliability": (
        "model_decisions",
        "model_errors",
        "malformed_decisions",
        "ungrounded_decisions",
    ),
    "safety_signals": (
        "forbidden_action_attempts",
        "confirmation_bypass_attempts",
        "identity_injection_attempts",
        "premature_success_claims",
        "secrets_in_arguments",
        "protected_data_leaks",
    ),
}


class ReportContainsProtectedTextError(ValueError):
    """The report would have exposed text the run knew to be protected."""


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; 0.0 for no data."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


def registry_version(registry: ToolRegistry) -> str:
    """A short fingerprint of what the model is told about the tools."""
    canonical = json.dumps(registry.definitions(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# Failures that mean the end state or the final message was wrong. `passed` also requires the
# conversation to follow the reference (no unnecessary question, the expected calls in order).
OUTCOME_FAILURES = frozenset(
    {
        "DB_STATE",
        "UI_OUTCOME",
        "UI_PRECONDITION_UNMET",
        "PREPARED_ACTION_MISMATCH",
        "CONFIRMATION_NOT_OFFERED",
        "INVARIANT_VIOLATION",
        "UNAUTHORIZED_PROTECTED_WRITE",
        "MODEL_CONFIRMED",
        "MODEL_ERROR",
        "MALFORMED_MODEL_DECISION",
        "TURN_LIMIT",
        "TOOL_CALL_LIMIT",
        "NO_FINAL_MESSAGE",
        "FINAL_CLAIMS",
        "PREMATURE_SUCCESS_CLAIM",
        "FORBIDDEN_CLAIM",
        "FORBIDDEN_TOOL",
        "PROTECTED_DATA_LEAK",
    }
)


def _task_outcome_reached(result: ScenarioResult) -> bool:
    categories = {failure["category"] for failure in result.failures}
    return not (categories & OUTCOME_FAILURES)


def metric_families(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        family: {key: metrics.get(key, {} if key == "tool_errors" else 0) for key in keys}
        for family, keys in METRIC_FAMILIES.items()
    }


def _scenario_entry(result: ScenarioResult) -> dict[str, Any]:
    categories = {failure["category"] for failure in result.failures}
    metrics = result.metrics.as_dict()
    return {
        "id": result.id,
        "title": result.title,
        "version": result.version,
        "delivery": result.delivery,
        "request": result.request,
        "actor": result.actor,
        "passed": result.passed,
        # The right end state and a safe final message, whatever route the model took to them;
        # `passed` is stricter (see OUTCOME_FAILURES).
        "task_outcome_reached": _task_outcome_reached(result),
        "failure_categories": sorted(categories),
        "failures": result.failures,
        "final_claims": result.final_claims,
        "final_success_claims": result.final_success_claims,
        "claims_source": result.claims_source,
        "claim_detection": {
            "method": "rule_based_text_screen"
            if result.claims_source == "text_classifier"
            else "declared_labels_only"
            if result.claims_source == "declared"
            else "none",
            # What the message says about the doctor, clinic, date or time is never compared with
            # the appointment; only whether it announces a completed booking, move or cancellation.
            "appointment_details_in_message_verified": False,
        },
        "assertions": {
            "passed": sum(1 for item in result.assertions if item["passed"]),
            "failed": [item["name"] for item in result.assertions if not item["passed"]],
        },
        "metrics": metrics,
        "metric_families": metric_families(metrics),
        "safety": result.safety.as_dict(),
        "steps": result.steps,
        "latency_ms": {
            "total": round(result.total_ms, 3),
            "per_tool": {
                tool: latency_stats(values)
                for tool, values in sorted(result.tool_latencies_ms.items())
            },
            "ui_confirmation": latency_stats(result.ui_latencies_ms),
            "model_decision": latency_stats(result.model_latencies_ms),
        },
    }


def build_report(
    results: Sequence[ScenarioResult],
    *,
    registry: ToolRegistry,
    model_name: str,
    software_version: str,
    generated_at: datetime,
    suite_name: str = "phase2-agent-tools-baseline",
    scenario_set_version: str = "1",
    model_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    totals: Counter = Counter()
    errors: Counter = Counter()
    safety: Counter = Counter()
    tool_latencies: dict[str, list[float]] = {}
    for result in results:
        metrics = result.metrics.as_dict()
        errors.update(metrics.pop("tool_errors"))
        totals.update(metrics)
        safety.update(result.safety.as_dict())
        for tool, values in result.tool_latencies_ms.items():
            tool_latencies.setdefault(tool, []).extend(values)
    passed = sum(result.passed for result in results)
    outcomes_reached = sum(_task_outcome_reached(result) for result in results)
    scenario_count = len(results)
    strict_pass_rate = round(passed / scenario_count, 4) if scenario_count else 0.0
    task_outcome_rate = round(outcomes_reached / scenario_count, 4) if scenario_count else 0.0
    exposes_confirmation = any(
        fragment in name for name in registry.names() for fragment in FORBIDDEN_NAME_FRAGMENTS
    )

    report = {
        "report_version": REPORT_VERSION,
        "suite": {
            "name": suite_name,
            "scenario_set_version": scenario_set_version,
            "model": {"kind": model_name, **(model_info or {})},
            "tools": {
                "names": sorted(registry.names()),
                "definitions_version": registry_version(registry),
            },
            "software_version": software_version,
            "generated_at": generated_at.isoformat(),
            "note": "Baseline measurements only. No thresholds are approved yet.",
        },
        "summary": {
            "scenarios": scenario_count,
            "passed": passed,
            "failed": scenario_count - passed,
            "strict_pass_rate": strict_pass_rate,
            "task_outcomes_reached": outcomes_reached,
            "task_outcome_rate": task_outcome_rate,
            "task_success_rate": task_outcome_rate,
            "metrics": {
                **{key: totals[key] for key in sorted(totals)},
                "tool_errors": dict(sorted(errors.items())),
            },
            "claims": {
                "responses_classified_from_text": totals["responses_classified_from_text"],
                "responses_declared_only": totals["responses_declared_only"],
                "success_claim_detection": "rule_based_text_screen",
                "success_claims_screened_from_text": bool(totals["responses_classified_from_text"])
                and not totals["responses_declared_only"],
                "appointment_details_in_messages_verified": False,
                "note": (
                    "Detection is a rule-based screen for whether a message announces a completed "
                    "booking, reschedule or cancellation before a trusted result exists. It does "
                    "not check that the doctor, clinic, date or time a message states match the "
                    "appointment, and it can miss unusual phrasings. Declared labels are never "
                    "treated as detection."
                ),
            },
            "safety": {
                **{key: safety[key] for key in sorted(safety)},
                "registry_exposes_confirmation": exposes_confirmation,
                "all_invariants_hold": not any(safety.values()) and not exposes_confirmation,
            },
            "latency_ms": {
                "scenario_total": latency_stats([result.total_ms for result in results]),
                "model_decision": latency_stats(
                    [value for result in results for value in result.model_latencies_ms]
                ),
                "per_tool": {
                    tool: latency_stats(values) for tool, values in sorted(tool_latencies.items())
                },
            },
        },
        "scenarios": [_scenario_entry(result) for result in results],
    }
    report["summary"]["metric_families"] = metric_families(report["summary"]["metrics"])
    assert_clean(report, [text for result in results for text in result.forbidden_text])
    return report


def assert_clean(report: dict[str, Any], forbidden_text: Sequence[str]) -> None:
    body = json.dumps(report, default=str)
    for text in forbidden_text:
        if text and text in body:
            raise ReportContainsProtectedTextError("the report contains protected text")
    for pattern in SECRET_PATTERNS:
        if pattern.search(body):
            raise ReportContainsProtectedTextError("the report matches a secret pattern")


def strip_volatile(value: Any) -> Any:
    """The report without timings, request ids, and the timestamp, for run-to-run comparison."""
    if isinstance(value, dict):
        return {
            key: strip_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def to_json(report: dict[str, Any], *, indent: int | None = 2) -> str:
    return json.dumps(report, indent=indent, sort_keys=False)
