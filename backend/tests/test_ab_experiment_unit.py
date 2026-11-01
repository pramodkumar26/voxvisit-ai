import json
from pathlib import Path

import pytest

from app.evaluation.ab_scenarios import AB_SCENARIOS
from app.evaluation.hybrid import PatientIntent, StructuredIntentExtractor
from app.evaluation.llm import APPROACH_A_SYSTEM_PROMPT, LlmConfig
from app.evaluation.model import MalformedModelResponse
from app.evaluation.report import METRIC_FAMILIES
from app.evaluation.runner import Metrics

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "backend" / "tests" / "agent-architecture-comparison-v1.json"


class CannedIntentProvider:
    def __init__(self, content: str):
        self.content = content
        self.payload = None

    def __call__(self, path, payload, timeout_s):
        self.payload = payload
        return {
            "message": {"role": "assistant", "content": self.content},
            "prompt_eval_count": 20,
            "eval_count": 10,
            "total_duration": 1_000_000,
        }


def test_the_comparison_manifest_matches_the_fresh_scenarios() -> None:
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["status"] == "FROZEN_BEFORE_EXECUTION"
    assert manifest["seeds"] == [7, 19]
    assert [item["id"] for item in manifest["scenarios"]] == [s.id for s in AB_SCENARIOS]
    assert [item["request"] for item in manifest["scenarios"]] == [s.request for s in AB_SCENARIOS]


def test_metric_families_partition_every_flat_metric_once() -> None:
    keys = [key for group in METRIC_FAMILIES.values() for key in group]
    expected = set(Metrics().as_dict())

    assert len(keys) == len(set(keys))
    assert set(keys) == expected


def test_approach_a_prompt_has_operation_specific_routing_and_progression() -> None:
    assert "For booking, resolve a named clinic" in APPROACH_A_SYSTEM_PROMPT
    assert "For cancellation or rescheduling, first use get_my_appointments" in (
        APPROACH_A_SYSTEM_PROMPT
    )
    assert "Availability results are ordered earliest first" in APPROACH_A_SYSTEM_PROMPT


def test_structured_extraction_uses_qwen_json_schema_and_preserves_stated_values() -> None:
    intent = PatientIntent(
        operation="BOOK",
        clinic_name="Pine Clinic",
        start_date="2026-11-10",
        end_date="2026-11-10",
        selection="EARLIEST",
    )
    provider = CannedIntentProvider(intent.model_dump_json())
    extractor = StructuredIntentExtractor(LlmConfig(), transport=provider)

    extracted = extractor.extract("Book the earliest visit at Pine Clinic on 2026-11-10.")

    assert extracted == intent
    assert provider.payload["model"] == "qwen2.5:7b"
    assert provider.payload["format"] == PatientIntent.model_json_schema()
    assert "tools" not in provider.payload
    assert extractor.served_replies == extractor.server_attested_replies == 1


def test_structured_extraction_rejects_an_unstated_preference() -> None:
    intent = PatientIntent(
        operation="BOOK",
        clinic_name="Invented Clinic",
        start_date="2026-11-10",
        selection="EARLIEST",
    )
    extractor = StructuredIntentExtractor(
        LlmConfig(), transport=CannedIntentProvider(intent.model_dump_json())
    )

    with pytest.raises(MalformedModelResponse, match="structured intent"):
        extractor.extract("Book the earliest visit at Pine Clinic on 2026-11-10.")
