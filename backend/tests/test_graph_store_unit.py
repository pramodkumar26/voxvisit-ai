"""LG-5B: properties of the checkpoint store that need no database."""

import dataclasses
import pickle
import uuid

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.evaluation.graph_store import (
    CheckpointStore,
    CheckpointStoreError,
    _conninfo,
    owner_of,
    strict_serializer,
    thread_for,
)


@dataclasses.dataclass
class Foreign:
    value: int = 1


def test_plain_conversation_state_round_trips() -> None:
    serde = strict_serializer()
    state = {
        "request": {"clinic_name": "Pine", "start_date": "2026-11-09", "extra": [1, None, "x"]},
        "dialogue": {"awaiting": "TIME", "offered": [{"label": "9:00 AM", "key": "s1"}]},
        "review": None,
        "turn": 3,
    }

    assert serde.loads_typed(serde.dumps_typed(state)) == state


def test_a_stored_object_of_a_foreign_type_is_not_rebuilt() -> None:
    kind, data = JsonPlusSerializer().dumps_typed(Foreign(3))  # What an attacker could write.

    rebuilt = strict_serializer().loads_typed((kind, data))

    assert not isinstance(rebuilt, Foreign)  # It stays plain data; nothing was imported or run.


def test_a_pickled_row_is_refused_outright() -> None:
    with pytest.raises(NotImplementedError):
        strict_serializer().loads_typed(("pickle", pickle.dumps({"a": 1})))


def test_the_thread_and_owner_are_opaque_stable_and_distinct() -> None:
    session, patient = uuid.uuid4(), uuid.uuid4()

    assert thread_for(session) == thread_for(session)
    assert thread_for(session) != thread_for(uuid.uuid4())
    assert str(session) not in thread_for(session) and str(patient) not in owner_of(patient)
    assert owner_of(patient) != thread_for(patient)  # Different purposes, different digests.
    assert len(thread_for(session)) == len(owner_of(patient)) == 32


def test_the_connection_string_drops_the_driver_and_keeps_the_credentials_for_the_driver() -> None:
    info = _conninfo("postgresql+psycopg://user:s3cret@db.example:5432/app")

    assert info.startswith("postgresql://user:s3cret@db.example:5432/app")


def test_an_unreachable_database_fails_closed_without_leaking_the_password() -> None:
    store = CheckpointStore("postgresql+psycopg://user:topsecret@127.0.0.1:1/none")

    with pytest.raises(CheckpointStoreError) as raised:
        store.open()

    assert "topsecret" not in str(raised.value) and "topsecret" not in repr(raised.value)
    assert not store.healthy()
    store.close()
