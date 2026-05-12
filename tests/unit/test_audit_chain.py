"""Tests for the hash-chained audit logger.

Covers: genesis case, normal append, multi-record chain integrity, tamper
detection, and restart-on-corrupt-tail. Pure stdlib — no model load required.
"""

from __future__ import annotations

import json
from pathlib import Path

from esta.audit import GENESIS_HASH, AuditLogger
from esta.audit.logger import BROKEN_HASH, iter_records, verify_chain


def _records(log_dir: Path) -> list[dict]:
    return list(iter_records(log_dir))


def test_genesis_hash_on_empty_dir(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    assert logger.last_hash == GENESIS_HASH


def test_single_append_produces_valid_chain(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    logger.write({"request_id": "r1", "prompt": "hi", "response": "hello"})

    records = _records(tmp_path)
    assert len(records) == 1
    assert records[0]["prev_hash"] == GENESIS_HASH
    assert "record_hash" in records[0]
    assert "record_timestamp" in records[0]

    ok, err = verify_chain(tmp_path)
    assert ok, err


def test_multi_record_chain_links_correctly(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    for i in range(5):
        logger.write({"request_id": f"r{i}", "payload": i})

    records = _records(tmp_path)
    assert len(records) == 5
    # First record links to GENESIS; each subsequent links to the previous record_hash.
    assert records[0]["prev_hash"] == GENESIS_HASH
    for prev, curr in zip(records, records[1:], strict=False):
        assert curr["prev_hash"] == prev["record_hash"]

    ok, err = verify_chain(tmp_path)
    assert ok, err


def test_tamper_with_payload_is_detected(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    logger.write({"request_id": "r0", "response": "the truth"})
    logger.write({"request_id": "r1", "response": "also true"})
    logger.write({"request_id": "r2", "response": "and this"})

    log_file = next(tmp_path.glob("esta-*.jsonl"))
    lines = log_file.read_text(encoding="utf-8").splitlines()
    # Tamper with the middle record's payload, keep record_hash unchanged.
    tampered = json.loads(lines[1])
    tampered["response"] = "ATTACKER REWRITE"
    lines[1] = json.dumps(tampered)
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, err = verify_chain(tmp_path)
    assert not ok
    assert err is not None
    assert "record 1" in err


def test_tamper_with_record_hash_is_detected(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    logger.write({"request_id": "r0", "response": "x"})

    log_file = next(tmp_path.glob("esta-*.jsonl"))
    record = json.loads(log_file.read_text(encoding="utf-8"))
    record["record_hash"] = "0" * 64
    log_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    ok, err = verify_chain(tmp_path)
    assert not ok
    assert err is not None


def test_logger_restarts_on_corrupt_tail(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    logger.write({"request_id": "r0", "response": "ok"})

    # Append a malformed line. Simulates a crash mid-write.
    log_file = next(tmp_path.glob("esta-*.jsonl"))
    with log_file.open("a", encoding="utf-8") as f:
        f.write('{"truncated":\n')

    # New logger initialized against corrupt tail: should not crash.
    new_logger = AuditLogger(tmp_path)
    assert new_logger.last_hash == BROKEN_HASH


def test_logger_recovers_last_hash_after_restart(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    logger.write({"request_id": "r0"})
    logger.write({"request_id": "r1"})
    expected = logger.last_hash

    new_logger = AuditLogger(tmp_path)
    assert new_logger.last_hash == expected

    # New write should chain off the recovered hash.
    new_logger.write({"request_id": "r2"})
    records = _records(tmp_path)
    assert records[-1]["prev_hash"] == expected
    ok, err = verify_chain(tmp_path)
    assert ok, err


def test_log_dir_created_on_init(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "audit"
    assert not nested.exists()
    AuditLogger(nested)
    assert nested.is_dir()


def test_write_returns_log_file_path(tmp_path: Path) -> None:
    logger = AuditLogger(tmp_path)
    returned = logger.write({"request_id": "r0"})
    actual = next(tmp_path.glob("esta-*.jsonl"))
    assert Path(returned) == actual
