"""JSONL audit logger with SHA-256 integrity chain.

Every record contains the hash of the previous record's canonical serialization.
Tampering with any record breaks the chain and is detectable on review with
`verify_chain()`.

LIMITATION: the chain is only locally verifiable. A determined attacker with
write access to the log directory can rewrite the entire chain. For evidentiary
use, anchor the chain to an external timestamping service (RFC 3161) or
transparency log. Deferred to Phase 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "GENESIS"
BROKEN_HASH = "BROKEN"


def _canonical(record: dict[str, Any]) -> str:
    """Deterministic JSON for hashing: sorted keys, no whitespace."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode()).hexdigest()


class AuditLogger:
    """Append-only JSONL logger with hash-chained records.

    The log directory is created lazily on first write. Files rotate daily by
    UTC date: `esta-YYYY-MM-DD.jsonl`. On restart, the logger recovers the last
    valid hash by reading the tail of the most recent log file.
    """

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._load_last_hash()

    def _current_log_path(self) -> Path:
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.log_dir / f"esta-{date}.jsonl"

    def _load_last_hash(self) -> str:
        log_files = sorted(self.log_dir.glob("esta-*.jsonl"))
        if not log_files:
            return GENESIS_HASH
        last_file = log_files[-1]
        last_line: bytes | None = None
        with last_file.open("rb") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            return GENESIS_HASH
        try:
            record = json.loads(last_line)
            return record.get("record_hash", GENESIS_HASH)
        except json.JSONDecodeError:
            log.warning("Last audit record in %s is malformed; chain restarting", last_file)
            return BROKEN_HASH

    def write(self, record: dict[str, Any]) -> str:
        """Append a record. Returns the path of the file it was written to."""
        record = dict(record)
        record["prev_hash"] = self._last_hash
        record["record_timestamp"] = datetime.now(UTC).isoformat()
        # Hash is computed over the record minus the hash field itself.
        record["record_hash"] = _record_hash(record)

        log_path = self._current_log_path()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._last_hash = record["record_hash"]
        return str(log_path)

    @property
    def last_hash(self) -> str:
        return self._last_hash


def iter_records(log_dir: Path | str) -> Iterator[dict[str, Any]]:
    """Yield records across all daily log files in chronological order."""
    log_dir = Path(log_dir)
    for path in sorted(log_dir.glob("esta-*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def verify_chain(log_dir: Path | str) -> tuple[bool, str | None]:
    """Verify the integrity chain across all records in log_dir.

    Returns (ok, error_message). On success, error_message is None.
    """
    prev_hash = GENESIS_HASH
    for i, record in enumerate(iter_records(log_dir)):
        stored_hash = record.get("record_hash")
        stored_prev = record.get("prev_hash")
        if stored_prev != prev_hash:
            return False, f"record {i}: prev_hash {stored_prev!r} != expected {prev_hash!r}"
        # Recompute hash by reconstructing the same dict we would have hashed.
        # The hash is over the record dict WITHOUT the record_hash key, so we strip it.
        check_record = {k: v for k, v in record.items() if k != "record_hash"}
        recomputed = _record_hash(check_record)
        if recomputed != stored_hash:
            return False, f"record {i}: record_hash mismatch (stored={stored_hash}, recomputed={recomputed})"
        prev_hash = stored_hash
    return True, None
