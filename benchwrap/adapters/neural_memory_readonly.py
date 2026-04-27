#!/usr/bin/env python3
"""
neural_memory_readonly.py — Read-only Neural Memory backend for benchwrap.

Wraps the production Neural Memory DB so benchmarks can query LIVE memories
without ever writing to the real database. Used by the suite's third mode:
"prod-readonly" — measures how the LLM does when retrieved context is whatever
production happens to surface for benchmark questions.

SAFETY MODEL — how we guarantee zero writes to the prod DB:

  At construction we use sqlite3's online backup API to snapshot the prod
  database into a tempfile under /tmp/. Neural Memory then opens THAT temp
  copy. Schema migrations, WAL commits, and any other writes Neural Memory
  performs go to the copy — the original DB at PROD_DB_PATH is never opened
  for writing by this process.

  store(), ingest(), and clear() are explicit no-ops (logged at WARNING) so
  benchmark harnesses that try to seed fixtures don't silently corrupt the
  copy either. Only recall() is allowed.

  The snapshot is deleted in __del__ / close().

USAGE:
    from benchwrap.adapters.neural_memory_readonly import (
        ReadOnlyNeuralMemoryBackend, PROD_DB_PATH,
    )
    backend = ReadOnlyNeuralMemoryBackend()              # snapshots prod
    results = backend.recall("where is the server?", top_k=5)
"""

import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

# Ensure neural-memory-adapter is importable
_NMA_DIR = str(Path.home() / "projects" / "neural-memory-adapter" / "python")
if _NMA_DIR not in sys.path:
    sys.path.insert(0, _NMA_DIR)

from benchwrap.adapters.memory_bench import MemoryBackend

logger = logging.getLogger(__name__)

PROD_DB_PATH = Path.home() / ".neural_memory" / "memory.db"


class ReadOnlyNeuralMemoryBackend(MemoryBackend):
    """A NeuralMemoryBackend that mirrors the production DB read-only.

    The mirror is a snapshot copy — writes by Neural Memory (schema ensure,
    WAL checkpoints, etc.) hit the copy, not the original. Benchmark-side
    writes (store/ingest/clear) are dropped with a warning.
    """

    def __init__(
        self,
        prod_db_path: str | Path = PROD_DB_PATH,
        embedding_backend: str = "auto",
        use_cpp: bool = True,
    ):
        self.prod_db_path = Path(prod_db_path)
        if not self.prod_db_path.exists():
            raise FileNotFoundError(
                f"Production Neural Memory DB not found at {self.prod_db_path}. "
                f"Run the system at least once to create it."
            )

        self._snapshot_dir = tempfile.mkdtemp(prefix="benchwrap_nm_ro_")
        self._snapshot_path = os.path.join(self._snapshot_dir, "memory.db")
        self._snapshot_prod_db()

        # Hard-assert the path Neural Memory will open is NOT the prod DB.
        assert Path(self._snapshot_path).resolve() != self.prod_db_path.resolve(), (
            "snapshot path equals prod path — refusing to open"
        )

        from memory_client import NeuralMemory

        self.mem = NeuralMemory(
            db_path=self._snapshot_path,
            embedding_backend=embedding_backend,
            use_cpp=use_cpp,
        )

        prod_stats = self.mem.store.get_stats()
        mode = "C++ SIMD" if self.mem._cpp else "Python linear"
        logger.info(
            "[neural-memory:RO] Mirrored prod DB → %s (%d memories, %d connections, %s)",
            self._snapshot_path,
            prod_stats["memories"],
            prod_stats["connections"],
            mode,
        )

        self._recall_count = 0
        self._total_recall_ms = 0.0
        self._dropped_writes = 0

    def _snapshot_prod_db(self) -> None:
        """Copy prod DB to the snapshot path using the online backup API.

        Using sqlite3.Connection.backup() avoids torn reads if the prod DB is
        actively being written by another process (e.g. a running agent).
        """
        src = sqlite3.connect(f"file:{self.prod_db_path}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(self._snapshot_path)
            try:
                with dst:
                    src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()

    # ------------------------------------------------------------------
    # MemoryBackend interface — recall is real, writes are silent no-ops
    # ------------------------------------------------------------------

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        start = time.time()
        results = self.mem.recall(query=query, k=top_k, temporal_weight=0.0)
        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms

        return [
            {
                "content": r.get("content", ""),
                "label": r.get("label", ""),
                "score": r.get("similarity", 0.0),
                "id": r.get("id"),
                "_similarity": r.get("similarity", 0.0),
                "_combined": r.get("combined", 0.0),
                "_latency_ms": elapsed_ms,
            }
            for r in results
        ]

    def store(self, content: str, label: str = "", metadata: dict = None) -> str:
        self._dropped_writes += 1
        logger.warning(
            "[neural-memory:RO] store() called in read-only mode — DROPPED "
            "(content=%r, total_dropped=%d)",
            content[:60],
            self._dropped_writes,
        )
        return ""

    def ingest(self, items: list[dict]) -> int:
        self._dropped_writes += len(items)
        logger.warning(
            "[neural-memory:RO] ingest() of %d items dropped — read-only mode",
            len(items),
        )
        return 0

    def clear(self) -> None:
        logger.warning("[neural-memory:RO] clear() called — DROPPED, prod DB is read-only")

    def stats(self) -> dict:
        store_stats = self.mem.store.get_stats()
        return {
            "backend": "neural-memory-readonly",
            "prod_db_path": str(self.prod_db_path),
            "snapshot_path": self._snapshot_path,
            "retrieval_mode": "C++ SIMD" if self.mem._cpp else "Python linear",
            "embedding_dim": self.mem.dim,
            "total_memories": store_stats["memories"],
            "total_connections": store_stats["connections"],
            "benchmark_recalls": self._recall_count,
            "dropped_writes": self._dropped_writes,
            "avg_recall_ms": (
                self._total_recall_ms / self._recall_count
                if self._recall_count > 0
                else 0
            ),
        }

    def close(self) -> None:
        if self._snapshot_dir and os.path.exists(self._snapshot_dir):
            try:
                shutil.rmtree(self._snapshot_dir)
            except Exception:
                pass
            self._snapshot_dir = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  ReadOnlyNeuralMemoryBackend — Self-Test")
    print("=" * 60)

    backend = ReadOnlyNeuralMemoryBackend()
    print(f"\nStats: {backend.stats()}")

    # Confirm writes are dropped
    backend.store("THIS SHOULD NEVER REACH PROD", label="canary")
    assert backend.stats()["dropped_writes"] == 1

    # Real recall against prod data
    for q in ["server", "deployment", "team"]:
        results = backend.recall(q, top_k=3)
        print(f"\nQuery: {q!r}")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:80]}")

    # Verify prod DB mtime didn't change
    mtime_before = PROD_DB_PATH.stat().st_mtime
    backend.recall("anything", top_k=1)
    mtime_after = PROD_DB_PATH.stat().st_mtime
    assert mtime_before == mtime_after, "Prod DB was modified — SAFETY VIOLATION"
    print("\n✓ Prod DB mtime unchanged — read-only guarantee holds")

    print(f"\nFinal stats: {backend.stats()}")
    backend.close()
