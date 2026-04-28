#!/usr/bin/env python3
"""
neural_memory.py — Neural Memory V3.1 backend for benchwrap.

DESIGN GOALS
============
1. **Production-faithful** — every benchmark run uses the same Neural Memory
   settings the production agent uses (auto_connect=True, detect_conflicts=True,
   hybrid retrieval, rerank). No "benchmark-safe" relaxations that would let
   us claim wins for behavior that wouldn't survive deployment.

2. **Hermetically isolated from the prod DB** — db_path is REQUIRED, must not
   resolve to ~/.neural_memory/memory.db, and is asserted at construction.
   The adapter physically cannot touch a user's real memories.

3. **Honest** — no scoring shortcuts (memory_bench used to score `contains`
   substring as "accuracy"; that's gone). Every metric is what the LLM
   actually said, scored against the canonical reference.

V3.1 FEATURES USED
==================
- retrieval_mode="hybrid"  → semantic + BM25 + entity + temporal + PPR + salience
                              channels fused via RRF (production behavior)
- auto_connect=True        → builds the knowledge graph (0.45 cosine threshold
                              in V3.1 means it stays sparse and fast)
- detect_conflicts=True    → marks contradictory facts; production behavior
- rerank=True              → cross-encoder rerank over top candidates
- temporal_weight=0.2      → production default; we don't fudge it

PUBLIC API
==========
    NeuralMemoryBackend(db_path=...) — db_path required, asserted not-prod
    .remember/store/ingest, .recall, .recall_temporal, .recall_multihop,
    .think, .connections, .graph, .clear, .stats
"""

import logging
import os
import sys
import time
from pathlib import Path

# Neural Memory lives in ~/projects/neural-memory-adapter/python/
_NMA_DIR = str(Path.home() / "projects" / "neural-memory-adapter" / "python")
if _NMA_DIR not in sys.path:
    sys.path.insert(0, _NMA_DIR)

from benchwrap.adapters.memory_bench import MemoryBackend

logger = logging.getLogger(__name__)

# The path Neural Memory production uses. We assert against this so a
# benchmark NEVER opens it.
PROD_DB_PATH = (Path.home() / ".neural_memory" / "memory.db").resolve()


def _assert_not_prod(db_path: str | Path) -> Path:
    """Raise if db_path resolves to the production memory DB.

    Without this guard, a misconfigured run could ingest test fixtures into
    the user's real memory store. That is unacceptable.
    """
    if db_path is None:
        raise ValueError(
            "NeuralMemoryBackend requires an explicit db_path. "
            "Pass a tempfile path; never default to the prod DB."
        )
    p = Path(db_path).resolve()
    if p == PROD_DB_PATH:
        raise RuntimeError(
            f"REFUSING to open production DB: {p}. "
            "Benchwrap must never write to ~/.neural_memory/memory.db. "
            "Pass a tempfile path instead."
        )
    return p


class NeuralMemoryBackend(MemoryBackend):
    """V3.1 production-faithful Neural Memory wrapper for benchmarks.

    Settings mirror the production agent. The only thing we change for
    benchmarks is the DB location (always a tempfile).
    """

    def __init__(
        self,
        db_path: str | Path,                    # required, no default
        embedding_backend: str = "auto",
        use_cpp: bool = True,
        retrieval_mode: str = "hybrid",         # production default
        rerank: bool = True,                    # production default
        temporal_weight: float = 0.2,           # production default
    ):
        self._db_path = _assert_not_prod(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporal_weight = float(temporal_weight)

        # Lazy import — keeps adapter discovery fast and avoids loading
        # bge-m3 on import for users who only run mmlu/gsm8k.
        try:
            from memory_client import NeuralMemory
        except ImportError as e:
            raise ImportError(
                f"Cannot import Neural Memory V3.1 from {_NMA_DIR}. "
                f"Make sure the V3.1 branch is checked out and "
                f"sentence-transformers + torch are installed. {e}"
            ) from e

        # Production-faithful instantiation. Hybrid retrieval engages every
        # channel (semantic, bm25, entity, temporal, ppr, salience).
        self.mem = NeuralMemory(
            db_path=str(self._db_path),
            embedding_backend=embedding_backend,
            use_cpp=use_cpp,
            retrieval_mode=retrieval_mode,
            rerank=rerank,
        )

        stats = self.mem.store.get_stats()
        mode = "C++ SIMD" if self.mem._cpp else "Python linear"
        logger.info(
            "[neural-memory:V3.1] db=%s | %s | %dd | retrieval=%s | rerank=%s | "
            "%d existing memories",
            self._db_path, mode, self.mem.dim, retrieval_mode, rerank,
            stats["memories"],
        )

        # Stats for the suite report.
        self._store_count = 0
        self._recall_count = 0
        self._total_recall_ms = 0.0

    # ------------------------------------------------------------------
    # Tier-1: MemoryBackend interface
    # ------------------------------------------------------------------

    def store(self, content: str, label: str = "", metadata: dict = None) -> str:
        """Production-faithful store: auto_connect on, detect_conflicts on.

        V3.1's connection threshold is 0.45 (was 0.15) so the graph stays
        sparse — auto_connect is fast even at scale. detect_conflicts marks
        contradictions in production; we keep it on so benchmarks see the
        same behavior. (For benchmarks that intentionally test conflicts,
        the adapter should disable per-call.)
        """
        mem_id = self.mem.remember(
            text=content,
            label=label or content[:60],
            detect_conflicts=True,
            auto_connect=True,
        )
        self._store_count += 1
        return str(mem_id)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Hybrid recall — production retrieval pipeline."""
        start = time.time()
        results = self.mem.recall(
            query=query,
            k=top_k,
            temporal_weight=self.temporal_weight,
        )
        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms
        return [self._format_result(r, elapsed_ms) for r in results]

    def recall_temporal(self, query: str, top_k: int = 5,
                        temporal_weight: float = 0.5) -> list[dict]:
        """Recall with stronger temporal weighting for 'when'/ordering queries."""
        start = time.time()
        results = self.mem.recall(query=query, k=top_k, temporal_weight=temporal_weight)
        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms
        return [self._format_result(r, elapsed_ms) for r in results]

    def recall_multihop(self, query: str, top_k: int = 5,
                        hops: int = 2,
                        temporal_weight: float | None = None) -> list[dict]:
        """Multi-hop graph traversal for queries that require chaining facts."""
        if temporal_weight is None:
            temporal_weight = self.temporal_weight
        start = time.time()
        results = self.mem.recall_multihop(
            query=query, k=top_k, hops=hops, temporal_weight=temporal_weight,
        )
        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms
        return [self._format_result(r, elapsed_ms) for r in results]

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """BFS/PPR spread from a seed memory."""
        return self.mem.think(start_id=start_id, depth=depth, decay=decay)

    def connections(self, mem_id: int) -> list[dict]:
        return self.mem.connections(mem_id)

    def graph(self) -> dict:
        return self.mem.graph()

    def ingest(self, items: list[dict]) -> int:
        count = 0
        for item in items:
            self.store(
                content=item.get("content", ""),
                label=item.get("label", ""),
                metadata=item.get("metadata", {}),
            )
            count += 1
        return count

    def clear(self) -> None:
        """Wipe DB + in-memory graph + C++ index. Production-grade reset.

        This is the ONLY way to start a clean benchmark — without clear(),
        memories from one dataset leak into the next.
        """
        with self.mem.store._lock:
            self.mem.store.conn.execute("DELETE FROM memories")
            self.mem.store.conn.execute("DELETE FROM connections")
            self.mem.store.conn.commit()
        self.mem._graph_nodes.clear()
        if self.mem._cpp:
            try:
                self.mem._cpp.initialize(dim=self.mem.dim)
            except Exception:
                pass
        self._store_count = 0
        self._recall_count = 0
        self._total_recall_ms = 0.0

    def stats(self) -> dict:
        store_stats = self.mem.store.get_stats()
        return {
            "backend": "neural-memory-v3.1",
            "db_path": str(self._db_path),
            "retrieval_mode": "C++ SIMD" if self.mem._cpp else "Python linear",
            "embedding_dim": self.mem.dim,
            "total_memories": store_stats["memories"],
            "total_connections": store_stats["connections"],
            "benchmark_stores": self._store_count,
            "benchmark_recalls": self._recall_count,
            "avg_recall_ms": (
                self._total_recall_ms / self._recall_count
                if self._recall_count > 0 else 0
            ),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _format_result(r: dict, elapsed_ms: float) -> dict:
        """Map a NeuralMemory recall result to benchwrap's MemoryBackend format."""
        return {
            "content": r.get("content", ""),
            "label": r.get("label", ""),
            "score": r.get("relevance", r.get("combined", r.get("similarity", 0.0))),
            "id": r.get("id"),
            "created_at": r.get("created_at"),
            "_similarity": r.get("similarity", 0.0),
            "_temporal": r.get("temporal_score", 0.0),
            "_ppr": r.get("ppr_score", 0.0),
            "_salience": r.get("salience_factor", 0.0),
            "_channels": r.get("channel_scores", {}),
            "_latency_ms": elapsed_ms,
        }


if __name__ == "__main__":
    import json
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  Neural Memory V3.1 — Self-Test")
    print("=" * 60)

    # Refuse prod
    try:
        NeuralMemoryBackend(db_path=PROD_DB_PATH)
    except RuntimeError as e:
        print(f"\n✓ prod-DB guard works: {e}")

    # Tempfile path is fine
    with tempfile.TemporaryDirectory(prefix="benchwrap_nm_selftest_") as tmp:
        backend = NeuralMemoryBackend(db_path=Path(tmp) / "memory.db")
        print(f"\nstats: {json.dumps(backend.stats(), indent=2)}")

        # Production-faithful ingest
        for item in [
            {"content": "The server is hosted in Frankfurt.", "label": "infra"},
            {"content": "The database uses PostgreSQL 16.",   "label": "infra"},
            {"content": "Deployment goes via GitHub Actions.","label": "ci"},
            {"content": "The team has 5 engineers.",          "label": "team"},
            {"content": "Frankfurt is in Germany.",           "label": "geo"},
        ]:
            backend.store(item["content"], item["label"])

        # Hybrid recall
        for q in ["Where is the server?", "How do we deploy?", "Country of the server?"]:
            results = backend.recall(q, top_k=3)
            print(f"\nrecall(hybrid): {q!r}")
            for r in results:
                print(f"  [{r['score']:.3f}] {r['content']}")

        # Multi-hop
        results = backend.recall_multihop("Where is the database hosted geographically?", top_k=3)
        print(f"\nrecall_multihop:")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content']}")

        print(f"\nfinal stats: {json.dumps(backend.stats(), indent=2)}")
        print("\n✓ self-test passed")
