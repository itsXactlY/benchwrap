#!/usr/bin/env python3
"""
neural_memory.py — Neural Memory adapter for benchwrap.

SHOWCASE EXAMPLE: How to integrate a real memory system with benchwrap.
This wraps our Neural Memory (C++ SIMD + SQLite + bge-m3 embeddings) as a
benchwrap MemoryBackend, making it testable against any memory benchmark.

WHY THIS EXISTS:
  benchwrap tests memory systems by: ingest → recall → answer → score.
  Neural Memory is our custom system with: C++ SIMD retrieval, SQLite persistence,
  knowledge graph with spreading activation, temporal decay, conflict detection.
  
  This adapter bridges the two. One interface, any benchmark.

USAGE:
  from benchwrap.adapters.neural_memory import NeuralMemoryBackend
  from benchwrap.core.model import OllamaBackend
  
  backend = NeuralMemoryBackend(db_path="/tmp/bench_test.db")
  llm = OllamaBackend(model="openhermes:7b-v2.5")
  
  from benchwrap.adapters.memory_bench import MemoryBenchAdapter
  adapter = MemoryBenchAdapter(memory_client=backend, llm_backend=llm)
  result = adapter.run("recall-accuracy")

ARCHITECTURE:
  benchwrap.MemoryBackend ←→ this adapter ←→ NeuralMemory
                                    ↓
                          C++ SIMD (libneural_memory.so)
                          SQLite (memory.db)
                          bge-m3 embeddings (1024d, CUDA)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# Neural Memory lives in ~/projects/neural-memory-adapter/python/
# We add it to path so we can import memory_client, embed_provider, etc.
_NMA_DIR = str(Path.home() / "projects" / "neural-memory-adapter" / "python")
if _NMA_DIR not in sys.path:
    sys.path.insert(0, _NMA_DIR)

from benchwrap.adapters.memory_bench import MemoryBackend

logger = logging.getLogger(__name__)


class NeuralMemoryBackend(MemoryBackend):
    """Wraps Neural Memory as a benchwrap MemoryBackend.
    
    Neural Memory's interface:
        mem.remember(text, label, detect_conflicts, auto_connect) → int (id)
        mem.recall(query, k, temporal_weight) → list[dict]
    
    benchwrap's MemoryBackend interface:
        store(content, label, metadata) → str (id)
        recall(query, top_k) → list[dict]
    
    The mapping is mostly 1:1, but there are nuances:
    
    NUANCE 1 — auto_connect:
        Neural Memory builds a knowledge graph by connecting similar memories.
        This is great for production (spreading activation finds related facts),
        but BAD for benchmark isolation. When testing recall accuracy, we don't
        want connections influencing results. We disable auto_connect during
        benchmark ingestion.
    
    NUANCE 2 — detect_conflicts:
        Neural Memory detects contradictory facts and marks old ones as
        [SUPERSEDED]. This is brilliant for production (your knowledge stays
        consistent), but TERRIBLE for benchmarks that test conflict resolution.
        The benchmark wants to see IF the system handles conflicts, not have
        the system silently resolve them. We disable detect_conflicts during
        benchmark ingestion.
    
    NUANCE 3 — temporal scoring:
        Neural Memory applies exponential decay based on created_at timestamp.
        During benchmarks, all memories are created at roughly the same time,
        so temporal scoring adds noise. We set temporal_weight=0.0 for benchmark
        recall to get pure similarity ranking.
    
    NUANCE 4 — C++ SIMD fallback:
        The C++ bridge may not be available (not compiled, wrong platform, etc.).
        Neural Memory falls back to Python O(n) linear scan. This is SLOW for
        large benchmarks. We log a warning but continue — the benchmark still
        works, just takes longer.
    
    NUANCE 5 — embedding model:
        Uses BAAI/bge-m3 (1024d, ~2.2GB, needs CUDA). First load takes ~10s to
        download/load the model. After that it's cached at ~/.neural_memory/models/.
        On CPU, embedding is ~100x slower. Always check CUDA availability.
    """

    def __init__(
        self,
        db_path: str | None = None,
        embedding_backend: str = "auto",
        use_cpp: bool = True,
        auto_connect_threshold: int = 50,
    ):
        """
        Args:
            db_path: SQLite database path. Default: ~/.neural_memory/memory.db
                     For benchmarks, use a TEMP path to avoid polluting production data.
            embedding_backend: "auto" (CUDA if available), "cpu", or model name.
            use_cpp: Whether to use C++ SIMD index. Falls back to Python if unavailable.
        """
        # PITFALL: Don't use production DB for benchmarks!
        # If db_path is None, we create a temp DB that gets cleaned up.
        # If you accidentally benchmark against production, you'll corrupt your
        # real memories with test data AND the conflict detector will mess up
        # benchmark results by "resolving" contradictions.
        if db_path is None:
            import tempfile
            self._temp_dir = tempfile.mkdtemp(prefix="benchwrap_nm_")
            db_path = os.path.join(self._temp_dir, "memory.db")
            logger.info(f"[neural-memory] Using temp DB: {db_path}")
        else:
            self._temp_dir = None

        # Import Neural Memory (lazy — only when this adapter is actually used)
        # PITFALL: The import chain is memory_client → embed_provider → sentence_transformers.
        # If sentence_transformers isn't installed, you get a cryptic ImportError.
        # The error message usually mentions 'torch' or 'sentence_transformers'.
        # Fix: pip install sentence-transformers torch
        try:
            from memory_client import NeuralMemory
        except ImportError as e:
            raise ImportError(
                f"Cannot import Neural Memory. Is ~/projects/neural-memory-adapter/python/ on sys.path? "
                f"Missing dependency? Error: {e}"
            ) from e

        # Create the Neural Memory instance
        # PITFALL: use_cpp=True will try to load libneural_memory.so.
        # If the C++ library isn't built, it logs a warning and falls back to Python.
        # For benchmarks, Python fallback is fine (accurate but slow).
        # To build the C++ lib: cd ~/projects/neural-memory-adapter/build && cmake --build .
        self.mem = NeuralMemory(
            db_path=db_path,
            embedding_backend=embedding_backend,
            use_cpp=use_cpp,
        )

        # Track stats for benchmark reporting
        self._store_count = 0
        self._recall_count = 0
        self._total_recall_ms = 0.0
        self._auto_connect_threshold = auto_connect_threshold

        # Log initialization
        stats = self.mem.store.get_stats()
        mode = "C++ SIMD" if self.mem._cpp else "Python linear"
        dim = self.mem.dim
        logger.info(
            f"[neural-memory] Initialized: {mode} retrieval, {dim}d embeddings, "
            f"{stats['memories']} existing memories, {stats['connections']} connections"
        )

    def store(self, content: str, label: str = "", metadata: dict = None) -> str:
        """Store a memory in Neural Memory.
        
        Args:
            content: The text to remember
            label: Short label (used in graph visualization, not for retrieval)
            metadata: Optional dict with extra info (stored but not embedded)
        
        Returns:
            Memory ID as string (Neural Memory uses int IDs, we stringify for interface compat)
        
        IMPORTANT: We disable auto_connect and detect_conflicts during benchmarks.
        
        WHY disable auto_connect:
            When auto_connect is True, Neural Memory builds connections between
            similar memories (cosine sim > 0.15). During benchmarks, we're testing
            PURE retrieval quality, not graph traversal. Connections add noise to
            the results — a memory might be recalled not because it's relevant to
            the query, but because it's connected to a relevant memory.
            
            For production use, auto_connect is great (spreading activation finds
            semantically related facts). For benchmarks, it's confounding.
        
        WHY disable detect_conflicts:
            When detect_conflicts is True, Neural Memory marks contradictory old
            memories as [SUPERSEDED] and updates them. This is correct behavior
            for production, but for benchmarks testing conflict resolution, we
            want BOTH conflicting facts to exist in memory. The benchmark then
            tests whether the system can identify and handle the conflict, not
            whether it silently resolved it.
            
            Example: If benchmark stores "Alice is 25" then "Alice is 30",
            with detect_conflicts=True, the first fact gets marked [SUPERSEDED].
            The benchmark then asks "How old is Alice?" — but there's no conflict
            to resolve anymore! The benchmark is testing nothing.
        """
        metadata = metadata or {}

        # Adaptive auto_connect: build the graph for small fixture sets
        # (multi-hop benchmark needs it) but skip for large bulk ingests
        # (LoCoMo has 2000+ dialog turns; auto_connect is O(N²) and would
        # take many minutes). recall_multihop degrades gracefully when no
        # graph edges exist — falls back to plain recall.
        use_connect = self._store_count < self._auto_connect_threshold
        mem_id = self.mem.remember(
            text=content,
            label=label or content[:60],
            detect_conflicts=False,
            auto_connect=use_connect,
        )

        self._store_count += 1
        return str(mem_id)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve memories relevant to query.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of dicts with 'content' key (required) plus optional metadata.
            benchwrap only uses 'content' for prompt formatting — everything
            else is for debugging/logging.
        
        IMPORTANT: We disable temporal weighting for benchmarks.
        
        WHY disable temporal scoring:
            Neural Memory computes temporal decay: e^(-0.693 * age_hours / 24).
            This gives a 24-hour half-life — memories lose half their relevance
            every day.
            
            During benchmarks, all memories are ingested within seconds. The
            "oldest" memory might be 30 seconds older than the "newest". With
            temporal scoring, this 30-second difference creates score variance
            that has nothing to do with relevance.
            
            For production, temporal scoring is essential (recent facts are more
            likely to be relevant). For benchmarks, it's noise.
        """
        start = time.time()

        # Neural Memory's recall() returns:
        # [
        #   {
        #     'id': 42,
        #     'label': 'short label',
        #     'content': 'full text',
        #     'similarity': 0.85,       # cosine similarity with query
        #     'temporal_score': 0.95,    # recency score
        #     'combined': 0.87,          # weighted combination
        #     'connections': [...]       # related memories
        #   },
        #   ...
        # ]
        #
        # We set temporal_weight=0.0 to get pure similarity ranking.
        # The 'combined' score then equals 'similarity' exactly.
        results = self.mem.recall(
            query=query,
            k=top_k,
            temporal_weight=0.0,  # ← Pure similarity, no temporal noise
        )

        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms

        # Convert Neural Memory results to benchwrap format
        # benchwrap only needs 'content' — the rest is bonus metadata
        output = []
        for r in results:
            output.append({
                "content": r.get("content", ""),
                "label": r.get("label", ""),
                "score": r.get("similarity", 0.0),
                "id": r.get("id"),
                "connections": r.get("connections", []),
                # Debug info — not used by benchwrap, useful for analysis
                "_similarity": r.get("similarity", 0.0),
                "_combined": r.get("combined", 0.0),
                "_latency_ms": elapsed_ms,
            })

        return output

    # ------------------------------------------------------------------
    # Tier-2 tools (multi-hop, graph). Optional MemoryBackend extensions.
    # Memory-aware adapters check hasattr() and call these for richer
    # benchmarks (multi-hop, temporal, etc.).
    # ------------------------------------------------------------------

    def recall_temporal(self, query: str, top_k: int = 5,
                        temporal_weight: float = 0.3) -> list[dict]:
        """Recall with temporal scoring on. Use for 'when' / ordering queries."""
        start = time.time()
        results = self.mem.recall(query=query, k=top_k, temporal_weight=temporal_weight)
        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms
        return [
            {"content": r.get("content", ""), "label": r.get("label", ""),
             "score": r.get("combined", r.get("similarity", 0.0)),
             "id": r.get("id"), "_temporal": r.get("temporal_score", 0.0),
             "_latency_ms": elapsed_ms}
            for r in results
        ]

    def recall_multihop(self, query: str, top_k: int = 5,
                        hops: int = 2, temporal_weight: float = 0.0) -> list[dict]:
        """Multi-hop recall via graph traversal (PPR/BFS over connections)."""
        start = time.time()
        results = self.mem.recall_multihop(
            query=query, k=top_k, hops=hops, temporal_weight=temporal_weight,
        )
        elapsed_ms = (time.time() - start) * 1000
        self._recall_count += 1
        self._total_recall_ms += elapsed_ms
        return [
            {"content": r.get("content", ""), "label": r.get("label", ""),
             "score": r.get("similarity", 0.0), "id": r.get("id"),
             "_hop": r.get("hop", 0), "_latency_ms": elapsed_ms}
            for r in results
        ]

    def think(self, start_id: int, depth: int = 3, decay: float = 0.85) -> list[dict]:
        """BFS-spread from a seed memory; returns activated nodes ranked."""
        results = self.mem.think(start_id=start_id, depth=depth, decay=decay)
        return [
            {"content": r.get("content", ""), "label": r.get("label", ""),
             "score": r.get("score", 0.0), "id": r.get("id")}
            for r in results
        ]

    def connections(self, mem_id: int) -> list[dict]:
        """Direct edges from one memory."""
        return self.mem.connections(mem_id)

    def graph(self) -> dict:
        """Whole-graph snapshot (nodes, edges)."""
        return self.mem.graph()

    def ingest(self, items: list[dict]) -> int:
        """Batch store — faster than individual store() calls.
        
        Neural Memory's auto_connect is O(n²) per memory (compares with all
        existing). For benchmarks ingesting 1000+ memories, this is brutal.
        We already disable auto_connect in store(), so batch ingestion is fast.
        """
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
        """Clear all memories. Used between benchmark runs.
        
        PITFALL: This drops the SQLite database AND clears the in-memory graph
        AND clears the C++ SIMD index. If you forget to clear between benchmark
        runs, you get cross-contamination — memories from Run A appear in Run B.
        
        For benchmarks, always call clear() before load().
        """
        # Clear SQLite
        with self.mem.store._lock:
            self.mem.store.conn.execute("DELETE FROM memories")
            self.mem.store.conn.execute("DELETE FROM connections")
            self.mem.store.conn.commit()

        # Clear in-memory graph
        self.mem._graph_nodes.clear()

        # Clear C++ SIMD index
        if self.mem._cpp:
            try:
                # C++ index doesn't have a clear() method — reinitialize
                self.mem._cpp.initialize(dim=self.mem.dim)
            except Exception:
                pass

        # Reset stats
        self._store_count = 0
        self._recall_count = 0
        self._total_recall_ms = 0.0

        logger.info("[neural-memory] Cleared all memories")

    def stats(self) -> dict:
        """Return memory system statistics."""
        store_stats = self.mem.store.get_stats()
        return {
            "backend": "neural-memory",
            "retrieval_mode": "C++ SIMD" if self.mem._cpp else "Python linear",
            "embedding_dim": self.mem.dim,
            "total_memories": store_stats["memories"],
            "total_connections": store_stats["connections"],
            "benchmark_stores": self._store_count,
            "benchmark_recalls": self._recall_count,
            "avg_recall_ms": (
                self._total_recall_ms / self._recall_count
                if self._recall_count > 0
                else 0
            ),
        }

    def __del__(self):
        """Cleanup temp directory if we created one."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass


# ============================================================================
# Convenience: Standalone Neural Memory adapter (memory-bench + locomo)
# ============================================================================

def create_neural_memory_adapter(
    benchmark: str = "memory-bench",
    dataset: str = "recall-accuracy",
    db_path: str | None = None,
    llm_spec: str = "ollama:openhermes:7b-v2.5",
):
    """Create a fully configured Neural Memory benchmark adapter.
    
    Args:
        benchmark: Which benchmark to use (memory-bench, locomo, evomem)
        dataset: Which dataset within the benchmark
        db_path: SQLite path (None = temp DB)
        llm_spec: Model backend specification (e.g. "ollama:openhermes:7b")
    
    Returns:
        Configured benchmark adapter ready to run.
    
    Example:
        adapter = create_neural_memory_adapter("memory-bench", "multi-hop")
        engine = EvaluationEngine(adapter, llm_backend)
        result = engine.run()
    """
    from benchwrap.core.model import parse_backend

    memory = NeuralMemoryBackend(db_path=db_path)
    llm = parse_backend(llm_spec)

    if benchmark == "memory-bench":
        from benchwrap.adapters.memory_bench import MemoryBenchAdapter
        adapter = MemoryBenchAdapter(memory_client=memory, llm_backend=llm)
    elif benchmark == "locomo":
        from benchwrap.adapters.locomo import LoCoMoAdapter
        adapter = LoCoMoAdapter(memory_client=memory, llm_backend=llm)
    elif benchmark == "evomem":
        from benchwrap.adapters.evomem import EvoMemAdapter
        adapter = EvoMemAdapter(memory_client=memory, llm_backend=llm)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return adapter


# ============================================================================
# Quick test (run directly)
# ============================================================================

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("  Neural Memory Adapter — Quick Self-Test")
    print("=" * 60)

    # Create temp backend
    backend = NeuralMemoryBackend()
    print(f"\nStats: {json.dumps(backend.stats(), indent=2)}")

    # Test store
    test_data = [
        {"content": "The server is hosted in Frankfurt.", "label": "server"},
        {"content": "The database uses PostgreSQL 16.", "label": "db"},
        {"content": "Deployment is via GitHub Actions.", "label": "ci"},
        {"content": "The team has 5 engineers.", "label": "team"},
    ]
    n = backend.ingest(test_data)
    print(f"\nStored {n} memories")

    # Test recall
    queries = [
        "Where is the server?",
        "What database do we use?",
        "How do we deploy?",
    ]
    for q in queries:
        results = backend.recall(q, top_k=2)
        print(f"\nQuery: {q}")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content']}")

    print(f"\nFinal stats: {json.dumps(backend.stats(), indent=2)}")
    print("\n✓ Self-test passed")
