"""
MemoryAgentBench (MAB) adapter for benchwrap.
ICLR 2026 memory-augmented agent benchmark.

Categories:
  - Conflict Resolution: FactConsolidation (single-hop, multi-hop)
  - Long Range Understanding: Detective QA, InfBench
  - Accurate Retrieval: EventQA, LongMemEval, Ruler
  - Test Time Learning: ICL (In-Context Learning), RecSys

Wraps existing MAB code at ~/projects/MemoryAgentBench/.
Uses benchwrap ModelBackend for LLM calls.
"""

import json
from pathlib import Path
from typing import Iterator, Optional

from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.model import ModelBackend


MAB_DIR = Path.home() / "projects/MemoryAgentBench"

# Dataset registry
# sub_dataset = HF source filter, max_test_samples = cap per dataset
DATASETS = {
    # Conflict Resolution
    "conflict-sh-6k": {
        "config": "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml",
        "category": "conflict_resolution",
        "subcategory": "single-hop",
        "context_size": "6k",
        "sub_dataset": "factconsolidation_sh_6k",
        "max_test_samples": 20,
    },
    "conflict-mh-6k": {
        "config": "configs/data_conf/Conflict_Resolution/Factconsolidation_mh_6k.yaml",
        "category": "conflict_resolution",
        "subcategory": "multi-hop",
        "context_size": "6k",
        "sub_dataset": "factconsolidation_mh_6k",
        "max_test_samples": 20,
    },
    "conflict-sh-32k": {
        "config": "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_32k.yaml",
        "category": "conflict_resolution",
        "subcategory": "single-hop",
        "context_size": "32k",
        "sub_dataset": "factconsolidation_sh_32k",
        "max_test_samples": 20,
    },
    "conflict-mh-32k": {
        "config": "configs/data_conf/Conflict_Resolution/Factconsolidation_mh_32k.yaml",
        "category": "conflict_resolution",
        "subcategory": "multi-hop",
        "context_size": "32k",
        "sub_dataset": "factconsolidation_mh_32k",
        "max_test_samples": 20,
    },
    # Long Range Understanding
    "detective-qa": {
        "config": "configs/data_conf/Long_Range_Understanding/Detective_QA.yaml",
        "category": "long_range_understanding",
        "subcategory": "detective",
        "sub_dataset": "detective_qa",
        "max_test_samples": 20,
    },
    "infbench-sum": {
        "config": "configs/data_conf/Long_Range_Understanding/InfBench_sum.yaml",
        "category": "long_range_understanding",
        "subcategory": "infbench",
        "sub_dataset": "infbench_sum_eng_shots2",
        "max_test_samples": 20,
    },
    # Accurate Retrieval
    "eventqa-64k": {
        "config": "configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_64k.yaml",
        "category": "accurate_retrieval",
        "subcategory": "eventqa",
        "context_size": "64k",
        "sub_dataset": "eventqa_65536",
        "max_test_samples": 20,
    },
    "eventqa-full": {
        "config": "configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_full.yaml",
        "category": "accurate_retrieval",
        "subcategory": "eventqa",
        "context_size": "full",
        "sub_dataset": "eventqa_full",
        "max_test_samples": 20,
    },
    "longmemeval-s": {
        "config": "configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s.yaml",
        "category": "accurate_retrieval",
        "subcategory": "longmemeval",
        "sub_dataset": "longmemeval_s_-1_500",
        "max_test_samples": 20,
    },
    # Test Time Learning
    "icl-nlu": {
        "config": "configs/data_conf/Test_Time_Learning/ICL/ICL_nlu.yaml",
        "category": "test_time_learning",
        "subcategory": "icl",
        "sub_dataset": "icl_nlu_8296shot_balance",
        "max_test_samples": 20,
    },
    "icl-banking77": {
        "config": "configs/data_conf/Test_Time_Learning/ICL/ICL_banking77.yaml",
        "category": "test_time_learning",
        "subcategory": "icl",
        "sub_dataset": "icl_banking77_5900shot_balance",
        "max_test_samples": 20,
    },
    "icl-clinic150": {
        "config": "configs/data_conf/Test_Time_Learning/ICL/ICL_clinic150.yaml",
        "category": "test_time_learning",
        "subcategory": "icl",
        "sub_dataset": "icl_clinic150_7050shot_balance",
        "max_test_samples": 20,
    },
}


class MemoryAgentBenchAdapter(BenchmarkAdapter):
    """MemoryAgentBench — ICLR 2026 memory-augmented agent evaluation.
    
    Tests memory systems on: conflict resolution, long-range understanding,
    accurate retrieval, and test-time learning.
    
    Data: ~/projects/MemoryAgentBench/data/
    Configs: ~/projects/MemoryAgentBench/configs/
    
    Memory backend is injected via set_memory_client().
    LLM backend is injected via set_llm_backend().
    """

    def __init__(
        self,
        memory_client=None,
        llm_backend: ModelBackend | None = None,
    ):
        self.memory_client = memory_client
        self.llm_backend = llm_backend
        self._data_cache = {}

    def name(self) -> str:
        return "memory-agent-bench"

    def datasets(self) -> list[str]:
        return ["all"] + list(DATASETS.keys())

    def load(
        self,
        dataset: str = "all",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load MAB samples."""
        if dataset == "all":
            dataset_list = list(DATASETS.keys())
        elif dataset in DATASETS:
            dataset_list = [dataset]
        else:
            # Try partial match
            dataset_list = [d for d in DATASETS if dataset in d]
            if not dataset_list:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. Available: {', '.join(DATASETS.keys())}"
                )

        count = 0
        for ds_name in dataset_list:
            ds_info = DATASETS[ds_name]
            samples = self._load_dataset(ds_name, ds_info)
            for sample in samples:
                if limit and count >= limit:
                    return
                sample.metadata["dataset_name"] = ds_name
                sample.metadata["category"] = ds_info["category"]
                sample.metadata["subcategory"] = ds_info.get("subcategory", "")
                yield sample
                count += 1

    def pre_evaluate(self, dataset: str = "all", backend=None):
        """Store context facts into memory for retrieval-augmented scoring."""
        if not self.memory_client:
            return
        
        # Load all samples for this dataset to extract contexts
        samples = list(self.load(dataset, limit=None))
        
        # Collect unique contexts (all QA pairs from same sample share context)
        seen_contexts = set()
        fact_count = 0
        for sample in samples:
            ctx = sample.metadata.get("context", "")
            ctx_hash = hash(ctx[:1000])  # First 1KB for dedup
            if ctx_hash in seen_contexts or not ctx:
                continue
            seen_contexts.add(ctx_hash)
            
            # Parse numbered facts from context (e.g., "0. fact\n1. fact\n...")
            facts = self._parse_context_facts(ctx)
            for i, fact in enumerate(facts):
                self.memory_client.store(
                    fact,
                    label=f"mab_{dataset}_fact_{i}",
                    metadata={"source": dataset, "fact_idx": i}
                )
                fact_count += 1
        
        print(f"[memory-agent-bench] Stored {fact_count} facts into memory")

    def _parse_context_facts(self, context: str) -> list[str]:
        """Parse numbered facts from context string."""
        facts = []
        for line in context.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Match patterns like "0. fact", "1. fact", "42. fact"
            import re
            m = re.match(r'^\d+\.\s*(.+)', line)
            if m:
                facts.append(m.group(1).strip())
            elif line.startswith('- '):
                facts.append(line[2:].strip())
            elif len(line) > 10:  # Arbitrary non-empty line
                facts.append(line)
        return facts

    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format MAB sample as a recall/reasoning prompt."""
        question = sample.input
        context = ""

        if self.memory_client:
            # Use memory recall instead of raw context
            try:
                results = self.memory_client.recall(question, top_k=20)
                if results:
                    context_parts = []
                    for r in results:
                        content = r.get("content", r.get("text", str(r)))
                        context_parts.append(f"- {content}")
                    context = "\n".join(context_parts)
            except Exception as e:
                context = f"[Memory recall error: {e}]"
        else:
            # No memory — use raw context from dataset
            context = sample.metadata.get("context", "")

        if context:
            text = (
                f"Context:\n{context[:50000]}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            text = f"Question: {question}\n\nAnswer:"

        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            raw_text=text[:500],  # Truncate for logging (context can be huge)
        )

    def score(
        self,
        prediction: str,
        reference: str,
        sample: Sample,
    ) -> Score:
        """Score using exact match (EM) and F1."""
        pred_clean = prediction.strip().lower()
        ref_clean = reference.strip().lower()

        # Exact match
        em = 1.0 if pred_clean == ref_clean else 0.0

        # Token-level F1
        f1 = _compute_f1(pred_clean, ref_clean)

        return Score(
            exact_match=em,
            f1=f1,
            raw_prediction=prediction[:200],
            raw_reference=reference,
            scoring_method="mab_em_f1",
        )

    def _load_dataset(self, ds_name: str, ds_info: dict) -> list[Sample]:
        """Load samples from MAB dataset via HuggingFace."""
        cache_key = ds_name
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        samples = []
        
        try:
            from datasets import load_dataset
        except ImportError:
            print(f"[memory-agent-bench] 'datasets' not installed. pip install datasets")
            self._data_cache[cache_key] = samples
            return samples

        hf_dataset_name = "ai-hyz/MemoryAgentBench"
        hf_split = ds_info.get("category", "").replace("_", " ").title().replace(" ", "_")
        # Map category names to actual HF split names
        split_map = {
            "conflict_resolution": "Conflict_Resolution",
            "long_range_understanding": "Long_Range_Understanding",
            "accurate_retrieval": "Accurate_Retrieval",
            "test_time_learning": "Test_Time_Learning",
        }
        hf_split = split_map.get(ds_info["category"], ds_info["category"])
        sub_dataset = ds_info.get("sub_dataset", "")
        max_samples = ds_info.get("max_test_samples", 20)

        try:
            print(f"[memory-agent-bench] Loading {sub_dataset} from {hf_dataset_name}/{hf_split}...")
            raw_data = load_dataset(hf_dataset_name, split=hf_split, revision="main")
            
            # Filter by source (sub_dataset)
            if sub_dataset:
                raw_data = raw_data.filter(
                    lambda s: s.get("metadata", {}).get("source", "") == sub_dataset
                )
            
            print(f"[memory-agent-bench] Got {len(raw_data)} samples, using up to {max_samples}")
            
            count = 0
            for sample in raw_data:
                if count >= max_samples:
                    break
                    
                questions = self._ensure_list(sample.get("questions", []))
                answers = self._ensure_list(sample.get("answers", []))
                
                # Build context from dialogue/context fields
                context = ""
                if "context" in sample:
                    context = sample["context"]
                elif "contexts" in sample:
                    ctxs = sample["contexts"]
                    if isinstance(ctxs, list):
                        context = "\n\n".join(str(c) for c in ctxs[:5])  # Cap context
                    else:
                        context = str(ctxs)
                
                # Yield one Sample per QA pair
                num_pairs = min(len(questions), len(answers))
                for j in range(num_pairs):
                    q = str(questions[j]).strip()
                    a = str(answers[j]).strip()
                    if not q:
                        continue
                    samples.append(Sample(
                        id=f"mab_{ds_name}_{count}_{j}",
                        input=q,
                        reference=a,
                        metadata={
                            "context": str(context)[:100000],
                            "file_id": ds_name,
                            "qa_pair_id": j,
                        },
                    ))
                count += 1
                
        except Exception as e:
            print(f"[memory-agent-bench] Failed to load {ds_name}: {e}")

        self._data_cache[cache_key] = samples
        return samples

    @staticmethod
    def _ensure_list(val):
        """Ensure a value is a list."""
        if isinstance(val, list):
            return val
        if val:
            return [val]
        return []

    def _parse_data_file(self, path: Path, ds_info: dict) -> list[Sample]:
        """Parse an MAB data file into Samples."""
        samples = []

        if path.is_dir():
            # Directory of files
            for f in sorted(path.glob("*.json")):
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        samples.append(self._item_to_sample(item, f.stem, i))
                elif isinstance(data, dict):
                    samples.append(self._item_to_sample(data, f.stem, 0))
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    samples.append(self._item_to_sample(item, path.stem, i))
            elif isinstance(data, dict):
                for key, item in data.items():
                    if isinstance(item, dict):
                        samples.append(self._item_to_sample(item, key, 0))
        elif path.suffix in (".jsonl", ".jsonl"):
            with open(path) as f:
                for i, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        samples.append(self._item_to_sample(item, path.stem, i))

        return samples

    def _item_to_sample(self, item: dict, file_id: str, idx: int) -> Sample:
        """Convert a data item to a Sample."""
        question = item.get("question", item.get("query", item.get("input", "")))
        answer = item.get("answer", item.get("reference", item.get("output", "")))
        context = item.get("context", item.get("passage", item.get("text", "")))

        if isinstance(answer, list):
            answer = "; ".join(str(a) for a in answer)
        answer = str(answer)

        return Sample(
            id=f"mab_{file_id}_{idx}",
            input=str(question),
            reference=answer,
            metadata={
                "context": str(context)[:100000],  # Cap context size
                "file_id": file_id,
            },
        )

    def set_memory_client(self, client):
        """Set the memory backend."""
        self.memory_client = client

    def set_llm_backend(self, backend: ModelBackend):
        """Set the LLM backend."""
        self.llm_backend = backend


def _compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
