#!/usr/bin/env python3
"""
benchview — Visualize benchwrap results in the terminal.
Reads JSON results, picks latest per benchmark, renders ASCII bar charts.

Usage:
    python3 benchview.py [results_dir]
    python3 benchview.py results/gpt-oss-120b-full-run/
    python3 benchview.py  # defaults to results/
"""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


def load_results(directory: str) -> dict:
    """Load all result JSONs, pick latest per (adapter, dataset, has_memory) triple."""
    latest = {}
    
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.json') or fname == 'summary.json':
            continue
        fpath = os.path.join(directory, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        
        adapter = data.get('adapter', '')
        dataset = data.get('dataset', '')
        timestamp = data.get('timestamp', 0)
        metrics = data.get('metrics', {})
        
        # Skip broken results
        if not metrics or metrics.get('n', 0) == 0:
            continue
        
        has_memory = _detect_memory(data)
        key = (adapter, dataset, has_memory)
        if key not in latest or timestamp > latest[key]['timestamp']:
            latest[key] = {
                'adapter': adapter,
                'dataset': dataset,
                'timestamp': timestamp,
                'metrics': metrics,
                'samples': data.get('num_samples', metrics.get('n', 0)),
                'duration_s': data.get('duration_s', 0),
                'model': data.get('model', '?'),
                'filename': fname,
                'has_memory': has_memory,
            }
    
    return latest


def _detect_memory(data: dict) -> bool:
    """Detect if neural memory was used based on prompt content."""
    samples = data.get('samples', [])
    if not samples:
        return False
    
    # Check if prompts contain recalled facts/context
    first_sample = samples[0]
    prompt = first_sample.get('prompt', {}).get('raw_text', '')
    
    # Memory benchmarks with memory include "Facts:" or "Memories:" in prompt
    if 'Facts:\n' in prompt or 'Memories:\n' in prompt:
        return True
    
    # MAB with memory uses "- fact" format in context (recalled list)
    # vs raw numbered "0. fact\n1. fact" (full context, no memory)
    adapter = data.get('adapter', '')
    if adapter == 'memory-agent-bench':
        # Full context starts with "Context:\nHere is a list of facts:\n0."
        # Recalled context starts with "Context:\n- fact"
        if 'Context:\n- ' in prompt:
            return True
        return False
    
    return False


def bar(value: float, width: int = 40, char: str = '█', empty: str = '░') -> str:
    """Render an ASCII bar."""
    filled = int(value * width)
    return char * filled + empty * (width - filled)


def render(results: dict):
    """Render results as a terminal dashboard."""
    
    # Group by adapter, separate memory vs no-memory
    grouped = defaultdict(lambda: {'no_mem': None, 'with_mem': None})
    
    for key, r in results.items():
        adapter, dataset, has_memory = key
        if has_memory:
            if grouped[adapter]['with_mem'] is None:
                grouped[adapter]['with_mem'] = r
        else:
            if grouped[adapter]['no_mem'] is None:
                grouped[adapter]['no_mem'] = r
    
    # Collect model name
    model = '?'
    for r in results.values():
        model = r.get('model', model)
        break
    
    # Header
    w = 78
    print()
    print('╔' + '═' * w + '╗')
    print('║' + f'  BENCHWRAP RESULTS — {model}'.ljust(w) + '║')
    print('╠' + '═' * w + '╣')
    
    # For each benchmark
    for adapter in ['mmlu', 'gsm8k', 'evomem', 'memory-bench', 'locomo', 'memory-agent-bench']:
        g = grouped.get(adapter)
        if not g:
            continue
        
        no_mem = g['no_mem']
        with_mem = g['with_mem']
        
        # Get the best score (accuracy > f1 > exact_match)
        def best_score(r):
            if not r:
                return None, None
            m = r['metrics']
            primary = m.get('accuracy', m.get('f1', m.get('exact_match', 0)))
            metric_name = 'accuracy' if 'accuracy' in m else ('f1' if 'f1' in m else 'exact_match')
            return primary, metric_name
        
        n_score, n_metric = best_score(no_mem)
        w_score, w_metric = best_score(with_mem)
        
        dataset = (no_mem or with_mem)['dataset']
        samples = (no_mem or with_mem)['samples']
        
        # Benchmark header
        label = f'{adapter}/{dataset}'
        print('║' + f'  ┌─ {label}'.ljust(w) + '║')
        print('║' + f'  │  samples: {samples}'.ljust(w) + '║')
        
        # No memory bar
        if n_score is not None:
            n_pct = n_score * 100
            print('║' + f'  │'.ljust(w) + '║')
            print('║' + f'  │  NO MEMORY ({n_metric})'.ljust(w) + '║')
            print('║' + f'  │  {bar(n_score)} {n_pct:5.1f}%'.ljust(w) + '║')
        
        # With memory bar
        if w_score is not None:
            w_pct = w_score * 100
            delta = ''
            if n_score is not None:
                d = (w_score - n_score) * 100
                delta = f'  ({d:+.1f}%)'
            print('║' + f'  │'.ljust(w) + '║')
            print('║' + f'  │  WITH NEURAL MEMORY ({w_metric}){delta}'.ljust(w) + '║')
            print('║' + f'  │  {bar(w_score)} {w_pct:5.1f}%'.ljust(w) + '║')
        
        # No memory comparison
        if n_score is None and w_score is not None:
            pass  # memory-only benchmark, no baseline needed
        
        print('║' + f'  └{"─" * (w - 3)}║')
    
    # Footer with key metrics summary
    print('╠' + '═' * w + '╣')
    print('║' + '  SUMMARY'.ljust(w) + '║')
    print('║' + '  ───────'.ljust(w) + '║')
    
    for adapter in ['mmlu', 'gsm8k', 'evomem', 'memory-bench', 'locomo', 'memory-agent-bench']:
        g = grouped.get(adapter)
        if not g:
            continue
        
        n_score, _ = best_score(g['no_mem'])
        w_score, _ = best_score(g['with_mem'])
        
        n_str = f'{n_score*100:5.1f}%' if n_score is not None else '    — '
        w_str = f'{w_score*100:5.1f}%' if w_score is not None else '    — '
        
        if n_score is not None and w_score is not None:
            d = (w_score - n_score) * 100
            delta = f'{d:+6.1f}%'
        else:
            delta = '      — '
        
        line = f'  {adapter:<20} {n_str}  →  {w_str}  {delta}'
        print('║' + line.ljust(w) + '║')
    
    print('╚' + '═' * w + '╝')
    print()


def render_json(results: dict):
    """Dump clean JSON summary."""
    output = {}
    for key, r in sorted(results.items()):
        adapter, dataset, has_memory = key
        m = r['metrics']
        suffix = '_with_mem' if has_memory else '_no_mem'
        output[f"{adapter}/{dataset}{suffix}"] = {
            'samples': r['samples'],
            'duration_s': round(r['duration_s'], 1),
            'exact_match': m.get('exact_match'),
            'accuracy': m.get('accuracy'),
            'f1': m.get('f1'),
            'avg_latency_ms': round(m['avg_latency_ms']) if 'avg_latency_ms' in m else None,
            'has_memory': r['has_memory'],
        }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / 'projects' / 'benchwrap' / 'results')
    
    # If a specific dir is given, use it; otherwise scan all subdirs
    if os.path.isdir(results_dir):
        results = load_results(results_dir)
        if not results:
            # Try scanning subdirectories
            for subdir in sorted(os.listdir(results_dir)):
                subpath = os.path.join(results_dir, subdir)
                if os.path.isdir(subpath):
                    results.update(load_results(subpath))
    
    if not results:
        print(f"No results found in {results_dir}")
        sys.exit(1)
    
    if '--json' in sys.argv:
        render_json(results)
    else:
        render(results)
