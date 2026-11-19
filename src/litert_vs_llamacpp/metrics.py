"""Performance aggregation and ROUGE scoring utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from tabulate import tabulate


def compute_rouge(records: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean ROUGE-1/2/L F1 with Porter stemming (matching lillm-summarizer)."""
    if not records:
        raise ValueError("Cannot compute ROUGE on empty record list")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for row in records:
        scores = scorer.score(row["reference"], row["prediction"])
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = len(records)
    return {key: round(val / n, 4) for key, val in totals.items()}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate throughput, latency, token counts, and ROUGE over a run."""
    if not records:
        raise ValueError("Cannot summarize empty record list")

    latencies = np.array([r["latency_s"] for r in records], dtype=float)
    ttfts = np.array([r["ttft_s"] for r in records if r.get("ttft_s") is not None], dtype=float)
    prefill_tps = np.array(
        [r["prefill_tok_s"] for r in records if r.get("prefill_tok_s") is not None], dtype=float
    )
    decode_tps = np.array(
        [r["decode_tok_s"] for r in records if r.get("decode_tok_s") is not None], dtype=float
    )
    prompt_tokens = [r["prompt_tokens"] for r in records if r.get("prompt_tokens") is not None]
    output_tokens = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]
    words = [len(r["prediction"].split()) for r in records]
    cap_stops = sum(1 for r in records if r.get("hit_max_tokens"))

    def _stats(arr: np.ndarray) -> dict[str, float] | None:
        if arr.size == 0:
            return None
        return {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
        }

    total_wall_s = float(np.sum(latencies))
    total_out_tok = int(sum(output_tokens))
    total_in_tok = int(sum(prompt_tokens))

    return {
        "count": len(records),
        "total_wall_s": round(total_wall_s, 3),
        "total_prompt_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "effective_output_tok_s": round(total_out_tok / total_wall_s, 2) if total_wall_s > 0 else 0.0,
        "latency_s": _stats(latencies),
        "ttft_s": _stats(ttfts),
        "prefill_tok_s": _stats(prefill_tps),
        "decode_tok_s": _stats(decode_tps),
        "mean_words": round(float(np.mean(words)), 1),
        "cap_stops": cap_stops,
        "rouge": compute_rouge(records),
    }


def format_comparison_markdown(runs: list[dict[str, Any]]) -> str:
    """Render a compact Markdown comparison table across engines/backends/quants."""
    headers = [
        "Engine",
        "Backend",
        "Quant",
        "Prefill (tok/s)",
        "Decode (tok/s)",
        "TTFT med (ms)",
        "Latency med (s)",
        "Total wall (s)",
        "ROUGE-1 / 2 / L",
        "Cap stops",
    ]
    rows = []
    for run in runs:
        s = run["summary"]
        pf = s["prefill_tok_s"]["median"] if s.get("prefill_tok_s") else None
        dc = s["decode_tok_s"]["median"] if s.get("decode_tok_s") else None
        ttft_ms = round(s["ttft_s"]["median"] * 1000, 1) if s.get("ttft_s") else None
        lat_med = s["latency_s"]["median"] if s.get("latency_s") else None
        r = s["rouge"]
        rows.append(
            [
                run["engine"],
                run["backend"],
                run["quant"],
                f"{pf:.1f}" if pf is not None else "n/a",
                f"{dc:.1f}" if dc is not None else "n/a",
                f"{ttft_ms:.1f}" if ttft_ms is not None else "n/a",
                f"{lat_med:.3f}" if lat_med is not None else "n/a",
                f"{s['total_wall_s']:.2f}",
                f"{r['rouge1']:.3f} / {r['rouge2']:.3f} / {r['rougeL']:.3f}",
                f"{s['cap_stops']}/{s['count']}",
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="github")
