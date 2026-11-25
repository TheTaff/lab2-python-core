from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from labcheck.eval_bench import benchmark_pipeline, evaluate_pairs, profile_hotspot, tune_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 2 Similarity & Evaluation CLI")
    parser.add_argument("--submissions", default="data/corpus_submissions.jsonl")
    parser.add_argument("--pairs", default="data/corpus_pairs.jsonl")
    parser.add_argument("--out", default="data/eval_summary.json")
    args = parser.parse_args()

    tok_th = tune_threshold(args.submissions, args.pairs, method="token")
    ast_th = tune_threshold(args.submissions, args.pairs, method="ast")

    tok_eval = evaluate_pairs(args.submissions, args.pairs, method="token", threshold=tok_th, split_filter="eval")
    ast_eval = evaluate_pairs(args.submissions, args.pairs, method="ast", threshold=ast_th, split_filter="eval")
    bench = benchmark_pipeline(sizes=(100, 1000, 5000), repeats=3, seed=42)
    hotspot = profile_hotspot(size=1000, seed=42)

    summary = {
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seed": 42,
        },
        "tuned_thresholds": {"token": tok_th, "ast": ast_th},
        "eval_results": {"token": tok_eval, "ast": ast_eval},
        "benchmark": bench,
        "cprofile_top10_unoptimized_1000": hotspot,
    }
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
