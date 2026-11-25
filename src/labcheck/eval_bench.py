from __future__ import annotations

import cProfile
import io
import json
import pstats
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from labcheck.models import Submission
from labcheck.pipeline import build_features, extract_template_ngrams, iter_feature_pipeline, iter_submissions
from labcheck.similarity import compare_pair_ast, compare_pair_tokens


def load_templates(jsonl_path: str | Path) -> dict[str, set[tuple[str, ...]]]:
    templates: dict[str, set[tuple[str, ...]]] = {}
    for sub in iter_submissions(jsonl_path):
        if sub.is_template:
            templates[sub.assignment_id] = extract_template_ngrams(sub.code)
    return templates


def evaluate_pairs(
    submissions_path: str | Path,
    pairs_path: str | Path,
    method: str,
    threshold: float,
    split_filter: str,
) -> dict:
    templates = load_templates(submissions_path)
    feat_map = {
        f.submission_id: f
        for f in iter_feature_pipeline(submissions_path, templates_by_assignment=templates, fast_ast=True)
    }

    tp = fp = tn = fn = undefined_cnt = 0
    errors: list[dict] = []

    with Path(pairs_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            pair = json.loads(line)
            if pair.get("split") != split_filter:
                continue

            left = feat_map[pair["left_id"]]
            right = feat_map[pair["right_id"]]
            reupload = left.student_id == right.student_id

            if method == "ast":
                cmp_res = compare_pair_ast(left, right, threshold=threshold, same_student_reupload=reupload)
            else:
                cmp_res = compare_pair_tokens(left, right, threshold=threshold, same_student_reupload=reupload)

            expected = bool(pair["derivation_relation"])
            if cmp_res.score is None:
                undefined_cnt += 1
                predicted = False
            else:
                predicted = cmp_res.review_required

            if predicted and expected:
                tp += 1
            elif predicted and not expected:
                fp += 1
                errors.append({
                    "type": "FP",
                    "left_id": pair["left_id"],
                    "right_id": pair["right_id"],
                    "score": cmp_res.score,
                    "template_only": pair.get("template_only", False),
                    "note": pair.get("note", ""),
                })
            elif not predicted and not expected:
                tn += 1
            else:
                fn += 1
                errors.append({
                    "type": "FN",
                    "left_id": pair["left_id"],
                    "right_id": pair["right_id"],
                    "score": cmp_res.score,
                    "template_only": pair.get("template_only", False),
                    "note": pair.get("note", ""),
                })

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    undefined_rate = undefined_cnt / total if total else 0.0

    return {
        "split": split_filter,
        "method": method,
        "threshold": threshold,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "undefined_count": undefined_cnt,
        "undefined_rate": round(undefined_rate, 4),
        "errors": errors,
    }


def tune_threshold(submissions_path: str | Path, pairs_path: str | Path, method: str) -> float:
    best_th = 0.5
    best_f1 = -1.0
    for step in range(20, 95, 5):
        th = step / 100.0
        res = evaluate_pairs(submissions_path, pairs_path, method=method, threshold=th, split_filter="tune")
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_th = th
    return best_th


def generate_synthetic_submissions(count: int, seed: int = 42) -> list[Submission]:
    rng = random.Random(seed)
    items: list[Submission] = []
    for idx in range(count):
        a_var = f"val_{rng.randint(1, 50)}"
        b_var = f"acc_{rng.randint(1, 50)}"
        limit = rng.randint(10, 200)
        code = "\n".join([
            f"def solve_{idx}(items):",
            f"    {b_var} = 0",
            f"    for {a_var} in items:",
            f"        if {a_var} > {limit}:",
            f"            {b_var} += {a_var} * 2",
            f"        else:",
            f"            {b_var} -= 1",
            f"    return {b_var}",
        ])
        items.append(
            Submission(
                submission_id=f"syn_{idx:05d}",
                assignment_id=f"hw_{idx % 5}",
                student_id=f"stud_{idx}",
                attempt=1,
                family_id=f"fam_{idx}",
                code=code,
            )
        )
    return items


def benchmark_pipeline(sizes: tuple[int, ...] = (100, 1000, 5000), repeats: int = 3, seed: int = 42) -> dict:
    """Измеряет медианное wall time и peak memory (tracemalloc) до и после оптимизации AST."""
    report: dict[str, dict] = {"unoptimized": {}, "optimized": {}}
    for mode, fast_flag in (("unoptimized", False), ("optimized", True)):
        for size in sizes:
            subs = generate_synthetic_submissions(size, seed=seed)
            times_ms: list[float] = []
            peaks_kb: list[float] = []
            for _ in range(repeats):
                tracemalloc.start()
                t0 = time.perf_counter()
                processed = 0
                for s in subs:
                    feat = build_features(s, fast_ast=fast_flag)
                    if not feat.is_empty:
                        processed += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                times_ms.append(elapsed_ms)
                peaks_kb.append(peak_bytes / 1024.0)

            report[mode][str(size)] = {
                "median_wall_ms": round(statistics.median(times_ms), 2),
                "median_peak_kb": round(statistics.median(peaks_kb), 2),
                "runs_ms": [round(x, 2) for x in times_ms],
            }
    return report


def profile_hotspot(size: int = 1000, seed: int = 42) -> str:
    subs = generate_synthetic_submissions(size, seed=seed)
    pr = cProfile.Profile()
    pr.enable()
    for s in subs:
        build_features(s, fast_ast=False)
    pr.disable()
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    stats.print_stats(10)
    return buf.getvalue()
