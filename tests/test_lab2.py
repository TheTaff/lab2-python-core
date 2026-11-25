from __future__ import annotations

import json
import unittest
from pathlib import Path

from labcheck.models import Submission
from labcheck.pipeline import build_features, extract_template_ngrams, iter_submissions
from labcheck.similarity import compare_all_in_assignment, compare_pair_ast, compare_pair_tokens

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestLab2Similarity(unittest.TestCase):
    def test_exact_copy_max_score(self) -> None:
        code = "def f(x):\n    y = x + 1\n    return y * 2\n"
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", code))
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam1", code))
        res_tok = compare_pair_tokens(f1, f2)
        res_ast = compare_pair_ast(f1, f2)
        self.assertEqual(res_tok.score, 1.0)
        self.assertEqual(res_ast.score, 1.0)

    def test_renamed_variables_ast_vs_token(self) -> None:
        c1 = "def calc(items):\n    total = 0\n    for x in items:\n        total += x\n    return total\n"
        c2 = "def compute(arr):\n    acc = 0\n    for val in arr:\n        acc += val\n    return acc\n"
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", c1))
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam1", c2))
        res_tok = compare_pair_tokens(f1, f2)
        res_ast = compare_pair_ast(f1, f2)
        self.assertIsNotNone(res_tok.score)
        self.assertIsNotNone(res_ast.score)
        self.assertEqual(res_ast.score, 1.0)
        self.assertGreater(res_ast.score, res_tok.score)

    def test_independent_solutions_low_score(self) -> None:
        c1 = "def f(a):\n    s = 0\n    for i in range(len(a)):\n        s += a[i]\n    return s\n"
        c2 = "def f(a):\n    return sum(x for x in a if x is not None)\n"
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", c1))
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam2", c2))
        res = compare_pair_ast(f1, f2, threshold=0.7)
        self.assertFalse(res.review_required)

    def test_allowed_template_not_flagged(self) -> None:
        tpl = "def read_data():\n    x = input().strip()\n    return x.split()\n"
        tpl_grams = extract_template_ngrams(tpl)
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", tpl), template_ngrams=tpl_grams)
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam2", tpl), template_ngrams=tpl_grams)
        res = compare_pair_tokens(f1, f2, threshold=0.5)
        self.assertTrue(res.template_only_overlap)
        self.assertFalse(res.review_required)
        self.assertEqual(res.reason, "template_only_match")

    def test_empty_input_returns_none_score(self) -> None:
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", "   \n# comment\n"))
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam2", ""))
        res = compare_pair_tokens(f1, f2)
        self.assertIsNone(res.score)
        self.assertEqual(res.reason, "insufficient_data_empty_union")

    def test_syntax_error_fallback(self) -> None:
        good = "def ok(x):\n    return x + 1\n"
        bad = "def broken(x:\n    return x + 1\n"
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", good))
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam2", bad))
        res = compare_pair_ast(f1, f2)
        self.assertTrue(res.fallback_used)
        self.assertEqual(res.method, "ast_fallback_token")
        self.assertIn("SyntaxError", res.reason or "")

    def test_no_self_match_and_separate_reuploads(self) -> None:
        c1 = "def f(x):\n    return x + 1\n"
        c2 = "def f(x):\n    return x + 2\n"
        f1 = build_features(Submission("s1_v1", "hw1", "st1", 1, "fam1", c1))
        f1_v2 = build_features(Submission("s1_v2", "hw1", "st1", 2, "fam1", c2))
        f2 = build_features(Submission("s2_v1", "hw2", "st2", 1, "fam2", c1))
        cross, reuploads = compare_all_in_assignment([f1, f1_v2, f2])
        self.assertEqual(len(cross), 0)
        self.assertEqual(len(reuploads), 1)
        self.assertTrue(reuploads[0].same_student_reupload)
        for item in cross + reuploads:
            self.assertNotEqual(item.left_id, item.right_id)

    def test_evidence_lines_in_range(self) -> None:
        c1 = "def f(x):\n    a = x + 1\n    return a\n"
        c2 = "def f(x):\n    a = x + 1\n    return a\n"
        max_lines = len(c1.splitlines())
        f1 = build_features(Submission("a", "hw1", "st1", 1, "fam1", c1))
        f2 = build_features(Submission("b", "hw1", "st2", 1, "fam1", c2))
        res = compare_pair_tokens(f1, f2)
        self.assertTrue(len(res.evidence) > 0)
        for ev in res.evidence:
            self.assertGreaterEqual(ev.start_line, 1)
            self.assertLessEqual(ev.end_line, max_lines)

    def test_corpus_no_family_leakage(self) -> None:
        subs = {s.submission_id: s for s in iter_submissions(DATA_DIR / "corpus_submissions.jsonl")}
        self.assertGreaterEqual(len(subs), 30)

        tune_families: set[str] = set()
        eval_families: set[str] = set()
        tpl_pairs = 0
        total_pairs = 0

        with (DATA_DIR / "corpus_pairs.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                p = json.loads(line)
                total_pairs += 1
                if p.get("template_only"):
                    tpl_pairs += 1
                f_left = subs[p["left_id"]].family_id
                f_right = subs[p["right_id"]].family_id
                if p["split"] == "tune":
                    tune_families.update([f_left, f_right])
                else:
                    eval_families.update([f_left, f_right])

        self.assertGreaterEqual(total_pairs, 40)
        self.assertGreaterEqual(tpl_pairs, 5)
        self.assertEqual(tune_families & eval_families, set())


if __name__ == "__main__":
    unittest.main()
