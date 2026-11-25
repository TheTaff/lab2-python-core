from __future__ import annotations

from labcheck.models import CodeFeatures, EvidenceSpan, PairComparison


def jaccard_score(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float | None:
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def compare_pair_tokens(
    left: CodeFeatures,
    right: CodeFeatures,
    threshold: float = 0.65,
    same_student_reupload: bool = False,
) -> PairComparison:
    union = left.token_ngrams | right.token_ngrams
    if not union:
        return PairComparison(
            assignment_id=left.assignment_id,
            left_id=left.submission_id,
            right_id=right.submission_id,
            left_student=left.student_id,
            right_student=right.student_id,
            same_student_reupload=same_student_reupload,
            method="token_3gram",
            score=None,
            reason="insufficient_data_empty_union",
        )

    inter = left.token_ngrams & right.token_ngrams
    score = len(inter) / len(union)

    tpl_common = left.template_ngrams & right.template_ngrams
    non_tpl_common = inter - tpl_common
    template_only = bool(inter) and (len(non_tpl_common) == 0)

    evidence: list[EvidenceSpan] = []
    for gram in sorted(inter)[:8]:
        l_span = left.token_line_spans.get(gram, [1, 1])
        r_span = right.token_line_spans.get(gram, [1, 1])
        kind = "template_overlap" if gram in tpl_common else "token_3gram"
        evidence.append(
            EvidenceSpan(
                start_line=min(l_span[0], r_span[0]),
                end_line=max(l_span[1], r_span[1]),
                kind=kind,
                detail=f"L:{l_span[0]}-{l_span[1]}, R:{r_span[0]}-{r_span[1]} | {' '.join(gram)}",
            )
        )

    # Общий шаблон без собственной совпадающей логики не должен вызывать ложную тревогу
    needs_review = (score >= threshold) and (not template_only) and (not same_student_reupload)
    reason = "template_only_match" if template_only else None

    return PairComparison(
        assignment_id=left.assignment_id,
        left_id=left.submission_id,
        right_id=right.submission_id,
        left_student=left.student_id,
        right_student=right.student_id,
        same_student_reupload=same_student_reupload,
        method="token_3gram",
        score=round(score, 6),
        reason=reason,
        template_only_overlap=template_only,
        review_required=needs_review,
        evidence=evidence,
    )


def compare_pair_ast(
    left: CodeFeatures,
    right: CodeFeatures,
    threshold: float = 0.70,
    same_student_reupload: bool = False,
) -> PairComparison:
    if left.ast_ngrams is None or right.ast_ngrams is None:
        fb = compare_pair_tokens(
            left,
            right,
            threshold=threshold,
            same_student_reupload=same_student_reupload,
        )
        fb.method = "ast_fallback_token"
        fb.fallback_used = True
        fb.reason = left.ast_error or right.ast_error or "ast_syntax_error"
        return fb

    score = jaccard_score(left.ast_ngrams, right.ast_ngrams)
    if score is None:
        return PairComparison(
            assignment_id=left.assignment_id,
            left_id=left.submission_id,
            right_id=right.submission_id,
            left_student=left.student_id,
            right_student=right.student_id,
            same_student_reupload=same_student_reupload,
            method="ast_3gram",
            score=None,
            reason="insufficient_data_empty_union",
        )

    inter = left.ast_ngrams & right.ast_ngrams
    # Проверяем долю нешаблонных токенов, чтобы разрешённый шаблон не давал ложный review_required
    tok_inter = left.token_ngrams & right.token_ngrams
    tpl_common = left.template_ngrams & right.template_ngrams
    template_only = bool(tok_inter) and (len(tok_inter - tpl_common) == 0)

    evidence = [
        EvidenceSpan(
            start_line=1,
            end_line=1,
            kind="ast_node",
            detail=" ".join(gram),
        )
        for gram in sorted(inter)[:8]
    ]
    needs_review = (score >= threshold) and (not template_only) and (not same_student_reupload)
    return PairComparison(
        assignment_id=left.assignment_id,
        left_id=left.submission_id,
        right_id=right.submission_id,
        left_student=left.student_id,
        right_student=right.student_id,
        same_student_reupload=same_student_reupload,
        method="ast_3gram",
        score=round(score, 6),
        reason="template_only_match" if template_only else None,
        template_only_overlap=template_only,
        review_required=needs_review,
        evidence=evidence,
    )


def compare_all_in_assignment(
    features: list[CodeFeatures],
    method: str = "token",
    threshold: float = 0.65,
) -> tuple[list[PairComparison], list[PairComparison]]:
    """
    Возвращает два списка:
    1) сравнения между разными студентами в пределах одного assignment_id;
    2) повторные версии одного и того же студента (учитываются отдельно, self-match исключён).
    """
    cross_student: list[PairComparison] = []
    same_student_versions: list[PairComparison] = []

    by_assignment: dict[str, list[CodeFeatures]] = {}
    for item in features:
        by_assignment.setdefault(item.assignment_id, []).append(item)

    for items in by_assignment.values():
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i], items[j]
                if a.submission_id == b.submission_id:
                    continue
                is_reupload = a.student_id == b.student_id
                if method == "ast":
                    res = compare_pair_ast(a, b, threshold=threshold, same_student_reupload=is_reupload)
                else:
                    res = compare_pair_tokens(a, b, threshold=threshold, same_student_reupload=is_reupload)
                if is_reupload:
                    same_student_versions.append(res)
                else:
                    cross_student.append(res)

    return cross_student, same_student_versions
