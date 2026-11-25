from __future__ import annotations

import ast
import io
import json
import tokenize
from pathlib import Path
from typing import Iterator

from labcheck.models import CodeFeatures, Submission

_SKIP_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
}


def iter_submissions(jsonl_path: str | Path) -> Iterator[Submission]:
    """Ленивое чтение JSONL построчно через генератор без загрузки всего корпуса в память."""
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if "submission_id" not in payload or "code" not in payload:
                raise ValueError(f"Строка {line_no}: нет обязательных полей submission_id/code")
            yield Submission(
                submission_id=str(payload["submission_id"]),
                assignment_id=str(payload.get("assignment_id", "default")),
                student_id=str(payload.get("student_id", "unknown")),
                attempt=int(payload.get("attempt", 1)),
                family_id=str(payload.get("family_id", payload["submission_id"])),
                code=str(payload["code"]),
                is_template=bool(payload.get("is_template", False)),
            )


def _extract_token_stream(code: str) -> list[tuple[str, int]]:
    """Извлекает значимые токены и номера строк, отбрасывая комментарии и форматирование."""
    if not code.strip():
        return []

    tokens: list[tuple[str, int]] = []
    reader = io.BytesIO(code.encode("utf-8")).readline
    try:
        for tok in tokenize.tokenize(reader):
            if tok.type in _SKIP_TOKEN_TYPES:
                continue
            if tok.type == tokenize.NAME:
                norm = f"NAME:{tok.string}"
            elif tok.type == tokenize.NUMBER:
                norm = "NUM"
            elif tok.type == tokenize.STRING:
                norm = "STR"
            elif tok.type == tokenize.OP:
                norm = f"OP:{tok.string}"
            else:
                norm = f"TOK:{tokenize.tok_name[tok.type]}"
            line_no = tok.start[0] if tok.start else 1
            tokens.append((norm, line_no))
    except tokenize.TokenError:
        # Возвращаем токены, собранные до синтаксического обрыва, чтобы работал token fallback
        pass
    return tokens


def _make_ngrams_with_lines(
    items: list[tuple[str, int]],
    n: int = 3,
) -> tuple[set[tuple[str, ...]], dict[tuple[str, ...], list[int]]]:
    ngrams: set[tuple[str, ...]] = set()
    spans: dict[tuple[str, ...], list[int]] = {}
    if len(items) < n:
        return ngrams, spans

    for i in range(len(items) - n + 1):
        chunk = items[i : i + n]
        gram = tuple(t for t, _ in chunk)
        ngrams.add(gram)
        lines = [ln for _, ln in chunk]
        existing = spans.get(gram)
        if existing is None:
            spans[gram] = [min(lines), max(lines)]
        else:
            existing[0] = min(existing[0], min(lines))
            existing[1] = max(existing[1], max(lines))
    return ngrams, spans


class _NameNormalizer(ast.NodeTransformer):
    """Нормализует имена переменных, аргументов и функций в порядке первого появления."""

    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}
        self._counter = 0

    def _norm(self, raw: str) -> str:
        if raw not in self._mapping:
            self._mapping[raw] = f"id_{self._counter}"
            self._counter += 1
        return self._mapping[raw]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._norm(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._norm(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._norm(node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = self._norm(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self._norm(node.name)
        return self.generic_visit(node)


def _extract_ast_ngrams_unoptimized(code: str, n: int = 3) -> tuple[set[tuple[str, ...]] | None, str | None]:
    """Базовый (неоптимизированный) вариант разбора AST через текстовый дамп ast.dump()."""
    if not code.strip():
        return set(), None
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    normalized = _NameNormalizer().visit(tree)
    dumped = ast.dump(normalized, include_attributes=False)
    parts = [p.strip("(), ") for p in dumped.replace("(", " ").replace(")", " ").split(",") if p.strip("(), ")]
    if len(parts) < n:
        return set(), None
    return {tuple(parts[i : i + n]) for i in range(len(parts) - n + 1)}, None


def _extract_ast_ngrams_fast(code: str, n: int = 3) -> tuple[set[tuple[str, ...]] | None, str | None]:
    """Оптимизированный вариант: обход ast.walk с нормализацией на лету без промежуточного ast.dump()."""
    if not code.strip():
        return set(), None
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    mapping: dict[str, str] = {}
    counter = 0

    def norm(name: str) -> str:
        nonlocal counter
        val = mapping.get(name)
        if val is None:
            val = f"id_{counter}"
            mapping[name] = val
            counter += 1
        return val

    seq: list[str] = []
    for node in ast.walk(tree):
        tname = type(node).__name__
        if isinstance(node, ast.Name):
            seq.append(f"{tname}:{norm(node.id)}")
        elif isinstance(node, ast.arg):
            seq.append(f"{tname}:{norm(node.arg)}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seq.append(f"{tname}:{norm(node.name)}")
        elif isinstance(node, ast.Constant):
            seq.append(f"Const:{type(node.value).__name__}")
        else:
            seq.append(tname)

    if len(seq) < n:
        return set(), None
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}, None


def extract_template_ngrams(template_code: str, ngram_size: int = 3) -> set[tuple[str, ...]]:
    """Строит множество 3-грамм из разрешённого шаблона задания."""
    tokens = _extract_token_stream(template_code)
    ngrams, _ = _make_ngrams_with_lines(tokens, n=ngram_size)
    return ngrams


def build_features(
    sub: Submission,
    template_ngrams: set[tuple[str, ...]] | None = None,
    ngram_size: int = 3,
    fast_ast: bool = True,
) -> CodeFeatures:
    """Конвейер validate -> features для одной сдачи."""
    tpl_set = template_ngrams or set()
    try:
        raw_tokens = _extract_token_stream(sub.code)
    except (tokenize.TokenError, IndentationError) as exc:
        return CodeFeatures(
            submission_id=sub.submission_id,
            assignment_id=sub.assignment_id,
            student_id=sub.student_id,
            attempt=sub.attempt,
            token_ngrams=set(),
            template_ngrams=set(),
            token_line_spans={},
            ast_ngrams=None,
            ast_error=f"TokenizeError: {exc}",
            fallback_used=True,
            is_empty=True,
        )

    token_ngrams, token_spans = _make_ngrams_with_lines(raw_tokens, n=ngram_size)
    tpl_overlap = token_ngrams & tpl_set
    extractor = _extract_ast_ngrams_fast if fast_ast else _extract_ast_ngrams_unoptimized
    ast_ngrams, ast_err = extractor(sub.code, n=ngram_size)
    fallback_used = ast_ngrams is None

    return CodeFeatures(
        submission_id=sub.submission_id,
        assignment_id=sub.assignment_id,
        student_id=sub.student_id,
        attempt=sub.attempt,
        token_ngrams=token_ngrams,
        template_ngrams=tpl_overlap,
        token_line_spans=token_spans,
        ast_ngrams=ast_ngrams,
        ast_error=ast_err,
        fallback_used=fallback_used,
        is_empty=(len(token_ngrams) == 0),
    )


def iter_feature_pipeline(
    jsonl_path: str | Path,
    templates_by_assignment: dict[str, set[tuple[str, ...]]] | None = None,
    ngram_size: int = 3,
    fast_ast: bool = True,
) -> Iterator[CodeFeatures]:
    """Ленивый конвейер read -> validate -> features."""
    tpl_map = templates_by_assignment or {}
    for sub in iter_submissions(jsonl_path):
        if sub.is_template:
            continue
        yield build_features(
            sub,
            template_ngrams=tpl_map.get(sub.assignment_id),
            ngram_size=ngram_size,
            fast_ast=fast_ast,
        )
