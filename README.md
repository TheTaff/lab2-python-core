# ЛР 2. Анализ сходства и оценка качества

Траектория: **Python Core**, основной уровень.

Расширение пакета `labcheck` потоковым чтением JSONL, извлечением признаков кода через `tokenize` и `ast`, попарным сравнением по коэффициенту Жаккара на 3-граммах, учётом шаблонов и повторных загрузок, а также воспроизводимой оценкой качества и производительности. Код студентов не выполняется.

## Установка

Нужен Python 3.10 или новее. Сторонних зависимостей нет (используется стандартная библиотека; при наличии можно запускать тесты через `pytest` или `unittest`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Запуск

```bash
# тесты
PYTHONPATH=src python -m unittest discover -s tests -v
# или через pytest:
pytest

# оценка порогов на tune, метрики на eval и замеры производительности (100, 1000, 5000)
PYTHONPATH=src python -m labcheck.cli --submissions data/corpus_submissions.jsonl --pairs data/corpus_pairs.jsonl --out data/eval_summary.json
```

## Формат корпуса и пар

Одна строка JSONL в корпусе сдач — одна работа или шаблон задания.

| Поле | Тип | Назначение |
|---|---|---|
| `submission_id` | строка | идентификатор сдачи |
| `assignment_id` | строка | идентификатор задания |
| `student_id` | строка | идентификатор студента |
| `attempt` | число | номер попытки/версии |
| `family_id` | строка | семейство исходного решения (без пересечений между `tune` и `eval`) |
| `code` | строка | исходный код на Python |
| `is_template` | булево | признак разрешённого шаблона (необязательное, по умолчанию `false`) |

Семейства решений разделены между выборками `tune` и `eval` в `data/corpus_pairs.jsonl`. Данные в `data/` учебные, реальных работ студентов в них нет.

## Устройство

```text
src/labcheck/
  models.py      dataclass: Submission, CodeFeatures, PairComparison, EvidenceSpan
  pipeline.py    потоковое чтение JSONL, токенизация, шаблоны, AST-нормализация и быстрый обход
  similarity.py  Jaccard по 3-граммам, правила сравнения, откат при SyntaxError, evidence
  eval_bench.py  подбор порога на tune, метрики на eval, cProfile, tracemalloc и бенчмарк
  cli.py         запуск оценки качества и профилирования с сохранением отчёта в JSON
tests/
  test_lab2.py   контрактные и модульные тесты
REPORT.md        отчёт с метриками, разбором ошибок и замерами времени/памяти
AI_USAGE.md      отметка об использовании ИИ
```

### Правила сравнения и граничные случаи

| Кейс | Поведение |
|---|---|
| Пустое объединение 3-грамм | Возвращается `score = None` (`null` в JSON) с причиной `insufficient_data_empty_union` |
| Разрешённый шаблон (`template_only`) | Общие 3-граммы шаблона выделяются отдельно; совпадение только по шаблону не выставляет `review_required` |
| `SyntaxError` в AST-анализе | Откат на токенный baseline с `fallback_used = True` и текстом ошибки в `reason` |
| `self-match` и повторные сдачи | `self-match` исключён; повторные версии одного студента в одном `assignment_id` учитываются отдельно |