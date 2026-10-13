# ROADMAP — план доработок NPR AI/Real Detector

Обновлён: 2026-10-13. Журнал выполненных изменений: `changes.log` (gitignored).

Файл — долгосрочная память проекта (см. «Roadmap as long-term memory» в `AGENTS.md`):
появился план → сразу добавляем строку; сделано и проверено → `✅` с датой;
отменённое/отложенное → в «Бэклог» с пометкой, без удаления контекста.

## Статус: сделано ранее

Аудит и ремонт пайплайна (2026-09-10, коммиты `b2b0020`, `2f7354e`):

- Локальный кэш-режим данных (`prepare_local_datasets`, `scripts/download_data.py`), streaming — fallback.
- RTX 4090: `torch.quantile` → `kthvalue` (снятие капа batch≈40), bf16-AMP, cudnn.benchmark,
  slim-чекпоинты (~2 МБ) с ротацией, PerformanceMonitor.
- Инференс/оценка: ключ `output['logits']` (raw logits), evaluate через `create_eval_loader`,
  ONNX-guard, ASCII-print'ы.
- Cleanup: 12 мёртвых флагов удалены, аугментации подключены к пайплайну, доки синхронизированы.

---

## Этап A — Метрики и калибровка порога (код, верифицируемо локально)

**Цель:** воспроизводимая методика оценки для отчёта НИРС вместо фиксированного порога 0.5.

| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| A1 | `compute_metrics` + F1-оптимальный порог (`precision_recall_curve`), числовой ECE (15 bins), Brier score | `utils.py` | ✅ |
| A2 | Калибровка порога на **val**-сплите → `outputs/threshold.json`; confusion matrix и classification report на test с калиброванным порогом | `evaluate.py` | ✅ |
| A3 | ECE/Brier в лог + отчёт; флаг `--no-calibrate` (fallback на 0.5) | `evaluate.py` | ✅ |

## Этап B — Ablation-таблица (вклад веток)

**Цель:** численно показать вклад spatial / frequency / semantic веток и их комбинаций.

| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| B1 | Пробы (sklearn LogisticRegression на StandardScaler-нормированных эмбеддингах) поверх эмбеддингов обученного детектора: spatial / freq / semantic / sp+fr / se+fr / sp+fr+se / final. Пробы обучаются на val, оцениваются на test (AUC + F1@0.5) | `ablation.py` (новый) | ✅ |
| B2 | Экспорт `outputs/ablation.csv` + `outputs/ablation.md`, вывод таблицы в лог; CLI `python ablation.py --checkpoint ...` | `ablation.py` | ✅ |

## Этап C — Модельные мелочи (код)

| # | Задача | Файлы | Статус |
|---|--------|-------|--------|
| C1 | Фикс интеграции unfreeze CLIP: при `semantic_unfreeze_last_n_blocks>0` оптимизатор строится из `semantic_branch.get_param_groups()` (CLIP @ `semantic_lr`, остальное @ `learning_rate`); иначе поведение прежнее | `train.py` | ✅ |
| C2 | Новый параметр `training.semantic_lr: float = 1e-6` | `config.py` | ✅ |
| C3 | Удалён мёртвый флаг `freeze_semantic` (управление разморозкой — только через `semantic_unfreeze_last_n_blocks`) | `config.py` | ✅ |
| C4 | `CrossAttentionFusion` остаётся MLP-фьюжном (единственный токен → attention вырожден); класс/атрибут/ключи чекпоинтов не меняются | — | ✅ (без изменений) |

## Этап D — Эксперименты на RTX 4090 (запуск вручную, не код)

### D0 — Среда на сервере (подготовлено 2026-10-13)

Сервер: Windows 10 Pro, RTX 3090 (24 ГБ), драйвер 591.74. Вход по паролю — решение пользователя (ключи не настраивались). Реквизиты в репозиторий не записываем.

| # | Задача | Статус |
|---|--------|--------|
| D0.1 | Python 3.12.10 (`C:\Program Files\Python312`), Git 2.56, клон `dev`, `.venv` в корне проекта | ✅ 2026-10-13 |
| D0.2 | `torch 2.11.0+cu128` ставится **до** `requirements.txt` (иначе pip оставит CPU-сборку с PyPI). Нужен VC++ 2015–2022 x64 ≥14.44 (иначе `WinError 1114` при импорте `c10.dll`). `PYTHONUTF8=1` для пользователя (иначе падает печать не-ASCII в перенаправленный вывод) | ✅ 2026-10-13 |
| D0.3 | Веса CLIP (`models/clip-ViT-B-32`, 577 МБ) и локальный кэш 120+40 скопированы по SFTP | ✅ 2026-10-13 |
| D0.4 | Проверка на GPU: CUDA matmul, `python -m models.detector`, 1 эпоха train с настройками по умолчанию (workers=8, pin_memory, bf16-AMP) | ✅ 2026-10-13 |
| D0.5 | Баги, найденные проверкой: `models/detector.py` `__main__` — модель и входы на одном устройстве, `predict()` с одним изображением; `train.py` — `.float()` перед `.numpy()` для bf16-выходов (падало в `train_epoch` при AMP); `scripts/download_data.py` — `count_existing` принимал `str` и падал на любом запуске | ✅ 2026-10-13 |
| D0.6 | Полный кэш 20000+20000: токен пользователя сохранён на сервере (стандартный `login`, `~/.cache/huggingface`); лицензия ImageNet-1k принята; `download_data.py` выполнен. Итог: `real` 20039 (20000 новых + 40 старых имён `*_lNN.jpg`; 1 точный дубликат удалён), `ai` 20000 (см. D0.8). JPEG повреждённых нет; `prepare_datasets` → train 30000 / val 6000 / test 4000 | ✅ 2026-10-13 |
| D0.7 | `evaluate.py` и `ablation.py` на GPU на полном кэше: отработали (exit 0) на необученном чекпойнте — проверка конвейера, не метрик; артефакты удалены | ✅ 2026-10-13 |
| D0.8 | ⚠️ Известная проблема данных: 18 полностью чёрных кадров (512×512, max=0) в классе `ai`, метки неверны. Не удалены: при `ai < ai_max_samples` (20000) `prepare_local_datasets` догружает AI из HF без фильтра и вернёт их. Решение пользователя: оставить (~0.1% класса AI); фильтр и перекачка AI — в бэклоге | ✅ решено 2026-10-13 (оставить) |

Тестовые артефакты (`checkpoints/`, `outputs/`, `logs/`) на сервере удалены — D1 стартует с чистого состояния.

### D1–D5 — Прогоны

Все прогоны: локальный кэш 20k+20k, 50 epochs, batch 64, bf16-AMP, early stopping по `val_auc`. Запуск — на сервере RTX 3090 (см. D0).

| # | Прогон | Что фиксируем |
|---|--------|---------------|
| D1 | **Baseline** — полный трейн текущего конфига | `metrics.json`, кривые, лучший чекпоинт |
| D2 | Оценка baseline: `evaluate.py` (калибровка порога, ECE/Brier) + `ablation.py` | `threshold.json`, `ablation.csv/md`, ROC/PR/confusion/reliability/gate-графики |
| D3 | **Вариант: multi-scale LBP** — `lbp_radii=[1,2]` (код готов, включается конфигом; dim LBP 16→32) | сравнение с baseline по AUC/F1 |
| D4 | **Вариант: частичная разморозка CLIP** — `semantic_unfreeze_last_n_blocks=1` (после фикса C1, CLIP @ 1e-6) | сравнение с baseline; чекпоинт станет полным (~600 МБ) |
| D5 | (опционально) ablation gate-регуляризации: `gate_entropy_weight=0`, `gate_penalty_weight=0` | вырождение gate к semantic-only? |

---

## Бэклог (не в текущей сессии)

- **Robustness-тесты на test-time** — JPEG q=30–90, resize, гауссов шум на test-вопросах: цифры устойчивости для защиты.
- **OOD-обобщение** — тест на генераторах вне трейна (`gasstation/generated-images` — единственный источник; добавить SDXL/Flux/Midjourney-выборки).
- **Cascade-режим** — калибровка early-exit порога или вырезание (`NPRDetectorWithEarlyExit`, experimental).
- **Настоящий multi-token cross-attention** — требует токенов CLIP patch-секвенции, ломает совместимость fusion-части чекпоинтов; отдельная исследовательская задача.
- **ONNX-экспорт** — заблокирован FFT-операциями frequency-ветки.
- **Калибровка вероятностей** (Platt/изотоническая на val), если reliability diagram покажет систематический сдвиг.
- **Датасет-разнообразие** — расширение real-части другими источниками (ImageNet-only в трейне).
- **Фильтр пустых кадров при выгрузке** — `download_hf_images`: пропускать однородные кадры (std<1) и точные дубликаты; нужно для чистой перекачки AI (см. D0.8).

## Верификация кода (без GPU)

`python -m py_compile <файлы>`; `python scripts/smoke_train.py --epochs 2` — полный
CPU-сanity-прогон (кэш как есть, без догрузки HF, воркеры off); смоук
evaluate/ablation на мини-кэше с подменёнными `data.*_max_samples`; полный
эксперимент — только на 4090.
