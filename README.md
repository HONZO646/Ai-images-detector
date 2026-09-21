# 🎯 NPR-based AI Image Detector

Многоэтапный детектор ИИ-сгенерированных изображений на основе анализа **пространственных (spatial)** и **частотных (frequency)** признаков с возможностью end-to-end обучения.

---

## 📋 Содержание

- [Архитектура](#архитектура)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Обучение](#обучение)
- [Инференс](#инференс)
- [Оценка](#оценка)
- [Структура проекта](#структура-проекта)

---

## 🔑 Архитектура

```
┌──────────────────────────────────────────────────┐
│  INPUT: Image (H×W×3)                             │
├──────────────────────────────────────────────────┤
│ 1. Spatial Branch                                 │
│    • NPR 3×3: 8 направлений × 5 статистик = 40   │
│    • Gradients: Sobel (dx, dy, magnitude)        │
│    • LBP: differentiable Local Binary Patterns   │
│    → Fusion → 128-dim embedding                   │
│                                                   │
│ 2. Frequency Branch                               │
│    • DWT (level 2): HH/HL/LH поддиапазоны        │
│    • DCT: AC-коэффициенты 8×8 блоков             │
│    • FFT: радиальные/угловые профили спектра     │
│    → Frequency Attention → 128-dim embedding      │
│                                                   │
│ 3. Semantic Branch (отдельная!)                   │
│    • Frozen CLIP/ViT → projector → 128-dim       │
│    • Опционально: inconsistency head              │
│                                                   │
│ FUSION: MLP Fusion + Adaptive Gating              │
 │    • Конкатенация semantic + spatial+freq          │
 │    • Динамическое взвешивание веток                │
│                                                   │
│ OUTPUT: P(AI) ∈ [0, 1] + uncertainty estimate     │
└──────────────────────────────────────────────────┘
```

### Почему такая архитектура?

| Решение | Обоснование |
|---------|-------------|
| **NPR 3×3** | Ловит checkerboard-артефакты, аномалии локальных корреляций |
| **MLP классификатор** | Вход — агрегированные статистики, CNN не добавит индукции |
| **Отдельная семантическая ветка** | Семантика (что?) и артефакты (как?) — разные уровни абстракции |
| **MLP fusion + gating** | Модель сама решает, какому сигналу доверять |
| **Differentiable всё** | End-to-end обучение: градиенты текут ко всем модулям |

---

## 🛠️ Установка

### Требования

```bash
Python >= 3.10 (3.12 рекомендован для CUDA-сборок PyTorch)
PyTorch >= 2.0.0
```

### Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Установка пакетов
pip install -r requirements.txt

# Для обучения на GPU — CUDA-сборка PyTorch (см. раздел «Развёртывание на GPU»)
```

### Основные зависимости

- `torch`, `torchvision` — нейросетевые модули
- `PyWavelets` — вейвлет-преобразования
- `sentence-transformers`, `transformers` — CLIP/ViT features
- `datasets` — загрузка данных из HuggingFace
- `scikit-learn` — метрики
- `gradio` — веб-интерфейс

---

## 🚀 Быстрый старт

### Инференс одного изображения

```bash
python inference.py \
  --image test-ai-1.jpg \
  --checkpoint checkpoints/best_model.pt \
  --device auto
```

### Инференс батча

```bash
python inference.py \
  --batch test-ai-1.jpg test-real-1.jpg test-ai-2.jpg \
  --checkpoint checkpoints/best_model.pt \
  --threshold 0.5
```

### Cascade mode (быстрый режим, экспериментальный)

```bash
python inference.py \
  --dir /path/to/images/ \
  --checkpoint checkpoints/best_model.pt \
  --cascade \
  --cascade-threshold 0.7
```

> ⚠️ Cascade-режим использует spatial embedding через общий classifier
> (BatchNorm-статистики откалиброваны на fused embedding), поэтому быстрый
> предикт смещён и порог некалиброван. По умолчанию выключен.
> ONNX-экспорт не поддерживается: частотная ветка использует `torch.fft`,
> у которого нет ONNX-операторов.

---

## 📚 Обучение

### Подготовка данных

**📁 Локальный кэш (рекомендуется для GPU)** — данные выгружаются на диск один раз:

```bash
# 1. Принять лицензию ImageNet-1k: https://huggingface.co/datasets/ILSVRC/imagenet-1k
# 2. huggingface-cli login
# 3. Выгрузить кэш (реально качаются только нужные сэмплы, streaming внутри)
python scripts/download_data.py                # 20000+20000 из config.py
# или меньше:
python scripts/download_data.py --real-samples 5000 --ai-samples 5000

# 4. Обучение с локальным кэшем (в config.py data.streaming = False — дефолт)
python train.py
```

Кэш хранится в `data/real/imagenet/` и `data/ai_generated/` (gitignored).

**🚀 Streaming mode (fallback)** — данные НЕ сохраняются на диск, подгружаются из HuggingFace по мере необходимости:

```python
# В config.py: data.streaming = True
```

Требует постоянного подключения к сети и CPU-декодирования — GPU простаивает.

Источники данных (настраиваются в `config.py → DataConfig`):

1. **Real изображения**: [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
2. **AI изображения**: HuggingFace `gasstation/generated-images` (только `media_type == 'synthetic'`)

> ⚠️ **Важно**: Для доступа к ImageNet-1k на HuggingFace необходимо принять лицензию на странице датасета и выполнить `huggingface-cli login`.

### Запуск обучения

```bash
python train.py
```

### Гиперпараметры (в `config.py`)

```python
BATCH_SIZE = 64   # RTX 4090 24GB: старт 64, после замера VRAM можно 96-128
LR = 3e-4 (AdamW, weight_decay=1e-4)
EPOCHS = 50 (early stop по val_auc, patience=8)
SCHEDULER = cosine с warmup 3 эпохи (min_lr=1e-6)
AMP = bfloat16 (на CPU автоматически отключается)
```

### Аугментации (train, симметрично обоим классам)

- Downscale+upscale (bilinear): prob 0.5, масштаб [0.5, 0.9] — имитация AI-апсемплинга
- Sharpen: prob 0.3, фактор 1.5–2.0 — имитация AI over-sharpening
- JPEG каскадом 1–3 раза: prob 0.4, quality [40, 70]
- Color jitter (brightness/contrast/saturation): prob 0.4, ±8%
- Gaussian blur: prob 0.2, σ ∈ [0.1, 2.0]

---

## 📊 Оценка

### Запуск evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Порог классификации калибруется по F1-max на **val**-сплите и сохраняется в
`outputs/threshold.json`; метрики test считаются с этим порогом. Флаг
`--no-calibrate` возвращает фиксированный порог 0.5. Дополнительно к метрикам
считаются ECE (калибровка вероятностей) и Brier score.

### Ablation (вклад веток)

```bash
python ablation.py --checkpoint checkpoints/best_model.pt
```

Линейные пробы (StandardScaler + LogisticRegression) поверх эмбеддингов
spatial / frequency / semantic / их комбинаций / final fusion. Пробы обучаются
на val, оцениваются на test → `outputs/ablation.csv` + `ablation.md`.

### Генерируемые графики

- ROC кривая
- Confusion matrix
- Precision-Recall curve
- Prediction distribution
- Reliability diagram
- Gate weights visualization

### Ожидаемые метрики

| Метрика | Ожидаемый диапазон |
|---------|-------------------|
| Train Accuracy | 92–96% |
| Val AUC-ROC | 0.88–0.94 |
| Test Accuracy | 85–91% |
| Inference time | <10 ms/image (CPU) |

---

## 🗂️ Структура проекта

```
project_root/
├── config.py                 # Все гиперпараметры в dataclass
├── train.py                  # Полный цикл обучения (+ --ui для веб-интерфейса)
├── evaluate.py               # Метрики + калибровка порога на val + графики
├── ablation.py               # Вклад веток: пробы на эмбеддингах
├── inference.py              # Инференс (single/batch/dir/cascade)
├── gradio_ui.py              # Веб-интерфейс (0.0.0.0:7860)
├── performance_monitor.py    # Throughput/VRAM мониторинг (подключён в Trainer)
├── test_dataset.py           # Визуализация одного батча датасета
├── utils.py                  # Logger, checkpoint, metrics, ECE/Brier
├── requirements.txt          # Зависимости
├── README.md                 # Документация
├── AGENTS.md                 # Инструкция для AI-агентов (English)
├── ROADMAP.md                # План доработок и 4090-экспериментов
│
├── data/
│   ├── __init__.py
│   ├── dataset.py            # NPRDataset + Streaming + prepare_datasets()
│   ├── real/imagenet/        # Локальный кэш real (gitignored)
│   └── ai_generated/         # Локальный кэш AI (gitignored)
│
├── models/
│   ├── spatial_branch.py     # NPR + Sobel + LBP
│   ├── frequency_branch.py   # DWT + DCT + FFT
│   ├── semantic_branch.py    # Frozen CLIP/ViT
│   ├── fusion.py             # MLP-фьюжн (исторически назван Cross-Attention) + Gating
│   ├── detector.py           # Main detector class
│   └── clip-ViT-B-32/        # Локальная копия CLIP (weights не в git)
│
├── scripts/
│   ├── download_data.py      # Выгрузка кэша данных
│   ├── setup_hf.py           # Проверка HF-доступа
│   └── DATA_SETUP.md
│
├── checkpoints/              # Сохранённые модели (без CLIP-весов)
├── outputs/                  # Графики и метрики
└── logs/                     # Логи обучения + performance_report.json
```

---

## ⚠️ Важные ограничения

| Проблема | Решение |
|----------|---------|
| **Постобработка (JPEG, blur)** | Аугментации при обучении + frequency branch |
| **Новые генераторы (OOD)** | Frozen CLIP даёт zero-shot обобщение |
| **Ложные срабатывания на арте** | Gate-механизм снижает weight semantic-ветки |
| **Дисбаланс в продакшене** | Калибровать пороги на валидации |
| **Интерпретируемость** | Per-branch scores + attention weights |

---

## 💡 Примеры использования

### Python API

# Предсказание (модель возвращает logits под ключом 'logits')
with torch.no_grad():
    output = model(gray, rgb)
    logits = output['logits']
    prob_ai = torch.sigmoid(logits).item()
    print(f"P(AI) = {prob_ai:.4f}")
```

### Cascade inference (экспериментально)

```python
from models.detector import NPRDetectorWithEarlyExit

model = NPRDetectorWithEarlyExit(config.model, early_exit_threshold=0.7)
result = model.predict_cascade(gray, rgb)

if result['early_exit']:
    print("Fast prediction ✓")
else:
    print("Full prediction")
```

---

## 🖥️ Развёртывание на GPU (RTX 4090, 24 ГБ)

### 1. Окружение (Python 3.12 рекомендован — стабильные CUDA-колёса)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. CUDA-сборка PyTorch

```bash
# Проверить версию CUDA-драйвера:
nvidia-smi   # правый верхний угол: "CUDA Version: 12.x"

# Для драйвера с CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Проверка:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Данные

```bash
huggingface-cli login          # токен: https://huggingface.co/settings/tokens
python scripts/setup_hf.py     # проверка доступа к ImageNet-1k
python scripts/download_data.py   # кэш 20000+20000 (в data/, ~10-15 ГБ)
```

### 4. Подбор batch size под VRAM (опционально, есть в performance_monitor)

```bash
python -c "from config import Config; from models.detector import NPRDetector; from performance_monitor import suggest_batch_size; c=Config(); m=NPRDetector(c.model); print(suggest_batch_size(m, (1,1,640,640), (1,3,640,640)))"
```

### 5. Обучение

```bash
python train.py   # в config.py: use_amp=True (bfloat16), num_workers=8, batch_size=64
```

- Начните с `batch_size=64`, посмотрите `[Perf] Peak VRAM` в логе
- При запасе (<18 ГБ) поднимайте до 96–128
- `reproducible=True` в config включает cudnn.deterministic (медленнее) — только для отладки
- Мониторинг: `logs/performance_report.json`, прогресс `[Perf]` в логе обучения

Ожидаемая ориентация: при 640×640 и bf16-AMP batch 64 занимает ~6–10 ГБ VRAM;
на batch 96–128 — 12–18 ГБ.
