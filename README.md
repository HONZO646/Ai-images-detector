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
│ FUSION: Cross-Attention + Adaptive Gating         │
│    • Query: semantic, Key/Value: spatial+freq     │
│    • Динамическое взвешивание веток               │
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
| **Cross-attention fusion** | Модель сама решает, какому сигналу доверять |
| **Differentiable всё** | End-to-end обучение: градиенты текут ко всем модулям |

---

## 🛠️ Установка

### Требования

```bash
Python >= 3.8
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
```

### Основные зависимости

- `torch`, `torchvision` — нейросетевые модули
- `PyWavelets` — вейвлет-преобразования
- `sentence-transformers` или `clip` — CLIP/ViT features
- `scikit-learn` — метрики
- `matplotlib`, `seaborn` — визуализация

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

### Cascade mode (быстрый режим)

```bash
python inference.py \
  --dir /path/to/images/ \
  --checkpoint checkpoints/best_model.pt \
  --cascade \
  --cascade-threshold 0.7
```

---

## 📚 Обучение

### Подготовка данных

**🚀 Streaming mode (рекомендуется)** — данные НЕ загружаются на диск!

Данные подгружаются по мере необходимости из HuggingFace, экономя место:

```python
# В config.py по умолчанию:
data.streaming = True  # ✅ Данные НЕ сохраняются локально
```

**📁 Локальное сохранение** — если нужно сохранить данные на диск:

```python
# В config.py:
data.streaming = False  # Данные будут сохранены в data/
```

Источники данных:

1. **Real изображения**: [ImageNet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) (validation split)
   - 15,000 изображений из ILSVRC/imagenet-1k
   - Загружается автоматически через `datasets` library

2. **AI изображения**: HuggingFace `gasstation/generated-images`
   - Можно изменить в `config.py`: `hf_dataset_name`

```bash
# Автоматическая загрузка данных произойдёт при первом запуске:
python train.py

# Или можно загрузить данные заранее:
python scripts/download_data.py --imagenet-samples 15000
```

> ⚠️ **Важно**: Для доступа к ImageNet-1k на HuggingFace необходимо принять лицензию на странице датасета.

### Запуск обучения

```bash
python train.py
```

### Гиперпараметры (в `config.py`)

```python
BATCH_SIZE = 256
LR = 1e-3 (AdamW, weight_decay=1e-4)
EPOCHS = 50 (early stop по val_auc, patience=8)
SCHEDULER = ReduceLROnPlateau(mode='max', factor=0.5, patience=4)
```

### Аугментации (только train)

- JPEG компрессия: quality ∈ [70, 100]
- Gaussian blur: radius ∈ {0.5, 1.0, 1.5}
- Аддитивный шум: σ=0.01
- Контраст: factor ∈ [0.9, 1.1]

---

## 📊 Оценка

### Запуск evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

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
├── npr.py                    # Original standalone модуль
├── freq_characteristics_script.py  # FFT analysis script
├── train.py                  # Полный цикл обучения
├── evaluate.py               # Метрики + графики
├── inference.py              # Инференс + ONNX export
├── utils.py                  # Logger, checkpoint, metrics
├── requirements.txt          # Зависимости
├── README.md                 # Документация
│
├── data/
│   ├── __init__.py
│   └── dataset.py           # NPRDataset + prepare_datasets()
│
├── models/
│   ├── __init__.py
│   ├── spatial_branch.py    # NPR + Sobel + LBP
│   ├── frequency_branch.py  # DWT + DCT + FFT
│   ├── semantic_branch.py   # Frozen CLIP/ViT
│   ├── fusion.py            # Cross-Attention + Gating
│   └── detector.py          # Main detector class
│
├── checkpoints/             # Сохранённые модели
├── outputs/                 # Графики и метрики
└── logs/                    # Логи обучения
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

```python
from config import Config
from models.detector import create_detector
import torch
from PIL import Image
import numpy as np

# Загрузка модели
config = Config()
model = create_detector(config.model, pretrained_path='checkpoints/best_model.pt')
model.eval()

# Препроцессинг
img = Image.open('test.jpg').convert('RGB')
rgb = torch.from_numpy(np.array(img)/255.0).permute(2,0,1).unsqueeze(0)
gray = rgb.mean(dim=1, keepdim=True)

# Предсказание (модель возвращает logits, применяем sigmoid для вероятности)
with torch.no_grad():
    output = model(gray, rgb)
    logits = output['probability']
    prob_ai = torch.sigmoid(logits).item()
    print(f"P(AI) = {prob_ai:.4f}")
```

### Cascade inference

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

## 📝 Лицензия

MIT License

---

## 👥 Авторы

Проект разработан в рамках НИРС 10 семестр, BMSTU

---

## 📧 Контакты

По вопросам: honzo@example.com
