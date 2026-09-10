# 📥 Подготовка данных (локальный кэш)

Основной режим обучения — **локальный кэш**: изображения выгружаются на диск один
раз, дальше обучение идет с shuffle и полной загрузкой GPU.

## 1️⃣ Лицензия и логин

Перед загрузкой ImageNet-1k необходимо принять лицензию на странице датасета:

🔗 **https://huggingface.co/datasets/ILSVRC/imagenet-1k**

Нажмите **"Agree"** / **"Accept Terms"**, затем:

```bash
huggingface-cli login          # токен: https://huggingface.co/settings/tokens
python scripts/setup_hf.py     # проверка доступа
```

## 2️⃣ Выгрузка кэша

```bash
# Полный кэш (20000 real + 20000 ai из config.py)
python scripts/download_data.py

# Уменьшенный кэш (быстрее, для отладки)
python scripts/download_data.py --real-samples 5000 --ai-samples 5000

# Свой каталог
python scripts/download_data.py --data-root data
```

Загрузка идёт потоково (качаются только нужные сэмплы). Повторный запуск
пропускает полные классы (resume).

## 3️⃣ Структура кэша

```
data/
├── real/
│   └── imagenet/
│       ├── real_000000.jpg
│       └── ...
└── ai_generated/
    ├── ai_000000.jpg       # только media_type == 'synthetic'
    └── ...
```

Пути настраиваются в `config.py → DataConfig` (`data_root`, `real_data_path`,
`ai_data_path`). Каталог `data/` gitignored.

## 4️⃣ Обучение с кэшем

В `config.py` (по умолчанию):

```python
data.streaming = False   # локальный кэш
```

```bash
python train.py
```

`prepare_local_datasets` балансирует классы 1:1 (min от каждого), перемешивает
с фиксированным сидом и делит на train/val/test по `data.{train,val,test}_ratio`.

Если кэш неполон — недостающее автоматически догружается из HuggingFace при
первом запуске.

## ⚠️ Возможные проблемы

| Проблема | Решение |
|----------|---------|
| **Dataset not found** | Примите лицензию на странице ImageNet-1k |
| **Authentication required** | `huggingface-cli login` |
| **Slow download** | `HF_HUB_ENABLE_HF_TRANSFER=1` |
| **Пустой кэш при streaming=False** | Запустите `python scripts/download_data.py` |

## 🔗 Streaming fallback

Если диск ограничен — включите `data.streaming = True` в `config.py`
(см. `STREAMING_MODE.md`). Учтите: GPU будет простаивать в ожидании сети и
CPU-декодирования.