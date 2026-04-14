# 📥 Инструкция по загрузке данных

## ImageNet-1k из HuggingFace

### 1️⃣ Принятие лицензии

Перед загрузкой необходимо принять лицензию на странице датасета:

🔗 **https://huggingface.co/datasets/ILSVRC/imagenet-1k**

Нажмите кнопку **"Agree"** или **"Accept Terms"** на странице датасета.

### 2️⃣ Аутентификация (если требуется)

```bash
# Установка HF CLI
pip install huggingface_hub

# Логин
huggingface-cli login
```

Получите токен на: https://huggingface.co/settings/tokens

### 3️⃣ Загрузка данных

#### Автоматическая загрузка при обучении

```bash
python train.py
```

Данные загрузятся автоматически при первом запуске.

#### Ручная загрузка заранее

```bash
python scripts/download_data.py --imagenet-samples 15000
```

### 4️⃣ Структура сохранённых данных

```
data/
├── real/
│   └── imagenet/
│       ├── real_000001_l234.png
│       ├── real_000002_l567.png
│       └── ...
└── ai_generated/
    ├── ai_000000.png
    ├── ai_000001.png
    └── ...
```

### 5️⃣ Кэш HuggingFace

Данные кэшируются в:

- **Windows**: `%USERPROFILE%\.cache\huggingface\datasets\`
- **Linux/Mac**: `~/.cache/huggingface/datasets/`

Для очистки кэша:

```bash
huggingface-cli logout
rm -rf ~/.cache/huggingface/datasets  # Linux/Mac
# или
rmdir /s %USERPROFILE%\.cache\huggingface\datasets  # Windows
```

### 6️⃣ Настройка в config.py

```python
# В config.py → DataConfig:

# ImageNet-1k
hf_imagenet_name: str = "ILSVRC/imagenet-1k"
hf_imagenet_split: str = "validation"  # или 'train'
hf_imagenet_max_samples: int = 15000

# AI generated
hf_dataset_name: str = "gasstation/generated-images"
hf_dataset_split: str = "train"
```

### ⚠️ Возможные проблемы

| Проблема | Решение |
|----------|---------|
| **Dataset not found** | Проверьте что приняли лицензию |
| **Authentication required** | Выполните `huggingface-cli login` |
| **Not enough disk space** | Требуется ~10-20 GB для ImageNet-1k |
| **Slow download** | Используйте `HF_HUB_ENABLE_HF_TRANSFER=1` |

### 🔗 Ссылки

- ImageNet-1k: https://huggingface.co/datasets/ILSVRC/imagenet-1k
- HuggingFace Datasets: https://huggingface.co/docs/datasets/
- HF CLI: https://huggingface.co/docs/huggingface_hub/
