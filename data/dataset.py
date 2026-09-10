"""
Dataset модуль для NPR AI/Real Classifier
Загрузка, препроцессинг и аугментация изображений
"""
import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose, GaussianBlur as TVGaussianBlur
from torch.utils.data import Dataset, DataLoader, Subset, IterableDataset, get_worker_info
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

def process_image(img_rgb: Image.Image, transform: Optional[Callable] = None, target_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process image: apply transform, resize, convert to tensors"""
    if transform:
        img_rgb = transform(img_rgb)
    
    img_rgb = img_rgb.resize((target_size, target_size), Image.LANCZOS)
    
    rgb_tensor = TF.to_tensor(img_rgb)
    gray_tensor = TF.to_tensor(img_rgb.convert('L'))
    
    return gray_tensor, rgb_tensor


class NPRDataset(Dataset):
    """
    Dataset для NPR детектора
    Возвращает пару: (grayscale_tensor, rgb_tensor, label)
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[callable] = None,
                 target_size: int = 256,
                 aug_factor: int = 1):
        """
        image_paths: список путей к изображениям
        labels: список меток (0=real, 1=ai)
        transform: функция аугментации
        target_size: размер для ресайза
        aug_factor: коэффициент умножения аугментаций
        """
        assert len(image_paths) == len(labels), "Длины paths и labels должны совпадать"
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        self.aug_factor = aug_factor
        
        logger.info(f"Создан dataset: {len(image_paths)} изображений")
        logger.info(f"  Real (0): {labels.count(0)}")
        logger.info(f"  AI (1):   {labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.image_paths) * self.aug_factor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Вычисляем реальный индекс исходного файла
        real_idx = idx // self.aug_factor
        
        # Загрузка изображения
        img_path = self.image_paths[real_idx]
        label = self.labels[real_idx]
        
        try:
            img_rgb = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Ошибка загрузки {img_path}: {e}")
            # Возвращаем чёрное изображение как fallback
            img_rgb = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # Обработка изображения через общую функцию
        gray_tensor, rgb_tensor = process_image(img_rgb, self.transform, self.target_size)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return gray_tensor, rgb_tensor, label_tensor


class CascadeJpegCompression:
    """Аугментация: Каскадное JPEG-сжатие (от 1 до 3 последовательных сжатий)"""
    
    def __init__(self, quality_range: Tuple[int, int] = (30, 100), prob: float = 0.5):
        self.quality_range = quality_range
        self.prob = prob
        
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
            
        import io
        # Случайное количество последовательных сжатий от 1 до 3
        num_compressions = random.randint(1, 3)
        
        for _ in range(num_compressions):
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
            
        return img


class ColorJitter:
    """Варьирование цвета для маскировки стиля датасета"""

    def __init__(self,
                 brightness_range: float = 0.08,
                 contrast_range: float = 0.08,
                 saturation_range: float = 0.08,
                 prob: float = 0.4):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img

        brightness = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
        contrast = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
        saturation = 1.0 + random.uniform(-self.saturation_range, self.saturation_range)

        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)

        return img


class DownscaleUpscale:
    """Аугментация: downscale + upscale — имитация AI-апсемплинга"""

    def __init__(self, size_range: Tuple[float, float] = (0.5, 0.9), prob: float = 0.5):
        self.size_range = size_range
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img

        w, h = img.size
        scale = random.uniform(*self.size_range)
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class ProbabilisticGaussianBlur:
    """Аугментация: Gaussian blur с вероятностью применения"""

    def __init__(self, prob: float = 0.2, kernel_size: int = 3,
                 sigma: Tuple[float, float] = (0.1, 2.0)):
        self.prob = prob
        self.blur = TVGaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        return self.blur(img)


class Sharpen:
    """Аугментация: имитация AI over-sharpening.
    Фактор = 1.0 + U(strength_range) — усиление резкости (ImageEnhance.Sharpness)"""

    def __init__(self, strength_range: Tuple[float, float] = (0.5, 1.0), prob: float = 0.3):
        self.strength_range = strength_range
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        factor = 1.0 + random.uniform(*self.strength_range)
        return ImageEnhance.Sharpness(img).enhance(factor)


def create_augmentation_pipeline(aug) -> Optional[Compose]:
    """Реалистичный pipeline аугментаций из AugmentationConfig (config.py)"""
    if not getattr(aug, 'enable', True):
        return None

    transforms = [
        DownscaleUpscale(size_range=aug.downscale_range, prob=aug.downscale_prob),
        Sharpen(strength_range=aug.sharpen_range, prob=aug.sharpen_prob),
        ColorJitter(
            brightness_range=aug.color_strength,
            contrast_range=aug.color_strength,
            saturation_range=aug.color_strength,
            prob=aug.color_prob
        ),
        ProbabilisticGaussianBlur(prob=aug.blur_prob),
        CascadeJpegCompression(quality_range=aug.jpeg_quality_range, prob=aug.jpeg_prob),
    ]
    return Compose(transforms)


class StreamingNPRDataset(IterableDataset):
    """
    Streaming Dataset для NPR детектора
    Данные загружаются по мере необходимости, не занимают диск
    Наследуется от IterableDataset для эффективной работы с DataLoader
    """
    
    def __init__(self,
                 hf_dataset,
                 label: int,
                 transform: Optional[callable] = None,
                 target_size: int = 256,
                 aug_factor: int = 1):
        """
        hf_dataset: HuggingFace dataset в streaming mode
        label: метка класса (0=real, 1=ai)
        transform: функция аугментации
        target_size: размер для ресайза
        max_samples: ограничение количества (None = все)
        aug_factor: коэффициент умножения аугментаций
        """
        super().__init__()
        self.hf_dataset = hf_dataset
        self.label = label
        self.transform = transform
        self.target_size = target_size
        self.aug_factor = aug_factor
        
        print(f"Создан Streaming dataset: label={label}, aug_factor={aug_factor}")
    
    def __iter__(self):
        """Итератор для streaming mode — бесконечный цикл"""
        worker_info = get_worker_info()
        ds = self.hf_dataset

        if worker_info is not None:
            ds = ds.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )
            
        for item in ds:
            try:
                img_rgb = item["image"].convert("RGB")
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения: {e}")
                img_rgb = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))

            label_tensor = torch.tensor(self.label, dtype=torch.float32)

            for _ in range(self.aug_factor):
                gray_tensor, rgb_tensor = process_image(
                    img_rgb, self.transform, self.target_size
                )
                yield gray_tensor, rgb_tensor, label_tensor
    
    def _rgb_to_grayscale_tensor(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL Image → [1, H, W] grayscale tensor"""
        img_gray = img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0)
        return tensor


class CombinedStreamingDataset(IterableDataset):
    """
    Комбинированный streaming dataset (real + ai)
    Балансировка классов автоматически
    Чередуёт real и ai примеры для баланса
    """
    
    def __init__(self, real_dataset, ai_dataset):
        super().__init__()
        self.real_ds = real_dataset
        self.ai_ds = ai_dataset

    def __iter__(self):
        real_iter = iter(self.real_ds)
        ai_iter = iter(self.ai_ds)

        while True:
            try:
                yield next(real_iter)
                yield next(ai_iter)
            except StopIteration:
                break

def filter_is_synthetic(item):
    return item.get("media_type", None) == "synthetic"

def create_streaming_datasets(config) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Создание train/val/test streaming split'ов
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")
    
    print("ЗАГРУЗКА ДАННЫХ В STREAMING MODE")

    # ImageNet-1k (real)
    print(f"\n📥 {config.data.hf_imagenet_name} (split={config.data.hf_imagenet_split})...")
    real_full_ds = load_dataset(
        config.data.hf_imagenet_name,
        split=config.data.hf_imagenet_split,
        streaming=True  # ← STREAMING MODE
    )

    # AI generated (оставляем только fully synthetic, исключая semisynthetic)
    print(f"📥 AI dataset: {config.data.hf_dataset_name}...")
    ai_full_ds = load_dataset(
        config.data.hf_dataset_name,
        split=config.data.hf_dataset_split,
        streaming=True
    )
    ai_full_ds = ai_full_ds.filter(filter_is_synthetic)
    print("✅ AI streaming dataset отфильтрован: media_type == 'synthetic'")

    real_full_ds = real_full_ds.shuffle(seed=config.training.seed, buffer_size=1000)
    ai_full_ds = ai_full_ds.shuffle(seed=config.training.seed, buffer_size=1000)

    # Единый размер на класс: баланс 1:1
    n_per_class = min(config.data.real_max_samples, config.data.ai_max_samples)
    train_size = int(n_per_class * config.data.train_ratio)
    val_size = int(n_per_class * config.data.val_ratio)
    test_size = int(n_per_class * config.data.test_ratio)

    print(f"На класс: train={train_size}, val={val_size}, test={test_size}")

    real_train_ds = real_full_ds.take(train_size)
    real_val_ds = real_full_ds.skip(train_size).take(val_size)
    # ВАЖНО: test ограничен take(), иначе итерация уйдёт в весь ImageNet train
    real_test_ds = real_full_ds.skip(train_size + val_size).take(test_size)

    ai_train_ds = ai_full_ds.take(train_size)
    ai_val_ds = ai_full_ds.skip(train_size).take(val_size)
    ai_test_ds = ai_full_ds.skip(train_size + val_size).take(test_size)

    # Создание streaming wrapper'ов с применением аугментаций к обоим классам
    train_transform = create_augmentation_pipeline(config.augmentation)
    
    real_train_stream = StreamingNPRDataset(
        hf_dataset=real_train_ds,
        label=0,
        transform=train_transform,
        target_size=config.data.image_size,
    )
    
    ai_train_stream = StreamingNPRDataset(
        hf_dataset=ai_train_ds,
        label=1,
        transform=train_transform,
        target_size=config.data.image_size,
    )
    
    # Комбинированный dataset
    print('---Combined Train dataset---')
    train_dataset = CombinedStreamingDataset(real_train_stream, ai_train_stream)

    # Валидация/тест: формируем сбалансированные наборы из обоих классов
    val_real_stream = StreamingNPRDataset(
        hf_dataset=real_val_ds,
        label=0,
        transform=None,
        target_size=config.data.image_size,
    )

    val_ai_stream = StreamingNPRDataset(
        hf_dataset=ai_val_ds,
        label=1,
        transform=None,
        target_size=config.data.image_size,
    )

    test_real_stream = StreamingNPRDataset(
        hf_dataset=real_test_ds,
        label=0,
        transform=None,
        target_size=config.data.image_size,
    )

    test_ai_stream = StreamingNPRDataset(
        hf_dataset=ai_test_ds,
        label=1,
        transform=None,
        target_size=config.data.image_size,
    )

    val_dataset = CombinedStreamingDataset(val_real_stream, val_ai_stream)
    test_dataset = CombinedStreamingDataset(test_real_stream, test_ai_stream)
    
    return train_dataset, val_dataset, test_dataset


def download_hf_images(dataset_name: str,
                       split: str,
                       output_dir: str,
                       prefix: str,
                       max_samples: int = 20000,
                       streaming: bool = True,
                       filter_fn: Callable = None) -> List[str]:
    """
    Выгрузка изображений из HuggingFace dataset на диск (локальный кэш).

    Args:
        dataset_name: имя датасета на HuggingFace
        split: split датасета ('train', 'validation', etc.)
        output_dir: директория для сохранения
        prefix: префикс для имен файлов ('real' или 'ai')
        max_samples: максимальное количество
        streaming: True = качать потоково (только нужные шардлы), False = целиком
        filter_fn: функция фильтрации (принимает item, возвращает bool)

    Returns:
        List[str]: пути к сохраненным изображениям
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")

    logger.info(f"Загрузка {dataset_name} (split={split}, max_samples={max_samples})...")

    ds = load_dataset(dataset_name, split=split, streaming=streaming)

    # Применение фильтра если задан
    if filter_fn is not None:
        ds = ds.filter(filter_fn)
        logger.info("✅ Dataset отфильтрован с помощью filter_fn")

    # Создание директории
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = []
    count = 0

    pbar = tqdm(ds, desc=f"Download {prefix}", total=max_samples)
    for idx, item in enumerate(pbar):
        if max_samples is not None and count >= max_samples:
            break

        try:
            img = item['image']
            if img is None:
                continue

            # Конвертация в RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Сохранение в JPEG
            filename = f"{prefix}_{idx:06d}.jpg"
            filepath = output_path / filename
            img.save(filepath, format='JPEG', quality=90)

            image_paths.append(str(filepath))
            count += 1

            pbar.update(1)
            if count % 1000 == 0:
                logger.info(f"  Сохранено {count}/{max_samples} изображений ({prefix})")

        except Exception as e:
            logger.warning(f"Ошибка обработки примера {idx}: {e}")
            continue

    logger.info(f"✅ Сохранено {len(image_paths)} изображений ({prefix})")

    return image_paths


def prepare_datasets(config, streaming: Optional[bool] = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Подготовка train/val/test split'ов

    Args:
        config: конфигурация проекта
        streaming: если None — берётся config.data.streaming;
                   True — стриминг из HuggingFace (fallback, без записи на диск)
                   False — локальный кэш (рекомендуется для GPU)

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if streaming is None:
        streaming = config.data.streaming

    if streaming:
        print("STREAMING mode active")
        return create_streaming_datasets(config)

    logger.info("📁 Локальный кэш-режим")
    return prepare_local_datasets(config)


def prepare_local_datasets(config) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Локальный кэш-режим: изображения уже выгружены на диск
    (scripts/download_data.py). При нехватке файлов кэш догружается из HuggingFace.

    Split'ы строятся из объединённого сбалансированного пула по data.{train,val,test}_ratio.
    """
    real_dir = Path(config.data.real_data_path)
    ai_dir = Path(config.data.ai_data_path)

    def cached_paths(directory: Path) -> List[str]:
        if not directory.exists():
            return []
        return sorted(str(p) for p in directory.glob("*.jpg"))

    real_paths = cached_paths(real_dir)
    ai_paths = cached_paths(ai_dir)

    # Догрузка кэша, если файлов меньше запрошенного
    if len(real_paths) < config.data.real_max_samples:
        logger.info(
            f"Real кэш: {len(real_paths)}/{config.data.real_max_samples} — догружаю из "
            f"{config.data.hf_imagenet_name}"
        )
        real_paths = download_hf_images(
            dataset_name=config.data.hf_imagenet_name,
            split=config.data.hf_imagenet_split,
            output_dir=config.data.real_data_path,
            prefix="real",
            max_samples=config.data.real_max_samples,
            streaming=True,
        )

    if len(ai_paths) < config.data.ai_max_samples:
        logger.info(
            f"AI кэш: {len(ai_paths)}/{config.data.ai_max_samples} — догружаю из "
            f"{config.data.hf_dataset_name}"
        )
        ai_paths = download_hf_images(
            dataset_name=config.data.hf_dataset_name,
            split=config.data.hf_dataset_split,
            output_dir=config.data.ai_data_path,
            prefix="ai",
            max_samples=config.data.ai_max_samples,
            streaming=True,
            filter_fn=filter_is_synthetic,
        )

    if not real_paths or not ai_paths:
        raise FileNotFoundError(
            "Локальный кэш пуст. Выгрузите данные: python scripts/download_data.py "
            "(нужны huggingface-cli login и принятая лицензия ImageNet-1k), "
            "либо включите data.streaming=True в config.py."
        )

    # Балансировка 1:1 с воспроизводимым сидом
    rng = random.Random(config.training.seed)
    n_per_class = min(len(real_paths), len(ai_paths))
    real_paths = rng.sample(real_paths, n_per_class)
    ai_paths = rng.sample(ai_paths, n_per_class)

    all_paths = real_paths + ai_paths
    all_labels = [0] * n_per_class + [1] * n_per_class

    # Перемешивание
    combined = list(zip(all_paths, all_labels))
    rng.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    # Сплиты
    n_total = len(all_paths)
    n_train = int(n_total * config.data.train_ratio)
    n_val = int(n_total * config.data.val_ratio)

    train_paths = all_paths[:n_train]
    train_labels = all_labels[:n_train]

    val_paths = all_paths[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]

    test_paths = all_paths[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]

    # Аугментации (только train)
    train_transform = create_augmentation_pipeline(config.augmentation)

    # Создание dataset'ов
    train_dataset = NPRDataset(list(train_paths), list(train_labels), train_transform, config.data.image_size)
    val_dataset = NPRDataset(list(val_paths), list(val_labels), None, config.data.image_size)
    test_dataset = NPRDataset(list(test_paths), list(test_labels), None, config.data.image_size)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def worker_init_fn(worker_id: int):
    """Установка сидов для каждого worker'а (модульный уровень для pickle)"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# TODO: Добить шардирование до рабочего
def _effective_pin_memory(pin_memory: bool) -> bool:
    """pin_memory имеет смысл только при наличии CUDA (иначе Warning)"""
    return pin_memory and torch.cuda.is_available()


def create_dataloaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Dataset,
                       batch_size: int = 256,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создание DataLoader'ов с оптимизациями производительности"""

    from torch.utils.data import IterableDataset

    # Генератор для перемешивания
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

    pin_memory = _effective_pin_memory(pin_memory)

    # Определяем тип датасета: IterableDataset (streaming) vs обычный Dataset (скачанный)
    is_train_streaming = isinstance(train_dataset, IterableDataset)

    # Настройки в зависимости от типа данных
    if is_train_streaming:
        # Streaming mode: ограничиваем workers, shuffle невозможен
        effective_workers = min(num_workers, 4)  # Меньше workers для streaming
        train_shuffle = False
        prefetch_factor = 2
        persistent_workers = False
        logger.info("📡 DataLoader: streaming mode (shuffle=False, persistent_workers=False)")
    else:
        # Локальные данные: полная производительность
        effective_workers = num_workers
        train_shuffle = True  # ✅ Shuffle для train при локальных данных
        prefetch_factor = 4
        persistent_workers = effective_workers > 0
        logger.info("💾 DataLoader: local data mode (shuffle=True, persistent_workers=True)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,  # True для локальных данных, False для streaming
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if effective_workers > 0 else None,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn if effective_workers > 0 else None,
        generator=generator if not is_train_streaming else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Для val/test можно больше workers
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader


def create_eval_loader(dataset: Dataset,
                       batch_size: int = 64,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> DataLoader:
    """DataLoader для оценки: без shuffle, c prefetch и persistent workers"""
    pin_memory = _effective_pin_memory(pin_memory)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    from config import Config

    config = Config()
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        config,
        streaming=config.data.streaming
    )

    print(train_dataset, val_dataset, test_dataset)
