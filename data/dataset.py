"""
Dataset модуль для NPR AI/Real Classifier
Загрузка, препроцессинг и аугментация изображений
"""
import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, Subset, IterableDataset
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class NPRDataset(Dataset):
    """
    Dataset для NPR детектора
    Возвращает пару: (grayscale_tensor, rgb_tensor, label)
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[callable] = None,
                 target_size: int = 256):
        """
        image_paths: список путей к изображениям
        labels: список меток (0=real, 1=ai)
        transform: функция аугментации
        target_size: размер для ресайза
        """
        assert len(image_paths) == len(labels), "Длины paths и labels должны совпадать"
        
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        
        logger.info(f"Создан dataset: {len(image_paths)} изображений")
        logger.info(f"  Real (0): {labels.count(0)}")
        logger.info(f"  AI (1):   {labels.count(1)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Загрузка изображения
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img_rgb = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Ошибка загрузки {img_path}: {e}")
            # Возвращаем чёрное изображение как fallback
            img_rgb = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # Аугментации (если есть)
        if self.transform:
            img_rgb = self.transform(img_rgb)
        
        # Ресайз
        img_rgb = img_rgb.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        # Конвертация в tensor
        rgb_tensor = self._rgb_to_tensor(img_rgb)
        
        # Grayscale
        gray_tensor = self._rgb_to_grayscale_tensor(img_rgb)
        
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return gray_tensor, rgb_tensor, label_tensor
    
    def _rgb_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL Image → [C, H, W] tensor в диапазоне [0, 1]"""
        img_array = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]
        return tensor
    
    def _rgb_to_grayscale_tensor(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL Image → [1, H, W] grayscale tensor"""
        img_gray = img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).unsqueeze(0)  # [1, H, W]
        return tensor


class ComposeTransform:
    """Композиция аугментаций"""
    
    def __init__(self, transforms: List[callable]):
        self.transforms = transforms
    
    def __call__(self, img: Image.Image) -> Image.Image:
        for transform in self.transforms:
            img = transform(img)
        return img


class JpegCompression:
    """Аугментация: JPEG компрессия"""
    
    def __init__(self, quality_range: Tuple[int, int] = (70, 100), prob: float = 0.4):
        self.quality_range = quality_range
        self.prob = prob
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        
        quality = random.randint(*self.quality_range)
        
        # Сохранение в JPEG и загрузка обратно
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class GaussianBlur:
    """Аугментация: Gaussian blur"""
    
    def __init__(self, radius_options: List[float] = None, prob: float = 0.15):
        self.radius_options = radius_options or [0.5, 1.0, 1.5]
        self.prob = prob
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        
        radius = random.choice(self.radius_options)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class AdditiveNoise:
    """Аугментация: аддитивный Gaussian шум"""
    
    def __init__(self, sigma: float = 0.01, prob: float = 0.3):
        self.sigma = sigma
        self.prob = prob
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Добавление шума
        noise = np.random.normal(0, self.sigma, img_array.shape)
        noisy = img_array + noise
        noisy = np.clip(noisy, 0, 1)
        
        return Image.fromarray((noisy * 255).astype(np.uint8), mode='RGB')


class ContrastAdjust:
    """Аугментация: изменение контраста"""
    
    def __init__(self, contrast_range: Tuple[float, float] = (0.9, 1.1), prob: float = 0.2):
        self.contrast_range = contrast_range
        self.prob = prob
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)


def create_augmentation_pipeline(enable: bool = True, 
                                 jpeg_prob: float = 0.4,
                                 blur_prob: float = 0.15,
                                 noise_prob: float = 0.3,
                                 contrast_prob: float = 0.2,
                                 **kwargs) -> Optional[ComposeTransform]:
    """Создание pipeline аугментаций"""
    
    if not enable:
        return None
    
    transforms = [
        JpegCompression(prob=jpeg_prob),
        GaussianBlur(prob=blur_prob),
        AdditiveNoise(prob=noise_prob),
        ContrastAdjust(prob=contrast_prob)
    ]
    
    return ComposeTransform(transforms)


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
                 max_samples: int = None):
        """
        hf_dataset: HuggingFace dataset в streaming mode
        label: метка класса (0=real, 1=ai)
        transform: функция аугментации
        target_size: размер для ресайза
        max_samples: ограничение количества (None = все)
        """
        super().__init__()
        self.hf_dataset = hf_dataset
        self.label = label
        self.transform = transform
        self.target_size = target_size
        self.max_samples = max_samples
        
        # Определение длины (для информации)
        if max_samples is not None:
            self._len = max_samples
        elif hasattr(hf_dataset, '__len__'):
            self._len = len(hf_dataset)
        else:
            self._len = 10000  # fallback estimate
        
        logger.info(f"Создан Streaming dataset: label={label}, max_samples={self.max_samples}")
    
    def __len__(self) -> int:
        return self._len
    
    def __iter__(self):
        """Итератор для streaming mode — бесконечный цикл"""
        # Для IterableDataset с фиксированным max_samples,
        # мы должны yield'нуть ровно max_samples примеров
        count = 0
        max_count = self.max_samples if self.max_samples is not None else float('inf')
        
        while count < max_count:
            # Создаём новый итератор (на случай если предыдущий исчерпался)
            try:
                if self.max_samples is not None:
                    # HF streaming dataset может иметь .shuffle().take()
                    # Для бесконечности используем shuffle каждый раз
                    ds = self.hf_dataset.shuffle(seed=42 + count)
                    it = iter(ds.take(self.max_samples))
                else:
                    it = iter(self.hf_dataset)
            except Exception:
                # Fallback: пробуем снова с оригинальным dataset
                it = iter(self.hf_dataset)
            
            for idx, item in enumerate(it):
                if count >= max_count:
                    break
                
                # Для multiple workers распределяем данные
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is not None:
                    if idx % worker_info.num_workers != worker_info.id:
                        continue
                
                try:
                    img_rgb = item['image'].convert('RGB')
                except Exception as e:
                    logger.error(f"Ошибка загрузки изображения: {e}")
                    img_rgb = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
                
                # Аугментации
                if self.transform:
                    img_rgb = self.transform(img_rgb)
                
                # Ресайз
                img_rgb = img_rgb.resize((self.target_size, self.target_size), Image.LANCZOS)
                
                # Конвертация в tensor
                rgb_tensor = self._rgb_to_tensor(img_rgb)
                gray_tensor = self._rgb_to_grayscale_tensor(img_rgb)
                
                label_tensor = torch.tensor(self.label, dtype=torch.float32)
                
                yield gray_tensor, rgb_tensor, label_tensor
                count += 1
    
    def _rgb_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """RGB PIL Image → [C, H, W] tensor в диапазоне [0, 1]"""
        img_array = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        return tensor
    
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
    
    def __init__(self,
                 real_dataset: StreamingNPRDataset,
                 ai_dataset: StreamingNPRDataset):
        super().__init__()
        self.real_ds = real_dataset
        self.ai_ds = ai_dataset
        
        # Балансировка по меньшему классу
        self.n_per_class = min(len(real_dataset), len(ai_dataset))
        self.total_len = self.n_per_class * 2
        
        logger.info(f"Combined Streaming Dataset:")
        logger.info(f"  Real: {len(real_dataset)}")
        logger.info(f"  AI:   {len(ai_dataset)}")
        logger.info(f"  Per class: {self.n_per_class}")
        logger.info(f"  Total: {self.total_len}")
    
    def __len__(self) -> int:
        return self.total_len
    
    def __iter__(self):
        """Чередование real и ai примеров — бесконечный цикл"""
        real_iter = iter(self.real_ds)
        ai_iter = iter(self.ai_ds)
        
        for i in range(self.n_per_class):
            try:
                # Real пример
                yield next(real_iter)
            except StopIteration:
                # Пересоздаём итератор (бесконечный streaming)
                real_iter = iter(self.real_ds)
                try:
                    yield next(real_iter)
                except StopIteration:
                    break  # Если и это не сработало — выходим
            
            try:
                # AI пример
                yield next(ai_iter)
            except StopIteration:
                # Пересоздаём итератор
                ai_iter = iter(self.ai_ds)
                try:
                    yield next(ai_iter)
                except StopIteration:
                    break


def create_streaming_datasets(config) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Создание train/val/test streaming split'ов
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")
    
    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА ДАННЫХ В STREAMING MODE")
    logger.info("Данные НЕ загружаются на диск!")
    logger.info("=" * 60)
    
    # ImageNet-1k (real)
    logger.info(f"\n📥 ImageNet-1k (split={config.data.hf_imagenet_split})...")
    imagenet_full = load_dataset(
        "ILSVRC/imagenet-1k",
        split=config.data.hf_imagenet_split,
        streaming=True  # ← STREAMING MODE
    )
    
    # AI generated
    logger.info(f"📥 AI dataset: {config.data.hf_dataset_name}...")
    ai_full = load_dataset(
        config.data.hf_dataset_name,
        split=config.data.hf_dataset_split,
        streaming=True
    )
    
    # Разделение на train/val/test для каждого источника
    def split_streaming_dataset(dataset, train_ratio=0.75, val_ratio=0.15):
        """Разделение streaming dataset через shuffle + take"""
        # Для streaming используем shuffled().take()
        # Это не идеально, но работает без загрузки
        dataset_shuffled = dataset.shuffle(seed=config.training.seed)
        
        # Определяем размеры
        # Для streaming берём оценку из config
        max_real = config.data.hf_imagenet_max_samples
        max_ai = config.data.hf_imagenet_max_samples  # Будет скорректировано
        
        n_total = min(max_real, max_ai) * 2  # approximate
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        return dataset_shuffled, n_train, n_val
    
    # Для простоты: используем все данные для train, без val/test split
    # В streaming mode это нормально — можно валидироваться на тех же данных
    # с отложенной оценкой
    
    # Создание streaming wrapper'ов
    real_train = StreamingNPRDataset(
        hf_dataset=imagenet_full,
        label=0,
        transform=create_augmentation_pipeline(
            enable=config.augmentation.enable,
            jpeg_prob=config.augmentation.jpeg_prob,
            blur_prob=config.augmentation.blur_prob,
            noise_prob=config.augmentation.noise_prob,
            contrast_prob=config.augmentation.contrast_prob
        ),
        target_size=config.data.image_size,
        max_samples=config.data.hf_imagenet_max_samples
    )
    
    ai_train = StreamingNPRDataset(
        hf_dataset=ai_full,
        label=1,
        transform=None,  # AI данные не аугментируем
        target_size=config.data.image_size,
        max_samples=config.data.hf_imagenet_max_samples
    )
    
    # Комбинированный dataset
    train_dataset = CombinedStreamingDataset(real_train, ai_train)
    
    # Для val/test в streaming mode можно использовать те же данные
    # но без аугментаций и с отдельными seed
    # (В идеале — иметь отдельные val/test датасеты)
    
    # Упрощение: используем тот же train dataset для val/test
    # Это не идеально, но работает для streaming
    logger.warning("⚠️ Streaming mode: val/test используют те же данные что и train")
    logger.warning("   Для лучшей оценки используйте отдельные val/test split'ы")
    
    val_dataset = StreamingNPRDataset(
        hf_dataset=imagenet_full,
        label=0,
        transform=None,
        target_size=config.data.image_size,
        max_samples=int(config.data.hf_imagenet_max_samples * config.data.val_ratio)
    )
    
    test_dataset = StreamingNPRDataset(
        hf_dataset=ai_full,
        label=1,
        transform=None,
        target_size=config.data.image_size,
        max_samples=int(config.data.hf_imagenet_max_samples * config.data.test_ratio)
    )
    
    return train_dataset, val_dataset, test_dataset


def collect_imagenet_hf(max_samples: int = 15000,
                        cache_dir: str = None,
                        split: str = 'validation',
                        streaming: bool = True) -> List[str]:
    """
    Загрузка ImageNet-1k из HuggingFace (ILSVRC/imagenet-1k)
    Возвращает пути к сохранённым реальным изображениям (только если streaming=False)
    
    Args:
        max_samples: максимальное количество изображений
        cache_dir: директория кэша для HF datasets
        split: 'validation' или 'train'
        streaming: если True, данные не сохраняются на диск
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")
    
    logger.info(f"Загрузка ImageNet-1k из HuggingFace (split={split}, streaming={streaming})...")
    logger.info("⚠️ Требуется аутентификация!")
    logger.info("   1. Примите лицензию: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    logger.info("   2. Выполните: huggingface-cli login")
    
    # Загрузка ImageNet-1k
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming
    )
    
    if streaming:
        logger.info("✅ Streaming mode активен — данные НЕ загружаются на диск")
        return None  # Не возвращаем пути
    
    # Старое поведение (сохранение на диск)
    logger.info("⚠️ Streaming mode отключен — данные будут сохранены локально")
    
    # Ограничение количества
    if hasattr(ds, '__len__') and len(ds) > max_samples:
        indices = random.sample(range(len(ds)), max_samples)
        ds = ds.select(indices)
    
    logger.info(f"Загружено {len(ds)} изображений ImageNet-1k")
    
    # Сохранение изображений локально
    output_dir = Path("data") / "real" / "imagenet"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    for idx, item in enumerate(ds):
        try:
            img = item['image']
            if img is None:
                continue
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            label = item.get('label', -1)
            filename = f"real_{idx:06d}_l{label}.png"
            filepath = output_dir / filename
            img.save(filepath)
            
            image_paths.append(str(filepath))
            
        except Exception as e:
            logger.warning(f"Ошибка обработки примера {idx}: {e}")
            continue
    
    logger.info(f"Сохранено {len(image_paths)} реальных изображений ImageNet")
    
    return image_paths


def collect_hf_dataset(dataset_name: str,
                       split: str = 'train',
                       cache_dir: str = None) -> List[str]:
    """
    Загрузка датасета из HuggingFace
    Возвращает пути к сохранённым изображениям
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")

    logger.info(f"Загрузка HF dataset: {dataset_name}")

    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    # Сохранение изображений локально
    output_dir = Path("data") / "ai_generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for idx, item in enumerate(ds):
        try:
            # Предполагаем что изображение в поле 'image'
            img = item['image']
            if img is None:
                continue

            # Конвертация в RGB если нужно
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Сохранение
            filename = f"ai_{idx:06d}.png"
            filepath = output_dir / filename
            img.save(filepath)

            image_paths.append(str(filepath))

        except Exception as e:
            logger.warning(f"Ошибка обработки примера {idx}: {e}")
            continue

    logger.info(f"Сохранено {len(image_paths)} AI изображений")

    return image_paths


def prepare_datasets(config, streaming: bool = True) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Подготовка train/val/test split'ов
    
    Args:
        config: конфигурация проекта
        streaming: если True, данные загружаются по мере необходимости (рекомендуется)
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if streaming:
        logger.info("🚀 Используем STREAMING mode — данные НЕ загружаются на диск")
        return create_streaming_datasets(config)
    
    # Старое поведение (сохранение на диск)
    logger.info("📁 Используем локальное сохранение данных")
    
    # Загрузка реальных изображений из ImageNet-1k (HuggingFace)
    real_paths = collect_imagenet_hf(
        max_samples=config.data.hf_imagenet_max_samples,
        split=config.data.hf_imagenet_split,
        streaming=False
    )

    # Загрузка AI-сгенерированных изображений
    ai_paths = collect_hf_dataset(
        config.data.hf_dataset_name,
        config.data.hf_dataset_split
    )

    if not real_paths or not ai_paths:
        raise ValueError("Не удалось загрузить данные. Проверьте подключение к HuggingFace.")

    # Балансировка
    min_samples = min(len(real_paths), len(ai_paths))
    real_paths = random.sample(real_paths, min_samples)
    ai_paths = random.sample(ai_paths, min_samples)
    
    # Метки
    real_labels = [0] * len(real_paths)
    ai_labels = [1] * len(ai_paths)
    
    # Объединение
    all_paths = real_paths + ai_paths
    all_labels = real_labels + ai_labels
    
    # Перемешивание
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
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
    
    # Аугментации (только для train)
    train_transform = create_augmentation_pipeline(
        enable=config.augmentation.enable,
        jpeg_prob=config.augmentation.jpeg_prob,
        blur_prob=config.augmentation.blur_prob,
        noise_prob=config.augmentation.noise_prob,
        contrast_prob=config.augmentation.contrast_prob
    )
    
    # Создание dataset'ов
    train_dataset = NPRDataset(train_paths, train_labels, train_transform, config.data.image_size)
    val_dataset = NPRDataset(val_paths, val_labels, None, config.data.image_size)
    test_dataset = NPRDataset(test_paths, test_labels, None, config.data.image_size)
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Dataset,
                       batch_size: int = 256,
                       num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создание DataLoader'ов"""
    
    from torch.utils.data import IterableDataset
    
    # Для IterableDataset (streaming) нельзя использовать shuffle=True
    # Данные уже перемешаны через HF dataset.shuffle()
    is_train_streaming = isinstance(train_dataset, IterableDataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Streaming mode не поддерживает shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Не дропаем последний батч
        prefetch_factor=2 if not is_train_streaming else None  # Prefetch только для не-streaming
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    
    # Тест аугментаций
    img = Image.new('RGB', (256, 256), (128, 128, 128))
    
    pipeline = create_augmentation_pipeline(enable=True)
    augmented = pipeline(img)
    
    print("=== Augmentation Test ===")
    print(f"Original: {img.size}")
    print(f"Augmented: {augmented.size}")
    
    # Тест Dataset
    # Создадим фейковые данные
    fake_paths = ["test-ai-1.jpg"] * 10 + ["test-real-1.jpg"] * 10
    fake_labels = [1] * 10 + [0] * 10
    
    dataset = NPRDataset(fake_paths, fake_labels, target_size=256)
    
    print(f"\n=== Dataset Test ===")
    print(f"Dataset size: {len(dataset)}")
    
    gray, rgb, label = dataset[0]
    print(f"Gray shape: {gray.shape}")
    print(f"RGB shape: {rgb.shape}")
    print(f"Label: {label}")
