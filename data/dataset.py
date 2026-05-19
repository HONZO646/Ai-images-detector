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
    
    # AI generated (оставляем только fully synthetic, исключая semisynthetic)
    logger.info(f"📥 AI dataset: {config.data.hf_dataset_name}...")
    ai_full = load_dataset(
        config.data.hf_dataset_name,
        split=config.data.hf_dataset_split,
        streaming=True
    )
    ai_full = ai_full.filter(lambda item: item.get('media_type', None) == 'synthetic')
    logger.info("✅ AI streaming dataset отфильтрован: media_type == 'synthetic'")
    
    # Разделение на train/val/test для каждого источника
    def split_streaming_dataset(dataset, train_ratio=0.75, val_ratio=0.15):
        """Разделение streaming dataset через shuffle + take"""
        # Для streaming используем shuffled().take()
        # Это не идеально, но работает без загрузки
        dataset_shuffled = dataset.shuffle(seed=config.training.seed)
        
        # Определяем размеры
        # Для streaming берём оценку из config
        max_real = config.data.hf_imagenet_max_samples
        max_ai = config.data.hf_dataset_max_samples
        
        n_total = min(max_real, max_ai) * 2  # approximate
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        return dataset_shuffled, n_train, n_val
    
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
        max_samples=config.data.hf_dataset_max_samples
    )
    
    # Комбинированный dataset
    train_dataset = CombinedStreamingDataset(real_train, ai_train)

    # Валидация/тест: формируем сбалансированные наборы из обоих классов
    val_real = StreamingNPRDataset(
        hf_dataset=imagenet_full,
        label=0,
        transform=None,
        target_size=config.data.image_size,
        max_samples=int(config.data.hf_imagenet_max_samples * config.data.val_ratio)
    )

    val_ai = StreamingNPRDataset(
        hf_dataset=ai_full,
        label=1,
        transform=None,
        target_size=config.data.image_size,
        max_samples=int(config.data.hf_dataset_max_samples * config.data.val_ratio)
    )

    test_real = StreamingNPRDataset(
        hf_dataset=imagenet_full,
        label=0,
        transform=None,
        target_size=config.data.image_size,
        max_samples=int(config.data.hf_imagenet_max_samples * config.data.test_ratio)
    )

    test_ai = StreamingNPRDataset(
        hf_dataset=ai_full,
        label=1,
        transform=None,
        target_size=config.data.image_size,
        max_samples=int(config.data.hf_dataset_max_samples * config.data.test_ratio)
    )

    val_dataset = CombinedStreamingDataset(val_real, val_ai)
    test_dataset = CombinedStreamingDataset(test_real, test_ai)
    
    return train_dataset, val_dataset, test_dataset


def collect_imagenet_hf(max_samples: int = 5000,
                         cache_dir: str = None,
                         split: str = 'validation',
                         streaming: bool = False,
                         output_dir: str = None) -> List[str]:
    """
    Загрузка ImageNet-1k из HuggingFace (ILSVRC/imagenet-1k)
    Возвращает пути к сохранённым реальным изображениям
    
    Args:
        max_samples: максимальное количество изображений
        cache_dir: директория кэша для HF datasets
        split: 'validation' или 'train'
        streaming: если True, данные будут стримиться и сохранены локально без полного кэша
        output_dir: путь для сохранения изображений
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")
    
    logger.info(f"Загрузка ImageNet-1k из HuggingFace (split={split}, max_samples={max_samples})...")
    logger.info("⚠️ Требуется аутентификация!")
    logger.info("   1. Примите лицензию: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    logger.info("   2. Выполните: huggingface-cli login")
    
    # Используем streaming чтобы не скачивать весь датасет на диск
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split=split,
        streaming=True
    )
    
    # Сохранение изображений локально
    if output_dir is None:
        output_dir = Path("\\\\192.168.12.19\\1119783\\project") / "real" / "imagenet"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    count = 0
    
    for idx, item in enumerate(ds):
        if count >= max_samples:
            break
            
        try:
            img = item['image']
            if img is None:
                continue
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            label = item.get('label', -1)
            filename = f"real_{idx:06d}_l{label}.jpg"
            filepath = output_dir / filename
            img.save(filepath, format='JPEG', quality=90)
            
            image_paths.append(str(filepath))
            count += 1
            
            if count % 1000 == 0:
                logger.info(f"  Сохранено {count}/{max_samples} изображений ImageNet")
            
        except Exception as e:
            logger.warning(f"Ошибка обработки примера {idx}: {e}")
            continue
    
    logger.info(f"✅ Сохранено {len(image_paths)} реальных изображений ImageNet")
    
    return image_paths


def collect_hf_dataset(dataset_name: str,
                       split: str = 'train',
                       cache_dir: str = None,
                       max_samples: int = None,
                       output_dir: str = None) -> List[str]:
    """
    Загрузка датасета из HuggingFace
    Возвращает пути к сохранённым изображениям
    
    Args:
        max_samples: максимальное количество изображений (None = все)
        output_dir: путь для сохранения изображений
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Установите datasets: pip install datasets")

    logger.info(f"Загрузка HF dataset: {dataset_name} (max_samples={max_samples})...")
    logger.info("Фильтрация AI dataset: сохраняем только media_type == 'synthetic'")

    # Используем streaming чтобы не скачивать весь датасет на диск
    ds = load_dataset(dataset_name, split=split, streaming=True)
    
    # Применяем фильтрацию synthetic на стороне HF (streaming)
    ds = ds.filter(lambda item: item.get('media_type', None) == 'synthetic')
    logger.info("✅ AI streaming dataset отфильтрован: media_type == 'synthetic'")
    
    # Сохранение изображений локально
    if output_dir is None:
        output_dir = Path("\\\\192.168.12.19\\1119783\\project") / "ai_generated"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    count = 0

    for idx, item in enumerate(ds):
        if max_samples is not None and count >= max_samples:
            break
            
        try:
            # Предполагаем что изображение в поле 'image'
            img = item['image']
            if img is None:
                continue
            
            # Конвертация в RGB если нужно
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Сохранение в JPEG для экономии места
            filename = f"ai_{idx:06d}.jpg"
            filepath = output_dir / filename
            img.save(filepath, format='JPEG', quality=90)
            
            image_paths.append(str(filepath))
            count += 1
            
            if max_samples is not None and count % 1000 == 0:
                logger.info(f"  Сохранено {count}/{max_samples} AI изображений")
        
        except Exception as e:
            logger.warning(f"Ошибка обработки примера {idx}: {e}")
            continue
    
    logger.info(f"✅ Сохранено {len(image_paths)} AI изображений (synthetic only)")
    
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
    
    # Проверяем наличие уже скачанных данных
    real_dir = Path(config.data.real_data_path)
    ai_dir = Path(config.data.ai_data_path)
    
    if real_dir.exists() and ai_dir.exists():
        real_existing = list(real_dir.glob("*.jpg"))
        ai_existing = list(ai_dir.glob("*.jpg"))
        
        if len(real_existing) > 0 and len(ai_existing) > 0:
            logger.info("✅ Найдены существующие данные на диске:")
            logger.info(f"   Real: {len(real_existing)} изображений")
            logger.info(f"   AI:   {len(ai_existing)} изображений")
            logger.info("   Пропускаем загрузку из HuggingFace")
            real_paths = [str(p) for p in real_existing]
            ai_paths = [str(p) for p in ai_existing]
        else:
            real_paths = None
            ai_paths = None
    else:
        real_paths = None
        ai_paths = None
    
    # Загрузка реальных изображений из ImageNet-1k (HuggingFace) если нужно
    if not real_paths:
        real_paths = collect_imagenet_hf(
            max_samples=config.data.hf_imagenet_max_samples,
            split=config.data.hf_imagenet_split,
            output_dir=config.data.real_data_path
        )

    # Загрузка AI-сгенерированных изображений если нужно
    if not ai_paths:
        ai_paths = collect_hf_dataset(
            config.data.hf_dataset_name,
            config.data.hf_dataset_split,
            max_samples=config.data.hf_dataset_max_samples,
            output_dir=config.data.ai_data_path
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
    """Создание DataLoader'ов с оптимизациями производительности"""
    
    from torch.utils.data import IterableDataset
    
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
        persistent_workers = True
        logger.info("💾 DataLoader: local data mode (shuffle=True, persistent_workers=True)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,  # True для локальных данных, False для streaming
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if effective_workers > 0 else None,
        persistent_workers=persistent_workers
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
