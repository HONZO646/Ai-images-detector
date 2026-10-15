"""
Конфигурация проекта NPR AI/Real Classifier
Все гиперпараметры и настройки в одном месте
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import os


@dataclass
class DataConfig:
    """Настройки данных"""
    # Локальный кэш изображений (используется в нестриминговом режиме
    # и заполняется scripts/download_data.py)
    data_root: str = "data"
    real_data_path: str = os.path.join(data_root, "real", "imagenet")  # ImageNet из HF
    ai_data_path: str = os.path.join(data_root, "ai_generated")  # HF generated

    # Режим загрузки
    # True  = стриминг из HuggingFace (fallback, без записи на диск)
    # False = локальный кэш (рекомендуется для обучения на GPU)
    streaming: bool = False

    # Размеры
    image_size: int = 640

    # Сплиты (доля от каждого класса)
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10

    # HuggingFace Datasets
    # ImageNet-1k: https://huggingface.co/datasets/ILSVRC/imagenet-1k
    hf_imagenet_name: str = "ILSVRC/imagenet-1k"
    hf_imagenet_split: str = "train"
    real_max_samples: int = 20000

    # AI generated dataset
    hf_dataset_name: str = "gasstation/generated-images"
    hf_dataset_split: str = "train"
    ai_max_samples: int = 20000  # Сколько AI изображений загружать

    # Источник обучающих данных
    # 'bfree' = выровненные пары GRIP-UNINA B-Free (COCO real vs SD 2.1 self-conditioned, пары по id;
    #           лицензия только для информационных и некоммерческих целей)
    # 'local' = локальный кэш ImageNet/gasstation (шорткаты по содержанию и геометрии, ROADMAP E5–E6)
    source: str = 'bfree'
    bfree_root: str = os.path.join(data_root, "bfree")  # распакованные training_data (CSV + папки)
    bfree_real_dir: str = "COCO_real_512"
    # Варианты фейков SD 2.1 из B-Free: для каждого id берётся один вариант (детерминированно по id)
    bfree_fake_variants: tuple = ("SD2.1_selfconditioned", "SD2.1_inpainted_samecat", "SD2.1_inpainted_diffcat")
    bfree_train_max: int = 20000  # пар для обучения (из train_list.csv)
    bfree_eval_max: int = 2000    # пар для val и для test (valid_list.csv делится по id пополам)


@dataclass
class AugmentationConfig:
    """Настройки аугментаций — имитируют артефакты постобработки AI-генерации.

    Применяются симметрично к обоим классам train-выборки.
    """
    enable: bool = True

    # Downscale + Upscale — имитирует AI-апсемплинг
    downscale_prob: float = 0.5
    downscale_range: tuple = (0.5, 0.9)  # до какого размера уменьшать

    # JPEG компрессия — агрессивное сжатие как у AI-изображений
    jpeg_prob: float = 0.4
    jpeg_quality_range: tuple = (40, 70)

    # Color Jitter — маскировка стиля датасета
    color_prob: float = 0.4
    color_strength: float = 0.08  # ±8% по brightness/contrast/saturation

    blur_prob: float = 0.2

    # Sharpening — имитация AI over-sharpening
    # Фактор = 1.0 + U(sharpen_range): 0.5–1.0 → усиление резкости в 1.5–2x
    sharpen_prob: float = 0.3
    sharpen_range: tuple = (0.5, 1.0)

    # Гауссов шум и cut-out — из inpainted++ (B-Free, arXiv:2412.17671); train-only, симметрично для классов
    noise_prob: float = 0.3
    noise_std_range: tuple = (2.0, 8.0)  # std в уровнях 0–255
    cutout_prob: float = 0.2
    cutout_size_range: tuple = (0.1, 0.3)  # доля стороны изображения


@dataclass
class ModelConfig:
    """Настройки модели"""
    # Spatial Branch
    spatial_embedding_dim: int = 128
    lbp_radius: int = 1
    lbp_n_points: int = 8
    lbp_radii: List[int] = field(default_factory=list)  # Multi-scale LBP radii (empty = single scale)
    
    # Frequency Branch
    freq_embedding_dim: int = 128
    dwt_level: int = 2
    dwt_wavelet: str = 'db1'
    dct_block_size: int = 8
    fft_radial_bins: int = 64
    fft_angular_bins: int = 36
    
    # Semantic Branch
    semantic_embedding_dim: int = 128
    semantic_model_name: str = 'clip-ViT-B-32'
    # 0 = полная заморозка CLIP; 1-2 = разморозка последних блоков
    # (обучаемые CLIP-параметры получают training.semantic_lr)
    semantic_unfreeze_last_n_blocks: int = 0
    
    # Fusion
    fusion_embedding_dim: int = 128
    n_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Classifier Head
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    """Настройки обучения"""
    # Основные
    # Для RTX 4090 (24 ГБ) при 640x640: старт 64, после замера VRAM можно 96-128
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    # LR для обучаемых CLIP-блоков при semantic_unfreeze_last_n_blocks > 0
    semantic_lr: float = 1e-6

    # Optimizer
    optimizer: str = 'adamw'  # adamw, adam, sgd
    gradient_clip_norm: float = 1.0

    # Scheduler
    scheduler_type: str = 'cosine'  # cosine или plateau
    scheduler_mode: Literal['min', 'max'] = 'max'  # используется только для plateau
    scheduler_factor: float = 0.5  # используется только для plateau
    scheduler_patience: int = 4  # используется только для plateau
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 3

    # Early Stopping
    early_stop_patience: int = 8
    early_stop_metric: str = 'val_auc'

    # Loss
    loss_function: str = 'bce'  # bce, bce_with_logits

    # Logging
    save_interval: int = 5

    # Checkpoint
    checkpoint_dir: str = os.path.join("checkpoints")
    best_model_path: str = os.path.join("checkpoints", "best_model.pt")

    # Device
    device: str = 'auto'  # auto, cpu, cuda

    # Mixed Precision Training (автоматически отключается на CPU)
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'  # bfloat16 (Ada/Ampere+) или float16 (нужен GradScaler)

    # Reproducibility
    seed: int = 42
    # False: cudnn.benchmark=True (быстрее на GPU). True: детерминизм ценой скорости
    reproducible: bool = False
    num_workers: int = 8
    pin_memory: bool = True  # ускоряет передачу на GPU (игнорируется на CPU)

    # Gate Regularization
    # Предотвращают вырождение AdaptiveGating к semantic-only режиму
    gate_entropy_weight: float = 0.1   # Entropy reg: поощряет равномерный вклад всех веток
    gate_penalty_weight: float = 0.05  # Penalty: штрафует semantic gate если он >> 1/n_branches


@dataclass
class Config:
    """Главная конфигурация"""
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Project
    project_name: str = "NPR_AI_Detector"
    output_dir: str = "outputs"
    log_dir: str = "logs"
