"""
Конфигурация проекта NPR AI/Real Classifier
Все гиперпараметры и настройки в одном месте
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Настройки данных"""
    # Пути к данным
    real_data_path: str = os.path.join("data", "real")  # ImageNet из HF
    ai_data_path: str = os.path.join("data", "ai")  # HF generated
    
    # Режим загрузки
    streaming: bool = True  # True = данные НЕ загружаются на диск
    
    # Размеры
    image_size: int = 256
    min_side: int = 256
    
    # Сплиты
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    
    # Баланс классов
    class_balance_ratio: float = 1.0  # 1:1 real:ai
    
    # HuggingFace Datasets
    # ImageNet-1k: https://huggingface.co/datasets/ILSVRC/imagenet-1k
    hf_imagenet_name: str = "ILSVRC/imagenet-1k"
    hf_imagenet_split: str = "validation"  # val split для реальных изображений
    hf_imagenet_max_samples: int = 15000
    
    # AI generated dataset
    hf_dataset_name: str = "gasstation/generated-images"
    hf_dataset_split: str = "train"


@dataclass
class AugmentationConfig:
    """Настройки аугментаций (только для train)"""
    enable: bool = True
    probability: float = 0.4
    
    # JPEG компрессия
    jpeg_prob: float = 0.4
    jpeg_quality_range: tuple = (70, 100)
    
    # Gaussian blur
    blur_prob: float = 0.15
    blur_radius_options: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    
    # Аддитивный шум
    noise_prob: float = 0.3
    noise_sigma: float = 0.01
    
    # Контраст
    contrast_prob: float = 0.2
    contrast_range: tuple = (0.9, 1.1)


@dataclass
class ModelConfig:
    """Настройки модели"""
    # NPR Features
    npr_bins: int = 32
    npr_feature_dim: int = 40  # 8 направлений × 5 статистик
    
    # Spatial Branch
    spatial_embedding_dim: int = 128
    lbp_radius: int = 1
    lbp_n_points: int = 8
    
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
    freeze_semantic: bool = True
    
    # Fusion
    fusion_embedding_dim: int = 128
    n_attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Classifier Head
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    """Настройки обучения"""
    # Основные
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = 'adamw'  # adamw, adam, sgd
    gradient_clip_norm: float = 1.0
    
    # Scheduler
    scheduler_mode: str = 'max'  # max для AUC
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    
    # Early Stopping
    early_stop_patience: int = 8
    early_stop_metric: str = 'val_auc'
    
    # Loss
    loss_function: str = 'bce'  # bce, bce_with_logits
    use_contrastive_aux: bool = False
    contrastive_weight: float = 0.1
    
    # Logging
    log_interval: int = 10
    save_interval: int = 5
    
    # Checkpoint
    checkpoint_dir: str = os.path.join("checkpoints")
    best_model_path: str = os.path.join("checkpoints", "best_model.pt")
    
    # Device
    device: str = 'auto'  # auto, cpu, cuda
    
    # Reproducibility
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = False  # True только для GPU


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
