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
    streaming: bool = False  # False = скачивать на диск, True = стриминг из HF
    
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
    hf_imagenet_max_samples: int = 5000
    
    # AI generated dataset
    hf_dataset_name: str = "gasstation/generated-images"
    hf_dataset_split: str = "train"
    hf_dataset_max_samples: int = 5000  # Сколько AI изображений загружать


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
    npr_feature_dim: int = 32  # 8 направлений × 4 статистики (без entropy)
    
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
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    """Настройки обучения с оптимизациями производительности"""
    # Основные
    batch_size: int = 512  # Увеличено с учётом 24GB VRAM
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = 'adamw'  # adamw, adam, sgd
    gradient_clip_norm: float = 1.0
    
    # Scheduler
    scheduler_type: str = 'cosine'  # cosine или plateau
    scheduler_mode: str = 'max'  # используется только для plateau
    scheduler_factor: float = 0.5  # используется только для plateau
    scheduler_patience: int = 4  # используется только для plateau
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 3
    
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
    device: str = 'cuda'  # auto, cpu, cuda
    
    # Mixed Precision Training
    use_amp: bool = False  # Automatic Mixed Precision (FP16/BF16)
    amp_dtype: str = 'float16'  # float16 or bfloat16
    
    # Gradient Accumulation (для эмуляции ещё больших батчей)
    gradient_accumulation_steps: int = 1  # Увеличьте если нужен effective batch > 512
    
    # Reproducibility
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True  # True для GPU (ускоряет передачу данных)
    
    # Performance optimizations
    compile_model: bool = False  # torch.compile (PyTorch 2.0+, может дать +20% скорости)
    enable_flash_attention: bool = True  # Использовать Flash Attention если доступен


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
