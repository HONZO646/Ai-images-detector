"""
Утилиты для проекта NPR AI/Real Classifier
Логирование, управление сидом, чекпоинты, метрики
"""
import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json


def set_seed(seed: int = 42):
    """Установка сида для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Настройка логгера"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файл
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None
):
    """Сохранение чекпоинта"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    
    if is_best and best_filepath:
        Path(best_filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, best_filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """Загрузка чекпоинта"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Чекпоинт не найден: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Подсчёт количества обучаемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Вычисление метрик классификации"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    finite_mask = np.isfinite(predictions) & np.isfinite(targets)
    if not np.all(finite_mask):
        dropped = int((~finite_mask).sum())
        logging.getLogger(__name__).warning(
            f"compute_metrics: отброшено {dropped} невалидных (NaN/Inf) предсказаний"
        )
        predictions = predictions[finite_mask]
        targets = targets[finite_mask]

    if predictions.size == 0 or targets.size == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0
        }

    preds_binary = (predictions >= threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(targets, preds_binary),
        'precision': precision_score(targets, preds_binary, zero_division=0),
        'recall': recall_score(targets, preds_binary, zero_division=0),
        'f1': f1_score(targets, preds_binary, zero_division=0),
    }
    
    # AUC только если есть оба класса
    if len(np.unique(targets)) > 1:
        try:
            metrics['auc'] = roc_auc_score(targets, predictions)
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Сохранение метрик в JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Конвертация numpy/torch типов в Python
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return obj
    
    metrics_serializable = {k: convert(v) for k, v in metrics.items()}
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)


def get_device(device_str: str = 'auto') -> torch.device:
    """Получение устройства"""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


class AverageMeter:
    """Счётчик средних значений"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
