"""
Training loop для NPR AI/Real Detector
Полный цикл обучения с early stopping, scheduler'ом и логированием
"""
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.amp import GradScaler, autocast

from config import Config
from models.detector import NPRDetector, create_detector
from data.dataset import prepare_datasets, create_dataloaders
from utils import (
    set_seed, setup_logger, save_checkpoint, load_checkpoint,
    compute_metrics, save_metrics, get_device, AverageMeter
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer класс для NPR Detector с поддержкой mixed precision
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Device
        self.device = get_device(config.training.device)
        logger.info(f"Using device: {self.device}")
        
        # Seed
        set_seed(config.training.seed)
        
        # Model
        self.model = create_detector(config.model, device=self.device)
        self.model = self.model.to(self.device)
        
        # Mixed precision scaler
        self.use_amp = config.training.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            logger.info("✅ Mixed Precision Training (AMP) enabled")
        
        # Data (download to disk)
        self.train_dataset, self.val_dataset, self.test_dataset = prepare_datasets(
            config,
            streaming=config.data.streaming  # False = download to disk
        )
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )
        
        # Loss function - BCEWithLogitsLoss is AMP-safe (combines sigmoid + BCE)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=config.training.scheduler_mode,
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience
        )
        
        # Метрики
        self.best_val_auc = 0.0
        self.current_epoch = 0
        self.epochs_without_improvement = 0
        
        # Логирование
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        self.logger = setup_logger(
            'training',
            log_file=os.path.join(config.log_dir, 'training.log')
        )
        
        # Для визуализации
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Создание оптимизатора"""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.config.training.optimizer == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'adam':
            return optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Один epoch обучения с mixed precision"""
        self.model.train()
        
        loss_meter = AverageMeter()
        predictions = []
        targets = []
        
        # Для IterableDataset len(loader) может быть неточным
        # Используем len(train_loader) только для tqdm
        total_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", 
                    total=total_batches)
        
        for batch_idx, (gray, rgb, labels) in enumerate(pbar):
            # Перенос на устройство
            gray = gray.to(self.device, non_blocking=True)
            rgb = rgb.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with autocast for mixed precision
            with autocast(device_type='cuda', enabled=self.use_amp):
                output = self.model(gray, rgb)
                # Model now returns logits (not probabilities)
                logits = output['probability'].squeeze()
                
                # Loss on logits (BCEWithLogitsLoss expects raw logits)
                loss = self.criterion(logits, labels)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Метрики — convert logits to probabilities for metrics
            probabilities = torch.sigmoid(logits)
            loss_meter.update(loss.item(), gray.shape[0])
            predictions.extend(probabilities.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())
            
            # Progress bar
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Вычисление метрик за epoch
        predictions = np.array(predictions)
        targets = np.array(targets)
        metrics = compute_metrics(predictions, targets)
        
        return {
            'loss': loss_meter.avg,
            'accuracy': metrics['accuracy'],
            'auc': metrics.get('auc', 0.0),
            'f1': metrics['f1']
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Валидация с mixed precision"""
        self.model.eval()
        
        loss_meter = AverageMeter()
        predictions = []
        targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for gray, rgb, labels in pbar:
            gray = gray.to(self.device, non_blocking=True)
            rgb = rgb.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with autocast
            with autocast(device_type='cuda', enabled=self.use_amp):
                output = self.model(gray, rgb)
                logits = output['probability'].squeeze()
                
                # Loss on logits
                loss = self.criterion(logits, labels)
            
            # Convert logits to probabilities for metrics
            probabilities = torch.sigmoid(logits)
            
            loss_meter.update(loss.item(), gray.shape[0])
            predictions.extend(probabilities.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss_meter.avg:.4f}"})
        
        # Метрики
        predictions = np.array(predictions)
        targets = np.array(targets)
        metrics = compute_metrics(predictions, targets)
        
        return {
            'loss': loss_meter.avg,
            'accuracy': metrics['accuracy'],
            'auc': metrics.get('auc', 0.0),
            'f1': metrics['f1']
        }
    
    @torch.no_grad()
    def evaluate_on_test(self) -> Dict[str, float]:
        """Финальная оценка на test set"""
        self.model.eval()
        
        predictions = []
        targets = []
        gate_weights = {
            'semantic': [],
            'spatial': [],
            'frequency': []
        }
        
        pbar = tqdm(self.test_loader, desc="Test Evaluation")
        
        for gray, rgb, labels in pbar:
            gray = gray.to(self.device)
            rgb = rgb.to(self.device)
            labels = labels.to(self.device)
            
            output = self.model(gray, rgb)
            logits = output['probability'].squeeze()
            # Convert logits to probabilities for metrics
            probabilities = torch.sigmoid(logits)
            
            predictions.extend(probabilities.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            if 'gate_weights' in output:
                gates = output['gate_weights'].cpu().numpy()
                gate_weights['semantic'].extend(gates[:, 0].tolist())
                gate_weights['spatial'].extend(gates[:, 1].tolist())
                gate_weights['frequency'].extend(gates[:, 2].tolist())
        
        # Метрики
        predictions = np.array(predictions)
        targets = np.array(targets)
        metrics = compute_metrics(predictions, targets)
        
        # Средние веса gate
        metrics['gate_weights'] = {
            k: np.mean(v) for k, v in gate_weights.items()
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_auc': self.best_val_auc,
            'config': self.config
        }
        
        # Текущий checkpoint
        current_path = os.path.join(
            self.config.training.checkpoint_dir,
            f"checkpoint_epoch_{self.current_epoch}.pt"
        )
        
        save_checkpoint(
            checkpoint,
            current_path,
            is_best=is_best,
            best_filepath=self.config.training.best_model_path
        )
        
        if is_best:
            self.logger.info(f"✅ Сохранён лучший checkpoint (AUC={self.best_val_auc:.4f})")
    
    def plot_training_curves(self):
        """Визуализация кривых обучения"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # AUC
        axes[1].plot(epochs, self.val_aucs, 'g-', label='Val AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        # Accuracy
        axes[2].plot(epochs, [m['accuracy'] for m in self.val_metrics], 'm-', label='Val Accuracy')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Validation Accuracy')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Сохранение
        plot_path = os.path.join(self.config.output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        self.logger.info(f"Графики сохранения: {plot_path}")
        
        plt.close()
    
    def train(self):
        """Основной цикл обучения"""
        self.logger.info("=" * 60)
        self.logger.info("НАЧАЛО ОБУЧЕНИЯ")
        self.logger.info("=" * 60)
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Val samples:   {len(self.val_dataset)}")
        self.logger.info(f"Test samples:  {len(self.test_dataset)}")
        self.logger.info(f"Batch size:    {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.info(f"Epochs:        {self.config.training.epochs}")
        self.logger.info("=" * 60)
        
        # Подсчёт параметров
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters:   {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info("=" * 60)
        
        self.val_metrics = []  # Для хранения метрик валидации
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_aucs.append(val_metrics['auc'])
            self.val_metrics.append(val_metrics)
            
            # Логирование
            elapsed = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.training.epochs} "
                f"[{elapsed:.0f}s] "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics['auc'])
            
            # Check for improvement
            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
                self.logger.info(f"🔥 Новый лучший AUC: {self.best_val_auc:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Периодическое сохранение
            if (epoch + 1) % self.config.training.save_interval == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.training.early_stop_patience:
                self.logger.info(
                    f"\n⚠️ Early stopping на epoch {epoch+1} "
                    f"(без улучшения {self.epochs_without_improvement} epochs)"
                )
                break
            
            # Plot curves каждые 10 epochs
            if (epoch + 1) % 10 == 0:
                try:
                    self.plot_training_curves()
                except Exception as e:
                    self.logger.warning(f"Ошибка при построении графиков: {e}")
        
        # Финальная оценка на test set
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET")
        self.logger.info("=" * 60)
        
        # Загрузка лучшей модели
        if os.path.exists(self.config.training.best_model_path):
            self.logger.info(f"Загрузка лучшей модели из: {self.config.training.best_model_path}")
            checkpoint = load_checkpoint(
                self.config.training.best_model_path,
                self.model,
                device=self.device
            )
        
        test_metrics = self.evaluate_on_test()
        
        self.logger.info(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Test AUC:       {test_metrics['auc']:.4f}")
        self.logger.info(f"Test F1:        {test_metrics['f1']:.4f}")
        self.logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        self.logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
        self.logger.info(f"Gate Weights:   {test_metrics['gate_weights']}")
        
        # Сохранение метрик
        all_metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_val_auc': self.best_val_auc,
            'test_metrics': test_metrics,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'training_time': time.time() - start_time
        }
        
        metrics_path = os.path.join(self.config.output_dir, 'metrics.json')
        save_metrics(all_metrics, metrics_path)
        
        # Финальные графики
        try:
            self.plot_training_curves()
        except Exception as e:
            self.logger.warning(f"Ошибка при построении финальных графиков: {e}")
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        self.logger.info("=" * 60)
        
        return test_metrics


def main():
    """Main entry point"""
    config = Config()
    
    trainer = Trainer(config)
    test_metrics = trainer.train()
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC:       {test_metrics['auc']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
