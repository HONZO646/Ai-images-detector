"""
Evaluation модуль для NPR AI/Real Detector
Метрики, ROC-кривые, confusion matrix, reliability diagrams
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import logging

from config import Config
from models.detector import NPRDetector, create_detector
from data.dataset import prepare_datasets, create_dataloaders
from utils import compute_metrics, get_device

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator для NPR Detector"""
    
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = get_device(config.training.device)
        
        # Model
        self.model = create_detector(config.model, device=self.device, 
                                     pretrained_path=checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Data
        _, _, self.test_dataset = prepare_datasets(config)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )
        
        # Results storage
        self.predictions = []
        self.targets = []
        self.gate_weights = {
            'semantic': [],
            'spatial': [],
            'frequency': []
        }
        
        # Output dir
        os.makedirs(config.output_dir, exist_ok=True)
    
    @torch.no_grad()
    def run_evaluation(self) -> Dict[str, float]:
        """Полная оценка модели на test set"""
        logger.info("Запуск evaluation на test set...")
        
        for gray, rgb, labels in self.test_loader:
            gray = gray.to(self.device)
            rgb = rgb.to(self.device)
            
            output = self.model(gray, rgb)
            logits = output['probability'].squeeze()
            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            self.predictions.extend(probabilities.flatten().tolist())
            self.targets.extend(labels.numpy().flatten().tolist())
            
            if 'gate_weights' in output:
                gates = output['gate_weights'].cpu().numpy()
                self.gate_weights['semantic'].extend(gates[:, 0].tolist())
                self.gate_weights['spatial'].extend(gates[:, 1].tolist())
                self.gate_weights['frequency'].extend(gates[:, 2].tolist())
        
        self.predictions = np.array(self.predictions)
        self.targets = np.array(self.targets)
        
        # Compute metrics
        metrics = compute_metrics(self.predictions, self.targets)
        
        # Additional metrics
        metrics['average_precision'] = average_precision_score(self.targets, self.predictions)
        metrics['gate_weights_mean'] = {
            k: np.mean(v) for k, v in self.gate_weights.items()
        }
        metrics['gate_weights_std'] = {
            k: np.std(v) for k, v in self.gate_weights.items()
        }
        
        logger.info(f"Test Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Test AUC:       {metrics['auc']:.4f}")
        logger.info(f"Test F1:        {metrics['f1']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {metrics['recall']:.4f}")
        
        return metrics
    
    def plot_roc_curve(self, save_path: str = None):
        """ROC кривая"""
        fpr, tpr, _ = roc_curve(self.targets, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.output_dir, 'roc_curve.png')
        
        plt.savefig(save_path, dpi=150)
        logger.info(f"ROC curve saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, threshold: float = 0.5, save_path: str = None):
        """Confusion matrix"""
        predictions_binary = (self.predictions >= threshold).astype(int)
        
        cm = confusion_matrix(self.targets, predictions_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Real (0)', 'AI (1)'],
                   yticklabels=['Real (0)', 'AI (1)'])
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix (threshold={threshold})', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved: {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.targets, self.predictions)
        avg_precision = average_precision_score(self.targets, self.predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.output_dir, 'precision_recall_curve.png')
        
        plt.savefig(save_path, dpi=150)
        logger.info(f"PR curve saved: {save_path}")
        plt.close()
    
    def plot_prediction_distribution(self, save_path: str = None):
        """Распределение предсказаний по классам"""
        plt.figure(figsize=(10, 6))
        
        # Real predictions
        real_preds = self.predictions[self.targets == 0]
        ai_preds = self.predictions[self.targets == 1]
        
        plt.hist(real_preds, bins=50, alpha=0.6, label='Real', color='green', density=True)
        plt.hist(ai_preds, bins=50, alpha=0.6, label='AI', color='red', density=True)
        
        plt.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
        plt.xlabel('P(AI)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Prediction Distribution by Class', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.output_dir, 'prediction_distribution.png')
        
        plt.savefig(save_path, dpi=150)
        logger.info(f"Prediction distribution saved: {save_path}")
        plt.close()
    
    def plot_reliability_diagram(self, n_bins: int = 10, save_path: str = None):
        """Reliability diagram для калибровки"""
        prob_true, prob_pred = calibration_curve(
            self.targets, self.predictions, n_bins=n_bins, strategy='uniform'
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 's-', color='blue', lw=2, label='Model')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Reliability Diagram', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.output_dir, 'reliability_diagram.png')
        
        plt.savefig(save_path, dpi=150)
        logger.info(f"Reliability diagram saved: {save_path}")
        plt.close()
    
    def plot_gate_weights(self, save_path: str = None):
        """Распределение весов adaptive gating"""
        plt.figure(figsize=(10, 6))
        
        branches = ['semantic', 'spatial', 'frequency']
        means = [np.mean(self.gate_weights[b]) for b in branches]
        stds = [np.std(self.gate_weights[b]) for b in branches]
        
        x = np.arange(len(branches))
        plt.bar(x, means, yerr=stds, capsize=5, color=['skyblue', 'lightgreen', 'salmon'])
        plt.xticks(x, branches)
        plt.ylabel('Mean Gate Weight', fontsize=12)
        plt.title('Adaptive Gating: Branch Importance', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.config.output_dir, 'gate_weights.png')
        
        plt.savefig(save_path, dpi=150)
        logger.info(f"Gate weights saved: {save_path}")
        plt.close()
    
    def generate_classification_report(self, threshold: float = 0.5) -> str:
        """Текстовый classification report"""
        predictions_binary = (self.predictions >= threshold).astype(int)
        report = classification_report(
            self.targets, 
            predictions_binary,
            target_names=['Real', 'AI'],
            output_dict=False
        )
        return report
    
    def run_full_evaluation(self):
        """Полная evaluation + все графики"""
        logger.info("=" * 60)
        logger.info("ЗАПУСК ПОЛНОЙ ОЦЕНКИ")
        logger.info("=" * 60)
        
        # Метрики
        metrics = self.run_evaluation()
        
        # Графики
        logger.info("\nГенерация графиков...")
        
        self.plot_roc_curve()
        self.plot_confusion_matrix()
        self.plot_precision_recall_curve()
        self.plot_prediction_distribution()
        self.plot_reliability_diagram()
        self.plot_gate_weights()
        
        # Classification report
        report = self.generate_classification_report()
        logger.info(f"\nClassification Report:\n{report}")
        
        # Сохранение отчёта
        report_path = os.path.join(self.config.output_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("NPR AI/Real Detector - Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write("\n\nGate Weights:\n")
            for k, v in metrics['gate_weights_mean'].items():
                f.write(f"  {k}: {v:.4f} ± {metrics['gate_weights_std'][k]:.4f}\n")
        
        logger.info(f"\nОтчёт сохранён: {report_path}")
        logger.info("=" * 60)
        logger.info("ОЦЕНКА ЗАВЕРШЕНА")
        logger.info("=" * 60)
        
        return metrics


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NPR Detector Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    config = Config()
    evaluator = Evaluator(config, args.checkpoint)
    metrics = evaluator.run_full_evaluation()
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
