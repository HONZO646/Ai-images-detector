"""
NPRDetector: Main detector class
Объединяет все три ветки и fusion модуль
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import os

from .spatial_branch import SpatialBranch
from .frequency_branch import FrequencyBranch
from .semantic_branch import SemanticBranch, SimpleSemanticBranch
from .fusion import CrossAttentionFusion, AdaptiveGating, ClassifierHead, UncertaintyEstimator


class NPRDetector(nn.Module):
    """
    Полный детектор AI/Real изображений
    Три паралельные ветки + Cross-Attention Fusion
    """
    
    def __init__(self, config, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Ветки
        self.spatial_branch = SpatialBranch(config)
        self.frequency_branch = FrequencyBranch(config)
        self.semantic_branch = SimpleSemanticBranch(config, device=device)
        
        # Fusion модули
        artifact_dim = config.spatial_embedding_dim + config.freq_embedding_dim
        
        self.cross_attention = CrossAttentionFusion(
            semantic_dim=config.semantic_embedding_dim,
            artifact_dim=artifact_dim,
            embedding_dim=config.fusion_embedding_dim,
            n_heads=config.n_attention_heads,
            dropout=config.attention_dropout
        )
        
        self.adaptive_gating = AdaptiveGating(
            input_dim=config.fusion_embedding_dim,
            n_branches=3
        )
        
        # Classifier
        self.classifier = ClassifierHead(
            input_dim=config.fusion_embedding_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        )
        
        # Uncertainty estimation (опционально)
        self.use_uncertainty = hasattr(config, 'use_uncertainty') and config.use_uncertainty
        if self.use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(n_samples=10)
    
    def forward(self, x_gray: torch.Tensor, x_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_gray: [B, 1, H, W] grayscale изображение (для spatial + freq)
        x_rgb: [B, 3, H', W'] RGB изображение (для semantic)
        
        Возвращает словарь:
            - 'probability': [B, 1] P(AI)
            - 'uncertainty': [B, 1] (опционально)
            - 'spatial_embedding': [B, spatial_dim]
            - 'frequency_embedding': [B, freq_dim]
            - 'semantic_embedding': [B, semantic_dim]
            - 'gate_weights': [B, 3]
        """
        # Извлечение признаков из каждой ветки
        spatial_emb = self.spatial_branch(x_gray)
        freq_emb = self.frequency_branch(x_gray)
        semantic_out = self.semantic_branch(x_rgb)
        semantic_emb = semantic_out['embedding']
        
        # Конкатенация spatial + frequency
        spatial_freq = torch.cat([spatial_emb, freq_emb], dim=1)
        
        # Cross-Attention fusion
        cross_attn_out = self.cross_attention(semantic_emb, spatial_freq)
        
        # Adaptive gating
        fused, gate_weights = self.adaptive_gating(
            semantic_emb, 
            spatial_emb, 
            freq_emb
        )
        
        # Комбинация cross-attention и gating
        final_embedding = (cross_attn_out + fused) / 2.0
        
        # Classification
        probability = self.classifier(final_embedding)
        
        result = {
            'probability': probability,
            'spatial_embedding': spatial_emb,
            'frequency_embedding': freq_emb,
            'semantic_embedding': semantic_emb,
            'gate_weights': gate_weights,
            'final_embedding': final_embedding
        }
        
        # Uncertainty estimation (опционально)
        if self.use_uncertainty:
            mean_pred, uncertainty = self.uncertainty_estimator(self.classifier, final_embedding)
            result['probability'] = mean_pred
            result['uncertainty'] = uncertainty
        
        return result
    
    def predict(self, x_gray: torch.Tensor, x_rgb: torch.Tensor, 
                threshold: float = 0.5) -> Dict[str, any]:
        """
        Инференс с постобработкой
        Возвращает словарь с интерпретируемыми результатами
        """
        self.eval()
        
        with torch.no_grad():
            output = self(x_gray, x_rgb)
            
            prob_ai = output['probability'].item()
            pred_class = "AI_GENERATED" if prob_ai >= threshold else "REAL"
            confidence = abs(prob_ai - 0.5) * 2  # 0.0 - 1.0
            
            result = {
                'probability_ai': prob_ai,
                'probability_real': 1.0 - prob_ai,
                'prediction': pred_class,
                'confidence': confidence,
                'threshold': threshold
            }
            
            # Добавляем uncertainty если есть
            if 'uncertainty' in output:
                result['uncertainty'] = output['uncertainty'].item()
            
            # Gate weights для интерпретируемости
            result['gate_weights'] = {
                'semantic': output['gate_weights'][0, 0].item(),
                'spatial': output['gate_weights'][0, 1].item(),
                'frequency': output['gate_weights'][0, 2].item()
            }
            
            return result
    
    def count_parameters(self) -> Dict[str, int]:
        """Подсчёт параметров по модулям"""
        params = {}
        
        for name, module in self.named_children():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            params[name] = {
                'total': total,
                'trainable': trainable,
                'frozen': total - trainable
            }
        
        return params


class NPRDetectorWithEarlyExit(NPRDetector):
    """
    Версия детектора с early exit по confidence
    Для cascade inference - быстрая классификация уверенных примеров
    """
    
    def __init__(self, config, device: str = 'cpu', 
                 early_exit_threshold: float = 0.7):
        super().__init__(config, device)
        self.early_exit_threshold = early_exit_threshold
    
    def predict_cascade(self, x_gray: torch.Tensor, x_rgb: torch.Tensor) -> Dict[str, any]:
        """
        Cascade prediction с early exit
        Если confidence > early_exit_threshold, пропускаем сложные ветки
        """
        self.eval()
        
        with torch.no_grad():
            # Сначала только быстрая spatial ветка
            spatial_emb = self.spatial_branch(x_gray)
            
            # Простой классификатор на spatial features
            quick_pred = self.classifier(spatial_emb).item()
            quick_confidence = abs(quick_pred - 0.5) * 2
            
            # Early exit если уверены
            if quick_confidence >= self.early_exit_threshold:
                return {
                    'probability_ai': quick_pred,
                    'probability_real': 1.0 - quick_pred,
                    'prediction': "AI_GENERATED" if quick_pred >= 0.5 else "REAL",
                    'confidence': quick_confidence,
                    'early_exit': True,
                    'branch_used': 'spatial_only'
                }
            
            # Иначе полный проход
            return {
                **self.predict(x_gray, x_rgb),
                'early_exit': False,
                'branch_used': 'full'
            }


# ============================================================
# Factory функции для удобства
# ============================================================
def create_detector(config, device: str = 'cpu', pretrained_path: Optional[str] = None) -> NPRDetector:
    """Создание детектора с опциональной загрузкой весов"""
    
    model = NPRDetector(config, device=device)
    
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Загружены веса из: {pretrained_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    
    return model


def load_npr_pipeline(checkpoint_path: str = None, device: str = 'auto'):
    """
    Обратная совместимость с оригинальным npr.py
    """
    from config import Config
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = Config()
    model = create_detector(config, device=device, pretrained_path=checkpoint_path)
    
    return model


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    from config import Config
    
    config = Config()
    
    # Создание модели
    model = NPRDetector(config.model)
    
    # Тестовые входы
    x_gray = torch.randn(2, 1, 256, 256)
    x_rgb = torch.randn(2, 3, 224, 224)
    
    # Прямой проход
    output = model(x_gray, x_rgb)
    
    print("\n=== NPRDetector Test ===")
    print(f"Input (gray): {x_gray.shape}")
    print(f"Input (RGB): {x_rgb.shape}")
    print(f"Output probability: {output['probability'].shape}")
    print(f"Output gate_weights: {output['gate_weights'].shape}")
    
    # Подсчёт параметров
    params = model.count_parameters()
    print(f"\n=== Parameters ===")
    for name, counts in params.items():
        print(f"  {name}: {counts['trainable']:,} trainable / {counts['total']:,} total")
    
    # Тест predict
    result = model.predict(x_gray, x_rgb)
    print(f"\n=== Prediction ===")
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value}")
