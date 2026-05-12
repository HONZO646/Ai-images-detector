"""
Fusion Module: Cross-Attention + Adaptive Gating
Объединение spatial, frequency и semantic веток
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossAttentionFusion(nn.Module):
    """
    Fusion MLP для semantic и spatial+freq признаков.
    NOTE: для единичного токена attention вырождается, поэтому используем MLP-фьюжн.
    """
    
    def __init__(self, 
                 semantic_dim: int = 128,
                 artifact_dim: int = 256,  # spatial + freq
                 embedding_dim: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        fusion_in_dim = semantic_dim + artifact_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                semantic: torch.Tensor,
                spatial_freq: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        semantic: [B, semantic_dim]
        spatial_freq: [B, artifact_dim]
        mask: optional attention mask
        Возвращает: [B, embedding_dim] fused representation
        """
        fused_input = torch.cat([semantic, spatial_freq], dim=1)
        return self.fusion(fused_input)


class AdaptiveGating(nn.Module):
    """
    Adaptive gating mechanism для динамического взвешивания веток
    Learns to trust certain branches more based on input characteristics
    """
    
    def __init__(self, input_dim: int = 128, n_branches: int = 3):
        super().__init__()
        self.n_branches = n_branches
        
        # Gate генератор
        self.gate_generator = nn.Sequential(
            nn.Linear(input_dim * n_branches, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_branches),
            nn.Softmax(dim=1)  # Веса суммируются в 1
        )
    
    def forward(self, 
                semantic: torch.Tensor,
                spatial: torch.Tensor,
                frequency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает: (weighted_combination, gate_weights)
        """
        B = semantic.shape[0]
        
        # Конкатенация всех веток
        combined = torch.cat([semantic, spatial, frequency], dim=1)  # [B, 3*dim]
        
        # Генерация весов
        gate_weights = self.gate_generator(combined)  # [B, 3]
        
        # Взвешивание
        # Стек всех эмбеддингов: [B, 3, dim]
        all_embeddings = torch.stack([semantic, spatial, frequency], dim=1)
        
        # Weighted sum: [B, 3, 1] * [B, 3, dim] → [B, dim]
        gate_weights_expanded = gate_weights.unsqueeze(2)  # [B, 3, 1]
        weighted = (gate_weights_expanded * all_embeddings).sum(dim=1)  # [B, dim]
        
        return weighted, gate_weights


class ClassifierHead(nn.Module):
    """
    MLP классификатор для финального предсказания
    Возвращает logits (без sigmoid) для совместимости с BCEWithLogitsLoss
    """
    
    def __init__(self, input_dim: int = 128, hidden_dims: list = None, dropout_rate: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Финальный слой — только logits, без sigmoid
        # Sigmoid применяется отдельно в inference
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        Возвращает: [B, 1] logits (не probability!)
        """
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Инференс: возвращает probability через sigmoid
        """
        return torch.sigmoid(self.forward(x))


class UncertaintyEstimator(nn.Module):
    """
    Оценка неопределённости предсказания
    Через Monte Carlo Dropout или энтропию предсказаний
    """
    
    def __init__(self, n_samples: int = 10):
        super().__init__()
        self.n_samples = n_samples
    
    def forward(self, classifier: ClassifierHead, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout для оценки uncertainty
        Возвращает: (mean_prediction, uncertainty)
        """
        predictions = []
        
        # Включаем dropout для инференса
        classifier.train()
        
        for _ in range(self.n_samples):
            pred = classifier(x)
            predictions.append(pred)
        
        classifier.eval()
        
        # Статистики
        predictions = torch.cat(predictions, dim=1)  # [B, n_samples]
        mean_pred = predictions.mean(dim=1, keepdim=True)  # [B, 1]
        uncertainty = predictions.std(dim=1, keepdim=True)  # [B, 1]
        
        return mean_pred, uncertainty


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    # Тест Cross-Attention Fusion
    fusion = CrossAttentionFusion(
        semantic_dim=128,
        artifact_dim=256,
        embedding_dim=128,
        n_heads=4
    )
    
    semantic = torch.randn(2, 128)
    spatial_freq = torch.randn(2, 256)
    
    fused = fusion(semantic, spatial_freq)
    print(f"Fused shape: {fused.shape}")
    print(f"Expected: [2, 128]")
    
    # Тест Adaptive Gating
    gating = AdaptiveGating(input_dim=128, n_branches=3)
    
    semantic = torch.randn(2, 128)
    spatial = torch.randn(2, 128)
    frequency = torch.randn(2, 128)
    
    weighted, gates = gating(semantic, spatial, frequency)
    print(f"Weighted shape: {weighted.shape}")
    print(f"Gate weights shape: {gates.shape}")
    print(f"Gate weights sum: {gates.sum(dim=1)}")  # Должно быть ~1.0
