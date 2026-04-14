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
    Cross-Attention: Query=semantic, Key/Value=spatial+freq
    Модель сама решает, какому сигналу доверять
    """
    
    def __init__(self, 
                 semantic_dim: int = 128,
                 artifact_dim: int = 256,  # spatial + freq
                 embedding_dim: int = 128,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        
        # Query projection (semantic)
        self.query_proj = nn.Linear(semantic_dim, embedding_dim)
        
        # Key/Value projections (spatial + freq)
        self.key_proj = nn.Linear(artifact_dim, embedding_dim)
        self.value_proj = nn.Linear(artifact_dim, embedding_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
    
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
        # Add sequence dimension for attention
        semantic_seq = self.query_proj(semantic).unsqueeze(1)  # [B, 1, embedding_dim]
        artifact_seq = self.key_proj(spatial_freq).unsqueeze(1)  # [B, 1, embedding_dim]
        value_seq = self.value_proj(spatial_freq).unsqueeze(1)  # [B, 1, embedding_dim]
        
        # Cross-attention
        attn_output, attn_weights = self.multihead_attn(
            query=semantic_seq,
            key=artifact_seq,
            value=value_seq,
            key_padding_mask=mask
        )  # [B, 1, embedding_dim]
        
        # Residual connection + norm
        attended = self.norm1(semantic_seq + attn_output)  # [B, 1, embedding_dim]
        
        # Output projection
        output = self.output_proj(attended.squeeze(1))  # [B, embedding_dim]
        
        return output


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
        
        # Финальный слой
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        Возвращает: [B, 1] probability P(AI)
        """
        return self.classifier(x)


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
