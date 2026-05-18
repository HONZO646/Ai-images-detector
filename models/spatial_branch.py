"""
Spatial Branch: NPR + Sobel Gradients + LBP
Различные пространственные признаки для детекции артефактов
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class NPRFeatureExtractor(nn.Module):
    """
    Differentiable NPR (Neighboring Pixel Relationship) extractor
    3x3 окно, 8 направлений, 4 статистики на каждое
    """
    
    def __init__(self):
        super().__init__()
        # Веса для 8 направлений (всё differentiable)
        self.register_buffer('directions', torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],  # NW, N, NE
            [0, -1],           [0, 1],   # W,      E
            [1, -1],  [1, 0],  [1, 1]    # SW, S, SE
        ], dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] grayscale изображение
        Возвращает: [B, 32] (8 направлений × 4 статистики)
        """
        B, C, H, W = x.shape
        
        # Pad для обработки границ
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        
        # Центральный пиксель
        center = x_padded[:, :, 1:-1, 1:-1]  # [B, 1, H, W]
        
        features = []
        eps = 1e-6
        
        for i in range(8):
            dy, dx = self.directions[i].long()
            
            # Соседнее направление
            neighbor = x_padded[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]
            
            # Разность
            diff = center - neighbor  # [B, 1, H, W]
            diff_flat = diff.reshape(B, -1)  # [B, H*W]
            
            # Статистики
            mean = diff_flat.mean(dim=1, keepdim=True)
            variance = diff_flat.var(dim=1, keepdim=True)
            std = torch.sqrt(variance + eps)
            
            # Skewness
            diff_centered = diff_flat - mean
            skew = (diff_centered ** 3).mean(dim=1, keepdim=True) / (std ** 3)
            
            # Kurtosis (excess)
            std4 = std ** 4
            kurtosis = (diff_centered ** 4).mean(dim=1, keepdim=True) / std4 - 3.0
            
            # Конкатенация 4 статистик
            features.append(torch.cat([mean, std, skew, kurtosis], dim=1))
        
        # [B, 32]
        return torch.cat(features, dim=1)


class SobelExtractor(nn.Module):
    """
    Differentiable Sobel gradient extractor
    Вычисляет dx, dy, magnitude градиенты
    """
    
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        Возвращает: [B, 3] (dx_mean, dy_mean, magnitude_mean)
        """
        # Градиенты
        dx = F.conv2d(x, self.sobel_x, padding=1)
        dy = F.conv2d(x, self.sobel_y, padding=1)
        
        # Magnitude
        magnitude = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        
        # Статистики
        dx_mean = dx.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        dy_mean = dy.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        mag_mean = magnitude.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        
        return torch.cat([dx_mean, dy_mean, mag_mean], dim=1)


class LBPExtractor(nn.Module):
    """
    Differentiable Local Binary Patterns (LBP)
    Упрощённая differentiable версия для извлечения текстурных признаков
    """
    
    def __init__(self, radius: int = 1, n_points: int = 8):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        Возвращает: [B, 3] (mean, std, uniform_ratio)
        """
        B, C, H, W = x.shape
        r = self.radius
        
        # Центр
        center = x[:, :, r:H-r, r:W-r].reshape(B, -1)  # [B, (H-2r)*(W-2r)]
        
        # 8 направлений (прямые соседи, без интерполяции)
        neighbors_offsets = [
            (-r, -r), (-r, 0), (-r, r),   # NW, N, NE
            (0, -r),          (0, r),      # W,      E
            (r, -r),  (r, 0),  (r, r)      # SW, S, SE
        ]
        
        lbp_codes = []
        
        for dy, dx in neighbors_offsets:
            # Соседний пиксель
            neighbor = x[:, :, r+dy:H-r+dy, r+dx:W-r+dx].reshape(B, -1)
            # Бинарное сравнение
            binary = (neighbor >= center).float()
            lbp_codes.append(binary)
        
        # Stack: [B, (H-2r)*(W-2r), 8]
        lbp_codes = torch.stack(lbp_codes, dim=2)
        
        # Статистики
        # Количество единиц
        ones_mean = lbp_codes.mean(dim=2).mean(dim=1, keepdim=True)  # [B, 1]
        
        # Количество переходов 0->1 и 1->0 (uniform patterns)
        diffs = torch.abs(lbp_codes[:, :, 1:] - lbp_codes[:, :, :-1])  # [B, N, 7]
        transitions = diffs.sum(dim=2)  # [B, N]
        uniform_ratio = (transitions <= 2).float().mean(dim=1, keepdim=True)  # [B, 1]
        
        # Standard deviation
        ones_std = torch.sqrt(lbp_codes.var(dim=2).clamp(min=1e-6)).mean(dim=1, keepdim=True)  # [B, 1]
        
        return torch.cat([ones_mean, ones_std, uniform_ratio], dim=1)  # [B, 3]


class SpatialFusion(nn.Module):
    """
    Fusion модуль для объединения NPR + Sobel + LBP
    """
    
    def __init__(self, npr_dim: int = 32, sobel_dim: int = 3, lbp_dim: int = 3,
                 embedding_dim: int = 128):
        super().__init__()
        
        total_dim = npr_dim + sobel_dim + lbp_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, npr_features: torch.Tensor, 
                sobel_features: torch.Tensor,
                lbp_features: torch.Tensor) -> torch.Tensor:
        """
        npr_features: [B, 32]
        sobel_features: [B, 3]
        lbp_features: [B, 3]
        Возвращает: [B, embedding_dim]
        """
        combined = torch.cat([npr_features, sobel_features, lbp_features], dim=1)
        return self.fusion(combined)


class SpatialBranch(nn.Module):
    """
    Полная пространственная ветка: NPR + Gradients + LBP → Fusion → Embedding
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extractors
        self.npr_extractor = NPRFeatureExtractor(bins=config.npr_bins)
        self.sobel_extractor = SobelExtractor()
        self.lbp_extractor = LBPExtractor(
            radius=config.lbp_radius,
            n_points=config.lbp_n_points
        )
        
        # Fusion
        npr_dim = 32  # 8 направлений × 4 статистики
        sobel_dim = 3  # dx, dy, magnitude
        lbp_dim = 3  # ones_mean, transitions_mean, uniform_ratio
        
        self.fusion = SpatialFusion(
            npr_dim=npr_dim,
            sobel_dim=sobel_dim,
            lbp_dim=lbp_dim,
            embedding_dim=config.spatial_embedding_dim
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] grayscale изображение
        Возвращает: [B, spatial_embedding_dim] embedding
        """
        # Извлечение признаков
        npr_features = self.npr_extractor(x)
        sobel_features = self.sobel_extractor(x)
        lbp_features = self.lbp_extractor(x)
        
        # Fusion
        embedding = self.fusion(npr_features, sobel_features, lbp_features)
        
        return embedding


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    from config import Config
    
    config = Config()
    model = SpatialBranch(config.model)
    
    # Тестовый вход
    x = torch.randn(2, 1, 256, 256)
    
    # Проверка прямого прохода
    embedding = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Expected: [2, {config.model.spatial_embedding_dim}]")
