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
        # bins не используется напрямую (8 направлений × 4 статистики = 32)
        # параметр оставлен для совместимости с конфигом
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
    Differentiable Sobel gradient extractor.
    Вычисляет расширенный набор статистик по градиентному полю:
    mean, std, p90, energy для dx/dy/magnitude,
    плюс edge_ratio, dx_dy_corr, isotropy, mag_cv.
    Итого: [B, 16]
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
        Возвращает: [B, 16]
          dx:  mean, std, p90, energy                (4)
          dy:  mean, std, p90, energy                (4)
          mag: mean, std, p90, energy                (4)
          structural: edge_ratio, dx_dy_corr, isotropy, mag_cv (4)
        """
        B = x.shape[0]
        eps = 1e-6
        
        # Градиенты
        dx = F.conv2d(x, self.sobel_x, padding=1)
        dy = F.conv2d(x, self.sobel_y, padding=1)
        
        # Magnitude
        magnitude = torch.sqrt(dx ** 2 + dy ** 2 + eps)
        
        def rich_stats(t: torch.Tensor):
            """mean, std, 90th percentile, normalised energy → [B, 4]"""
            flat = t.reshape(B, -1)
            m   = flat.mean(dim=1, keepdim=True)
            s   = flat.std(dim=1, keepdim=True).clamp(min=eps)
            p90 = torch.quantile(flat, 0.90, dim=1, keepdim=True)
            energy = (flat ** 2).mean(dim=1, keepdim=True)
            return torch.cat([m, s, p90, energy], dim=1)
        
        dx_stats  = rich_stats(dx)   # [B, 4]
        dy_stats  = rich_stats(dy)   # [B, 4]
        mag_stats = rich_stats(magnitude) # [B, 4]
        
        # Дополнительные структурные признаки [B, 4]
        mag_flat = magnitude.reshape(B, -1)
        mag_mean = mag_flat.mean(dim=1, keepdim=True)
        edge_ratio = (mag_flat > mag_mean).float().mean(dim=1, keepdim=True)
        
        dx_c = dx.reshape(B, -1) - dx.reshape(B, -1).mean(dim=1, keepdim=True)
        dy_c = dy.reshape(B, -1) - dy.reshape(B, -1).mean(dim=1, keepdim=True)
        dx_std = dx.reshape(B, -1).std(dim=1, keepdim=True).clamp(min=eps)
        dy_std = dy.reshape(B, -1).std(dim=1, keepdim=True).clamp(min=eps)
        dx_dy_corr = (dx_c * dy_c).mean(dim=1, keepdim=True) / (dx_std * dy_std)
        
        isotropy = dx.reshape(B, -1).abs().mean(dim=1, keepdim=True) / (
            dy.reshape(B, -1).abs().mean(dim=1, keepdim=True) + eps
        )
        
        mag_cv = mag_flat.std(dim=1, keepdim=True) / (mag_mean + eps)
        
        extra = torch.cat([edge_ratio, dx_dy_corr, isotropy, mag_cv], dim=1)  # [B, 4]
        
        return torch.cat([dx_stats, dy_stats, mag_stats, extra], dim=1)  # [B, 16]


class LBPExtractor(nn.Module):
    """
    Differentiable Local Binary Patterns (LBP).
    Расширенная differentiable версия:
    Гистограмма uniform/non-uniform паттернов (8 bins) + структурные признаки.
    Итого: [B, 16]
    """
    
    def __init__(self, radius: int = 1, n_points: int = 8):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        Возвращает: [B, 16]
          [0..7]  гистограмма числа единиц в LBP-коде (0..8 единиц)  (8 bins)
          [8]     uniform_ratio (переходов ≤ 2)
          [9]     transitions_mean
          [10]    transitions_std
          [11]    ones_mean (норм.)
          [12]    ones_std (норм.)
          [13]    uniform_ratio верхняя треть
          [14]    uniform_ratio центральная треть
          [15]    uniform_ratio нижняя треть
        """
        B, C, H, W = x.shape
        r = self.radius
        eps = 1e-6
        
        # Центр
        center = x[:, :, r:H-r, r:W-r].reshape(B, -1)  # [B, (H-2r)*(W-2r)]
        
        # 8 направлений (прямые соседи, без интерполяции)
        neighbors_offsets = [
            (-r, -r), (-r, 0), (-r, r),
            (0, -r),          (0, r),
            (r, -r),  (r, 0),  (r, r)
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
        
        # Число единиц в каждом LBP-коде [B, N]
        ones_per_pix = lbp_codes.sum(dim=2)  # range [0, 8]
        
        # 8-бинная гистограмма [B, 8]
        hist_bins = 8
        hist = torch.zeros(B, hist_bins, device=x.device)
        for b in range(hist_bins):
            if b < hist_bins - 1:
                hist[:, b] = (ones_per_pix == b).float().mean(dim=1)
            else:
                hist[:, b] = (ones_per_pix >= b).float().mean(dim=1)
        
        # Переходы между соседними битами
        diffs = torch.abs(lbp_codes[:, :, 1:] - lbp_codes[:, :, :-1])  # [B, N, 7]
        transitions = diffs.sum(dim=2)  # [B, N], range [0, 7]
        
        uniform_ratio = (transitions <= 2).float().mean(dim=1, keepdim=True)  # [B, 1]
        trans_mean = transitions.mean(dim=1, keepdim=True)                    # [B, 1]
        trans_std  = transitions.std(dim=1, keepdim=True).clamp(min=eps)     # [B, 1]
        
        ones_mean = ones_per_pix.mean(dim=1, keepdim=True) / 8.0             # [B, 1] нормализовано
        ones_std  = ones_per_pix.std(dim=1, keepdim=True).clamp(min=eps) / 8.0  # [B, 1] нормализовано
        
        # Uniform ratio по 3 горизонтальным полосам изображения
        H_crop = H - 2 * r
        W_crop = W - 2 * r
        transitions_spatial = transitions.reshape(B, H_crop, W_crop)  # [B, H', W']
        h3 = H_crop // 3
        top_ur    = (transitions_spatial[:, :h3, :] <= 2).float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        center_ur = (transitions_spatial[:, h3:2*h3, :] <= 2).float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        bot_ur    = (transitions_spatial[:, 2*h3:, :] <= 2).float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        
        return torch.cat([
            hist,           # [B, 8]
            uniform_ratio,  # [B, 1]
            trans_mean,     # [B, 1]
            trans_std,      # [B, 1]
            ones_mean,      # [B, 1]
            ones_std,       # [B, 1]
            top_ur,         # [B, 1]
            center_ur,      # [B, 1]
            bot_ur,         # [B, 1]
        ], dim=1)  # [B, 16]


class SpatialFusion(nn.Module):
    """
    Fusion модуль для объединения NPR + Sobel + LBP.
    Входные размерности: NPR=32, Sobel=16, LBP=16 → total=64
    Баланс: NPR 50%, Sobel 25%, LBP 25%.
    """
    
    def __init__(self, npr_dim: int = 32, sobel_dim: int = 16, lbp_dim: int = 16,
                 embedding_dim: int = 128):
        super().__init__()
        
        total_dim = npr_dim + sobel_dim + lbp_dim  # 64
        
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
        self.npr_extractor = NPRFeatureExtractor()
        self.sobel_extractor = SobelExtractor()
        self.lbp_extractor = LBPExtractor(
            radius=config.lbp_radius,
            n_points=config.lbp_n_points
        )
        
        # Fusion
        npr_dim   = 32  # 8 направлений × 4 статистики
        sobel_dim = 16  # 4×3 карты + 4 структурных признака
        lbp_dim   = 16  # 8-bin гистограмма + 8 структурных признаков
        
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
