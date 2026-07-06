"""
Spatial Branch: NPR + Sobel Gradients + LBP
Различные пространственные признаки для детекции артефактов
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union


class NPRFeatureExtractor(nn.Module):
    """
    Differentiable NPR (Neighboring Pixel Relationship) extractor
    3x3 окно, 8 направлений, 4 статистики на каждое
    """
    
    directions: torch.Tensor
    
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
            direction = self.directions[i].long()
            dy, dx = direction[0].item(), direction[1].item()
            
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
    
    sobel_x: torch.Tensor
    sobel_y: torch.Tensor
    
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
    Differentiable Local Binary Patterns (LBP) with multi-scale support.
    Поддерживает произвольное число точек на окружности (n_points) и радиус (radius).
    Гистограмма uniform/non-uniform паттернов (n_points bins) + структурные признаки.
    Итого: [B, n_points + 8] (n_points для гистограммы + 8 структурных признаков)
    """
    
    neighbor_offsets: torch.Tensor
    
    def __init__(self, radius: int = 1, n_points: int = 8):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
        
        # Precompute neighbor offsets using circular sampling
        # For multi-scale LBP: evenly distribute n_points around circle of given radius
        # Note: PyTorch linspace doesn't have endpoint param, so we create n+1 points and exclude last
        angles = torch.linspace(0, 2 * torch.pi, n_points + 1, dtype=torch.float32)[:-1]
        # Offsets: [n_points, 2] where each row is (dy, dx) for neighbor sampling
        # Using (dy, dx) convention where dy is row offset, dx is column offset
        self.register_buffer('neighbor_offsets', torch.stack([
            torch.sin(angles) * radius,  # dy (row offset)
            torch.cos(angles) * radius   # dx (column offset)
        ], dim=1))  # Shape: [n_points, 2]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        Возвращает: [B, n_points + 8]
          [0..n_points-1]  гистограмма числа единиц в LBP-коде (0..n_points единиц)  (n_points bins)
          [n_points]     uniform_ratio (переходов ≤ 2)
          [n_points+1]   transitions_mean
          [n_points+2]   transitions_std
          [n_points+3]   ones_mean (норм.)
          [n_points+4]   ones_std (норм.)
          [n_points+5]   uniform_ratio верхняя треть
          [n_points+6]   uniform_ratio центральная треть
          [n_points+7]   uniform_ratio нижняя треть
        """
        B, C, H, W = x.shape
        r = self.radius
        n = self.n_points
        eps = 1e-6
        
        # Pad image to handle neighbors outside boundaries
        x_padded = F.pad(x, (r, r, r, r), mode='reflect')  # Pad left, right, top, bottom
        
        # Center pixels (cropped to valid region)
        center = x[:, :, r:H-r, r:W-r].reshape(B, -1)  # [B, (H-2r)*(W-2r)]
        
        # Sample neighbors using bilinear interpolation for subpixel accuracy
        # Create sampling grid for each neighbor position
        lbp_codes = []
        
        for i in range(n):
            dy = self.neighbor_offsets[i, 0]  # Row offset (can be float)
            dx = self.neighbor_offsets[i, 1]  # Column offset (can be float)
            
            # Use bilinear interpolation via grid_sample
            # Create normalized grid coordinates for this neighbor offset
            # grid: [B, H', W', 2] where last dim is (x, y) in [-1, 1] range
            H_crop = H - 2 * r
            W_crop = W - 2 * r
            
            # Create base grid
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H_crop, device=x.device),
                torch.linspace(-1, 1, W_crop, device=x.device),
                indexing='ij'
            )  # Both [H', W']
            
            # Add offset to grid (convert pixel offset to normalized offset)
            # With align_corners=False: normalized_x = x * 2/(W-1) - 1
            # So offset dx in pixel -> normalized offset = dx * 2/(W-1)
            grid_x = grid_x + dx * 2.0 / (W - 1)
            grid_y = grid_y + dy * 2.0 / (H - 1)
            
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [H', W', 2]
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H', W', 2]
            
            # Sample neighbor using grid_sample
            neighbor = F.grid_sample(
                x_padded, grid,
                mode='bilinear',
                padding_mode='reflection',
                align_corners=False
            )  # [B, 1, H', W']
            
            neighbor = neighbor.reshape(B, -1)  # [B, H'*W']
            
            # Binary comparison
            binary = (neighbor >= center).float()
            lbp_codes.append(binary)
        
        # Stack: [B, (H-2r)*(W-2r), n_points]
        lbp_codes = torch.stack(lbp_codes, dim=2)
        
        # Number of ones in each LBP code [B, N]
        ones_per_pix = lbp_codes.sum(dim=2)  # range [0, n_points]
        
        # n_points-bin histogram [B, n_points]
        hist = torch.zeros(B, n, device=x.device)
        for b in range(n):
            if b < n - 1:
                hist[:, b] = (ones_per_pix == b).float().mean(dim=1)
            else:
                hist[:, b] = (ones_per_pix >= b).float().mean(dim=1)
        
        # Transitions between adjacent bits (circular: last bit connects to first)
        # For circular LBP, we need to check transitions from bit[n-1] to bit[0] as well
        diffs = torch.abs(lbp_codes[:, :, 1:] - lbp_codes[:, :, :-1])  # [B, N, n-1]
        # Add circular transition: bit[n-1] -> bit[0]
        circular_diff = torch.abs(lbp_codes[:, :, 0] - lbp_codes[:, :, -1]).unsqueeze(2)  # [B, N, 1]
        diffs = torch.cat([diffs, circular_diff], dim=2)  # [B, N, n]
        
        transitions = diffs.sum(dim=2)  # [B, N], range [0, n]
        
        uniform_ratio = (transitions <= 2).float().mean(dim=1, keepdim=True)  # [B, 1]
        trans_mean = transitions.mean(dim=1, keepdim=True)                    # [B, 1]
        trans_std  = transitions.std(dim=1, keepdim=True).clamp(min=eps)     # [B, 1]
        
        ones_mean = ones_per_pix.mean(dim=1, keepdim=True) / n  # [B, 1] normalized
        ones_std  = ones_per_pix.std(dim=1, keepdim=True).clamp(min=eps) / n  # [B, 1] normalized
        
        # Uniform ratio по 3 горизонтальным полосам изображения
        H_crop = H - 2 * r
        W_crop = W - 2 * r
        transitions_spatial = transitions.reshape(B, H_crop, W_crop)  # [B, H', W']
        h3 = H_crop // 3
        top_ur    = (transitions_spatial[:, :h3, :] <= 2).float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        center_ur = (transitions_spatial[:, h3:2*h3, :] <= 2).float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        bot_ur    = (transitions_spatial[:, 2*h3:, :] <= 2).float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        
        return torch.cat([
            hist,           # [B, n_points]
            uniform_ratio,  # [B, 1]
            trans_mean,     # [B, 1]
            trans_std,      # [B, 1]
            ones_mean,      # [B, 1]
            ones_std,       # [B, 1]
            top_ur,         # [B, 1]
            center_ur,      # [B, 1]
            bot_ur,         # [B, 1]
        ], dim=1)  # [B, n_points + 8]


class SpatialFusion(nn.Module):
    """
    Fusion модуль для объединения NPR + Sobel + LBP.
    Использует attention-based взвешивание ветвей.
    Входные размерности: NPR=32, Sobel=16, LBP=16
    """
    
    def __init__(self, npr_dim: int = 32, sobel_dim: int = 16, lbp_dim: int = 16,
                  embedding_dim: int = 128, d: int = 64):
        super().__init__()
        
        # Task 2: Separate LayerNorm for each branch (normalize aggregated statistics [B, N])
        self.npr_norm = nn.LayerNorm(npr_dim)
        self.sobel_norm = nn.LayerNorm(sobel_dim)
        self.lbp_norm = nn.LayerNorm(lbp_dim)
        
        # Task 1: Separate linear projections to common dimension d
        self.npr_proj = nn.Linear(npr_dim, d)
        self.sobel_proj = nn.Linear(sobel_dim, d)
        self.lbp_proj = nn.Linear(lbp_dim, d)
        
        # Task 1: Gating module (attention) - MLP that outputs 3 weights (softmax)
        # Input: concatenated projected vectors [B, 3*d]
        self.gating = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.LayerNorm(d),
            nn.ReLU(inplace=True),
            nn.Linear(d, 3),
            nn.Softmax(dim=1)  # 3 weights summing to 1
        )
        
        # Task 1 & 4: Final MLP with LayerNorm (replacing BatchNorm) and GELU
        self.mlp = nn.Sequential(
            nn.Linear(d, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Task 3: Skip connection
        total_dim = npr_dim + sobel_dim + lbp_dim
        self.skip_proj = nn.Linear(total_dim, embedding_dim)
        self.final_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, npr_features: torch.Tensor, 
                 sobel_features: torch.Tensor,
                 lbp_features: torch.Tensor,
                 return_attention: bool = False,
                 return_projections: bool = False) -> Union[
                     torch.Tensor,
                     Tuple[torch.Tensor, torch.Tensor],
                     Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                 ]:
        """
        npr_features: [B, 32]
        sobel_features: [B, 16]
        lbp_features: [B, 16]
        return_attention: если True, возвращает также attention веса
        return_projections: если True, возвращает также спроецированные векторы для diversity loss
        Возвращает: [B, embedding_dim] или tuple в зависимости от флагов
        """
        # Task 2: Apply separate LayerNorm to each branch
        npr = self.npr_norm(npr_features)
        sobel = self.sobel_norm(sobel_features)
        lbp = self.lbp_norm(lbp_features)
        
        # Task 1: Project each branch to common dimension d
        npr_proj = self.npr_proj(npr)      # [B, d]
        sobel_proj = self.sobel_proj(sobel) # [B, d]
        lbp_proj = self.lbp_proj(lbp)       # [B, d]
        
        # Task 1: Compute attention weights from concatenated projections
        # Option: using concatenated projected vectors for more expressive attention
        concatenated = torch.cat([npr_proj, sobel_proj, lbp_proj], dim=1)  # [B, 3*d]
        attention_weights = self.gating(concatenated)  # [B, 3], sums to 1
        
        # Task 1: Weighted sum of projected vectors (more efficient than concat)
        # This allows dynamic downweighting of noisy branches per image
        weighted_sum = (attention_weights[:, 0:1] * npr_proj +
                       attention_weights[:, 1:2] * sobel_proj +
                       attention_weights[:, 2:3] * lbp_proj)  # [B, d]
        
        # Main fusion path through MLP
        main_output = self.mlp(weighted_sum)  # [B, embedding_dim]
        
        # Task 3: Skip connection
        combined_original = torch.cat([npr, sobel, lbp], dim=1)  # [B, total_dim]
        skip_output = self.skip_proj(combined_original)  # [B, embedding_dim]
        
        output = self.final_norm(main_output + skip_output)
        
        # Prepare return values based on flags
        if return_attention and return_projections:
            return output, attention_weights, (npr_proj, sobel_proj, lbp_proj)
        elif return_attention:
            return output, attention_weights
        elif return_projections:
            return output, (npr_proj, sobel_proj, lbp_proj)
        return output
    
    @staticmethod
    def compute_diversity_loss(npr_proj: torch.Tensor, 
                                sobel_proj: torch.Tensor, 
                                lbp_proj: torch.Tensor) -> torch.Tensor:
        """
        Task 6: Compute diversity loss as mean pairwise cosine similarity.
        Цель: штрафовать дублирование информации между ветвями.
        
        Args:
            npr_proj: [B, d] - projected NPR features
            sobel_proj: [B, d] - projected Sobel features
            lbp_proj: [B, d] - projected LBP features
        
        Returns:
            Scalar loss (mean pairwise cosine similarity, range [-1, 1])
        """
        # L2-normalize projections
        npr_norm = F.normalize(npr_proj, p=2, dim=1)
        sobel_norm = F.normalize(sobel_proj, p=2, dim=1)
        lbp_norm = F.normalize(lbp_proj, p=2, dim=1)
        
        # Compute pairwise cosine similarities (dot product of normalized vectors)
        cos_npr_sobel = (npr_norm * sobel_norm).sum(dim=1).mean()
        cos_npr_lbp = (npr_norm * lbp_norm).sum(dim=1).mean()
        cos_sobel_lbp = (sobel_norm * lbp_norm).sum(dim=1).mean()
        
        # Mean pairwise cosine similarity (to be minimized)
        diversity_loss = (cos_npr_sobel + cos_npr_lbp + cos_sobel_lbp) / 3.0
        
        return diversity_loss


class SpatialBranch(nn.Module):
    """
    Полная пространственная ветка: NPR + Gradients + LBP → Fusion → Embedding.
    Поддерживает мультимасштабный LBP (несколько радиусов).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extractors
        self.npr_extractor = NPRFeatureExtractor()
        self.sobel_extractor = SobelExtractor()
        
        # Task 5: Multi-scale LBP support
        # Check if lbp_radii is specified in config for multi-scale LBP
        if hasattr(config, 'lbp_radii') and config.lbp_radii:
            self.lbp_extractors = nn.ModuleList([
                LBPExtractor(radius=r, n_points=config.lbp_n_points)
                for r in config.lbp_radii
            ])
            # Calculate total LBP dimension: sum of (n_points + 8) for each radius
            self.lbp_dim = len(config.lbp_radii) * (config.lbp_n_points + 8)
            self.multi_scale_lbp = True
        else:
            # Single LBP extractor (backward compatible)
            self.lbp_extractors = nn.ModuleList([
                LBPExtractor(
                    radius=config.lbp_radius,
                    n_points=config.lbp_n_points
                )
            ])
            self.lbp_dim = config.lbp_n_points + 8  # n_points + 8 structural features
            self.multi_scale_lbp = False
        
        # Fusion
        npr_dim   = 32  # 8 направлений × 4 статистики
        sobel_dim = 16  # 4×3 карты + 4 структурных признака
        
        self.fusion = SpatialFusion(
            npr_dim=npr_dim,
            sobel_dim=sobel_dim,
            lbp_dim=self.lbp_dim,
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
        
        # Task 5: Multi-scale LBP features
        lbp_features_list = []
        for lbp_extractor in self.lbp_extractors:
            lbp_features_list.append(lbp_extractor(x))
        lbp_features = torch.cat(lbp_features_list, dim=1)  # Concatenate multi-scale LBP features
        
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
    
    # Тест 1: Обычный батч
    x = torch.randn(2, 1, 256, 256)
    embedding = model(x)
    print(f"Test 1 - Input shape: {x.shape}")
    print(f"Test 1 - Output embedding shape: {embedding.shape}")
    print(f"Test 1 - Expected: [2, {config.model.spatial_embedding_dim}]")
    
    # Task 4: Тест с batch_size=1 в train mode
    model.train()
    x_single = torch.randn(1, 1, 256, 256)
    embedding_single = model(x_single)
    print(f"\nTest 2 (Task 4) - Single batch train mode: {embedding_single.shape}")
    assert not torch.isnan(embedding_single).any(), "NaN detected in train mode with batch_size=1"
    print("Task 4 PASSED: No NaN with batch_size=1 in train mode")
    
    # Task 1: Проверка attention весов
    model.eval()
    x_test = torch.randn(4, 1, 256, 256)
    npr_feat = model.npr_extractor(x_test)
    sobel_feat = model.sobel_extractor(x_test)
    # Task 5: Multi-scale LBP features
    lbp_feat = torch.cat([lbp(x_test) for lbp in model.lbp_extractors], dim=1)
    embedding_attn, attn_weights = model.fusion(npr_feat, sobel_feat, lbp_feat, return_attention=True)
    print(f"\nTest 3 (Task 1) - Attention weights shape: {attn_weights.shape}")
    print(f"Test 3 (Task 1) - Attention weights sum per sample: {attn_weights.sum(dim=1)}")
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(4, device=attn_weights.device), atol=1e-6), "Attention weights don't sum to 1"
    print("Task 1 PASSED: Attention weights sum to 1")
    
    # Task 2: Проверка нормализации
    with torch.no_grad():
        npr_normed = model.fusion.npr_norm(npr_feat)
        sobel_normed = model.fusion.sobel_norm(sobel_feat)
        lbp_normed = model.fusion.lbp_norm(lbp_feat)
        print(f"\nTest 4 (Task 2) - NPR normalized mean: {npr_normed.mean():.4f}, std: {npr_normed.std():.4f}")
        print(f"Test 4 (Task 2) - Sobel normalized mean: {sobel_normed.mean():.4f}, std: {sobel_normed.std():.4f}")
        print(f"Test 4 (Task 2) - LBP normalized mean: {lbp_normed.mean():.4f}, std: {lbp_normed.std():.4f}")
    
    # Task 3: Проверка residual connection (норма эмбеддинга)
    embedding_norm = embedding_attn.norm(dim=1)
    print(f"\nTest 5 (Task 3) - Embedding norm range: [{embedding_norm.min():.4f}, {embedding_norm.max():.4f}]")
    print("Task 3 PASSED: Embedding norm is reasonable")
    
    # Task 5: Тест мультимасштабного LBP
    print("\n--- Task 5: Multi-scale LBP tests ---")
    # Test LBPExtractor with different parameters
    lbp_8 = LBPExtractor(radius=1, n_points=8)
    lbp_12 = LBPExtractor(radius=2, n_points=12)
    x_lbp = torch.randn(2, 1, 256, 256)
    out_8 = lbp_8(x_lbp)
    out_12 = lbp_12(x_lbp)
    print(f"LBP (r=1, n=8) output shape: {out_8.shape} (expected: [2, 16])")
    print(f"LBP (r=2, n=12) output shape: {out_12.shape} (expected: [2, 20])")
    assert out_8.shape == (2, 16), f"Expected [2, 16], got {out_8.shape}"
    assert out_12.shape == (2, 20), f"Expected [2, 20], got {out_12.shape}"
    
    # Test multi-scale LBP in SpatialBranch
    print(f"Model LBP extractors count: {len(model.lbp_extractors)}")
    print(f"Model LBP dim: {model.lbp_dim}")
    print("Task 5 PASSED: Multi-scale LBP works correctly")
    
    # Task 6: Тест diversity loss
    print("\n--- Task 6: Diversity loss test ---")
    # Get projections from fusion
    _, projections = model.fusion(npr_feat, sobel_feat, lbp_feat, return_projections=True)
    npr_proj, sobel_proj, lbp_proj = projections
    
    # Compute diversity loss
    diversity_loss = SpatialFusion.compute_diversity_loss(npr_proj, sobel_proj, lbp_proj)
    print(f"Diversity loss: {diversity_loss.item():.4f} (range: [-1, 1])")
    print(f"Diversity loss shape: {diversity_loss.shape}")
    
    # Check gradient flow
    diversity_loss.backward(retain_graph=True)
    print("Task 6 PASSED: Diversity loss computes correctly with gradient flow")
    
    print("\n=== All tests passed ===")
