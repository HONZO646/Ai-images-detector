"""
Frequency Branch: DWT, DCT, FFT анализ изображений
Differentiable преобразования для end-to-end обучения
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import pywt


class DWTExtractor(nn.Module):
    """
    Extractor дискретного вейвлет-преобразования (Level 2)
    Возвращает HH, HL, LH поддиапазоны для каждого уровня
    """
    
    def __init__(self, wavelet: str = 'db1', level: int = 2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
        # Фильтры для DWT
        dec_lo, dec_hi, rec_lo, rec_hi = pywt.Wavelet(wavelet).filter_bank
        self.dec_lo = nn.Parameter(torch.tensor(dec_lo, dtype=torch.float32), requires_grad=False)
        self.dec_hi = nn.Parameter(torch.tensor(dec_hi, dtype=torch.float32), requires_grad=False)
    
    def _dwt2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D DWT на одноуровневом разложении
        x: [B, 1, H, W]
        Возвращает: LL, LH, HL, HH
        """
        B, C, H, W = x.shape
        
        # Фильтрация по строкам
        pad_h = self.dec_lo.shape[0] - 1
        x_padded = F.pad(x, (pad_h, pad_h, 0, 0), mode='reflect')
        
        # Низкочастотная и высокочастотная фильтрация по строкам
        L_rows = F.conv2d(x_padded, self.dec_lo.view(1, 1, 1, -1), padding=0)
        H_rows = F.conv2d(x_padded, self.dec_hi.view(1, 1, 1, -1), padding=0)
        
        # Фильтрация по столбцам
        pad_w = self.dec_lo.shape[0] - 1
        LL = F.conv2d(L_rows, self.dec_lo.view(1, 1, -1, 1), padding=0)
        LH = F.conv2d(L_rows, self.dec_hi.view(1, 1, -1, 1), padding=0)
        HL = F.conv2d(H_rows, self.dec_lo.view(1, 1, -1, 1), padding=0)
        HH = F.conv2d(H_rows, self.dec_hi.view(1, 1, -1, 1), padding=0)
        
        # Downsampling (stride slicing → non-contiguous)
        LL = LL[:, :, ::2, ::2].contiguous()
        LH = LH[:, :, ::2, ::2].contiguous()
        HL = HL[:, :, ::2, ::2].contiguous()
        HH = HH[:, :, ::2, ::2].contiguous()
        
        return LL, LH, HL, HH
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 1, H, W]
        Возвращает словарь с поддиапазонами для каждого уровня
        """
        features = {}
        current = x
        
        for lvl in range(1, self.level + 1):
            LL, LH, HL, HH = self._dwt2d(current)
            features[f'LH_l{lvl}'] = LH
            features[f'HL_l{lvl}'] = HL
            features[f'HH_l{lvl}'] = HH
            current = LL  # Для следующего уровня
        
        return features


class DCTExtractor(nn.Module):
    """
    Extractor DCT (Discrete Cosine Transform) AC-коэффициентов
    Разбивает изображение на 8x8 блоки и вычисляет DCT
    """
    
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        
        # DCT-II матрица (нормализованная)
        N = block_size
        dct_matrix = torch.zeros(N, N)
        for k in range(N):
            for n in range(N):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
            if k == 0:
                dct_matrix[k] *= np.sqrt(1 / N)
            else:
                dct_matrix[k] *= np.sqrt(2 / N)
        
        self.register_buffer('dct_matrix', dct_matrix)
    
    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        2D DCT на блоках
        x: [B, 1, H, W]
        Возвращает: [B, block_size, block_size, num_blocks_h, num_blocks_w]
        """
        B, C, H, W = x.shape
        
        # Разбиение на блоки
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        
        if num_blocks_h == 0 or num_blocks_w == 0:
            raise ValueError(
                f"Input is too small for DCT block_size={self.block_size}: got H={H}, W={W}"
            )

        # Reshape в блоки
        h_crop = num_blocks_h * self.block_size
        w_crop = num_blocks_w * self.block_size
        x = x[:, :, :h_crop, :w_crop]
        x = x.view(B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # [B, C, num_h, num_w, bs, bs]
        x = x.reshape(B * num_blocks_h * num_blocks_w, C, self.block_size, self.block_size)
        
        # Применение DCT
        x_dct = self.dct_matrix @ x @ self.dct_matrix.t()
        
        # Reshape обратно
        x_dct = x_dct.view(B, num_blocks_h, num_blocks_w, C, self.block_size, self.block_size)
        x_dct = x_dct.permute(0, 3, 1, 4, 2, 5)  # [B, C, num_h, bs_h, num_w, bs_w]
        x_dct = x_dct.reshape(B, C, h_crop, w_crop)
        
        return x_dct
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        Возвращает AC-коэффициенты (без DC компонента)
        """
        dct_coeffs = self._dct2d(x)
        
        # Извлечение AC-коэффициентов (без [0,0])
        B, C, H, W = dct_coeffs.shape
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size
        
        # Reshape для маскирования DC
        dct_coeffs = dct_coeffs.view(B, C, num_blocks_h, self.block_size, num_blocks_w, self.block_size)
        
        # Маска для AC-коэффициентов
        mask = torch.ones(self.block_size, self.block_size, device=x.device)
        mask[0, 0] = 0  # Убираем DC
        
        ac_coeffs = dct_coeffs * mask.view(1, 1, 1, self.block_size, 1, self.block_size)
        
        # Возвращаем статистику AC-коэффициентов
        ac_flat = ac_coeffs.view(B, -1)
        ac_mean = ac_flat.mean(dim=1, keepdim=True)
        ac_std = torch.sqrt(ac_flat.var(dim=1, keepdim=True) + 1e-6)
        # Mean energy keeps feature scale stable and AMP-friendly
        ac_energy = (ac_flat ** 2).mean(dim=1, keepdim=True)
        
        return torch.cat([ac_mean, ac_std, ac_energy], dim=1)


class FFTExtractor(nn.Module):
    """
    Extractor FFT (Fast Fourier Transform) признаков
    Радиальные и угловые профили амплитудного спектра
    """
    
    def __init__(self, radial_bins: int = 64, angular_bins: int = 36):
        super().__init__()
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
    
    def _create_frequency_coords(self, H: int, W: int, device: torch.device):
        """Создание координат в частотной области"""
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device) - cy
        x = torch.arange(W, device=device) - cx
        
        # Радиус и угол
        Y, X = torch.meshgrid(y, x, indexing='ij')
        R = torch.sqrt(Y.float() ** 2 + X.float() ** 2)
        Theta = torch.atan2(Y.float(), X.float())
        
        # Нормализация
        max_r = torch.sqrt(torch.tensor(float(cy ** 2 + cx ** 2), device=device))
        R = R / max_r  # [0, 1]
        Theta = (Theta + np.pi) / (2 * np.pi)  # [0, 1]
        
        return R, Theta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]
        Возвращает: [B, radial_bins + angular_bins]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # FFT
        x_fft = torch.fft.fft2(x)
        x_fft_shifted = torch.fft.fftshift(x_fft)
        
        # Амплитудный спектр (логарифмический)
        magnitude = torch.log(torch.abs(x_fft_shifted) + 1e-6)
        
        # Координаты
        R, Theta = self._create_frequency_coords(H, W, device)
        
        mag_flat = magnitude.reshape(B, -1)
        r_flat = R.reshape(-1)
        theta_flat = Theta.reshape(-1)

        # Радиальный профиль (векторизованный)
        r_idx = torch.clamp((r_flat * self.radial_bins).long(), 0, self.radial_bins - 1)
        radial_profile = torch.zeros(B, self.radial_bins, device=device)
        radial_profile.scatter_add_(1, r_idx.unsqueeze(0).expand(B, -1), mag_flat)
        radial_counts = torch.bincount(r_idx, minlength=self.radial_bins).float().clamp_min(1.0)
        radial_profile = radial_profile / radial_counts.unsqueeze(0)

        # Угловой профиль (векторизованный)
        theta_idx = torch.clamp((theta_flat * self.angular_bins).long(), 0, self.angular_bins - 1)
        angular_profile = torch.zeros(B, self.angular_bins, device=device)
        angular_profile.scatter_add_(1, theta_idx.unsqueeze(0).expand(B, -1), mag_flat)
        angular_counts = torch.bincount(theta_idx, minlength=self.angular_bins).float().clamp_min(1.0)
        angular_profile = angular_profile / angular_counts.unsqueeze(0)
        
        # Конкатенация
        features = torch.cat([radial_profile, angular_profile], dim=1)
        
        return features


class FrequencyAttention(nn.Module):
    """
    Attention-механизм для взвешивания частотных признаков
    """
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Attention: input_dim → input_dim (веса для каждого признака)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim),
            nn.Softmax(dim=1)
        )
        
        # Projection: input_dim → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        Возвращает: [B, embedding_dim]
        """
        weights = self.attention(x)  # [B, input_dim]
        weighted_x = x * weights  # [B, input_dim]
        embedding = self.projection(weighted_x)  # [B, embedding_dim]
        return embedding


class FrequencyBranch(nn.Module):
    """
    Полная частотная ветка: DWT + DCT + FFT → Attention → Embedding
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extractors
        self.dwt_extractor = DWTExtractor(
            wavelet=config.dwt_wavelet,
            level=config.dwt_level
        )
        
        self.dct_extractor = DCTExtractor(
            block_size=config.dct_block_size
        )
        
        self.fft_extractor = FFTExtractor(
            radial_bins=config.fft_radial_bins,
            angular_bins=config.fft_angular_bins
        )
        
        # FFT dim (единственная фиксированная)
        fft_dim = config.fft_radial_bins + config.fft_angular_bins
        
        # DCT dim
        dct_dim = 3  # mean, std, energy
        
        # DWT dim: 3 поддиапазона × level × 3 статистики
        dwt_feature_dim = 3 * config.dwt_level * 3
        
        # Total input dim для attention
        total_dim = dwt_feature_dim + dct_dim + fft_dim
        
        # Fusion & Attention
        self.attention = FrequencyAttention(
            input_dim=total_dim,
            embedding_dim=config.freq_embedding_dim
        )
    
    def _process_dwt_features(self, dwt_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Агрегация DWT поддиапазонов в вектор признаков"""
        features = []

        for key, tensor in dwt_features.items():
            # Глобальная статистика для каждого поддиапазона
            B = tensor.shape[0]
            mean_feat = tensor.reshape(B, -1).mean(dim=1, keepdim=True)
            std_feat = torch.sqrt(tensor.reshape(B, -1).var(dim=1, keepdim=True) + 1e-6)
            # Use mean instead of sum to avoid very large magnitudes
            energy_feat = (tensor ** 2).reshape(B, -1).mean(dim=1, keepdim=True)
            features.append(torch.cat([mean_feat, std_feat, energy_feat], dim=1))

        return torch.cat(features, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] grayscale изображение
        Возвращает: [B, freq_embedding_dim] embedding
        """
        # DWT features
        dwt_features = self.dwt_extractor(x)
        dwt_vec = self._process_dwt_features(dwt_features)
        
        # DCT features
        dct_vec = self.dct_extractor(x)
        
        # FFT features
        fft_vec = self.fft_extractor(x)
        
        # Конкатенация всех частотных признаков
        combined = torch.cat([dwt_vec, dct_vec, fft_vec], dim=1)
        
        # Attention и проекция
        embedding = self.attention(combined)
        
        return embedding


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    from config import Config
    
    config = Config()
    model = FrequencyBranch(config.model)
    
    # Тестовый вход
    x = torch.randn(2, 1, 256, 256)
    
    # Проверка прямого прохода
    embedding = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Expected: [2, {config.model.freq_embedding_dim}]")
