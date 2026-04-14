"""
Semantic Branch: Frozen CLIP/ViT extractor
Использует предобученную модель для семантического понимания изображения
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os


class CLIPFeatureExtractor(nn.Module):
    """
    Feature extractor на основе CLIP/ViT
    Замороженные веса для zero-shot обобщения
    """
    
    def __init__(self, model_name: str = 'clip-ViT-B-32', device: str = 'cpu'):
        super().__init__()
        self.model_name = model_name
        
        # Попытка загрузить sentence-transformers (CLIP)
        try:
            from sentence_transformers import SentenceTransformer
            self._use_sentence_transformers = True
            
            model_path = os.path.join("models", "clip-ViT-B-32")
            
            # Загрузка модели
            if os.path.exists(model_path):
                self.model = SentenceTransformer(model_path)
            else:
                self.model = SentenceTransformer(model_name)
                # Сохранение для быстрого доступа
                try:
                    self.model.save(model_path)
                except:
                    pass
            
            # Заморозка всех параметров
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model.eval()
            
            # CLIP ViT-B-32 имеет 512-dim embeddings
            self.clip_dim = 512
            
        except ImportError:
            # Fallback: используем torchvision CLIP
            self._use_sentence_transformers = False
            self._load_torchvision_clip(device)
    
    def _load_torchvision_clip(self, device: str):
        """Загрузка CLIP через torchvision"""
        try:
            import clip
            self.model, self.preprocess = clip.load('ViT-B/32', device=device)
            
            # Заморозка
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model.eval()
            self.clip_dim = 512
            
        except ImportError:
            raise ImportError(
                "Установите clip: pip install git+https://github.com/openai/CLIP.git"
                "или sentence-transformers: pip install sentence-transformers"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] RGB изображение (нормализованное для CLIP)
        Возвращает: [B, 512] CLIP embedding
        """
        if self._use_sentence_transformers:
            # Sentence Transformers ожидают PIL изображения или numpy
            # Конвертируем tensor -> numpy
            with torch.no_grad():
                # x: [B, 3, H, W] -> list of numpy
                embeddings = self.model.encode(
                    self._tensor_to_pil_list(x),
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return torch.tensor(embeddings, dtype=torch.float32, device=x.device)
        else:
            # Torchvision CLIP
            with torch.no_grad():
                return self.model.encode_image(x)


class SemanticProjector(nn.Module):
    """
    Проекция CLIP embedding (512-dim) → semantic_embedding_dim (128-dim)
    Обучаемый модуль
    """
    
    def __init__(self, input_dim: int = 512, output_dim: int = 128):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 512] CLIP embedding
        Возвращает: [B, output_dim] projected embedding
        """
        return self.projector(x)


class InconsistencyHead(nn.Module):
    """
    Опциональный модуль для детекции несоответствий
    (освещение, геометрия, перспектива)
    Помогает выявить AI-артефакты в семантике
    """
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(32, 4)  # [lighting_consistency, geometry_consistency, perspective, texture_realism]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 128] semantic embedding
        Возвращает: [B, 4] inconsistency scores
        """
        return self.head(x)


class SemanticBranch(nn.Module):
    """
    Полная семантическая ветка:
    Frozen CLIP → Projector → [Optional: Inconsistency Head]
    """
    
    def __init__(self, config, device: str = 'cpu'):
        super().__init__()
        self.config = config
        
        # CLIP extractor (frozen)
        self.clip_extractor = CLIPFeatureExtractor(
            model_name=config.semantic_model_name,
            device=device
        )
        
        # Projector (trainable)
        self.projector = SemanticProjector(
            input_dim=self.clip_extractor.clip_dim,
            output_dim=config.semantic_embedding_dim
        )
        
        # Inconsistency head (optional)
        self.use_inconsistency = hasattr(config, 'use_inconsistency') and config.use_inconsistency
        if self.use_inconsistency:
            self.inconsistency_head = InconsistencyHead(
                input_dim=config.semantic_embedding_dim
            )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B, 3, H, W] RGB изображение
        Возвращает словарь:
            - 'embedding': [B, semantic_embedding_dim]
            - 'inconsistency': [B, 4] (опционально)
        """
        # CLIP features (без градиентов)
        with torch.no_grad():
            clip_features = self.clip_extractor(x)
        
        # Project to semantic space
        embedding = self.projector(clip_features)
        
        result = {'embedding': embedding}
        
        # Inconsistency analysis (optional)
        if self.use_inconsistency:
            inconsistency = self.inconsistency_head(embedding)
            result['inconsistency'] = inconsistency
        
        return result


# ============================================================
# Альтернативная реализация с torchvision для простоты
# ============================================================
class SimpleSemanticBranch(nn.Module):
    """
    Упрощённая семантическая ветка без внешних зависимостей
    Использует torchvision.models для извлечения features
    """
    
    def __init__(self, config, device: str = 'cpu'):
        super().__init__()
        
        # Загрузка pretrained модели (ResNet-18 для скорости)
        try:
            from torchvision import models
        except ImportError:
            raise ImportError("Установите torchvision: pip install torchvision")
        
        # Используем ResNet без classifier
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Заморозка всех слоёв
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Удаляем последний fully connected слой
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.backbone.eval()
        
        # ResNet-18 имеет 512-dim features
        backbone_dim = 512
        
        # Projector
        self.projector = SemanticProjector(
            input_dim=backbone_dim,
            output_dim=config.semantic_embedding_dim
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        x: [B, 3, H, W] RGB изображение
        Возвращает: {'embedding': [B, semantic_embedding_dim]}
        """
        with torch.no_grad():
            # ResNet backbone
            features = self.backbone(x)  # [B, 512, 1, 1]
            features = features.view(features.shape[0], -1)  # [B, 512]
        
        # Projection
        embedding = self.projector(features)
        
        return {'embedding': embedding}


# ============================================================
# Тестирование
# ============================================================
if __name__ == '__main__':
    from config import Config
    
    config = Config()
    
    # Тест с упрощённой версией (без CLIP)
    model = SimpleSemanticBranch(config.model)
    
    # Тестовый вход (RGB)
    x = torch.randn(2, 3, 224, 224)
    
    # Проверка прямого прохода
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output embedding shape: {output['embedding'].shape}")
    print(f"Expected: [2, {config.model.semantic_embedding_dim}]")
