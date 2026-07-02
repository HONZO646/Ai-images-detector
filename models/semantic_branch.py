"""
Semantic Branch: CLIP/ViT extractor with optional partial fine-tuning
Использует предобученную модель для семантического понимания изображения
Поддерживает разморозку последних N transformer-блоков для fine-tuning
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class CLIPFeatureExtractor(nn.Module):
    """
    Feature extractor на основе CLIP/ViT
    Поддерживает частичную разморозку последних transformer-блоков для fine-tuning
    """
    
    _hf_clip_model: Optional[nn.Module]
    model: Optional[nn.Module]
    _clip_preprocess: Optional[object]
    _use_sentence_transformers: bool
    clip_dim: int

    def __init__(self, model_name: str = 'clip-ViT-B-32', device: str = 'cpu',
                 unfreeze_last_n_blocks: int = 0):
        super().__init__()
        self.model_name = model_name
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.device = device

        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        )

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
                try:
                    self.model.save(model_path)
                except:
                    pass

            # Получаем доступ к внутренней HuggingFace CLIP-модели
            self._hf_clip_model = self._extract_hf_clip_model() # type: ignore

            # Настройка градиентов (заморозка/разморозка)
            self._setup_gradient_requirements()

            # CLIP ViT-B-32 имеет 512-dim embeddings
            self.clip_dim = 512

        except ImportError:
            # Fallback: используем openai/clip
            self._use_sentence_transformers = False
            self._load_openai_clip(device)

    def _extract_hf_clip_model(self):
        """Извлечение внутренней HuggingFace CLIP-модели из sentence-transformers."""
        # В sentence-transformers CLIP структура может варьироваться
        # Пытаемся найти CLIPModel в modules
        if hasattr(self.model, 'modules') and callable(getattr(self.model, 'modules')):
            for module in self.model.modules(): # pyright: ignore[reportOptionalMemberAccess]
                class_name = type(module).__name__
                # Ищем классы, связанные с CLIP
                if any(x in class_name for x in ['CLIP', 'VisionText', 'Transformer']):
                    # Пытаемся получить доступ к HuggingFace модели
                    if hasattr(module, 'auto_model'):
                        return module.auto_model
                    # Иногда модель доступна напрямую
                    if hasattr(module, 'model'):
                        return module.model
                    # Или это уже сама HuggingFace модель
                    if hasattr(module, 'vision_model'):
                        return module

        # Альтернативный путь: через [0] индекс
        try:
            first_module = self.model[0] # type: ignore
            # Пытаемся получить auto_model
            if hasattr(first_module, 'auto_model'):
                return first_module.auto_model
            # Или model
            if hasattr(first_module, 'model'):
                return first_module.model
            # Или это уже модель
            return first_module
        except:
            pass

        # Ещё один вариант: сама модель может быть CLIP моделью
        if hasattr(self.model, 'vision_model'):
            return self.model

        raise AttributeError(
            "Cannot find internal CLIP model in sentence-transformers. "
            "The model structure may have changed. Please check the sentence-transformers version."
        )

    def _load_openai_clip(self, device: str):
        """Загрузка CLIP через openai/clip"""
        try:
            import clip  # type: ignore[import]
            self.model, self._clip_preprocess = clip.load('ViT-B/32', device=device)

            # Настройка градиентов
            self._setup_gradient_requirements()

            self.clip_dim = 512

        except ImportError:
            raise ImportError(
                "Install clip: pip install git+https://github.com/openai/CLIP.git"
                " or sentence-transformers: pip install sentence-transformers"
            )

    def _setup_gradient_requirements(self):
        """
        Настройка requires_grad для параметров CLIP.
        Размораживает последние unfreeze_last_n_blocks transformer-блоков.
        """
        # Сначала замораживаем все параметры
        if self._use_sentence_transformers:
            for param in self._hf_clip_model.parameters(): # type: ignore
                param.requires_grad = False
        else:
            for param in self.model.parameters(): # type: ignore
                param.requires_grad = False

        # Если unfreeze_last_n_blocks > 0, размораживаем последние блоки
        if self.unfreeze_last_n_blocks > 0:
            if self._use_sentence_transformers:
                self._unfreeze_last_blocks_hf(self._hf_clip_model)
            else:
                self._unfreeze_last_blocks_openai(self.model)

        # Выводим статистику
        self._log_parameter_status()

    def _unfreeze_last_blocks_hf(self, hf_clip_model):
        """
        Разморозка последних блоков для HuggingFace CLIP.
        Структура: CLIPVisionModel -> vision_model -> encoder -> layers
        """
        try:
            # Путь к transformer-блокам в HuggingFace CLIP
            encoder_layers = hf_clip_model.vision_model.encoder.layers

            # Размораживаем последние N блоков
            num_layers = len(encoder_layers)
            start_idx = max(0, num_layers - self.unfreeze_last_n_blocks)

            for idx in range(start_idx, num_layers):
                for param in encoder_layers[idx].parameters():
                    param.requires_grad = True
                print(f"  Unfrozen: vision_model.encoder.layers[{idx}]")

            # Размораживаем финальный LayerNorm и projection
            if hasattr(hf_clip_model.vision_model, 'post_layernorm'):
                for param in hf_clip_model.vision_model.post_layernorm.parameters():
                    param.requires_grad = True
                print(f"  Unfrozen: vision_model.post_layernorm")

            # Визуальная projection (преобразует output в embedding)
            if hasattr(hf_clip_model, 'visual_projection'):
                for param in hf_clip_model.visual_projection.parameters():
                    param.requires_grad = True
                print(f"  Unfrozen: visual_projection")

        except AttributeError as e:
            print(f"Warning: Could not unfreeze blocks in HF CLIP: {e}")
            print("Falling back to full freeze")

    def _unfreeze_last_blocks_openai(self, clip_model):
        """
        Разморозка последних блоков для openai/clip.
        Структура: CLIP -> visual -> transformer -> resblocks
        """
        try:
            # Путь к transformer-блокам в openai/clip
            resblocks = clip_model.visual.transformer.resblocks

            # Размораживаем последние N блоков
            num_blocks = len(resblocks)
            start_idx = max(0, num_blocks - self.unfreeze_last_n_blocks)

            for idx in range(start_idx, num_blocks):
                for param in resblocks[idx].parameters():
                    param.requires_grad = True
                print(f"  Unfrozen: visual.transformer.resblocks[{idx}]")

            # Размораживаем ln_post и proj
            if hasattr(clip_model.visual, 'ln_post'):
                for param in clip_model.visual.ln_post.parameters():
                    param.requires_grad = True
                print(f"  Unfrozen: visual.ln_post")

            if hasattr(clip_model.visual, 'proj'):
                if clip_model.visual.proj is not None:
                    for param in clip_model.visual.proj.parameters():
                        param.requires_grad = True
                    print(f"  Unfrozen: visual.proj")

        except AttributeError as e:
            print(f"Warning: Could not unfreeze blocks in openai CLIP: {e}")
            print("Falling back to full freeze")

    def _log_parameter_status(self):
        """Вывод статистики по обучаемым/замороженным параметрам."""
        if self._use_sentence_transformers:
            model = self._hf_clip_model
        else:
            model = self.model

        total_params = sum(p.numel() for p in model.parameters()) # type: ignore
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # type: ignore
        frozen_params = total_params - trainable_params

        print(f"\n[CLIPFeatureExtractor] Parameter status:")
        print(f"  Total CLIP parameters: {total_params:,}")
        print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"  unfreeze_last_n_blocks: {self.unfreeze_last_n_blocks}")

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Дифференцируемая предобработка для CLIP.
        x: [B, 3, H, W] в произвольном диапазоне
        Returns: [B, 3, 224, 224] нормализованный для CLIP
        """
        # Изменение размера через дифференцируемую интерполяцию
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Нормализация для CLIP
        x = (x - self.clip_mean) / self.clip_std # type: ignore

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через CLIP с поддержкой градиентов.
        x: [B, 3, H, W] RGB изображение
        Возвращает: [B, 512] CLIP embedding
        """
        if self._use_sentence_transformers:
            return self._forward_sentence_transformers(x)
        else:
            return self._forward_openai_clip(x)

    def _forward_sentence_transformers(self, x: torch.Tensor) -> torch.Tensor:
        """Дифференцируемый forward через HuggingFace CLIP (из sentence-transformers)."""
        # Предобработка (дифференцируемая)
        x_norm = self._preprocess(x)

        # Прямой проход через vision model
        # Используем внутреннюю модель напрямую для сохранения графа автоградиента
        vision_outputs = self._hf_clip_model.vision_model(pixel_values=x_norm) # type: ignore

        # Получаем эмбеддинг из последнего hidden state
        # CLIP использует CLS token (первый токен) как глобальный дескриптор
        hidden_state = vision_outputs.last_hidden_state  # [B, num_patches+1, hidden_size]
        cls_embedding = hidden_state[:, 0, :]  # [B, hidden_size]

        # Проекция в пространство CLIP (visual_projection)
        if hasattr(self._hf_clip_model, 'visual_projection'):
            embeddings = self._hf_clip_model.visual_projection(cls_embedding) # type: ignore
        else:
            embeddings = cls_embedding

        return embeddings

    def _forward_openai_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Дифференцируемый forward через openai/clip."""
        # Предобработка (дифференцируемая)
        x_norm = self._preprocess(x)

        # encode_image без torch.no_grad() - управление градиентом через requires_grad
        embeddings = self.model.encode_image(x_norm) # type: ignore

        return embeddings

    def get_trainable_clip_params(self) -> List[torch.nn.Parameter]:
        """Возвращает список обучаемых параметров CLIP."""
        if self._use_sentence_transformers:
            return [p for p in self._hf_clip_model.parameters() if p.requires_grad] # type: ignore
        else:
            return [p for p in self.model.parameters() if p.requires_grad] # type: ignore

    def get_frozen_clip_params(self) -> List[torch.nn.Parameter]:
        """Возвращает список замороженных параметров CLIP."""
        if self._use_sentence_transformers:
            return [p for p in self._hf_clip_model.parameters() if not p.requires_grad] # type: ignore
        else:
            return [p for p in self.model.parameters() if not p.requires_grad] # type: ignore

    @property
    def is_partially_trainable(self) -> bool:
        """Возвращает True, если хотя бы часть CLIP разморожена."""
        return self.unfreeze_last_n_blocks > 0

    def train(self, mode: bool = True):
        """
        Переводит модель в train/eval режим.
        Размороженные блоки переключаются, замороженные остаются в eval.
        """
        super().train(mode)

        if self._use_sentence_transformers:
            # Переводим всю модель в нужный режим
            self._hf_clip_model.train(mode) # type: ignore
            # Но отключаем градиенты для замороженных параметров
            if not mode:
                # В eval режиме все параметры должны быть в eval
                pass  # HuggingFace модели сами обрабатывают .eval()
        else:
            self.model.train(mode) # type: ignore

        return self

    def eval(self):
        """Переводит модель в eval режим."""
        return self.train(False)


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
    CLIP (frozen или частично разморожен) → Projector → [Optional: Inconsistency Head]

    Если unfreeze_last_n_blocks > 0, градиент течёт через последние блоки CLIP;
    иначе поведение идентично полностью замороженной версии.
    """

    def __init__(self, config, device: str = 'cpu'):
        super().__init__()
        self.config = config

        # Получаем параметр разморозки из конфига (по умолчанию 0 = полная заморозка)
        unfreeze_last_n_blocks = getattr(config, 'semantic_unfreeze_last_n_blocks', 0)

        # CLIP extractor (может быть частично разморожен)
        self.clip_extractor = CLIPFeatureExtractor(
            model_name=config.semantic_model_name,
            device=device,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks
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
        # CLIP features (градиент управляется через requires_grad внутри CLIPFeatureExtractor)
        clip_features = self.clip_extractor(x)

        # Project to semantic space
        embedding = self.projector(clip_features)

        result = {'embedding': embedding}

        # Inconsistency analysis (optional)
        if self.use_inconsistency:
            inconsistency = self.inconsistency_head(embedding)
            result['inconsistency'] = inconsistency

        return result

    def get_param_groups(self, lr_clip: float = 1e-6, lr_head: float = 1e-4) -> list:
        """
        Возвращает список групп параметров для оптимизатора с дифференцированными LR.

        Группы:
        1. Параметры размороженных блоков CLIP (lr_clip, по умолчанию 1e-6)
        2. Параметры projector (lr_head, по умолчанию 1e-4)
        3. Параметры inconsistency_head (lr_head, если используется)

        Args:
            lr_clip: Learning rate для обучаемых параметров CLIP
            lr_head: Learning rate для новых слоёв (projector, inconsistency head)

        Returns:
            list[dict]: Список групп параметров для torch.optim.Optimizer

        Example:
            optimizer = torch.optim.AdamW(
                model.get_param_groups(lr_clip=1e-6, lr_head=1e-4)
            )
        """
        param_groups = []

        # Группа 1: Обучаемые параметры CLIP
        clip_params = self.clip_extractor.get_trainable_clip_params()
        if len(clip_params) > 0:
            param_groups.append({
                'params': clip_params,
                'lr': lr_clip,
                'name': 'clip_trainable'
            })
        else:
            if self.clip_extractor.unfreeze_last_n_blocks > 0:
                print("Warning: unfreeze_last_n_blocks > 0, but no trainable CLIP params found")

        # Группа 2: Projector
        param_groups.append({
            'params': list(self.projector.parameters()),
            'lr': lr_head,
            'name': 'projector'
        })

        # Группа 3: Inconsistency head (если используется)
        if self.use_inconsistency:
            param_groups.append({
                'params': list(self.inconsistency_head.parameters()),
                'lr': lr_head,
                'name': 'inconsistency_head'
            })

        return param_groups


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
    import torch.optim as optim

    print("=" * 60)
    print("Testing CLIPFeatureExtractor and SemanticBranch")
    print("=" * 60)

    # Тестовый вход (RGB)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)

    # ============================================================
    # Тест A: unfreeze_last_n_blocks=0 (полная заморозка)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST A: unfreeze_last_n_blocks=0 (full freeze)")
    print("=" * 60)

    try:
        extractor_0 = CLIPFeatureExtractor(
            model_name='clip-ViT-B-32',
            device='cpu',
            unfreeze_last_n_blocks=0
        )

        # Проверка: все параметры должны быть заморожены
        all_frozen = all(p.requires_grad == False for p in extractor_0.get_trainable_clip_params())
        if all_frozen or len(extractor_0.get_trainable_clip_params()) == 0:
            print("[OK] All CLIP parameters are frozen (requires_grad=False)")
        else:
            print("[FAIL] Some CLIP parameters are trainable (unexpected for unfreeze_last_n_blocks=0)")

        # Forward pass
        with torch.no_grad():
            output_0 = extractor_0(x)
        print(f"[OK] Forward pass works: output shape = {output_0.shape}")

        # Подсчёт параметров
        trainable_params = sum(p.numel() for p in extractor_0.get_trainable_clip_params())
        frozen_params = sum(p.numel() for p in extractor_0.get_frozen_clip_params())
        print(f"  Trainable CLIP parameters: {trainable_params:,}")
        print(f"  Frozen CLIP parameters: {frozen_params:,}")

    except Exception as e:
        print(f"[FAIL] Test A failed: {e}")

    # ============================================================
    # Тест B: unfreeze_last_n_blocks=1 (разморозка 1 блока)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST B: unfreeze_last_n_blocks=1 (unfreeze 1 block)")
    print("=" * 60)

    try:
        extractor_1 = CLIPFeatureExtractor(
            model_name='clip-ViT-B-32',
            device='cpu',
            unfreeze_last_n_blocks=1
        )

        # Проверка: должны быть размороженные параметры
        trainable_params = extractor_1.get_trainable_clip_params()
        if len(trainable_params) > 0:
            print(f"[OK] Found {len(trainable_params)} trainable parameter tensors")
            print(f"[OK] Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        else:
            print("[FAIL] No trainable parameters found (unexpected for unfreeze_last_n_blocks=1)")

        # Forward pass с градиентом
        extractor_1.train()  # Важно: переключаем в train mode
        output_1 = extractor_1(x)
        print(f"[OK] Forward pass works: output shape = {output_1.shape}")
        print(f"[OK] Output requires_grad: {output_1.requires_grad}")

        # Backward pass
        fake_loss = output_1.sum()
        fake_loss.backward()
        print("[OK] Backward pass completed")

        # Проверка: градиенты должны быть у размороженных параметров
        params_with_grad = [p for p in trainable_params if p.grad is not None]
        if len(params_with_grad) > 0:
            print(f"[OK] {len(params_with_grad)} parameter tensors have gradients after backward")
        else:
            print("[FAIL] No gradients found for trainable parameters")

    except Exception as e:
        print(f"[FAIL] Test B failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # Тест C: unfreeze_last_n_blocks=2 (разморозка 2 блоков)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST C: unfreeze_last_n_blocks=2 (unfreeze 2 blocks)")
    print("=" * 60)

    try:
        extractor_2 = CLIPFeatureExtractor(
            model_name='clip-ViT-B-32',
            device='cpu',
            unfreeze_last_n_blocks=2
        )

        # Проверка: должны быть размороженные параметры
        trainable_params = extractor_2.get_trainable_clip_params()
        if len(trainable_params) > 0:
            print(f"[OK] Found {len(trainable_params)} trainable parameter tensors")
            print(f"[OK] Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        else:
            print("[FAIL] No trainable parameters found (unexpected for unfreeze_last_n_blocks=2)")

        # Forward + Backward
        extractor_2.train()
        output_2 = extractor_2(x)
        fake_loss = output_2.sum()
        fake_loss.backward()
        print("[OK] Forward + Backward pass completed")

        # Проверка: градиенты должны быть у размороженных параметров
        params_with_grad = [p for p in trainable_params if p.grad is not None]
        if len(params_with_grad) > 0:
            print(f"[OK] {len(params_with_grad)} parameter tensors have gradients after backward")

    except Exception as e:
        print(f"[FAIL] Test C failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # Тест D: get_param_groups() и инициализация AdamW
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST D: get_param_groups() and AdamW initialization")
    print("=" * 60)

    try:
        # Создаём конфиг с unfreeze_last_n_blocks=1
        class MockConfig:
            semantic_embedding_dim = 128
            semantic_model_name = 'clip-ViT-B-32'
            semantic_unfreeze_last_n_blocks = 1
            use_inconsistency = False

        config = MockConfig()
        semantic_branch = SemanticBranch(config, device='cpu')
        semantic_branch.train()

        # Получаем группы параметров
        param_groups = semantic_branch.get_param_groups(lr_clip=1e-6, lr_head=1e-4)
        print(f"[OK] get_param_groups() returned {len(param_groups)} groups")

        for i, group in enumerate(param_groups):
            n_params = sum(p.numel() for p in group['params'])
            print(f"  Group {i}: name={group.get('name', 'N/A')}, lr={group['lr']}, params={n_params:,}")

        # Инициализация AdamW
        optimizer = optim.AdamW(param_groups)
        print("[OK] AdamW optimizer initialized successfully")

        # Тест оптимизации
        x_opt = torch.randn(2, 3, 224, 224)
        output = semantic_branch(x_opt)
        loss = output['embedding'].sum()
        loss.backward()
        optimizer.step()
        print("[OK] Optimization step completed")

    except Exception as e:
        print(f"[FAIL] Test D failed: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # Тест E: обратная совместимость (unfreeze_last_n_blocks=0)
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST E: Backward compatibility (unfreeze_last_n_blocks=0)")
    print("=" * 60)

    try:
        class MockConfigFrozen:
            semantic_embedding_dim = 128
            semantic_model_name = 'clip-ViT-B-32'
            semantic_unfreeze_last_n_blocks = 0
            use_inconsistency = False

        config_frozen = MockConfigFrozen()
        semantic_branch_frozen = SemanticBranch(config_frozen, device='cpu')

        # Forward pass
        with torch.no_grad():
            output_frozen = semantic_branch_frozen(x)
        print(f"[OK] Forward pass works with unfreeze_last_n_blocks=0")
        print(f"  Output shape: {output_frozen['embedding'].shape}")

        # Проверка: градиенты не должны течь через CLIP
        print(f"  CLIP is_partially_trainable: {semantic_branch_frozen.clip_extractor.is_partially_trainable}")

    except Exception as e:
        print(f"[FAIL] Test E failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
