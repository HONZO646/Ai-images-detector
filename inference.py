#!/usr/bin/env python3
"""
NPR AI/Real Detector - Inference Script
Инференс для одиночных изображений и батчей
Поддержка ONNX export и cascade inference
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from typing import List, Dict, Optional
import logging

from config import Config
from models.detector import create_detector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NPRInference:
    """
    Inference pipeline для NPR Detector
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'auto', 
                 use_cascade: bool = False, cascade_threshold: float = 0.7):
        self.device = self._get_device(device)
        self.use_cascade = use_cascade
        self.cascade_threshold = cascade_threshold
        self.config = Config()
        
        # Загрузка модели
        self.model = create_detector(
            self.config.model,
            device=self.device,
            pretrained_path=checkpoint_path
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on: {self.device}")
        logger.info(f"Cascade mode: {'ON' if use_cascade else 'OFF'}")
    
    def _get_device(self, device_str: str) -> torch.device:
        """Получение устройства"""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)
    
    def preprocess_image(self, image_path: str) -> tuple:
        """
        Препроцессинг изображения
        Возвращает: (gray_tensor, rgb_tensor)
        """
        # Загрузка
        img_rgb = Image.open(image_path).convert('RGB')

        # Приведение к размеру обучения (стабильная форма для всех веток)
        target_size = self.config.data.image_size
        img_rgb = img_rgb.resize((target_size, target_size), Image.LANCZOS)
        
        # RGB tensor [1, 3, H, W]
        rgb_array = np.array(img_rgb, dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)
        
        # Grayscale tensor [1, 1, H, W]
        img_gray = img_rgb.convert('L')
        gray_array = np.array(img_gray, dtype=np.float32) / 255.0
        gray_tensor = torch.from_numpy(gray_array).unsqueeze(0).unsqueeze(0)
        
        return gray_tensor, rgb_tensor
    
    def predict_single(self, image_path: str, threshold: float = 0.5) -> Dict:
        """Предсказание для одного изображения"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        gray, rgb = self.preprocess_image(image_path)
        gray = gray.to(self.device)
        rgb = rgb.to(self.device)
        
        start_time = time.time()
        
        if self.use_cascade:
            result = self._predict_cascade(gray, rgb, threshold)
        else:
            result = self._predict_full(gray, rgb, threshold)
        
        inference_time = time.time() - start_time
        
        # Добавление метаинформации
        result['image_path'] = image_path
        result['inference_time_ms'] = inference_time * 1000
        
        return result
    
    def _predict_full(self, gray: torch.Tensor, rgb: torch.Tensor, 
                     threshold: float) -> Dict:
        """Полный проход через все ветки"""
        with torch.no_grad():
            output = self.model(gray, rgb)
            
            # Model returns logits, apply sigmoid for probability
            logits = output['probability']
            prob_ai = torch.sigmoid(logits).item()
            pred_class = "AI_GENERATED" if prob_ai >= threshold else "REAL"
            confidence = abs(prob_ai - 0.5) * 2
            
            result = {
                'probability_ai': round(prob_ai, 4),
                'probability_real': round(1.0 - prob_ai, 4),
                'prediction': pred_class,
                'confidence': round(confidence, 4),
                'threshold': threshold,
                'branch_used': 'full'
            }
            
            # Gate weights
            if 'gate_weights' in output:
                gates = output['gate_weights'][0].cpu().numpy()
                result['gate_weights'] = {
                    'semantic': round(float(gates[0]), 4),
                    'spatial': round(float(gates[1]), 4),
                    'frequency': round(float(gates[2]), 4)
                }
            
            # Uncertainty
            if 'uncertainty' in output:
                result['uncertainty'] = round(output['uncertainty'].item(), 4)
            
            return result
    
    def _predict_cascade(self, gray: torch.Tensor, rgb: torch.Tensor, 
                        threshold: float) -> Dict:
        """Cascade prediction с early exit"""
        with torch.no_grad():
            # Сначала только spatial branch (быстрая)
            spatial_emb = self.model.spatial_branch(gray)
            
            # Quick prediction (classifier returns logits, apply sigmoid)
            quick_logits = self.model.classifier(spatial_emb)
            quick_pred = torch.sigmoid(quick_logits).item()
            quick_confidence = abs(quick_pred - 0.5) * 2
            
            # Early exit?
            if quick_confidence >= self.cascade_threshold:
                return {
                    'probability_ai': round(quick_pred, 4),
                    'probability_real': round(1.0 - quick_pred, 4),
                    'prediction': "AI_GENERATED" if quick_pred >= threshold else "REAL",
                    'confidence': round(quick_confidence, 4),
                    'threshold': threshold,
                    'branch_used': 'spatial_only',
                    'early_exit': True
                }
            
            # Полный проход
            return {
                **self._predict_full(gray, rgb, threshold),
                'early_exit': False
            }
    
    def predict_batch(self, image_paths: List[str], 
                     threshold: float = 0.5) -> List[Dict]:
        """Предсказание для батча изображений"""
        results = []
        
        for path in image_paths:
            try:
                result = self.predict_single(path, threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append({
                    'image_path': path,
                    'error': str(e)
                })
        
        return results
    
    def export_to_onnx(self, output_path: str, opset_version: int = 14):
        """Export модели в ONNX формат"""
        logger.info(f"Exporting to ONNX: {output_path}")
        
        # Dummy inputs
        gray = torch.randn(1, 1, 256, 256).to(self.device)
        rgb = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Export
        torch.onnx.export(
            self.model,
            (gray, rgb),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['gray', 'rgb'],
            output_names=['probability'],
            dynamic_axes={
                'gray': {0: 'batch_size'},
                'rgb': {0: 'batch_size'},
                'probability': {0: 'batch_size'}
            }
        )
        
        logger.info(f"ONNX model saved: {output_path}")


def format_result(result: Dict) -> str:
    """Форматирование результата для вывода"""
    if 'error' in result:
        return f"❌ {result['image_path']}: {result['error']}"
    
    emoji = "🤖" if result['prediction'] == "AI_GENERATED" else "✅"
    
    output = f"{emoji} {result['image_path']}\n"
    output += f"   Prediction: {result['prediction']}\n"
    output += f"   P(AI):      {result['probability_ai']:.4f}\n"
    output += f"   Confidence: {result['confidence']:.4f}\n"
    output += f"   Time:       {result['inference_time_ms']:.2f} ms\n"
    output += f"   Branch:     {result['branch_used']}"
    
    if result.get('early_exit'):
        output += " (early exit)"
    
    if 'gate_weights' in result:
        output += f"\n   Gates:      semantic={result['gate_weights']['semantic']:.3f}, "
        output += f"spatial={result['gate_weights']['spatial']:.3f}, "
        output += f"freq={result['gate_weights']['frequency']:.3f}"
    
    return output


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='NPR AI/Real Detector - Inference')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, nargs='+', help='Paths to multiple images')
    parser.add_argument('--dir', type=str, help='Directory with images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--cascade', action='store_true', help='Enable cascade mode')
    parser.add_argument('--cascade-threshold', type=float, default=0.7, help='Cascade confidence threshold')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    parser.add_argument('--export-onnx', type=str, help='Export to ONNX file')
    
    args = parser.parse_args()
    
    # Проверка входных данных
    if not args.image and not args.batch and not args.dir and not args.export_onnx:
        parser.error("Укажите --image, --batch, --dir или --export-onnx")
    
    # ONNX export
    if args.export_onnx:
        inference = NPRInference(
            args.checkpoint,
            device=args.device,
            use_cascade=args.cascade,
            cascade_threshold=args.cascade_threshold
        )
        inference.export_to_onnx(args.export_onnx)
        return
    
    # Создание inference pipeline
    inference = NPRInference(
        args.checkpoint,
        device=args.device,
        use_cascade=args.cascade,
        cascade_threshold=args.cascade_threshold
    )
    
    # Сбор путей к изображениям
    image_paths = []
    
    if args.image:
        image_paths.append(args.image)
    
    if args.batch:
        image_paths.extend(args.batch)
    
    if args.dir:
        dir_path = Path(args.dir)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        for ext in extensions:
            image_paths.extend([str(p) for p in dir_path.glob(ext)])
    
    if not image_paths:
        logger.error("No images found!")
        return
    
    # Инференс
    logger.info(f"Processing {len(image_paths)} images...")
    results = inference.predict_batch(image_paths, args.threshold)
    
    # Вывод
    print("\n" + "=" * 70)
    for result in results:
        print(format_result(result))
        print("-" * 70)
    
    # Статистика
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        ai_count = sum(1 for r in valid_results if r['prediction'] == 'AI_GENERATED')
        real_count = sum(1 for r in valid_results if r['prediction'] == 'REAL')
        avg_time = np.mean([r['inference_time_ms'] for r in valid_results])
        
        print(f"\n📊 Summary:")
        print(f"   Total:     {len(valid_results)}")
        print(f"   AI:        {ai_count}")
        print(f"   Real:      {real_count}")
        print(f"   Avg Time:  {avg_time:.2f} ms/image")
    
    # Сохранение в JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
