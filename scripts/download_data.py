#!/usr/bin/env python3
"""
Скрипт для загрузки данных из HuggingFace
ImageNet-1k (real) + AI generated dataset
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import collect_imagenet_hf, collect_hf_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Загрузка данных из HuggingFace')
    parser.add_argument('--imagenet-samples', type=int, default=15000,
                       help='Количество изображений ImageNet (default: 15000)')
    parser.add_argument('--ai-dataset', type=str, default='gasstation/generated-images',
                       help='HF путь к AI dataset')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Директория для сохранения данных')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА ДАННЫХ ИЗ HUGGINGFACE")
    logger.info("=" * 60)

    # Создание директорий
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Загрузка ImageNet-1k
    logger.info("\n📥 Загрузка ImageNet-1k (real images)...")
    logger.info("   Источник: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    logger.info("   Split: validation")

    try:
        real_paths = collect_imagenet_hf(
            max_samples=args.imagenet_samples,
            cache_dir=None
        )
        logger.info(f"✅ ImageNet загружен: {len(real_paths)} изображений")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки ImageNet: {e}")
        logger.info("   Убедитесь что принята лицензия на странице датасета")
        return

    # Загрузка AI dataset
    logger.info(f"\n📥 Загрузка AI dataset: {args.ai_dataset}")

    try:
        ai_paths = collect_hf_dataset(
            dataset_name=args.ai_dataset,
            split='train'
        )
        logger.info(f"✅ AI dataset загружен: {len(ai_paths)} изображений")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки AI dataset: {e}")
        return

    # Итоговая статистика
    logger.info("\n" + "=" * 60)
    logger.info("ЗАГРУЗКА ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Real images: {len(real_paths)}")
    logger.info(f"AI images:   {len(ai_paths)}")
    logger.info(f"Total:       {len(real_paths) + len(ai_paths)}")
    logger.info(f"Data dir:    {os.path.abspath(args.output_dir)}")

    # Проверка баланса
    min_count = min(len(real_paths), len(ai_paths))
    if len(real_paths) != len(ai_paths):
        logger.warning(f"\n⚠️ Дисбаланс классов:")
        logger.warning(f"   Real: {len(real_paths)}, AI: {len(ai_paths)}")
        logger.warning(f"   Будет использовано: {min_count} каждого класса")


if __name__ == '__main__':
    main()
