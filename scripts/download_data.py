#!/usr/bin/env python3
"""
Выгрузка и кэширование данных на диск для обучения в нестриминговом режиме.

Real:    ImageNet-1k (ILSVRC/imagenet-1k) — требуется принятая лицензия и huggingface-cli login
AI:      gasstation/generated-images (только media_type == 'synthetic')

После выгрузки запускайте обучение с data.streaming=False в config.py.
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import download_hf_images

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_existing(directory) -> int:
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.jpg")))


def main():
    config = Config()

    parser = argparse.ArgumentParser(description='Выгрузка данных HuggingFace в локальный кэш')
    parser.add_argument('--real-samples', type=int, default=config.data.real_max_samples,
                        help=f'Количество ImageNet-изображений (default: {config.data.real_max_samples})')
    parser.add_argument('--ai-samples', type=int, default=config.data.ai_max_samples,
                        help=f'Количество AI-изображений (default: {config.data.ai_max_samples})')
    parser.add_argument('--data-root', type=str, default=config.data.data_root,
                        help=f'Корень кэша (default: {config.data.data_root})')
    args = parser.parse_args()

    real_dir = os.path.join(args.data_root, "real", "imagenet")
    ai_dir = os.path.join(args.data_root, "ai_generated")

    logger.info("=" * 60)
    logger.info("ВЫГРУЗКА ДАННЫХ В ЛОКАЛЬНЫЙ КЭШ")
    logger.info("=" * 60)
    logger.info(f"Real: {args.real_samples} → {os.path.abspath(real_dir)}")
    logger.info(f"AI:   {args.ai_samples} → {os.path.abspath(ai_dir)}")
    logger.info("Если кэш уже полон — загрузка пропускается (resume).")

    # Real: ImageNet-1k
    real_count = count_existing(real_dir)
    if real_count >= args.real_samples:
        logger.info(f"✅ Real кэш уже полон: {real_count}/{args.real_samples} — пропуск")
        real_paths = None
    else:
        logger.info(f"\n📥 {config.data.hf_imagenet_name} (split={config.data.hf_imagenet_split})")
        logger.info("   https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        try:
            real_paths = download_hf_images(
                dataset_name=config.data.hf_imagenet_name,
                split=config.data.hf_imagenet_split,
                output_dir=real_dir,
                prefix="real",
                max_samples=args.real_samples,
                streaming=True,
            )
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки ImageNet: {e}")
            logger.info("   Убедитесь что: 1) принята лицензия на странице датасета;")
            logger.info("                  2) выполнен huggingface-cli login")
            return

    # AI: gasstation/generated-images
    ai_count = count_existing(ai_dir)
    if ai_count >= args.ai_samples:
        logger.info(f"✅ AI кэш уже полон: {ai_count}/{args.ai_samples} — пропуск")
        ai_paths = None
    else:
        logger.info(f"\n📥 AI dataset: {config.data.hf_dataset_name} (split={config.data.hf_dataset_split})")
        try:
            ai_paths = download_hf_images(
                dataset_name=config.data.hf_dataset_name,
                split=config.data.hf_dataset_split,
                output_dir=ai_dir,
                prefix="ai",
                max_samples=args.ai_samples,
                streaming=True,
                filter_fn=lambda item: item.get('media_type', None) == 'synthetic',
            )
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки AI dataset: {e}")
            return

    # Итоговая статистика
    real_total = count_existing(real_dir)
    ai_total = count_existing(ai_dir)

    logger.info("\n" + "=" * 60)
    logger.info("ВЫГРУЗКА ЗАВЕРШЕНА")
    logger.info("=" * 60)
    logger.info(f"Real images: {real_total} ({os.path.abspath(real_dir)})")
    logger.info(f"AI images:   {ai_total} ({os.path.abspath(ai_dir)})")
    logger.info(f"Данные кэшированы в data_root: {os.path.abspath(args.data_root)}")

    if real_total != ai_total:
        logger.warning(f"⚠️ Дисбаланс классов: real={real_total}, ai={ai_total}")
        logger.warning("   prepare_local_datasets возьмёт min() от каждого класса")

    logger.info("\nДалее: убедитесь что в config.py data.streaming = False и запустите python train.py")


if __name__ == '__main__':
    main()