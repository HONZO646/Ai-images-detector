"""Мини-прогон обучения на CPU (sanity check пайплайна без GPU-машины).

Ограничивает выборку размером локального кэша (не триггерит догрузку из HF),
отключает DataLoader-воркеров (Windows-спавн) и тренирует пару эпох.
Чекпоинты пишутся в стандартные checkpoints/ (ротация сохранится).

Запуск: python scripts/smoke_train.py [--epochs 2] [--batch-size 8]

Полное обучение (RTX 4090): python train.py — см. README «Развёртывание на GPU».
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description='Мини-прогон обучения (CPU)')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    config = Config()

    # Размеры выборки = сколько есть в кэше, иначе prepare_local_datasets
    # начнёт догружать датасеты из HuggingFace
    real_n = len(list(Path(config.data.real_data_path).glob('*.jpg')))
    ai_n = len(list(Path(config.data.ai_data_path).glob('*.jpg')))
    if real_n == 0 or ai_n == 0:
        raise SystemExit(
            f"Локальный кэш пуст (real={real_n}, ai={ai_n}). "
            "Выгрузите данные: python scripts/download_data.py"
        )
    config.data.real_max_samples = real_n
    config.data.ai_max_samples = ai_n

    # CPU-режим: без воркеров и AMP
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.num_workers = 0
    config.training.pin_memory = False
    config.training.use_amp = False
    config.training.save_interval = 1

    print(f"SMOKE: real={real_n}, ai={ai_n}, epochs={args.epochs}, "
          f"batch={args.batch_size}, device=cpu")

    trainer = Trainer(config)
    trainer.train()

    print("SMOKE TRAIN DONE")


if __name__ == '__main__':
    main()
