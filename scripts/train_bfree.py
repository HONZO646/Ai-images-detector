"""Обучение M1 на выровненных парах B-Free: 20k пар train, по 2k val и test, варианты фейков SD 2.1.

Запуск (из корня проекта):
    python scripts/train_bfree.py [--epochs 15] [--workers 4] [--train-pairs 20000] [--eval-pairs 2000]

Отличия от python train.py — только параметры запуска:
- --workers 4: на сервере с файлом подкачки ~7 ГБ 8 воркеров DataLoader + pin_memory
  упираются в лимит commit Windows (WinError 1455, см. ROADMAP D1); архитектура и данные не меняются;
- --epochs 15: 20k пар на эпоху, лучший чекпойнт по val AUC (early stopping, patience 8).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from train import Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='M1 на выровненных парах B-Free (20k пар)')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--train-pairs', type=int, default=20000)
    parser.add_argument('--eval-pairs', type=int, default=2000)
    args = parser.parse_args()

    config = Config()
    assert config.data.source == 'bfree', "config.data.source должен быть 'bfree'"
    config.data.bfree_train_max = args.train_pairs
    config.data.bfree_eval_max = args.eval_pairs
    config.training.epochs = args.epochs
    config.training.num_workers = args.workers
    print(f"RUN: source={config.data.source} train_pairs={config.data.bfree_train_max} "
          f"eval_pairs={config.data.bfree_eval_max} variants={list(config.data.bfree_fake_variants)} "
          f"epochs={config.training.epochs} workers={config.training.num_workers} "
          f"batch={config.training.batch_size} amp={config.training.amp_dtype}", flush=True)

    trainer = Trainer(config)
    test_metrics = trainer.train()

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ (test, B-Free held-out):")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Test AUC:       {test_metrics['auc']:.4f}")
    print(f"  Test F1:        {test_metrics['f1']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
