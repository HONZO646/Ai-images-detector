"""
Ablation-анализ: вклад веток детектора через линейные пробы на эмбеддингах.

Идея: обученный NPRDetector выдаёт эмбеддинги spatial / frequency / semantic
(и final — результат fusion). Поверх каждого набора признаков обучается
простая проба (LogisticRegression на стандартизованных векторах):
  - на val-сплите — обучение пробы;
  - на test-сплите — оценка AUC и F1@0.5.

Получается таблица, показывающая дискриминативность каждой ветки и их
комбинаций без переобучения самого детектора. Полный fusion-эмбеддинг
(final) служит верхней планкой: пробы не могут его превзойти по
построению (линейность против обученного классификатора).

Использование:
    python ablation.py --checkpoint checkpoints/best_model.pt

Выход:
    outputs/ablation.csv, outputs/ablation.md
"""
import os
import logging
import numpy as np
import torch
from typing import Dict

from config import Config
from models.detector import create_detector
from data.dataset import prepare_datasets, create_eval_loader
from utils import get_device, find_f1_optimal_threshold

logger = logging.getLogger(__name__)

# Наборы признаков: имя → ключи эмбеддингов из output детектора
FEATURE_SETS = [
    ('spatial',      ['spatial_embedding']),
    ('frequency',    ['frequency_embedding']),
    ('semantic',     ['semantic_embedding']),
    ('sp+fr',        ['spatial_embedding', 'frequency_embedding']),
    ('sp+se',        ['spatial_embedding', 'semantic_embedding']),
    ('se+fr',        ['semantic_embedding', 'frequency_embedding']),
    ('sp+fr+se',     ['spatial_embedding', 'frequency_embedding', 'semantic_embedding']),
    ('final (fusion)', ['final_embedding']),
]


# Пайплайн проб: StandardScaler → LogisticRegression
def build_probe():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'))
    ])


@torch.no_grad()
def collect_embeddings(model, loader, device) -> Dict[str, np.ndarray]:
    """Один прогон по loader: эмбеддинги всех веток + final + метки."""
    model.eval()
    feats = {
        'spatial_embedding': [], 'frequency_embedding': [],
        'semantic_embedding': [], 'final_embedding': []
    }
    targets = []

    for gray, rgb, labels in loader:
        gray = gray.to(device)
        rgb = rgb.to(device)

        output = model(gray, rgb)
        for key in feats:
            feats[key].append(output[key].cpu().numpy())
        targets.append(labels.numpy().reshape(-1))

    result = {k: np.concatenate(v, axis=0) for k, v in feats.items()}
    result['targets'] = np.concatenate(targets, axis=0)
    return result


def run_ablation(config: Config, checkpoint_path: str) -> list:
    """Полный ablation-прогон. Возвращает список строк результатов."""
    device = get_device(config.training.device)
    logger.info(f"Using device: {device}")

    model = create_detector(config.model, device=str(device),
                            pretrained_path=checkpoint_path)
    model = model.to(device)

    _, val_dataset, test_dataset = prepare_datasets(config)
    val_loader = create_eval_loader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    test_loader = create_eval_loader(
        test_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    logger.info("Сбор эмбеддингов: val...")
    val_data = collect_embeddings(model, val_loader, device)
    logger.info("Сбор эмбеддингов: test...")
    test_data = collect_embeddings(model, test_loader, device)

    y_val = val_data['targets']
    y_test = test_data['targets']

    from sklearn.metrics import roc_auc_score, f1_score

    results = []
    for name, keys in FEATURE_SETS:
        x_val = np.concatenate([val_data[k] for k in keys], axis=1)
        x_test = np.concatenate([test_data[k] for k in keys], axis=1)

        probe = build_probe()
        probe.fit(x_val, y_val)

        p_val = probe.predict_proba(x_val)[:, 1]
        p_test = probe.predict_proba(x_test)[:, 1]

        # AUC (при вырожденном наборе классов — 0.0)
        val_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.0
        test_auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.0

        # F1 на калиброванном по val порогу (справедливо для всех проб)
        thr, _ = find_f1_optimal_threshold(p_val, y_val)
        test_f1 = f1_score(y_test, (p_test >= thr).astype(int), zero_division=0)

        results.append({
            'set': name,
            'n_features': int(x_val.shape[1]),
            'val_auc': float(val_auc),
            'test_auc': float(test_auc),
            'test_f1': float(test_f1),
            'threshold': float(thr),
        })
        logger.info(
            f"[Ablation] {name:16s} dim={x_val.shape[1]:4d} "
            f"val_auc={val_auc:.4f} test_auc={test_auc:.4f} test_f1={test_f1:.4f}"
        )

    return results


def save_results(results: list, output_dir: str):
    """Экспорт таблицы в CSV + Markdown."""
    os.makedirs(output_dir, exist_ok=True)
    header = ['set', 'n_features', 'val_auc', 'test_auc', 'test_f1', 'threshold']

    csv_path = os.path.join(output_dir, 'ablation.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in results:
            f.write(','.join(
                r[h] if isinstance(r[h], str) else f"{r[h]:.4f}" for h in header
            ) + '\n')
    logger.info(f"CSV сохранён: {csv_path}")

    md_path = os.path.join(output_dir, 'ablation.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Ablation: вклад веток (пробы на эмбеддингах)\n\n")
        f.write("Пробы (StandardScaler + LogisticRegression) обучены на val, оценены на test.\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in results:
            cells = [r[h] if isinstance(r[h], str) else f"{r[h]:.4f}" for h in header]
            f.write("| " + " | ".join(str(c) for c in cells) + " |\n")
    logger.info(f"Markdown сохранён: {md_path}")

    logger.info("\n" + "=" * 72)
    logger.info("ABLATION РЕЗУЛЬТАТЫ")
    logger.info("=" * 72)
    logger.info(f"{'set':16s} {'dim':>5s} {'val_auc':>9s} {'test_auc':>9s} {'test_f1':>9s}")
    for r in results:
        logger.info(
            f"{r['set']:16s} {r['n_features']:5d} {r['val_auc']:9.4f} "
            f"{r['test_auc']:9.4f} {r['test_f1']:9.4f}"
        )


def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='NPR Detector — ablation веток')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Каталог результатов (по умолчанию config.output_dir)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = Config()
    output_dir = args.output_dir or config.output_dir

    results = run_ablation(config, args.checkpoint)
    save_results(results, output_dir)


if __name__ == '__main__':
    main()
