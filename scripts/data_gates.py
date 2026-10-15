"""
Ворота качества данных до обучения: нет ли шорткатов по простым признакам.

Для подвыборки пар (real, AI) из train-сплита B-Free считаются признаки и линейная проба
(логистическая регрессия, 5-fold CV, ROC AUC). 0.5 = признак не различает классы.

  pixel_stats  — средние и std RGB, насыщенность, Laplacian, байт на пиксель;
  npr_residual — остаток x - up(down(x)) по яркости (интерполяционные следы NPR-подобного типа);
  clip_probe   — проба по замороженным CLIP-эмбеддингам (содержание и стиль).

Геометрия: все изображения B-Free — 512x512 (проверяется в выводе).
Запуск (из корня проекта): python scripts/data_gates.py --per-class 800
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config import Config  # noqa: E402
from data.bfree_split import assign_variant, select_ids  # noqa: E402
from data.dataset import NPRDataset  # noqa: E402
from models.detector import create_detector  # noqa: E402


def pixel_features(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    flat = arr.reshape(-1, 3)
    bpp = path.stat().st_size * 8 / (h * w)
    return np.concatenate([flat.mean(0), flat.std(0), [sat.mean(), np.abs(lap).mean(), bpp]])


def npr_features(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    g = gray[:h2, :w2]
    down = g[::2, ::2]
    up = np.repeat(np.repeat(down, 2, axis=0), 2, axis=1)
    res = g - up
    return np.array([np.abs(res).mean(), res.std(), (res ** 2).mean()], dtype=np.float64)


def cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, p)), 4)


@torch.no_grad()
def clip_embeddings(model, paths, labels, target_size, device) -> np.ndarray:
    ds = NPRDataset(paths, labels, transform=None, target_size=target_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    out = []
    for _, rgb, _ in loader:
        out.append(model.semantic_branch.clip_extractor(rgb.to(device)).float().cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(description="Ворота качества пар B-Free")
    ap.add_argument("--per-class", type=int, default=800)
    args = ap.parse_args()

    cfg = Config()
    root = ROOT / cfg.data.bfree_root
    train_ids, _, _ = select_ids(root / "train_list.csv", root / "valid_list.csv",
                                 cfg.training.seed, args.per_class, 1)
    real_dir = root / cfg.data.bfree_real_dir
    real_paths = [real_dir / f"{i}.png" for i in train_ids]
    fake_paths = [root / assign_variant(i, cfg.data.bfree_fake_variants, cfg.training.seed) / f"{i}.png"
                  for i in train_ids]
    paths = real_paths + fake_paths
    y = np.array([0] * len(real_paths) + [1] * len(fake_paths))
    print(f"gate sample: {len(real_paths)} real + {len(fake_paths)} AI pairs (ids from train split)", flush=True)

    sizes = set()
    for p in paths[:50]:
        with Image.open(p) as im:
            sizes.add(im.size)
    print("image sizes (first 50 files):", sorted(sizes), flush=True)

    pix = np.vstack([pixel_features(p) for p in paths])
    npr = np.vstack([npr_features(p) for p in paths])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_detector(cfg.model, device=device, pretrained_path=None).to(device).eval()
    emb = clip_embeddings(model, [str(p) for p in paths], y.tolist(), cfg.data.image_size, device)

    results = {
        "n_pairs_per_class": len(real_paths),
        "pixel_stats_cv_auc": cv_auc(pix, y),
        "npr_residual_cv_auc": cv_auc(npr, y),
        "clip_probe_cv_auc": cv_auc(emb, y),
        "interpretation": "0.5 = no difference; high values = shortcut signal in this sample",
    }
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "data_gates_bfree.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
