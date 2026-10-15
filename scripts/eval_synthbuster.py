"""
Оценка детектора на новых генераторах (расширенный Synthbuster, GRIP-UNINA / B-Free):
реальные RAISE (класс 0) против FLUX и SD3.5 (класс 1). Генераторы не участвуют в обучении.

Два конвейера (одни и те же изображения):
  standard   — как при обучении: квадратный ресайз до config.data.image_size;
  centre512  — контроль: центральный квадрат -> 512 -> config.data.image_size
               (выравнивает размеры и пропорции классов, которые различаются в исходных данных).

Выход: outputs/synthbuster_eval.json и таблица в stdout.
Запуск: python scripts/eval_synthbuster.py [--checkpoint checkpoints/best_model.pt]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from config import Config  # noqa: E402
from data.dataset import NPRDataset  # noqa: E402
from models.detector import create_detector  # noqa: E402


def centre_square(img: Image.Image, size: int = 512) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)


@torch.no_grad()
def predict(model, paths, labels, transform, target_size, device):
    ds = NPRDataset(paths, labels, transform=transform, target_size=target_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    out = []
    for gray, rgb, _ in loader:
        logits = model(gray.to(device), rgb.to(device))["logits"]
        out.append(torch.sigmoid(logits).float().cpu().numpy().reshape(-1))
    return np.concatenate(out)


def summarize(p, y):
    y = np.asarray(y)
    pred = (p >= 0.5).astype(int)
    tpr = float(pred[y == 1].mean())
    tnr = float(1 - pred[y == 0].mean())
    return {
        "n_real": int((y == 0).sum()),
        "n_ai": int((y == 1).sum()),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "bacc_at_0.5": round((tpr + tnr) / 2, 4),
        "ai_recall": round(tpr, 4),
        "real_specificity": round(tnr, 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Оценка на новых генераторах (Synthbuster-ext)")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--eval-dir", default="data/eval/synthbuster")
    args = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_detector(cfg.model, device=device, pretrained_path=args.checkpoint).to(device).eval()

    base = Path(args.eval_dir)
    real = sorted((base / "real_raise").glob("*.png"))
    gens = {"flux": sorted((base / "flux").glob("*.png")), "sd3_large": sorted((base / "sd3_large").glob("*.png"))}

    results = {"checkpoint": args.checkpoint, "image_size": cfg.data.image_size, "pipelines": {}}
    for pipe_name, transform in [("standard", None), ("centre512", lambda im: centre_square(im, 512))]:
        per_gen = {}
        for gen_name, ai in gens.items():
            paths = [str(p) for p in real] + [str(p) for p in ai]
            labels = [0] * len(real) + [1] * len(ai)
            p = predict(model, paths, labels, transform, cfg.data.image_size, device)
            per_gen[gen_name] = summarize(p, labels)
            print(f"[{pipe_name}] RAISE vs {gen_name}: {per_gen[gen_name]}", flush=True)
        results["pipelines"][pipe_name] = per_gen

    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "synthbuster_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("saved", out / "synthbuster_eval.json", flush=True)


if __name__ == "__main__":
    main()
