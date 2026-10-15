"""
Проверка асимметрии ресемплинга в B-Free (ROADMAP F8).

Реальные изображения COCO в B-Free: кроп 480 -> растяжка до 512 (след апскейлинга).
Фейки SD 2.1: нативные 512 без такого следа. Если детектор опирается на след апскейлинга
реальных, то тот же след у фейков должен заметно снизить AUC.

  V0 — тестовые пары как есть;
  V1 — только фейки проходят даунскейл 512->480 и апскейл 480->512 (тот же след, что у реальных).

Запуск (из корня проекта):
  python scripts/resampling_parity.py --checkpoint checkpoints/pilot_m1_best.pt --train-pairs 4000 --eval-pairs 500
Выход: outputs/resampling_parity.json
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
from data.bfree_split import assign_variant, select_ids  # noqa: E402
from data.dataset import NPRDataset  # noqa: E402
from models.detector import create_detector  # noqa: E402


def resample_trace(img: Image.Image) -> Image.Image:
    """Тот же след интерполяции, что у реальных B-Free: вниз до 480, вверх до 512."""
    return img.resize((480, 480), Image.LANCZOS).resize((512, 512), Image.LANCZOS)


@torch.no_grad()
def predict(model, paths, labels, transform, size, device):
    ds = NPRDataset(paths, labels, transform=transform, target_size=size)
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
        "n": int(len(y)),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "bacc_at_0.5": round((tpr + tnr) / 2, 4),
        "ai_recall": round(tpr, 4),
        "real_specificity": round(tnr, 4),
        "mean_p_real": round(float(p[y == 0].mean()), 4),
        "mean_p_ai": round(float(p[y == 1].mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Проверка асимметрии ресемплинга B-Free")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-pairs", type=int, default=4000, help="как при обучении пилота")
    ap.add_argument("--eval-pairs", type=int, default=500, help="как при обучении пилота")
    ap.add_argument("--variant", default=None, help="фиксированный вариант фейков; по умолчанию как при обучении")
    args = ap.parse_args()

    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = ROOT / cfg.data.bfree_root
    # тестовые id — те же правила выбора, что при обучении (prefix-свойство select_ids)
    _, _, ids = select_ids(root / "train_list.csv", root / "valid_list.csv",
                           cfg.training.seed, args.train_pairs, args.eval_pairs)
    variants = list(cfg.data.bfree_fake_variants)

    def fake_path(image_id):
        folder = args.variant or assign_variant(image_id, variants, cfg.training.seed)
        return root / folder / f"{image_id}.png"

    ids = [i for i in ids
           if (root / cfg.data.bfree_real_dir / f"{i}.png").exists() and fake_path(i).exists()]
    if not ids:
        raise SystemExit("no complete test pairs found on disk")
    real = [str(root / cfg.data.bfree_real_dir / f"{i}.png") for i in ids]
    fake = [str(fake_path(i)) for i in ids]
    print(f"complete test pairs: {len(ids)}", flush=True)
    model = create_detector(cfg.model, device=device, pretrained_path=args.checkpoint).to(device).eval()

    p_real = predict(model, real, [0] * len(real), None, cfg.data.image_size, device)
    p_fake = predict(model, fake, [1] * len(fake), None, cfg.data.image_size, device)
    p_fake_trace = predict(model, fake, [1] * len(fake), resample_trace, cfg.data.image_size, device)

    y = np.array([0] * len(real) + [1] * len(fake))
    results = {
        "checkpoint": args.checkpoint,
        "variant": args.variant or "as in training (assign_variant)",
        "n_pairs": len(ids),
        "V0_as_is": summarize(np.concatenate([p_real, p_fake]), y),
        "V1_fakes_with_real_trace": summarize(np.concatenate([p_real, p_fake_trace]), y),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "resampling_parity.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
