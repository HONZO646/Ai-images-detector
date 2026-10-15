"""
B-Free (GRIP-UNINA, CVPR 2025): выровненные пары реальных и сгенерированных изображений.

Реальные — COCO (largest central crop -> 512x512). Сгенерированные — варианты SD 2.1 из B-Free:
self-conditioned (регенерация того же изображения) и inpainted same/diff category (правка объекта).
Для каждого id берётся один вариант фейка (детерминированно), поэтому баланс классов 1:1.
Пара связана одним COCO id, поэтому пары всегда попадают в один и тот же сплит.

Лицензия B-Free: только информационные и некоммерческие цели.
"""
import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from data.bfree_split import assign_variant, select_ids
from data.dataset import NPRDataset, create_augmentation_pipeline

logger = logging.getLogger(__name__)

IMG_EXTS = (".png", ".jpg", ".jpeg")


def _locate(folder: Path, image_id: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = folder / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def _build_pairs(real_dir: Path, root: Path, ids: List[str], variants: Sequence[str], seed: int
                 ) -> Tuple[List[str], List[int], int]:
    """Пары (real, fake) по id; вариант фейка — по assign_variant. Возвращает пути, метки (0=real, 1=AI), пропуски."""
    paths: List[str] = []
    labels: List[int] = []
    missing = 0
    for image_id in ids:
        real = _locate(real_dir, image_id)
        fake = _locate(root / assign_variant(image_id, variants, seed), image_id)
        if real is None or fake is None:
            missing += 1
            continue
        paths += [str(real), str(fake)]
        labels += [0, 1]
    return paths, labels, missing


def prepare_bfree_datasets(config):
    """Train/val/test из B-Free. Вызывается из data.dataset.prepare_datasets при data.source='bfree'."""
    root = Path(config.data.bfree_root)
    real_dir = root / config.data.bfree_real_dir
    variants = list(config.data.bfree_fake_variants)
    seed = config.training.seed

    train_ids, val_ids, test_ids = select_ids(
        root / "train_list.csv", root / "valid_list.csv",
        seed, config.data.bfree_train_max, config.data.bfree_eval_max,
    )

    tr_paths, tr_labels, tr_miss = _build_pairs(real_dir, root, train_ids, variants, seed)
    va_paths, va_labels, va_miss = _build_pairs(real_dir, root, val_ids, variants, seed)
    te_paths, te_labels, te_miss = _build_pairs(real_dir, root, test_ids, variants, seed)
    variant_counts = Counter(assign_variant(i, variants, seed) for i in train_ids)
    logger.info(
        f"B-Free пары: train={len(tr_paths) // 2} val={len(va_paths) // 2} test={len(te_paths) // 2} "
        f"(пропущено id: train={tr_miss}, val={va_miss}, test={te_miss}); "
        f"фейки в train по вариантам: {dict(variant_counts)}"
    )

    train_transform = create_augmentation_pipeline(config.augmentation)
    train_ds = NPRDataset(tr_paths, tr_labels, train_transform, config.data.image_size)
    val_ds = NPRDataset(va_paths, va_labels, None, config.data.image_size)
    test_ds = NPRDataset(te_paths, te_labels, None, config.data.image_size)

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "bfree_split.json", "w", encoding="utf-8") as f:
        json.dump({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
                   "variants": variants, "seed": seed}, f)

    return train_ds, val_ds, test_ds
