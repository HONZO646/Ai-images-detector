"""
Выбор id и вариантов для B-Free (без torch: используется загрузчиком, скриптом выгрузки и воротами).

В авторском наборе valid_list.csv — подмножество train_list.csv (все 10 304 id входят в train).
Поэтому обучение берём из train минус valid, а val/test делим из valid пополам по id
(группы id не пересекаются, пары одного id всегда в одном сплите).

Вариант фейка SD 2.1 выбирается детерминированно по id (SHA-256 от seed и id), поэтому
не зависит от порядка и размера списков: выгрузка и загрузчик всегда согласованы.
"""
import csv
import hashlib
import random
from typing import List, Sequence, Tuple


def read_ids(csv_path) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row["id"].strip() for row in csv.DictReader(f)]


def select_ids(train_csv, valid_csv, seed: int, train_max: int, eval_max: int) -> Tuple[List[str], List[str], List[str]]:
    """Возвращает (train_ids, val_ids, test_ids). Детерминировано при одинаковом seed."""
    train_all = read_ids(train_csv)
    valid_all = read_ids(valid_csv)
    valid_set = set(valid_all)

    pool = [i for i in train_all if i not in valid_set]
    rng = random.Random(seed)
    rng.shuffle(pool)
    train_ids = pool[:train_max]

    rng.shuffle(valid_all)
    half = len(valid_all) // 2
    val_ids = valid_all[:half][:eval_max]
    test_ids = valid_all[half:][:eval_max]
    return train_ids, val_ids, test_ids


def assign_variant(image_id: str, variants: Sequence[str], seed: int) -> str:
    """Вариант фейка для id: равномерно по variants, детерминированно (SHA-256 от seed и id)."""
    digest = hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
    return variants[int.from_bytes(digest[:8], "big") % len(variants)]
