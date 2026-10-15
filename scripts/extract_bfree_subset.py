"""
Выборочная выгрузка пар B-Free из удалённых zip (HTTP Range, параллельно).

Скачиваются только нужные файлы: центральный каталог каждого zip читается один раз,
затем каждый PNG тянется отдельным Range-запросом (несколько потоков). Файлы кладутся в
data/bfree/<папка>/<id>.png — там их читает data/bfree.py. Выбор id и вариантов фейков
совпадает с загрузчиком (data.bfree_split с тем же seed).

Запуск (из корня проекта):
    python scripts/extract_bfree_subset.py --train-pairs 20000 --eval-pairs 2000 --workers 16
Требуется: pip install remotezip requests; train_list.csv и valid_list.csv в data/bfree/.
"""
import argparse
import json
import struct
import sys
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402  (модуль без torch)
from data.bfree_split import assign_variant, select_ids  # noqa: E402

BASE = "https://www.grip.unina.it/download/prog/B-Free/training_data/"
REAL_ZIP = {"COCO_real_512": BASE + "COCO_real_512.zip"}
VARIANT_ZIP = {
    "SD2.1_selfconditioned": BASE + "SD2.1_selfconditioned.zip",
    "SD2.1_inpainted_samecat": BASE + "SD2.1_inpainted_samecat.zip",
    "SD2.1_inpainted_diffcat": BASE + "SD2.1_inpainted_diffcat.zip",
}
_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "session"):
        _tls.session = requests.Session()
    return _tls.session


def _fetch(url: str, info) -> bytes:
    """Одна запись zip по Range: локальный заголовок + данные, deflate/stored."""
    start = info.header_offset
    span = 30 + len(info.filename.encode("utf-8")) + len(info.extra or b"") + info.compress_size + 512
    headers = {"Range": f"bytes={start}-{start + span - 1}"}
    for attempt in range(4):
        try:
            resp = _session().get(url, headers=headers, timeout=120)
            resp.raise_for_status()
            buf = resp.content
            if buf[:4] != b"PK\x03\x04":
                raise ValueError("bad local file header")
            n, m = struct.unpack("<HH", buf[26:30])
            data_start = 30 + n + m
            raw = buf[data_start:data_start + info.compress_size]
            if info.compress_type == 8:
                data = zlib.decompressobj(-15).decompress(raw)
            elif info.compress_type == 0:
                data = raw
            else:
                raise ValueError(f"unsupported compress_type {info.compress_type}")
            if len(data) != info.file_size:
                raise ValueError(f"size mismatch {len(data)} != {info.file_size}")
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser(description="Выборочная выгрузка пар B-Free")
    ap.add_argument("--train-pairs", type=int, default=20000)
    ap.add_argument("--eval-pairs", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = Config()
    bfree_root = ROOT / cfg.data.bfree_root
    variants = list(cfg.data.bfree_fake_variants)
    seed = cfg.training.seed
    train_ids, val_ids, test_ids = select_ids(
        bfree_root / "train_list.csv", bfree_root / "valid_list.csv",
        seed, args.train_pairs, args.eval_pairs,
    )
    wanted = train_ids + val_ids + test_ids
    fake_of = {i: assign_variant(i, variants, seed) for i in wanted}
    print(f"ids: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} | "
          f"fake variants: {dict(Counter(fake_of.values()))}", flush=True)

    jobs = [(folder, url, wanted) for folder, url in REAL_ZIP.items()]
    for folder in variants:
        jobs.append((folder, VARIANT_ZIP[folder], [i for i in wanted if fake_of[i] == folder]))

    tasks = []
    for folder, url, ids in jobs:
        out_dir = bfree_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        t_dir = time.time()
        with RemoteZip(url) as z:
            index = {i.filename: i for i in z.infolist()}
        print(f"{folder}: central directory {len(index)} entries in {time.time() - t_dir:.1f}s, ids={len(ids)}",
              flush=True)
        before = len(tasks)
        for image_id in ids:
            name = f"{folder}/{image_id}.png"
            info = index.get(name)
            if info is None:
                print("missing in zip:", name, flush=True)
                continue
            dest = out_dir / f"{image_id}.png"
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            tasks.append((url, info, dest))
        print(f"{folder}: to fetch {len(tasks) - before}", flush=True)

    t0 = time.time()
    done = failed = 0
    nbytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_fetch, url, info): dest for url, info, dest in tasks}
        for fut in as_completed(futures):
            dest = futures[fut]
            try:
                data = fut.result()
            except Exception as e:
                failed += 1
                print(f"FAILED {dest.name}: {type(e).__name__}: {e}", flush=True)
                continue
            dest.write_bytes(data)
            done += 1
            nbytes += len(data)
            if done % 500 == 0 or done + failed == len(tasks):
                dt = max(time.time() - t0, 1e-6)
                print(f"{done + failed}/{len(tasks)} files | ok={done} failed={failed} | "
                      f"{nbytes / 1e6:.0f} MB | {nbytes / 1e6 / dt:.2f} MB/s", flush=True)

    manifest = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
                "variants": variants, "seed": seed}
    (bfree_root / "subset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"DONE ok={done} failed={failed} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
