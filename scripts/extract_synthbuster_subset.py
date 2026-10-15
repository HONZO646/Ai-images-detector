"""
Выборочная выгрузка тестового набора расширенного Synthbuster (GRIP-UNINA, B-Free):
реальные RAISE против FLUX и SD3.5 (подписи взяты из RAISE). Только тест, в обучении не участвует.

Использует тот же Range-механизм, что scripts/extract_bfree_subset.py.
Запуск: python scripts/extract_synthbuster_subset.py --per-class 500 --workers 16
Результат: data/eval/synthbuster/{real_raise,flux,sd3_large}/*.png
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_bfree_subset import _fetch  # noqa: E402  (тот же Range-загрузчик)

BASE = "https://www.grip.unina.it/download/prog/B-Free/extended_synthbuster/"
SOURCES = [
    # (zip, папка внутри zip, подпапка результата)
    ("real_RAISE_1k.zip", "real_RAISE_1k/", "real_raise"),
    ("sd3_flux.zip", "flux/", "flux"),
    ("sd3_flux.zip", "sd3_large/", "sd3_large"),
]


def main():
    ap = argparse.ArgumentParser(description="Выборка Synthbuster-ext для теста на новых генераторах")
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_root = ROOT / "data" / "eval" / "synthbuster"
    tasks = []
    for zip_name, prefix, sub in SOURCES:
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        with RemoteZip(BASE + zip_name) as z:
            infos = [i for i in z.infolist() if i.filename.startswith(prefix) and i.filename.lower().endswith(".png")]
        infos.sort(key=lambda i: i.filename)
        chosen = infos[: args.per_class]
        print(f"{sub}: available png={len(infos)} selected={len(chosen)}", flush=True)
        for info in chosen:
            dest = out_dir / Path(info.filename).name
            if dest.exists() and dest.stat().st_size == info.file_size:
                continue
            tasks.append((BASE + zip_name, info, dest))

    print(f"to fetch: {len(tasks)}", flush=True)
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
            if done % 200 == 0 or done + failed == len(tasks):
                dt = max(time.time() - t0, 1e-6)
                print(f"{done + failed}/{len(tasks)} | ok={done} failed={failed} | "
                      f"{nbytes / 1e6:.0f} MB | {nbytes / 1e6 / dt:.2f} MB/s", flush=True)
    print(f"DONE ok={done} failed={failed} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
