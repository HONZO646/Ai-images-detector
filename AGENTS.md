# AGENTS.md

NPR-based AI-generated image detector (PyTorch, single flat Python project). Three branches — spatial (NPR/Sobel/LBP), frequency (DWT/DCT/FFT), semantic (CLIP ViT-B/32) — fused via MLP + adaptive gating. README and code comments are in Russian; this file is the English quick-map.

## Commands
- Env: `.venv` (Python 3.14, CPU-only `torch 2.11` on this machine). Use `.venv\Scripts\python.exe <script>` or activate first.
- Install: `pip install -r requirements.txt`
- Train (streaming from HuggingFace): `python train.py` — needs `huggingface-cli login` and an accepted ImageNet-1k license.
- Web UI: `python gradio_ui.py` or `python train.py --ui` (serves `0.0.0.0:7860`).
- Inference: `python inference.py --checkpoint checkpoints/best_model.pt --image <path>` (also `--batch`, `--dir`, `--cascade`, `--export-onnx`).
- Evaluation: `python evaluate.py --checkpoint checkpoints/best_model.pt`.
- No pytest/lint/typecheck config. Verify with module smoke tests (`python -m models.fusion`, `python -m models.detector`) or `python -m py_compile <file>`; `python test_dataset.py` visualizes one streamed train batch.

## Non-obvious facts
- `output['probability']` from the detector is **raw logits**, not a probability — apply `torch.sigmoid` before thresholding/metrics. Same for `classifier(...)` / spatial-branch cascade outputs.
- `gate_weights` columns are `[semantic, spatial, frequency]` (0/1/2); `train.py` gate regularization assumes index 0 = semantic.
- Input contract: spatial/frequency take grayscale `[B,1,640,640]`; semantic takes RGB `[B,3,640,640]` (CLIP resizes to 224 internally). `config.data.image_size` is shared by dataset/inference/UI — keep consistent.
- `CrossAttentionFusion` in `models/fusion.py` is actually a plain MLP (single-token attention is degenerate); README still calls it cross-attention.
- Checkpoints are dicts with `model_state_dict`; load via `create_detector(..., pretrained_path=...)` or `utils.load_checkpoint` (`weights_only=False`), not `torch.load` directly.
- CLIP loads from local `models/clip-ViT-B-32` if present (it is here), else downloads `clip-ViT-B-32` and saves there. Only tokenizer/config files are git-tracked, weights are not.
- `models/`, `outputs/`, `checkpoints/`, `logs/` are gitignored — new files under `models/` need `git add -f`.
- Development happens on `dev`; `main` receives merge PRs (see `git log`).

## Stale / currently broken paths (verify before trusting docs)
- `data.streaming=True` (default) is the only maintained data path. `streaming=False` references removed config fields (`hf_imagenet_max_samples`, `hf_dataset_max_samples`); `scripts/download_data.py` imports `collect_imagenet_hf`/`collect_hf_dataset` that no longer exist in `data/dataset.py` (only `download_hf_images`). README and `scripts/*.md` describe that old local-download workflow.
- Working tree has uncommitted edits (`config.py`, `data/dataset.py`, `train.py`, `utils.py`). `prepare_datasets(config)` currently takes no `streaming=` kwarg, but `train.py` still passes it, so `python train.py` raises `TypeError`. Check `git status` / `git diff` before editing these.
- `DataConfig.data_root` hardcodes `C:\Users\honzo\.cache\huggingface`; streaming ignores it, non-streaming depends on it.
