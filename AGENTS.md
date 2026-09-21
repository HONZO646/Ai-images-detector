# AGENTS.md

NPR-based AI-generated image detector (PyTorch, single flat Python project). Three branches — spatial (NPR/Sobel/LBP), frequency (DWT/DCT/FFT), semantic (CLIP ViT-B/32) — fused via MLP + adaptive gating. README and code comments are in Russian; this file is the English quick-map.

## Commands
- Env: `.venv` (Python 3.14, CPU-only `torch 2.11` on this machine). Use `.venv\Scripts\python.exe <script>` or activate first. GPU training targets a separate RTX 4090 machine (Python 3.12 + CUDA torch — see README «Развёртывание на GPU»).
- Install: `pip install -r requirements.txt`
- Data cache (main path): `python scripts/download_data.py --real-samples 5000 --ai-samples 5000` — needs `huggingface-cli login` and accepted ImageNet-1k license. Stores to `data/real/imagenet/` + `data/ai_generated/`.
- Train (local cache, `data.streaming=False` default): `python train.py`. Streaming fallback: set `data.streaming=True`.
- Web UI: `python gradio_ui.py` or `python train.py --ui` (serves `0.0.0.0:7860`).
- Inference: `python inference.py --checkpoint checkpoints/best_model.pt --image <path>` (also `--batch`, `--dir`; `--cascade` is experimental/uncalibrated; `--export-onnx` is disabled — FFT has no ONNX ops).
- Evaluation: `python evaluate.py --checkpoint checkpoints/best_model.pt` — calibrates F1-max threshold on val (saved to `outputs/threshold.json`, `--no-calibrate` keeps 0.5) and reports ECE/Brier.
- Ablation: `python ablation.py --checkpoint ...` — logistic probes on branch embeddings (fit on val, eval on test) → `outputs/ablation.{csv,md}`.
- Roadmap: `ROADMAP.md` — improvement plan (this session + RTX 4090 experiment phases).
- No pytest/lint/typecheck config. Verify with module smoke tests (`python -m models.fusion`, `python -m models.detector`) or `python -m py_compile <file>`; `python test_dataset.py` visualizes one train batch.

## Session logging (mandatory)
- After every user request that changes anything in the project, append one entry to `changes.log` (repo root): `<YYYY-MM-DD HH:mm> | <короткий запрос> | <краткая выжимка изменений>`. Write in Russian, keep it compact.
- Get the timestamp fresh at write time (`Get-Date -Format "yyyy-MM-dd HH:mm"`), do not guess it.
- `changes.log` is gitignored via the `*.log` rule — do not commit it unless asked.

## Non-obvious facts
- Detector output key is `output['logits']` — **raw logits**, not a probability. Apply `torch.sigmoid` before thresholding/metrics. Same for `classifier(...)` / spatial-branch cascade outputs. (Was renamed from `probability`.)
- `gate_weights` columns are `[semantic, spatial, frequency]` (0/1/2); `train.py` gate regularization assumes index 0 = semantic.
- Input contract: spatial/frequency take grayscale `[B,1,640,640]`; semantic takes RGB `[B,3,640,640]` (CLIP resizes to 224 internally). `config.data.image_size` is shared by dataset/inference/UI — keep consistent.
- `CrossAttentionFusion` in `models/fusion.py` is actually a plain MLP (single-token attention is degenerate); README diagram says MLP Fusion.
- Checkpoints are slim: frozen CLIP weights (`semantic_branch.clip_extractor.*`) are excluded unless CLIP blocks are unfrozen (~2 MB instead of 600 MB). Loading uses `strict=False` and pulls CLIP from the pretrained model — use `create_detector(..., pretrained_path=...)` or `utils.load_checkpoint`, not `torch.load` directly. Note: when CLIP is unfrozen, checkpoints store CLIP fully — keep `batch_size` modest.
- CLIP unfreeze: `semantic_unfreeze_last_n_blocks > 0` → `Trainer._build_param_groups` splits trainable params into two groups (CLIP @ `training.semantic_lr` = 1e-6, rest @ `learning_rate`); never raise CLIP LR to 3e-4.
- Evaluate calibrates the decision threshold on val (F1-max → `outputs/threshold.json`); test metrics use that threshold, not 0.5. `ablation.py` probes (logreg on embeddings) fit on val, eval on test — expect noisy results on tiny sets.
- `torch.quantile` is banned in `SobelExtractor` (CUDA limit ~2^24 elements caps batch ≈ 40 at 640×640) — p90 computed via `kthvalue`.
- AMP: `use_amp=True` + `amp_dtype='bfloat16'` by default; auto-disabled on CPU; GradScaler only for float16. `pin_memory` auto-off without CUDA. `reproducible=True` in config enables cudnn.deterministic (slower).
- `models/`, `outputs/`, `checkpoints/`, `logs/`, `data/real/`, `data/ai_generated/` are gitignored — new files under `models/` need `git add -f`. Epoch checkpoints rotate (only latest kept + best).
- CLIP loads from local `models/clip-ViT-B-32` if present (it is here), else downloads `clip-ViT-B-32` and saves there. Only tokenizer/config files are git-tracked, weights are not.
- Development happens on `dev`; `main` receives merge PRs (see `git log`).

## Data pipeline notes
- `prepare_datasets(config, streaming=None)` — `None` falls back to `config.data.streaming`.
- Streaming is a fallback: fixed order per epoch (no reshuffle), unknown length, val/test bounded by `.take()` on `n_per_class = min(real_max_samples, ai_max_samples)`.
- Local cache: `prepare_local_datasets` balances 1:1, seeds with `random.Random(training.seed)`, splits by `data.{train,val,test}_ratio`.
- Augmentations (train only, both classes symmetric) come from `config.augmentation`: downscale/upscale, sharpen (factor = 1 + U(range)), color jitter, probabilistic Gaussian blur, cascade JPEG with `jpeg_quality_range`.