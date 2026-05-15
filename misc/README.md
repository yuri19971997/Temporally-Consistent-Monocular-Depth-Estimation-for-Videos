# Depth extraction and visualization scripts

This README covers five entry points in **Depth_Anything_V2_Stage_3**:

| Script | Path |
|--------|------|
| `depth_extract_v2_final_modified_v1.py` | [`code/scripts/depth/depth_extract_v2_final_modified_v1.py`](../code/scripts/depth/depth_extract_v2_final_modified_v1.py) |
| `depth_extract_v5.py` | [`code/scripts/depth/depth_extract_v5.py`](../code/scripts/depth/depth_extract_v5.py) |
| `depth_extract_v8.py` | [`code/scripts/depth/depth_extract_v8.py`](../code/scripts/depth/depth_extract_v8.py) |
| `depth_extract_v9.py` | [`code/scripts/depth/depth_extract_v9.py`](../code/scripts/depth/depth_extract_v9.py) |
| `depth_visualize.py` | [`code/scripts/visualization/depth_visualize.py`](../code/scripts/visualization/depth_visualize.py) |

---

## 1. Environment requirements

### 1.1 Python and hardware

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10+ recommended (match your CUDA PyTorch build). |
| **GPU** | **Strongly recommended** for all `depth_extract_*` scripts: Depth Anything V2 + RAFT on CUDA. CPU is not practical for video. |
| **`depth_visualize.py`** | Runs on **CPU** (OpenCV + NumPy); **no PyTorch** needed for visualization only. |

### 1.2 Python packages

**Repository root** [`requirements.txt`](../requirements.txt) lists:

- `torch`, `torchvision`
- `opencv-python`
- `tqdm`, `matplotlib`
- `gradio` (for other app entry points; not required for these five scripts)

**Per script:**

| Script | Extra dependencies |
|--------|---------------------|
| **v2_final_modified_v1** | `torch`, `torchvision`, `opencv-python`, `numpy`, `tqdm`; RAFT from [`raft_repo`](../raft_repo) (no pip package). |
| **v5** | Same as v2 (DA2 + RAFT). |
| **v8** | Same + **`ultralytics`** (`pip install ultralytics`) for YOLOv8-seg. |
| **v9** | Same as v8 (RANSAC uses OpenCV `cv2`, already covered). |
| **depth_visualize** | `opencv-python`, `numpy`, `tqdm`; **`ffmpeg`** in `PATH` if you use `--h264`. |

**PyTorch:** install a build that matches your CUDA version from [pytorch.org](https://pytorch.org). Example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 1.3 Weights and on-disk layout

Inside the repo, the depth scripts set `PROJECT_ROOT = <repo>/code` (the directory containing the `scripts/` tree). Under `code/`, **symlinks** usually point to assets at the **repository root**:

| Asset | Typical location (after symlinks) |
|-------|----------------------------------|
| Depth Anything V2 weights | `code/checkpoints/depth_anything_v2_{vits,vitb,vitl}.pth` → repo `checkpoints/` |
| RAFT weights | `code/raft_models/raft-things.pth` → repo `raft_models/` |
| RAFT code | `code/raft_repo` → repo `raft_repo/` |
| Model Python package | `code/depth_anything_v2` → repo `depth_anything_v2/` |

**YOLO (v8 / v9):** `yolov8n-seg.pt` (or other sizes) is downloaded automatically by Ultralytics on first use unless you point to a local file.

---

## 2. How to run (working directory and paths)

**Recommended:** open a terminal whose **current working directory is the repository root**:

```text
Depth_Anything_V2_Stage_3/
```

Run scripts with module-relative paths from there:

```bash
cd /path/to/Depth_Anything_V2_Stage_3

python code/scripts/depth/depth_extract_v5.py \
  --video-path data/input_videos/example.mp4 \
  --output data/depth_maps/example_v5.npz \
  --use-rotation
```

**Videos:** often stored under [`data/input_videos/`](../data/input_videos).

**Outputs:** save `.npz` depth runs under e.g. `data/depth_maps/` (create if needed).

**Visualization:** pass the same `.npz` (and optional RGB `--video-path`) to `depth_visualize.py`.

**Important:** `--video-path` and `--output` / `--input` are whatever paths you use; the scripts do not require a fixed output folder name.

---

## 3. Command-line arguments

### 3.1 `depth_extract_v2_final_modified_v1.py`

**2D** temporal pipeline: RAFT flow, photometric + forward–backward consistency, scale/shift alignment, foreground/background fusion, optional hole filling and smoothing (no Kabsch 3D path; no YOLO).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video-path` | str | **required** | Input video. |
| `--output` | str | **required** | Output `.npz`. |
| `--encoder` | str | `vitl` | `vits`, `vitb`, `vitl`. |
| `--input-size` | int | `518` | Depth network input size. |
| `--photo-threshold` | float | `0.25` | Photometric L1 threshold (RGB consistency). |
| `--fb-threshold` | float | `4.0` | Forward–backward flow error threshold (pixels). |
| `--max-scale-diff` | float | `0.1` | Max deviation of scale from 1.0 per frame. |
| `--max-shift-diff` | float | `0.1` | Max shift deviation. |
| `--min-valid-pixels` | int | `100` | Minimum reliable pixels to align. |
| `--morph-kernel-size` | int | `7` | Morphology kernel size. |
| `--apply-light-dilation` | flag | off | Lighter dilation of validity mask. |
| `--fg-percentile` | float | `60` | Foreground depth percentile. |
| `--fg-blend` | float | `0.15` | History blend weight for foreground. |
| `--fusion-kernel-size` | int | `15` | Fusion smoothing kernel. |
| `--fill-fg-holes` | flag | off | Aggressive foreground hole fill. |
| `--fg-hole-threshold` | float | `0.7` | FG hole detection threshold. |
| `--bilateral-smooth` | flag | off | Bilateral smoothing. |
| `--bilateral-spatial` | float | `5` | Bilateral spatial sigma. |
| `--bilateral-intensity` | float | `0.08` | Bilateral intensity sigma. |
| `--inpaint-holes` | flag | off | Inpaint small holes. |
| `--max-hole-size` | int | `200` | Max hole size (px) for inpainting. |
| `--interpolate-holes` | flag | off | Interpolate holes from neighbors. |
| `--max-interpolate-hole-size` | int | `500` | Max hole size for interpolation. |
| `--detect-gaps` | flag | off | Detect large failure regions. |
| `--gap-area-threshold` | int | `1000` | Min area for gap detection. |
| `--gap-total-threshold` | int | `600` | Min total gap pixels. |
| `--fill-gaps` | flag | off | Fill gaps from temporal neighbors. |
| `--gap-window` | int | `3` | Frames before/after for gap fill. |
| `--flow-iters` | int | `20` | RAFT refinement iterations. |
| `--min-depth-value` | float | `1e-6` | Depth clamp floor. |
| `--scene-threshold` | float | `30.0` | Scene-cut histogram threshold. |

```bash
python code/scripts/depth/depth_extract_v2_final_modified_v1.py \
  --video-path data/input_videos/clips.mp4 \
  --output data/depth_maps/clips_v2.npz
```

---

### 3.2 `depth_extract_v5.py`

**3D Kabsch–Umeyama** alignment: low–optical-flow anchors, optional rotation, scale clamp.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video-path` | str | **required** | Input video. |
| `--output` | str | **required** | Output `.npz`. |
| `--encoder` | str | `vitl` | `vits`, `vitb`, `vitl`. |
| `--input-size` | int | `518` | Depth input size. |
| `--fov` | float | `70.0` | Assumed horizontal FOV (degrees) for intrinsics. |
| `--use-rotation` | flag | off | Enable full 3D similarity (rotation + scale + translation). |
| `--anchor-percent` | float | `0.15` | Fraction of pixels with **smallest** flow used as anchors. |
| `--max-scale-diff` | float | `0.1` | Per-frame scale clamp around 1.0. |
| `--min-valid-pixels` | int | `100` | Minimum anchors required. |
| `--scene-threshold` | float | `30.0` | Scene-cut threshold (0 disables if implemented in script). |

```bash
python code/scripts/depth/depth_extract_v5.py \
  --video-path data/input_videos/clips.mp4 \
  --output data/depth_maps/clips_v5.npz \
  --use-rotation
```

---

### 3.3 `depth_extract_v8.py`

V5-style Kabsch anchors + **YOLOv8 segmentation** to remove dynamic classes (default: person) from the anchor set.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video-path` | str | **required** | Input video. |
| `--output` | str | **required** | Output `.npz`. |
| `--yolo-model` | str | `yolov8n-seg.pt` | Ultralytics seg model name/path. |
| `--blacklist` | int… | `0` | COCO class IDs to exclude (e.g. `0 2` person + car). |
| `--yolo-confidence` | float | `0.5` | Min conf. to apply mask. |
| `--encoder` | str | `vitl` | `vits`, `vitb`, `vitl`. |
| `--input-size` | int | `518` | Depth input size. |
| `--anchor-percent` | float | `0.15` | Low-flow fraction among valid+semantic-safe pixels. |
| `--max-scale-diff` | float | `0.1` | Scale clamp. |
| `--use-rotation` | flag | off | Kabsch with rotation. |
| `--fov` | float | `70.0` | Horizontal FOV (degrees). |
| `--scene-threshold` | float | `30.0` | Scene-cut threshold. |

```bash
python code/scripts/depth/depth_extract_v8.py \
  --video-path data/input_videos/clips.mp4 \
  --output data/depth_maps/clips_v8.npz \
  --use-rotation \
  --blacklist 0 2
```

---

### 3.4 `depth_extract_v9.py`

**RANSAC homography inliers** (from RAFT correspondences) ∩ **YOLO background** ∩ low-flow **anchor-percent** subset, then Kabsch.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video-path` | str | **required** | Input video. |
| `--output` | str | **required** | Output `.npz`. |
| `--ransac-thresh` | float | `3.0` | Max reprojection error (px) for homography inliers. |
| `--ransac-stride` | int | `8` | Grid stride when subsampling points for homography. |
| `--yolo-model` | str | `yolov8n-seg.pt` | YOLO seg weights. |
| `--blacklist` | int… | `0` | COCO IDs to exclude. |
| `--yolo-confidence` | float | `0.5` | Min detection confidence. |
| `--anchor-percent` | float | `0.15` | Of (inlier ∩ background), keep lowest-flow fraction. |
| `--encoder` | str | `vitl` | `vits`, `vitb`, `vitl`. |
| `--input-size` | int | `518` | Depth input size. |
| `--max-scale-diff` | float | `0.1` | Scale clamp. |
| `--use-rotation` | flag | off | Kabsch rotation on. |
| `--fov` | float | `70.0` | Horizontal FOV (degrees). |
| `--scene-threshold` | float | `30.0` | Scene-cut threshold. |

```bash
python code/scripts/depth/depth_extract_v9.py \
  --video-path data/input_videos/clips.mp4 \
  --output data/depth_maps/clips_v9.npz \
  --use-rotation \
  --ransac-thresh 3.0
```

---

### 3.5 `depth_visualize.py`

Renders **existing** `.npz` depth sequences to a video. Modes: single file, `--compare` (2 NPZs), `--grid` (2–4 NPZs), `--triple` (RGB + 2 NPZs).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | str | `None` | Single `.npz` (single-file mode). |
| `--output` | str | `None` | Output video path (auto if omitted). |
| `--video-path` | str | `None` | Source RGB for overlays / side-by-side / triple. |
| `--grid` | flag | off | 2×2 layout; use with `--inputs` (2–4 files). |
| `--compare` | flag | off | Side-by-side **two** depth streams; `--inputs` with 2 paths. |
| `--triple` | flag | off | RGB \| depth1 \| depth2; requires `--video-path` and 2 `--inputs`. |
| `--inputs` | str… | `None` | List of `.npz` for grid/compare/triple. |
| `--labels` | str… | `None` | Labels for panels in comparison modes. |
| `--norm-mode` | str | `global` | `global`, `adaptive`, `per-frame`. |
| `--norm-alpha` | float | `0.05` | Adaptive norm smoothing (smaller = slower adaptation). |
| `--reset-norm-on-scene-cut` | flag | off | Reset adaptive norm on scene cuts (if metadata present). |
| `--colormap` | str | `inferno` | `inferno`, `viridis`, `plasma`, `magma`, `jet`, `hot`, `bone`, `turbo`. |
| `--grayscale` | flag | off | Grayscale instead of colormap. |
| `--side-by-side` | flag | off | Single mode: append RGB next to depth. |
| `--dashboard` | flag | off | Single mode: 2×2 dashboard (needs `trust_maps` in `.npz` if used). |
| `--h264` | flag | off | Re-encode with FFmpeg to H.264. |
| `--h264-crf` | int | `23` | FFmpeg CRF (lower = better quality). |

```bash
python code/scripts/visualization/depth_visualize.py \
  --input data/depth_maps/clips_v5.npz \
  --video-path data/input_videos/clips.mp4 \
  --side-by-side \
  --colormap turbo \
  --output data/output_videos/clips_v5_vis.mp4
```

---

## 4. Quick reference

```bash
# Optional: create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics   # v8 / v9 only

cd Depth_Anything_V2_Stage_3

# Extract (example v5)
python code/scripts/depth/depth_extract_v5.py \
  --video-path data/input_videos/orient_day.mp4 \
  --output data/depth_maps/orient_day_v5.npz --use-rotation

# Visualize
python code/scripts/visualization/depth_visualize.py \
  --input data/depth_maps/orient_day_v5.npz \
  --video-path data/input_videos/orient_day.mp4 \
  --side-by-side --h264
```

For fuller project context, see [`code/scripts/README.md`](../code/scripts/README.md) and [`docs/PROJECT_SUMMARY_3_STAGES.md`](PROJECT_SUMMARY_3_STAGES.md).
