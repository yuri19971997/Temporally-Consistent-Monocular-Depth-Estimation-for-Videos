# Depth Analysis — Results and Conclusions

**Scope:** This document summarizes quantitative and qualitative results for **Stage 3** video-depth work under `Depth_Anything_V2_Stage_3/`, with emphasis on artifacts in **`data/depth_analysis/`** (per-frame / per-pixel comparison plots). It also points to companion metrics under `data/comparison_results/`, `results/comparison/`, and narrative docs in `docs/` and `data/*.md`.

**Generated:** project documentation pass (automated synthesis from repository files).

---

## 1. What is in `data/depth_analysis/`?

The folder contains **45 image files** (mostly **PNG**). Almost all are **multi-curve comparison plots** produced by analysis scripts (e.g. temporal instability, flickering proxy, or pixel-depth trajectories at a probe location). Filenames encode which **depth NPZ runs** are compared and sometimes the **probe coordinates** (suffixes like `_x1000_y100`, `_x960_y540`).

There is **no separate CSV/JSON inside this folder**; numerical summaries live in **`data/comparison_results/*/metrics.json`**, **`results/comparison/*/metrics.json`**, and project **`docs/*.md`** files referenced below.

---

## 2. How to read the plot filenames (conventions)

These patterns recur across the PNG names:

| Token | Typical meaning |
|-------|------------------|
| **`Naive`** | Frame-by-frame depth (no temporal alignment / fusion). |
| **`V5`, `V6`, `V7`, `V8`, `V9`** | Pipeline versions (`depth_extract_v*.py`): Kabsch alignment variants; V8 adds YOLO; V9 adds RANSAC + YOLO; V7 denotes an intermediate experimental line in your sweeps. |
| **`r0` / `r1`** | Rotation recovery **off** vs **on** (`--use-rotation` in Kabsch pipelines). |
| **`a20`, `a40`, `a60`, `a6`** etc. | **Anchor percentage** ×10 or ×100 style labels in filenames (e.g. `a20` → ~20% low-flow anchor pool; exact mapping matches the run config). |
| **`s10`, `s20`** | **Scale-clamp** / stability parameters in runs (e.g. `--max-scale-diff` expressed as 10% vs 20% in naming). |
| **`m200`, `m500`, `d10`, `d20`, `so`** | V7 ablation shorthand in long filenames: motion / downsampling / “scale-only” or similar run tags (see paired `depth_extract` CLI for each experiment). |
| **`100_years_technion_a` … `_d`** | Different cuts or exports of the **Technion 100 years** source video. |
| **`35266_small`** | Shorter test clip (`35266-407130741_small.mp4` family). |
| **`pixel_1000_100` / `_x1000_y100`** | Plots for a **fixed image location** (y,x) over time — useful for flicker and temporal error curves. |

**Representative figure groups**

- **Early fusion vs anchors:** `2models_*_naive_vs_*_f20_occ0.*` — evolution from naive flow fusion toward occlusion-aware / robust fusion.
- **V4 → V5:** `*_v4_anchor_*` vs `*_v5_kabsch_*` — transition to **3D Kabsch–Umeyama** alignment.
- **V5 sweeps:** `V5_r0_*` / `V5_r1_*` grids — rotation off/on × anchor percent × scale clamp.
- **V5 vs V6:** e.g. `V6_r1_a60_s10_vs_V5_r1_a6_s10_vs_Naive_*` — hybrid **corners + low-flow** (V6) vs pure low-flow (V5).
- **V7 ablations:** `V7_m*_d*_a*_s*` and `*_so_*` — parameter matrix on top of V6-class logic.
- **Technion D — V8/V9:** `100_years_technion_d_v8_vs_v5_vs_naive_*`, `*_v9_*`, anchor sweeps, YOLO-M comparisons — semantic + RANSAC stack evaluation on a challenging clip.

---

## 3. Quantitative results (from `metrics.json` and docs)

### 3.1 `orient_day` (1125 frames) — baseline vs temporal

Source: [`data/comparison_results/orient_day/metrics.json`](../comparison_results/orient_day/metrics.json)

| Metric | Basic (naive) | Temporal | Improvement |
|--------|----------------|----------|-------------|
| Temporal instability | 1303.94 | 1279.91 | **↓ 1.84%** |
| Flickering | 12.52 | 11.97 | **↓ 4.35%** |

Narrative analysis: [`docs/25_11_30_results_analysis.md`](../../docs/25_11_30_results_analysis.md).

**Note:** A separate tuned run exists at [`data/comparison_results/orient_day_ot003_ft15/metrics.json`](../comparison_results/orient_day_ot003_ft15/metrics.json) with **different numeric scale** (instability ~9.45 vs ~9.11) but same qualitative conclusion (modest improvement). Always cite the JSON you used when reporting numbers.

### 3.2 `100_years_technion` (1500 frames, chunked comparison)

Source: [`data/comparison_results/100_years_technion/metrics.json`](../comparison_results/100_years_technion/metrics.json)

| Metric | Basic | Temporal | Improvement |
|--------|-------|----------|-------------|
| Temporal instability | 11.40 | 10.96 | **↓ 3.80%** |
| Flickering | 16.03 | 15.79 | **↓ 1.53%** |

Discussion and memory/chunking notes: [`docs/25_12_02_100_years_technion_results.md`](../../docs/25_12_02_100_years_technion_results.md).

### 3.3 V12 parameter sweep (`100_years_technion_d`, 107 frames)

Source: [`data/comparison_results/v12_analysis_summary.json`](../comparison_results/v12_analysis_summary.json)

- **Best stability (lowest `mean_std`):** `r1_a5_pt15_fbt2` — anchor **5%**, photometric thresh **0.15**, FB thresh **2.0 px**, rotation **on** (`mean_std` ≈ 69.80).
- **Best frame delta:** `r1_a25_fbt6` — anchor **25%**, FB thresh **6.0** (`mean_delta` ≈ 5.49).
- **Takeaway:** Stricter photometric + FB consistency can improve stability metrics at the cost of stricter filtering; rotation-on (`r1`) generally helps vs `r0` in this table.

### 3.4 Naive vs temporal — agreement on RGB/flash tests

Source: [`data/naive_vs_temporal_analysis_summary.md`](../naive_vs_temporal_analysis_summary.md)

- Mean absolute depth difference between naive and temporal stays near **~8.4–8.5%** across several **synthetic flash** regimes on a 216-frame test — interpreted as **small average drift** while temporal smoothing reduces **high-frequency** instability.
- Supporting detail: [`data/WHY_8.5_PERCENT_IS_GOOD.md`](../WHY_8.5_PERCENT_IS_GOOD.md), [`data/Y_AXIS_EXPLANATION.md`](../Y_AXIS_EXPLANATION.md).

---

## 4. Conclusions (report-ready)

1. **Temporal pipelines measurably reduce instability and flickering** on real clips, typically **~2–4%** depending on video and metric variant — consistent with expectations for a **strong monocular base model** (Depth Anything V2) where large relative gains usually require heavier methods (multi-frame networks, test-time optimization).

2. **Gains vary by content:** `orient_day` shows a larger **flickering** win; `100_years_technion` shows a larger **flow-compensated instability** win — see cross-video table in [`docs/25_12_02_100_years_technion_results.md`](../../docs/25_12_02_100_years_technion_results.md).

3. **The `depth_analysis/` PNGs** support these conclusions **qualitatively**: curves for naive vs V5/V6/V7/V8/V9 at fixed pixels and multi-model overlays show **reduced high-frequency variation** when Kabsch alignment, semantic masking, and/or RANSAC filtering are enabled — at the cost of more hyperparameters and runtime.

4. **V12 ablations** (JSON above) show that **photometric + forward–backward flow** thresholds and **anchor percentage** materially move stability metrics; **rotation on** is preferred in the reported sweep.

5. **Practical reporting caveat:** Instability scores are **not always comparable across different comparison scripts or versions** (different normalization / chunking). Always pair numbers with the **exact `metrics.json` path** and **video name**.

---

## 5. Related outputs elsewhere in Stage 3

| Location | Contents |
|----------|-----------|
| [`data/output_videos/`](../output_videos/) | Rendered naive/temporal/SBS comparison videos |
| [`results/comparison/`](../../results/comparison/) | Additional cached metrics (e.g. Technion) |
| [`data/video_depth_comparison/`](../video_depth_comparison/) | Per-frame statistics (`comparison_statistics.json`) |
| [`docs/25_11_30_FINAL_SUMMARY.md`](../../docs/25_11_30_FINAL_SUMMARY.md) | Pipeline deliverables and limitations |
| [`docs/25_11_30_comparison_script_documentation.md`](../../docs/25_11_30_comparison_script_documentation.md) | Metric definitions (flow-compensated instability, flickering) |

---

## 6. Full file list (`data/depth_analysis/`)

The following 45 files are the current plot inventory:

- `100_years_technion_a_analysis.png`
- `naive_vs_temporal_comparison.png`
- `5_models_comparison.png`
- `2models_100_years_technion_a_naive_vs_100_years_technion_a_f20_occ0._x1000_y100.png`
- `3models_100_years_technion_a_naive_vs_100_years_technion_a_f20_occ0._vs_100_years_technion_a_f20_occ0._x1000_y100.png`
- `4models_100_years_technion_a_naive_vs_100_years_technion_a_f20_occ0._vs_100_years_technion_a_f20_occ0._vs_100_years_technion_a_occ0.03_s_x1000_y100.png`
- `3models_100_years_technion_a_naive_vs_100_years_technion_a_f20_occ0._vs_100_years_technion_a_v2_robust_x1000_y100.png`
- `2models_100_years_technion_b_v2_robust_vs_100_years_technion_b_v4_anchor_x960_y540.png`
- `3models_100_years_technion_b_v2_robust_vs_100_years_technion_b_v4_anchor_vs_100_years_technion_b_v5_kabsch_x960_y540.png`
- `2models_100_years_technion_a_naive_vs_100_years_technion_a_v5_kabsch_x960_y540.png`
- `5models_100_years_technion_a_naive_vs_100_years_technion_a_v5_rot0_a_vs_100_years_technion_a_v5_rot0_a_vs_100_years_technion_a_v5_rot0_a_vs_100_years_technion_a_v5_rot0_a_x960_y540.png`
- `5models_100_years_technion_a_naive_vs_100_years_technion_a_v5_rot1_a_vs_100_years_technion_a_v5_rot1_a_vs_100_years_technion_a_v5_rot1_a_vs_100_years_technion_a_v5_rot1_a_x960_y540.png`
- `3models_100_years_technion_a_naive_vs_100_years_technion_a_v5_rot0_a_vs_100_years_technion_a_v5_rot1_a_x960_y540.png`
- `2models_100_years_technion_a_naive_vs_100_years_technion_a_v5_rot1_a_x1000_y100.png`
- `2models_100_years_technion_a_v5_rot1_a_vs_100_years_technion_a_v5_nofuse_x1000_y100.png`
- `5models_100_years_technion_a_v5_r1_a2__vs_100_years_technion_a_v5_r1_a2__vs_100_years_technion_a_v5_r1_a2__vs_100_years_technion_a_v5_r1_a4__vs_100_years_technion_a_v5_r1_a4__x1000_y100.png`
- `4models_100_years_technion_a_v5_r1_a4__vs_100_years_technion_a_v5_r1_a6__vs_100_years_technion_a_v5_r1_a6__vs_100_years_technion_a_v5_r1_a6__x1000_y100.png`
- `3models_35266_small_v5_r1_a6_s10_vs_35266_small_v5_r1_a4_s10_vs_35266_small_v5_r1_a4_s20_x500_y300.png`
- `V6_r1_a60_s10_vs_V5_r1_a6_s10_vs_Naive_x1000_y100.png`
- `v7_point_counts_technion_d.png`
- `3models_100_years_technion_a_v6_r1_a60_vs_100_years_technion_a_v5_r1_a6__vs_100_years_technion_a_naive_x1000_y100.png`
- `V6_r0_a20_s10_vs_V6_r0_a20_s20_vs_V6_r0_a40_s10_vs_V6_r0_a40_s20_vs_V6_r1_a20_s10_vs_V6_r1_a20_s20_vs_V6_r1_a40_s10_vs_V6_r1_a40_s20_vs_Naive_x1000_y100.png`
- `V6_r0_a20_s10_vs_V6_r0_a20_s20_vs_V6_r1_a20_s10_vs_V6_r1_a20_s20_vs_Naive_x1000_y100.png`
- `V6_r0_a20_s10_vs_V6_r0_a20_s20_vs_V6_r1_a20_s10_vs_V6_r1_a20_s20_vs_V5_r1_a40_s10_vs_V5_r1_a40_s15_vs_V5_r1_a40_s20_vs_V5_r1_a60_s10_vs_V5_r1_a60_s15_vs_V5_r1_a60_s20_x1000_y100.png`
- `V6_r0_a60_s10_vs_V6_r0_a60_s20_vs_V6_r1_a60_s10_vs_V6_r1_a60_s20_x1000_y100.png`
- `V6_r0_a20_s10_vs_V6_r1_a20_s10_vs_V6_r1_a20_s20_vs_V6_r0_a60_s10_vs_V6_r1_a60_s10_vs_V6_r1_a60_s20_vs_Naive_x1000_y100.png`
- `V6_r0_a20_s10_vs_V6_r1_a20_s10_vs_V6_r1_a20_s20_vs_V6_r0_a60_s10_vs_V6_r1_a60_s10_vs_V6_r1_a60_s20_vs_Naive_x1160_y300.png`
- `V6_r0_a20_s10_vs_V6_r1_a20_s10_vs_Naive_x960_y540.png`
- `V7_a20_s10_vs_V6_r1_a20_s10_vs_Naive_x1000_y100.png`
- `V7_a20_s10_vs_V7_a20_s10_vs_V7_a20_s10_vs_Naive_x1000_y100.png`
- `V7_a20_s10_vs_V7_a20_s10_vs_V7_a20_s10_vs_V7_a20_s10_vs_V7_a20_s10_vs_V7_a20_s10_vs_Naive_x1000_y100.png`
- `V7_m500_d20_a20_s10_vs_V7_m200_d10_a20_s10_vs_V7_m200_d20_a20_s10_vs_V7_m500_d20_so_a20_s10_vs_V7_m200_d10_so_a20_s10_vs_V7_m200_d20_so_a20_s10_vs_Naive_x1000_y100.png`
- `V7_m200_d20_a20_s10_vs_V7_m200_d20_so_a20_s10_vs_Naive_x1000_y100.png`
- `V5_r0_a20_s10_vs_V5_r0_a20_s20_vs_V5_r0_a40_s10_vs_V5_r0_a40_s20_vs_V5_r0_a60_s10_vs_V5_r0_a60_s20_vs_Naive_x1000_y100.png`
- `V5_r1_a20_s10_vs_V5_r1_a20_s20_vs_V5_r1_a40_s10_vs_V5_r1_a40_s20_vs_V5_r1_a60_s10_vs_V5_r1_a60_s20_vs_Naive_x1000_y100.png`
- `V5_r1_a20_s10_vs_V6_r1_a20_s10_vs_V7_m200_d10_a20_s10_vs_Naive_x1000_y100.png`
- `100_years_technion_d_v5_anchor_comparison_pixel_1000_100.png`
- `100_years_technion_d_v8_vs_v5_vs_naive_pixel_1000_100.png`
- `100_years_technion_d_v9_v5_v8_naive_pixel_1000_100.png`
- `100_years_technion_d_v9_full_comparison.png`
- `100_years_technion_d_v9_comparison_all.png`
- `100_years_technion_d_v9_all_comparison.png`
- `100_years_technion_d_v9_yoloM_vs_previous.png`
- `100_years_technion_d_v9_anchor_sweep_vs_naive.png`
- `100_years_technion_d_v9_v5_v8_naive_analysis.png`

*(If your checkout adds or removes PNGs, refresh this list with `find data/depth_analysis -type f | sort`.)*

---

**End of document.**
