# Scripts Directory

**v2_final_modified_v1, v5, v8, v9, depth_visualize — environment, paths, CLI:** [`../../docs/README_depth_extract_v2_v5_v8_v9_visualize.md`](../../docs/README_depth_extract_v2_v5_v8_v9_visualize.md)

**Overview of the video → depth → visualization pipeline:** [`SUMMARY_depth_generalization_process.md`](SUMMARY_depth_generalization_process.md)

Scripts are organized into subdirectories:

| Directory | Contents |
|-----------|----------|
| **depth/** | Depth extraction: `depth_extract*.py`, `temporal_video_depth.py`, `naive_video_depth.py`, etc. |
| **analysis/** | Analysis & comparison: `depth_analyze.py`, `analyze_depth_evolution.py`, `compare_depth_methods*.py`, etc. |
| **visualization/** | Visualization: `depth_visualize.py`, `visualize_depth_grid.py`, `visualize_anchors*.py`, etc. |
| **other/** | Utilities, shell scripts, debug tools: `generate_depth_video.sh`, `depth_debug.py`, etc. |

Run from project root, e.g.:
```bash
python code/scripts/depth/depth_extract_v12.py --video-path ... --output ...
python code/scripts/analysis/depth_analyze.py --compare-models ...
python code/scripts/visualization/visualize_depth_grid.py --video ... --depth-npz ...
```
