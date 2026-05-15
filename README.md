# Temporally Consistent Monocular Depth Estimation for Video

[cite_start]This repository implements a pipeline for generating temporally stable depth maps from monocular video[cite: 3]. [cite_start]While single-frame predictors like **Depth Anything V2** offer high-quality results, they often suffer from temporal flickering and inconsistent depth when applied to video[cite: 2, 8]. [cite_start]Our solution stabilizes both global scale and local temporal consistency to enable reliable downstream visualization and analysis[cite: 4, 10].

---

## 🚀 Key Features

* **State-of-the-art Backbone:** Leverages Depth Anything V2 (ViT encoder + DPT decoder) for high-quality single-image depth maps.
* **Geometric Alignment:** Stabilizes global scale by converting depth maps into 3D point clouds and aligning them across frames.
* **Temporal Fusion:** Ensures local consistency by warping previous depth maps using **RAFT optical flow**.
* **Adaptive Smoothing:** Applies stronger smoothing to background regions and weaker smoothing to dynamic foregrounds to preserve detail.
* **Noise Robustness:** Utilizes a multi-signal reliability mask based on photometric consistency and forward–backward flow agreement.

---
