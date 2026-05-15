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

## 🛠️ Methodology

### 1. Geometric Alignment
Monocular depth predictors estimate relative depth, which results in scale changes between frames[cite: 5]. To correct this, we:
* Identify low-motion background anchors using RAFT, YOLO, and RANSAC.
* Compute the optimal scale, rotation, and translation ($S, R, T$) using the **Kabsch–Umeyama algorithm**.
* Apply the resulting transformation to the full 3D point cloud to minimize Mean Squared Error (MSE) between corresponding points.

### 2. Temporal Fusion
To reduce high-frequency noise and local flicker, the pipeline:
* Warps the previous depth map into the current frame.
* Constructs a **Trust Map** to guard against warp invalidity and photometric inconsistency.
* Performs morphological hole filling followed by FG/BG aware blending.

---

## 📊 Results
Our stabilization pipelines significantly reduce temporal flicker and scale drift compared to naive predictions.
* **Stability:** Fixed-pixel depth trajectories show smoother curves with reduced high-frequency noise.
* **Consistency:** The methods enable more reliable interpretation of depth over time
---

## 💻 Tech Stack
* **Language:** Python 
* **Optical Flow:** RAFT (Winner of Best Paper at ECCV 2020)
* **Geometric Alignment:** Kabsch–Umeyama Algorithm 
* **Depth Backbone:** Depth Anything V2 

---

## 👥 Credits
**Authors:** Noam Murciano and Yuri Minin  
**Supervisor:** Dr. Meir Barzohar 
**Institution:** Signal and Image Processing Lab (SIPL), Technion - Israel Institute of Technology 
