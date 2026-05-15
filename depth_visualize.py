"""
Depth Extraction Script V9 - Geometric RANSAC + Semantic YOLO Alignment

Key Innovation:
Combines RANSAC geometric filtering (Homography inliers) with V8-style semantic
filtering (YOLO background). Anchors must satisfy ALL three conditions:

1. RANSAC Inlier: Motion fits the dominant planar homography
2. YOLO Background: Not person/car/etc. (excluded by semantic mask)
3. Lowest Flow: Among the intersection, keep anchor_percent with lowest flow magnitude

================================================================================
THE LOGIC
================================================================================

1. RANSAC: Find Homography that explains dominant motion. Inliers = geometric background.
2. YOLO: Exclude dynamic object pixels (person, car, etc.) from anchor candidates.
3. Intersection: Valid = (RANSAC Inlier) AND (YOLO Background) AND (Valid Warp/Depth)
4. Anchor Selection: Take anchor_percent of that intersection with LOWEST flow magnitude.

================================================================================
ADVANTAGES
================================================================================

- Triple filtering: Geometric (RANSAC) + Semantic (YOLO) + Motion (low flow)
- Rejects standing-still people (YOLO) and moving people (RANSAC outlier)
- Final anchor set is robust: background-only, geometrically consistent, low motion

================================================================================
LIMITATIONS
================================================================================

- Planar Assumption: Homography models a single plane. Scenes with strong depth
  variation may have fewer inliers. The ransac_thresh helps by allowing some
  geometric tolerance.

- Static Scene: When the camera is completely still, flow is near-zero. We add
  a static_scene check (like V5/V8) to skip alignment in that case.

================================================================================
USAGE
================================================================================

# Basic V9 extraction
python depth_extract_v9.py --video-path input.mp4 --output output.npz

# With rotation recovery (recommended for moving cameras)
python depth_extract_v9.py --video-path input.mp4 --output output.npz --use-rotation

# Stricter RANSAC (fewer, higher-quality anchors)
python depth_extract_v9.py --video-path input.mp4 --output output.npz --ransac-thresh 2.0

# Looser RANSAC (more anchors, may include some outliers)
python depth_extract_v9.py --video-path input.mp4 --output output.npz --ransac-thresh 5.0

================================================================================
CLI ARGUMENTS
================================================================================

Required:
  --video-path PATH     Input video file (e.g. .mp4, .avi).
  --output PATH         Output .npz file. A mask file {stem}_mask.npz is also saved.

RANSAC (geometric inlier detection):
  --ransac-thresh FLOAT   Reprojection error threshold in pixels. Pixels with error
                          below this are RANSAC inliers. Lower = stricter, fewer
                          anchors. Higher = looser, more anchors. Default: 3.0
  --ransac-stride INT     Grid stride for RANSAC point sampling. Higher = faster
                          but less accurate. Default: 8

YOLO (semantic background filter):
  --yolo-model NAME       YOLOv8 segmentation model. Options: yolov8n-seg.pt,
                          yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt.
                          Default: yolov8n-seg.pt (fastest)
  --blacklist ID [ID...]  COCO class IDs to exclude from anchors. Common: 0=Person,
                          1=Bicycle, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck.
                          Default: 0 (Person only)
  --yolo-confidence FLOAT Minimum detection confidence to exclude. Default: 0.5

Anchor selection:
  --anchor-percent FLOAT  Among pixels that pass (RANSAC inlier AND YOLO background),
                          keep this fraction with LOWEST flow magnitude. E.g. 0.15
                          = keep 15% of candidates. Default: 0.15

Depth model:
  --encoder {vits,vitb,vitl}  Depth model size. vitl = best quality, vits = fastest.
                              Default: vitl
  --input-size INT            Input resolution for depth model. Default: 518

Alignment:
  --max-scale-diff FLOAT   Maximum allowed scale change per frame (clamp). Default: 0.1
  --use-rotation           Enable 3D rotation recovery. Use for moving/panning cameras.

Camera and scene:
  --fov FLOAT             Estimated camera horizontal FOV in degrees. Default: 70.0
  --scene-threshold FLOAT  Histogram distance threshold for scene cut detection.
                           Higher = fewer cuts. Default: 30.0

================================================================================
"""

import argparse
import cv2
import json
import numpy as np
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Import Ultralytics for YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Error: 'ultralytics' library not found.")
    print("   Please install it using: pip install ultralytics")
    sys.exit(1)

# =============================================================================
# SETUP PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Depth Anything V2
from depth_anything_v2.dpt import DepthAnythingV2

# Import RAFT
RAFT_PATH = PROJECT_ROOT / "raft_repo"
sys.path.insert(0, str(RAFT_PATH / "core"))
from raft import RAFT
from utils.utils import InputPadder


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_depth_model(encoder='vitl', device='cuda'):
    """
    Loads the Depth Anything V2 model architecture and weights.
    
    Args:
        encoder: Model size ('vits', 'vitb', 'vitl')
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Loaded and initialized depth model
    """
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    checkpoint_paths = {
        'vits': PROJECT_ROOT / 'checkpoints' / 'depth_anything_v2_vits.pth',
        'vitb': PROJECT_ROOT / 'checkpoints' / 'depth_anything_v2_vitb.pth',
        'vitl': PROJECT_ROOT / 'checkpoints' / 'depth_anything_v2_vitl.pth',
    }
    
    print(f"Loading Depth Anything V2 ({encoder.upper()})...")
    model = DepthAnythingV2(**model_configs[encoder])
    
    if not checkpoint_paths[encoder].exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_paths[encoder]}")
        
    model.load_state_dict(torch.load(checkpoint_paths[encoder], map_location='cpu'))
    model = model.to(device).eval()
    print("✅ Depth model loaded")
    return model


def load_raft_model(raft_checkpoint=None, device='cuda'):
    """
    Loads the RAFT optical flow model for motion estimation.
    
    Args:
        raft_checkpoint: Path to RAFT weights
        device: Device to load model on
        
    Returns:
        Loaded RAFT model
    """
    if raft_checkpoint is None:
        raft_checkpoint = PROJECT_ROOT / 'raft_models' / 'raft-things.pth'
    
    print("Loading RAFT optical flow model...")
    args = argparse.Namespace(
        model=str(raft_checkpoint),
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )
    model = RAFT(args)
    
    state_dict = torch.load(raft_checkpoint, map_location='cpu')
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device).eval()
    print("✅ RAFT model loaded")
    return model


def load_yolo_model(model_name='yolov8n-seg.pt', device='cuda'):
    """Loads YOLOv8 Segmentation model for semantic background filtering."""
    print(f"Loading YOLOv8 Segmentation model ({model_name})...")
    model = YOLO(model_name)
    model.to(device)
    print("✅ YOLO model loaded")
    return model


def get_semantic_exclude_mask(frame_rgb, yolo_model, blacklist_classes=[0],
                               confidence_threshold=0.5, device='cuda'):
    """
    Creates boolean mask where True = background (safe for anchors).
    Excludes blacklisted object classes (e.g. Person) detected by YOLO.
    Returns: safe_mask [1, 1, H, W] (True = keep)
    """
    H, W = frame_rgb.shape[:2]
    results = yolo_model(frame_rgb, verbose=False, retina_masks=True)
    safe_mask = torch.ones((H, W), dtype=torch.bool, device=device)
    result = results[0]
    if result.masks is None:
        return safe_mask.unsqueeze(0).unsqueeze(0)
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    masks_data = result.masks.data
    if masks_data.shape[1:] != (H, W):
        masks_data = F.interpolate(
            masks_data.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(1)
        masks_data = masks_data > 0.5
    for i, (cls_id, conf) in enumerate(zip(classes, confidences)):
        if int(cls_id) in blacklist_classes and conf >= confidence_threshold:
            safe_mask = safe_mask & (~masks_data[i])
    return safe_mask.unsqueeze(0).unsqueeze(0)


# =============================================================================
# GEOMETRIC UTILITIES: RANSAC HOMOGRAPHY
# =============================================================================

def get_ransac_inlier_mask(flow, stride=8, ransac_thresh=3.0, device='cuda'):
    """
    Computes a binary mask of background pixels using RANSAC and Homography.
    
    Algorithm:
    1. Sampling: Select points on a grid (stride) to reduce computation for RANSAC.
    2. Correspondences: Map (x,y) -> (x+u, y+v) using optical flow.
    3. RANSAC: Use cv2.findHomography with RANSAC to find the dominant geometric
       transform (Homography) that explains the majority of point correspondences.
    4. Dense Verification: Apply the found Homography to ALL pixels (not just samples).
       Pixels whose actual flow destination matches the Homography-predicted destination
       (within ransac_thresh pixels) are classified as Inliers (Background).
       
    Args:
        flow: Optical flow tensor [1, 2, H, W] (u, v per pixel)
        stride: Subsampling step for RANSAC (higher = faster, less accurate)
        ransac_thresh: Maximum allowed reprojection error in pixels.
                       Stricter = fewer but cleaner anchors.
                       Looser = more anchors, may include outliers.
        device: Torch device
        
    Returns:
        inlier_mask: Boolean tensor [1, 1, H, W] (True = Background/Inlier)
        
    Note:
        When M is None (RANSAC failed to find a dominant plane), we return
        zeros to avoid using potentially corrupted anchors.
    """
    B, C, H, W = flow.shape
    
    # 1. Create coordinate grids
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    # Source coordinates [H, W, 2]
    coords0 = torch.stack([x_grid, y_grid], dim=-1).float()
    
    # Apply flow to get destination coordinates (actual flow destination)
    flow_permuted = flow.squeeze(0).permute(1, 2, 0)  # [H, W, 2]
    coords1 = coords0 + flow_permuted
    
    # 2. Extract sparse points for RANSAC (CPU-side via OpenCV)
    pts0_flat = coords0[::stride, ::stride, :].reshape(-1, 2).cpu().numpy()
    pts1_flat = coords1[::stride, ::stride, :].reshape(-1, 2).cpu().numpy()
    
    # Minimum 4 point correspondences required for Homography (we use 10 for stability)
    if pts0_flat.shape[0] < 10:
        return torch.ones((1, 1, H, W), dtype=torch.bool, device=device)
    
    # 3. Find Homography with RANSAC
    # M: 3x3 transformation matrix
    # status: inlier mask for the sparse sample points (not used for dense output)
    M, status = cv2.findHomography(pts0_flat, pts1_flat, cv2.RANSAC, ransac_thresh)
    
    if M is None:
        # RANSAC failed (no dominant plane found) -> treat all as outliers
        return torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)
    
    # 4. Dense verification: apply Homography to ALL pixels
    M_tensor = torch.from_numpy(M).float().to(device)
    
    # Homogeneous coordinates [H, W, 3] -> (x, y, 1)
    ones = torch.ones((H, W, 1), device=device)
    coords0_homo = torch.cat([coords0, ones], dim=-1)
    coords0_flat_homo = coords0_homo.reshape(-1, 3).permute(1, 0)  # [3, N]
    
    # Project all points: [3, N]
    coords1_pred_homo = torch.mm(M_tensor, coords0_flat_homo)
    
    # Convert from homogeneous (divide by z)
    # Use clone() to avoid in-place modification of original tensor
    z_vec = coords1_pred_homo[2, :].clone()
    z_vec[torch.abs(z_vec) < 1e-6] = 1.0  # Safety: avoid division by zero
    
    x_pred = coords1_pred_homo[0, :] / z_vec
    y_pred = coords1_pred_homo[1, :] / z_vec
    
    coords1_pred = torch.stack([x_pred, y_pred], dim=-1).reshape(H, W, 2)
    
    # 5. Compute reprojection error:
    # Error = || Actual_Flow_Dest - Homography_Predicted_Dest ||
    diff = coords1 - coords1_pred
    dist_sq = (diff ** 2).sum(dim=-1)
    error_map = torch.sqrt(dist_sq + 1e-8)  # [H, W]
    
    # 6. Threshold: inliers are pixels with error below threshold
    inlier_mask = error_map < ransac_thresh
    
    return inlier_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


# =============================================================================
# CORE ALGORITHMS: FLOW & WARPING
# =============================================================================

def compute_optical_flow(raft_model, frame1, frame2, device='cuda', iters=20):
    """
    Computes optical flow from frame2 to frame1 using RAFT.
    
    Args:
        raft_model: Loaded RAFT model
        frame1: Current frame RGB [H, W, 3]
        frame2: Previous frame RGB [H, W, 3]
        device: Torch device
        iters: RAFT refinement iterations
        
    Returns:
        flow: Optical flow tensor [1, 2, H, W]
    """
    with torch.no_grad():
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        
        _, flow = raft_model(img1, img2, iters=iters, test_mode=True)
        return padder.unpad(flow)


def warp_depth_gpu(depth, flow):
    """
    Warps previous depth map to current frame using optical flow.
    
    Args:
        depth: Depth map [1, 1, H, W]
        flow: Optical flow [1, 2, H, W]
        
    Returns:
        warped_depth: Warped depth map [1, 1, H, W]
        valid_mask: Mask of valid warped pixels [1, 1, H, W]
    """
    B, C, H, W = depth.shape
    
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing='ij'
    )
    coords = torch.stack([x, y], dim=0).float().unsqueeze(0)
    sample_coords = coords + flow
    
    # Normalize to [-1, 1] for grid_sample
    sample_coords[:, 0, :, :] = 2.0 * sample_coords[:, 0, :, :] / (W - 1) - 1.0
    sample_coords[:, 1, :, :] = 2.0 * sample_coords[:, 1, :, :] / (H - 1) - 1.0
    sample_coords = sample_coords.permute(0, 2, 3, 1)
    
    warped_depth = F.grid_sample(
        depth, sample_coords,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    valid_mask = torch.ones_like(depth)
    valid_mask = F.grid_sample(
        valid_mask, sample_coords,
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
    
    return warped_depth, valid_mask


def estimate_camera_intrinsics(width, height, fov_degrees=70.0):
    """
    Estimates camera intrinsic matrix from field of view.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_degrees: Horizontal field of view in degrees
        
    Returns:
        K: 3x3 camera intrinsic matrix
    """
    fov_rad = np.radians(fov_degrees)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx
    cx = width / 2
    cy = height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def depth_to_point_cloud_torch(depth, K, device='cuda'):
    """
    Converts depth map to 3D point cloud using camera intrinsics.
    
    Args:
        depth: Depth map [H, W] or [1, 1, H, W]
        K: Camera intrinsic matrix [3, 3]
        device: Torch device
        
    Returns:
        points: Point cloud [H, W, 3] (X, Y, Z)
    """
    if depth.dim() == 4:
        depth = depth.squeeze(0).squeeze(0)
    elif depth.dim() == 3:
        depth = depth.squeeze(0)
    
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    v, u = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return torch.stack([X, Y, Z], dim=-1)


def solve_kabsch_umeyama_torch(P, Q, compute_rotation=True):
    """
    Solves for optimal similarity transform (scale + rotation + translation)
    that aligns point set P to point set Q.
    
    Args:
        P: Source points [N, 3]
        Q: Target points [N, 3]
        compute_rotation: If False, only compute scale (faster)
        
    Returns:
        s: Scale factor (float)
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
    """
    device = P.device
    
    if P.shape[0] < 3:
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    var_P = (P_centered ** 2).sum()
    
    if not compute_rotation:
        var_Q = (Q_centered ** 2).sum()
        s = torch.sqrt(var_Q / (var_P + 1e-8))
        return s.item(), torch.eye(3, device=device), centroid_Q - s * centroid_P
    
    H = torch.mm(P_centered.T, Q_centered)
    U, S, Vt = torch.linalg.svd(H)
    R = torch.mm(Vt.T, U.T)
    
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = torch.mm(Vt.T, U.T)
    
    s = torch.trace(torch.mm(R, H)) / (var_P + 1e-8)
    t = centroid_Q - s * torch.mv(R, centroid_P)
    
    return s.item(), R, t


def detect_scene_cut(frame1, frame2, threshold=30.0):
    """
    Detects scene cuts using color histogram comparison.
    
    Args:
        frame1: First frame RGB
        frame2: Second frame RGB
        threshold: Chi-square threshold for cut detection
        
    Returns:
        is_cut: True if scene cut detected
    """
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return distance > threshold


# =============================================================================
# V9: GEOMETRIC RANSAC ALIGNMENT
# =============================================================================

def align_depth_kabsch_ransac(curr_depth, prev_depth_warped, flow, valid_mask, K,
                              semantic_mask=None,
                              ransac_thresh=3.0, stride=8, max_scale_diff=0.1,
                              min_valid_pixels=100, anchor_percent=0.15, use_rotation=False,
                              device='cuda'):
    """
    V9 Geometric + Semantic Alignment: Anchors must satisfy ALL three:
    1. RANSAC Inlier (geometric consistency with dominant homography)
    2. YOLO Background (not person/car/etc.)
    3. Lowest flow magnitude among anchor_percent of the intersection
    
    Valid Anchor = (RANSAC Inlier) AND (YOLO Background) AND (Valid Warp) AND (Valid Depth)
    Then: take anchor_percent of those candidates with LOWEST flow magnitude.
    
    Args:
        curr_depth: Current frame depth [1, 1, H, W]
        prev_depth_warped: Previous depth warped to current frame [1, 1, H, W]
        flow: Optical flow [1, 2, H, W]
        valid_mask: Warp validity mask [1, 1, H, W]
        K: Camera intrinsics [3, 3]
        semantic_mask: YOLO background mask [1, 1, H, W] (True=background). If None, skip.
        ransac_thresh: RANSAC reprojection error threshold (pixels)
        stride: Grid stride for RANSAC sampling (higher = faster)
        max_scale_diff: Maximum allowed scale change (prevents outliers)
        min_valid_pixels: Minimum anchors required for alignment
        anchor_percent: Fraction of candidates (inlier & background) to keep, by lowest flow
        use_rotation: Enable rotation recovery
        device: Torch device
        
    Returns:
        aligned_depth: Aligned depth map [1, 1, H, W]
        info: Dictionary with alignment statistics and final_mask
    """
    
    # 1. Static scene check (like V5/V8)
    flow_mag = torch.norm(flow, dim=1, keepdim=True)
    H_img, W_img = curr_depth.shape[2], curr_depth.shape[3]
    empty_mask = torch.zeros((1, 1, H_img, W_img), dtype=torch.bool, device=device)
    
    if flow_mag.max() < 0.5:
        return curr_depth, {
            'scale': 1.0,
            'status': 'static_scene',
            'anchor_count': 0,
            'final_mask': empty_mask
        }
    
    # 2. Generate RANSAC Inlier Mask (Geometric Background Detection)
    inlier_mask = get_ransac_inlier_mask(
        flow, stride=stride, ransac_thresh=ransac_thresh, device=device
    )
    
    # 3. Combine masks: Inlier AND Semantic (YOLO background) AND Geometric Validity
    base_valid = (
        inlier_mask &
        (valid_mask > 0.5) &
        (prev_depth_warped > 0.001) &
        (curr_depth > 0.001)
    )
    if semantic_mask is not None:
        base_valid = base_valid & (semantic_mask > 0.5)
    
    final_valid_mask = base_valid
    
    # 4. Select anchor_percent of candidates with LOWEST flow magnitude
    flat_flow = flow_mag.view(-1)
    flat_mask = final_valid_mask.view(-1)
    flat_indices = torch.nonzero(flat_mask).squeeze(-1)
    num_candidates = flat_indices.numel()
    
    if num_candidates > 0:
        k_elements = max(min_valid_pixels, int(num_candidates * anchor_percent))
        if num_candidates > k_elements:
            candidate_flows = flat_flow[flat_indices]
            _, sorted_idx = torch.topk(candidate_flows, k=min(k_elements, num_candidates), largest=False)
            selected_indices = flat_indices[sorted_idx]
            final_valid_mask = torch.zeros_like(flat_mask, dtype=torch.bool)
            final_valid_mask[selected_indices] = True
            final_valid_mask = final_valid_mask.view_as(inlier_mask)
    
    anchor_count = final_valid_mask.sum().item()
    
    # 5. Fallback: insufficient inliers
    if anchor_count < min_valid_pixels:
        return curr_depth, {
            'status': 'insufficient_ransac_inliers',
            'anchor_count': anchor_count,
            'final_mask': empty_mask
        }
    
    # 5. Extract 3D point clouds for Kabsch
    curr_points_3d = depth_to_point_cloud_torch(curr_depth, K, device)
    prev_points_3d = depth_to_point_cloud_torch(prev_depth_warped, K, device)
    
    final_valid_2d = final_valid_mask.squeeze()
    P_anchors = curr_points_3d[final_valid_2d]
    Q_anchors = prev_points_3d[final_valid_2d]
    
    # 6. Solve Kabsch-Umeyama
    s_raw, R, t = solve_kabsch_umeyama_torch(
        P_anchors, Q_anchors,
        compute_rotation=use_rotation
    )
    s = np.clip(s_raw, 1.0 - max_scale_diff, 1.0 + max_scale_diff)
    
    # 7. Apply transform
    H_img, W_img = curr_depth.shape[2], curr_depth.shape[3]
    
    if use_rotation:
        curr_flat = curr_points_3d.view(-1, 3)
        curr_rotated = torch.mm(curr_flat, R.T)
        curr_transformed = s * curr_rotated + t.unsqueeze(0)
        aligned_depth = curr_transformed[:, 2].view(1, 1, H_img, W_img)
        
        R_np = R.cpu().numpy()
        rot_y = np.arcsin(-R_np[2, 0]) * 180 / np.pi
        rot_x = np.arctan2(R_np[2, 1], R_np[2, 2]) * 180 / np.pi
        rot_z = np.arctan2(R_np[1, 0], R_np[0, 0]) * 180 / np.pi
        rot_info = f"({rot_x:.1f}°, {rot_y:.1f}°, {rot_z:.1f}°)"
    else:
        aligned_depth = s * curr_depth + t[2].item()
        rot_info = "disabled"
    
    return aligned_depth, {
        'scale': s,
        'rotation': rot_info,
        'translation': t.cpu().numpy().tolist(),
        'anchor_count': anchor_count,
        'status': 'aligned',
        'final_mask': final_valid_mask
    }


# =============================================================================
# MAIN EXTRACTION LOOP
# =============================================================================

def extract_depth_v9(args):
    """
    Main V9 depth extraction with RANSAC geometric filtering.
    
    Process Flow:
    1. Load models (Depth, RAFT)
    2. For each frame:
       a. Infer raw depth
       b. Detect scene cuts
       c. Compute optical flow
       d. Get RANSAC inlier mask
       e. Align using Kabsch on inliers
       f. Save aligned depth
    3. Save results to .npz
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    depth_model = load_depth_model(encoder=args.encoder, device=device)
    raft_model = load_raft_model(device=device)
    yolo_model = load_yolo_model(model_name=args.yolo_model, device=device)
    
    # Open video
    video_path = Path(args.video_path)
    video_name = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    K = estimate_camera_intrinsics(width, height, fov_degrees=args.fov)
    
    # Print configuration
    print(f"\n🎥 Video: {video_name}")
    print(f"📊 Video: {width}x{height} @ {fps}fps, {frame_count} frames")
    print(f"📐 Camera intrinsics (estimated, FOV={args.fov}°):")
    print(f"   fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"🎯 V9 Geometric + Semantic Alignment (RANSAC + YOLO + Anchor%):")
    print(f"   RANSAC Threshold: {args.ransac_thresh} px")
    print(f"   RANSAC Stride: {args.ransac_stride}")
    print(f"   YOLO Blacklist: {args.blacklist}")
    print(f"   Anchor Percent: {args.anchor_percent}")
    print(f"   Max Scale Diff: {args.max_scale_diff}")
    print(f"   Rotation Recovery: {'ENABLED' if args.use_rotation else 'DISABLED'}")
    
    # Storage
    all_depths = []
    all_masks = []  # Per-frame anchor masks for export
    prev_frame_rgb = None
    prev_depth_stable = None
    
    # Statistics
    status_counts = {
        'aligned': 0,
        'static_scene': 0,
        'insufficient_ransac_inliers': 0,
        'first_frame': 0,
        'scene_cut': 0
    }
    anchor_history = []
    scale_history = []
    
    # Main loop
    pbar = tqdm(total=frame_count, desc="Extracting Depth (V9 Geometric)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Scene cut detection
        is_scene_cut = False
        if prev_frame_rgb is not None and args.scene_threshold > 0:
            is_scene_cut = detect_scene_cut(prev_frame_rgb, frame_rgb, args.scene_threshold)
            if is_scene_cut:
                status_counts['scene_cut'] += 1
        
        # 2. Raw depth inference
        with torch.no_grad():
            depth_raw = depth_model.infer_image(frame_rgb, args.input_size)
            depth_raw = torch.from_numpy(depth_raw).float().to(device).unsqueeze(0).unsqueeze(0)
        
        # 3. Alignment
        if prev_frame_rgb is not None and prev_depth_stable is not None and not is_scene_cut:
            flow = compute_optical_flow(raft_model, frame_rgb, prev_frame_rgb, device)
            prev_warped, valid_mask = warp_depth_gpu(prev_depth_stable, flow)
            
            semantic_mask = get_semantic_exclude_mask(
                frame_rgb, yolo_model,
                blacklist_classes=args.blacklist,
                confidence_threshold=args.yolo_confidence,
                device=device
            )
            
            curr_aligned, info = align_depth_kabsch_ransac(
                depth_raw, prev_warped, flow, valid_mask, K,
                semantic_mask=semantic_mask,
                ransac_thresh=args.ransac_thresh,
                stride=args.ransac_stride,
                max_scale_diff=args.max_scale_diff,
                anchor_percent=args.anchor_percent,
                use_rotation=args.use_rotation,
                device=device
            )
            
            depth_stable = curr_aligned
            status_counts[info['status']] = status_counts.get(info['status'], 0) + 1
            
            if 'anchor_count' in info:
                anchor_history.append(info['anchor_count'])
            if 'scale' in info:
                scale_history.append(info['scale'])
            
            # Store mask for export
            frame_mask = info.get('final_mask', torch.zeros((1, 1, *depth_stable.shape[2:]), dtype=torch.bool, device=device))
            all_masks.append(frame_mask.squeeze().cpu().numpy())
        else:
            depth_stable = depth_raw
            if prev_frame_rgb is None:
                status_counts['first_frame'] += 1
            
            # No alignment: trivial mask (zeros)
            H, W = depth_stable.shape[2], depth_stable.shape[3]
            all_masks.append(np.zeros((H, W), dtype=bool))
        
        # 4. Store Result
        all_depths.append(depth_stable.squeeze().cpu().numpy().astype(np.float16))
        
        # 5. Update state
        prev_frame_rgb = frame_rgb.copy()
        prev_depth_stable = depth_stable.clone()
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Convert to numpy
    all_depths = np.array(all_depths)
    global_min = all_depths.min()
    global_max = all_depths.max()
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'depths': all_depths,
        'fps': fps,
        'width': width,
        'height': height,
        'global_min': global_min,
        'global_max': global_max,
        'scene_cuts': status_counts.get('scene_cut', 0),
        'source_video': str(video_path.name),
        'config_json': json.dumps(vars(args))
    }
    
    np.savez_compressed(str(output_path), **save_dict)
    
    # Save mask to data/anchors_mask
    mask_dir = PROJECT_ROOT / "data" / "anchors_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{output_path.stem}_mask.npz"
    np.savez_compressed(str(mask_path), masks=np.array(all_masks), fps=fps)
    print(f"   Mask saved to: {mask_path}")
    
    # Print summary
    print(f"\n💾 Saving to: {output_path}")
    print(f"✅ Saved! File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"   Frames: {len(all_depths)}")
    print(f"   Depth range: {global_min:.4f} - {global_max:.4f}")
    print(f"   Scene cuts: {status_counts.get('scene_cut', 0)}")
    
    print(f"\n📊 Alignment Status Breakdown:")
    for status, count in status_counts.items():
        if count > 0:
            print(f"   {status}: {count}")
    
    if anchor_history:
        print(f"\n📊 RANSAC Inlier Statistics:")
        print(f"   Mean: {int(np.mean(anchor_history))}")
        print(f"   Min: {int(np.min(anchor_history))}")
        print(f"   Max: {int(np.max(anchor_history))}")
    
    if scale_history:
        print(f"\n📏 Scale Statistics:")
        print(f"   Mean: {np.mean(scale_history):.4f}")
        print(f"   Std: {np.std(scale_history):.4f}")
        print(f"   Range: [{np.min(scale_history):.4f}, {np.max(scale_history):.4f}]")
    
    print(f"\n✅ Extraction complete!")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

_CLI_EPILOG = """
CLI Arguments Summary
────────────────────
Required: --video-path, --output

RANSAC:    --ransac-thresh (px), --ransac-stride (grid step)
YOLO:      --yolo-model, --blacklist (class IDs), --yolo-confidence
Anchors:   --anchor-percent (fraction of inlier+background to keep by lowest flow)
Depth:     --encoder (vits/vitb/vitl), --input-size
Alignment: --max-scale-diff, --use-rotation (for moving cameras)
Scene:     --fov, --scene-threshold
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V9 Depth Extraction: RANSAC + YOLO + Anchor% geometric alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EPILOG
    )
    
    # Required
    parser.add_argument('--video-path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output .npz file')
    
    # RANSAC settings
    parser.add_argument('--ransac-thresh', type=float, default=3.0,
                        help='RANSAC reprojection error threshold in pixels (default: 3.0)')
    parser.add_argument('--ransac-stride', type=int, default=8,
                        help='Grid stride for RANSAC sampling (default: 8)')
    
    # YOLO settings (semantic background filter)
    parser.add_argument('--yolo-model', type=str, default='yolov8n-seg.pt',
                        help='YOLO model variant (n/s/m/l). Default: yolov8n-seg.pt')
    parser.add_argument('--blacklist', type=int, nargs='+', default=[0],
                        help='COCO class IDs to exclude. Default: 0 (Person)')
    parser.add_argument('--yolo-confidence', type=float, default=0.5,
                        help='Minimum YOLO confidence for exclusion. Default: 0.5')
    
    # Anchor selection (like V8)
    parser.add_argument('--anchor-percent', type=float, default=0.15,
                        help='Fraction of (inlier & background) candidates to keep by lowest flow. Default: 0.15')
    
    # Depth model
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Depth model size (default: vitl)')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Depth model input resolution (default: 518)')
    
    # Alignment
    parser.add_argument('--max-scale-diff', type=float, default=0.1,
                        help='Maximum allowed scale change (default: 0.1)')
    parser.add_argument('--use-rotation', action='store_true',
                        help='Enable rotation recovery (for moving cameras)')
    
    # Camera and scene
    parser.add_argument('--fov', type=float, default=70.0,
                        help='Camera horizontal FOV in degrees (default: 70.0)')
    parser.add_argument('--scene-threshold', type=float, default=30.0,
                        help='Scene cut detection threshold (default: 30.0)')
    
    args = parser.parse_args()
    extract_depth_v9(args)
