"""
Depth Extraction Script V5 - Kabsch-Umeyama 3D Alignment

Key Improvement:
This version lifts the problem to 3D space using the Kabsch-Umeyama algorithm.
Instead of aligning depth values directly (2D), we:
1. Convert depth maps to 3D point clouds using estimated camera intrinsics
2. Use optical flow to establish point correspondences
3. Find the optimal similarity transform (Scale + Rotation + Translation)
4. Apply the transform to align current frame to previous frame

This is geometrically more correct, especially for:
- Panning/tilting cameras (rotation recovery)
- Dolly movements (proper 3D scale)
- Mixed camera motion

The anchor filtering from V4 is preserved for robustness.

================================================================================
USAGE
================================================================================

# Basic extraction (scale-only, no rotation)
python depth_extract_v5.py --video-path input.mp4 --output output.npz

# Enable rotation recovery (for moving cameras)
python depth_extract_v5.py --video-path input.mp4 --output output.npz --use-rotation

# Adjust anchor percentage
python depth_extract_v5.py --video-path input.mp4 --output output.npz --anchor-percent 0.2

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
    
    # Load weights
    if not checkpoint_paths[encoder].exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_paths[encoder]}")
        
    model.load_state_dict(torch.load(checkpoint_paths[encoder], map_location='cpu'))
    model = model.to(device).eval()
    print("✅ Depth model loaded")
    return model


def load_raft_model(raft_checkpoint=None, device='cuda'):
    """
    Loads the RAFT optical flow model for motion estimation.
    """
    if raft_checkpoint is None:
        raft_checkpoint = PROJECT_ROOT / 'raft_models' / 'raft-things.pth'
    
    print("Loading RAFT optical flow model...")
    
    # Prepare arguments required by RAFT class
    args = argparse.Namespace()
    args.model = str(raft_checkpoint)
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    
    model = RAFT(args)
    
    # Load weights ensuring compatibility with DataParallel checkpoints
    state_dict = torch.load(raft_checkpoint, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model = model.to(device).eval()
    print("✅ RAFT model loaded")
    return model


# =============================================================================
# CORE ALGORITHMS: FLOW & WARPING
# =============================================================================

def compute_optical_flow(raft_model, frame1, frame2, device='cuda', iters=20):
    """
    Calculates the optical flow vector field between two frames.
    Input: frame1 (Current), frame2 (Previous)
    Returns: Flow (Current -> Previous)
    """
    with torch.no_grad():
        # Convert to tensor and normalize
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Pad images to be divisible by 8 (requirement for RAFT)
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        
        # Inference
        _, flow = raft_model(img1, img2, iters=iters, test_mode=True)
        
        # Remove padding
        flow = padder.unpad(flow)
        return flow


def warp_depth_gpu(depth, flow):
    """
    Warps a depth map from the previous frame to the current frame using flow.
    """
    B, C, H, W = depth.shape
    
    # Generate grid of coordinates
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), 
                          torch.arange(W, device=depth.device), indexing='ij')
    coords = torch.stack([x, y], dim=0).float().unsqueeze(0)
    
    # Apply flow to coordinates
    sample_coords = coords + flow
    
    # Normalize coordinates to range [-1, 1] for grid_sample
    sample_coords[:, 0, :, :] = 2.0 * sample_coords[:, 0, :, :] / (W - 1) - 1.0
    sample_coords[:, 1, :, :] = 2.0 * sample_coords[:, 1, :, :] / (H - 1) - 1.0
    sample_coords = sample_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    # Sample pixels from the previous depth map
    warped_depth = F.grid_sample(depth, sample_coords, mode='bilinear', 
                                 padding_mode='zeros', align_corners=True)
    
    # Create a mask to identify pixels that fell out of image bounds
    valid_mask = torch.ones_like(depth)
    valid_mask = F.grid_sample(valid_mask, sample_coords, mode='nearest',
                               padding_mode='zeros', align_corners=True)
    
    return warped_depth, valid_mask


# =============================================================================
# 3D POINT CLOUD UTILITIES
# =============================================================================

def estimate_camera_intrinsics(width, height, fov_degrees=70.0):
    """
    Estimates camera intrinsic matrix K for a given image size.
    
    We assume a standard pinhole camera with:
    - Principal point at image center
    - Square pixels (fx = fy)
    - FOV of approximately 70 degrees (common for phone/webcam)
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_degrees: Horizontal field of view in degrees
        
    Returns:
        K: 3x3 intrinsic matrix
    """
    # Convert FOV to focal length
    # fov = 2 * arctan(W / (2 * fx))
    # => fx = W / (2 * tan(fov/2))
    fov_rad = np.radians(fov_degrees)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Square pixels
    
    cx = width / 2
    cy = height / 2
    
    K = np.array([
        [fx,  0,  cx],
        [0,  fy,  cy],
        [0,   0,   1]
    ], dtype=np.float32)
    
    return K


def depth_to_point_cloud_torch(depth, K, device='cuda'):
    """
    Converts a 2D depth map to a 3D point cloud using pinhole camera model.
    
    Back-projection formula:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth value
    
    Args:
        depth: Depth tensor [1, 1, H, W] or [H, W]
        K: Intrinsic matrix (3x3 numpy array)
        device: Torch device
        
    Returns:
        points_3d: Tensor [H, W, 3] containing (X, Y, Z) for each pixel
    """
    # Handle different input shapes
    if depth.dim() == 4:
        depth = depth.squeeze(0).squeeze(0)  # [H, W]
    elif depth.dim() == 3:
        depth = depth.squeeze(0)  # [H, W]
    
    H, W = depth.shape
    
    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel coordinate grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Back-project to 3D
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack into [H, W, 3]
    points_3d = torch.stack([X, Y, Z], dim=-1)
    
    return points_3d


def point_cloud_to_depth(points_3d):
    """
    Extracts depth (Z values) from a point cloud.
    
    Args:
        points_3d: Tensor [H, W, 3] or [N, 3]
        
    Returns:
        depth: Tensor [H, W] or [N]
    """
    if points_3d.dim() == 3:
        return points_3d[:, :, 2]  # [H, W]
    else:
        return points_3d[:, 2]  # [N]


# =============================================================================
# KABSCH-UMEYAMA ALGORITHM
# =============================================================================

def solve_kabsch_umeyama_torch(P, Q, compute_rotation=True):
    """
    Kabsch-Umeyama algorithm: finds optimal Scale (s), Rotation (R), and Translation (t)
    that aligns source points P to target points Q.
    
    Minimizes: E = || Q - (s * R @ P + t) ||²
    
    Args:
        P: Source points (Current Frame) [N, 3] torch tensor
        Q: Target points (Previous Frame, warped) [N, 3] torch tensor
        compute_rotation: If False, only compute scale (faster, R=I)
        
    Returns:
        s: Scale factor (scalar)
        R: Rotation matrix [3, 3] (identity if compute_rotation=False)
        t: Translation vector [3]
    """
    device = P.device
    N = P.shape[0]
    
    if N < 3:
        # Not enough points for meaningful computation
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    # 1. Compute Centroids
    centroid_P = P.mean(dim=0)  # [3]
    centroid_Q = Q.mean(dim=0)  # [3]
    
    # 2. Center the points (remove translation)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 3. Compute Scale
    # Variance of P (sum of squared distances from centroid)
    var_P = (P_centered ** 2).sum()
    
    if not compute_rotation:
        # Scale-only mode: s = std(Q) / std(P)
        var_Q = (Q_centered ** 2).sum()
        s = torch.sqrt(var_Q / (var_P + 1e-8))
        R = torch.eye(3, device=device)
        t = centroid_Q - s * centroid_P
        return s.item(), R, t
    
    # 4. Compute Covariance Matrix H
    # H = P_centered.T @ Q_centered
    H = torch.mm(P_centered.T, Q_centered)  # [3, 3]
    
    # 5. SVD Decomposition
    U, S, Vt = torch.linalg.svd(H)
    
    # 6. Compute Rotation Matrix R = V @ U.T
    V = Vt.T
    R = torch.mm(V, U.T)
    
    # Handle reflection case (ensure det(R) = +1)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = torch.mm(V, U.T)
    
    # 7. Compute Scale s
    # s = trace(R @ H) / var(P)
    trace_RH = torch.trace(torch.mm(R, H))
    s = trace_RH / (var_P + 1e-8)
    
    # 8. Compute Translation t
    # t = centroid_Q - s * R @ centroid_P
    t = centroid_Q - s * torch.mv(R, centroid_P)
    
    return s.item(), R, t


# =============================================================================
# V5: KABSCH ALIGNMENT WITH ANCHOR FILTERING
# =============================================================================

def align_depth_kabsch_anchored(curr_depth, prev_depth_warped, flow, valid_mask, K,
                                 anchor_percent=0.15, max_scale_diff=0.1,
                                 min_valid_pixels=100, use_rotation=False,
                                 device='cuda'):
    """
    Aligns Current Frame -> Previous Frame using Kabsch-Umeyama in 3D space.
    Uses only low-motion anchor pixels for robustness.
    
    Process:
    1. Convert both depth maps to 3D point clouds
    2. Filter to keep only low-motion anchor pixels
    3. Solve Kabsch-Umeyama for optimal (s, R, t)
    4. Apply transform to entire current frame
    
    Args:
        curr_depth: Current raw depth [1, 1, H, W]
        prev_depth_warped: Previous history warped to current [1, 1, H, W]
        flow: Optical flow vectors [1, 2, H, W]
        valid_mask: Validity mask from warping [1, 1, H, W]
        K: Camera intrinsic matrix [3, 3] numpy array
        anchor_percent: Percentage of slowest pixels to use (0.15 = 15%)
        max_scale_diff: Maximum scale deviation from 1.0
        min_valid_pixels: Minimum anchor pixels required
        use_rotation: Whether to compute rotation (True) or scale-only (False)
        device: Torch device
        
    Returns:
        curr_aligned: Aligned depth map [1, 1, H, W]
        info: Diagnostic dictionary
    """
    
    # 1. Calculate Motion Magnitude
    flow_mag = torch.norm(flow, dim=1, keepdim=True)  # [1, 1, H, W]
    
    # --- PROTECTION: Static Scene Check ---
    if flow_mag.max() < 0.5:
        return curr_depth, {
            'scale': 1.0, 'rotation': 'identity', 'translation': [0, 0, 0],
            'status': 'static_scene'
        }
    
    # 2. Identify Anchor Pixels (lowest motion)
    flat_flow = flow_mag.view(-1)
    num_pixels = flat_flow.numel()
    k_elements = int(num_pixels * anchor_percent)
    
    _, indices = torch.topk(flat_flow, k=k_elements, largest=False)
    anchor_mask_flat = torch.zeros_like(flat_flow, dtype=torch.bool)
    anchor_mask_flat[indices] = True
    anchor_mask = anchor_mask_flat.view_as(flow_mag)
    
    # 3. Combine with Validity Mask
    final_valid_mask = (
        anchor_mask & 
        (valid_mask > 0.5) & 
        (prev_depth_warped > 0.001) & 
        (curr_depth > 0.001)
    )
    
    if final_valid_mask.sum() < min_valid_pixels:
        return curr_depth, {'status': 'fallback_no_anchors'}
    
    # 4. Convert to 3D Point Clouds
    curr_points_3d = depth_to_point_cloud_torch(curr_depth, K, device)  # [H, W, 3] 
    prev_points_3d = depth_to_point_cloud_torch(prev_depth_warped, K, device)  # [H, W, 3]
    
    # 5. Extract Anchor Points
    final_valid_2d = final_valid_mask.squeeze()  # [H, W]
    
    P_anchors = curr_points_3d[final_valid_2d]  # [N, 3] - Current frame anchors
    Q_anchors = prev_points_3d[final_valid_2d]  # [N, 3] - Previous frame anchors (warped)
    
    # 6. Solve Kabsch-Umeyama
    s_raw, R, t = solve_kabsch_umeyama_torch(P_anchors, Q_anchors, compute_rotation=use_rotation)
    
    # 7. Safety Clamping for Scale
    s = np.clip(s_raw, 1.0 - max_scale_diff, 1.0 + max_scale_diff)
    
    # 8. Apply Transform to Entire Frame
    # Q_aligned = s * R @ P + t
    H, W = curr_depth.shape[2], curr_depth.shape[3]
    
    if use_rotation:
        # Full transform: s * R @ P + t
        curr_flat = curr_points_3d.view(-1, 3)  # [H*W, 3]
        # Apply rotation: [H*W, 3] @ [3, 3].T = [H*W, 3]
        curr_rotated = torch.mm(curr_flat, R.T)
        # Apply scale and translation
        curr_transformed = s * curr_rotated + t.unsqueeze(0)
        # Extract Z (depth)
        aligned_depth_flat = curr_transformed[:, 2]
        aligned_depth = aligned_depth_flat.view(1, 1, H, W)
    else:
        # Scale + Z-translation only (simpler, more stable)
        # Z_aligned = s * Z_curr + t_z
        aligned_depth = s * curr_depth + t[2].item()
    
    # Prepare rotation info for logging
    if use_rotation:
        # Extract approximate rotation angles (for logging)
        R_np = R.cpu().numpy()
        rot_y = np.arcsin(-R_np[2, 0]) * 180 / np.pi  # Pitch
        rot_x = np.arctan2(R_np[2, 1], R_np[2, 2]) * 180 / np.pi  # Roll
        rot_z = np.arctan2(R_np[1, 0], R_np[0, 0]) * 180 / np.pi  # Yaw
        rotation_info = f"({rot_x:.2f}°, {rot_y:.2f}°, {rot_z:.2f}°)"
    else:
        rotation_info = "disabled"
    
    return aligned_depth, {
        'scale': s,
        'scale_raw': s_raw,
        'rotation': rotation_info,
        'translation': t.cpu().numpy().tolist(),
        'anchor_count': final_valid_mask.sum().item(),
        'status': 'aligned'
    }


# =============================================================================
# SCENE CUT DETECTION
# =============================================================================

def detect_scene_cut(frame1, frame2, threshold=30.0):
    """
    Detects drastic changes in the video content (Scene Cuts).
    """
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    return diff > threshold


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def extract_depth(args):
    """Main depth extraction function with V5 Kabsch-Umeyama alignment."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    depth_model = load_depth_model(encoder=args.encoder, device=device)
    raft_model = load_raft_model(device=device)
    
    # Open video
    video_path = Path(args.video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Estimate camera intrinsics
    K = estimate_camera_intrinsics(width, height, fov_degrees=args.fov)
    
    print(f"\n🎥 Extracting depth from: {video_path.stem}")
    print(f"📊 Video: {width}x{height} @ {fps}fps, {frame_count} frames")
    print(f"📐 Camera intrinsics (estimated, FOV={args.fov}°):")
    print(f"   fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"🎯 V5 Kabsch-Umeyama Alignment:")
    print(f"   anchor_percent={args.anchor_percent}, max_scale_diff={args.max_scale_diff}")
    print(f"   rotation_recovery={'ENABLED' if args.use_rotation else 'DISABLED'}")
    
    # Storage
    all_depths = []
    scene_cuts = []
    alignment_stats = []
    
    # State
    prev_frame = None
    prev_depth_stable = None
    
    # Global tracking
    global_min = float('inf')
    global_max = float('-inf')
    
    # Status counters
    status_counts = {'aligned': 0, 'static_scene': 0, 'fallback_no_anchors': 0, 'scene_cut': 0, 'first_frame': 0}
    
    pbar = tqdm(total=frame_count, desc="Extracting Depth (V5 Kabsch)")
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect scene cut
            is_scene_cut = False
            if prev_frame is not None and args.scene_threshold > 0:
                is_scene_cut = detect_scene_cut(prev_frame, frame_rgb, args.scene_threshold)
                if is_scene_cut:
                    scene_cuts.append(frame_idx)
                    status_counts['scene_cut'] += 1
            
            # Get raw depth
            with torch.no_grad():
                depth_raw = depth_model.infer_image(frame_rgb, args.input_size)
                depth_raw = torch.from_numpy(depth_raw).float().to(device)
                depth_raw = depth_raw.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Temporal processing
            if prev_frame is not None and prev_depth_stable is not None and not is_scene_cut:
                # Compute optical flow (current -> previous)
                flow = compute_optical_flow(raft_model, frame_rgb, prev_frame, device)
                
                # Warp previous stable depth to current frame geometry
                prev_depth_warped, valid_mask = warp_depth_gpu(prev_depth_stable, flow)
                
                # Align current depth using KABSCH-UMEYAMA (V5 approach)
                curr_aligned, align_info = align_depth_kabsch_anchored(
                    depth_raw, prev_depth_warped, flow, valid_mask, K,
                    anchor_percent=args.anchor_percent,
                    max_scale_diff=args.max_scale_diff,
                    min_valid_pixels=args.min_valid_pixels,
                    use_rotation=args.use_rotation,
                    device=device
                )
                alignment_stats.append(align_info)
                status_counts[align_info['status']] = status_counts.get(align_info['status'], 0) + 1
                
                # Use aligned depth directly (no fusion)
                depth_stable = curr_aligned
            else:
                # First frame or scene cut: use raw depth
                depth_stable = depth_raw
                if frame_idx == 0:
                    status_counts['first_frame'] += 1
            
            # Convert to numpy and store
            depth_np = depth_stable.squeeze().cpu().numpy()
            all_depths.append(depth_np.astype(np.float16))
            
            # Update global stats
            global_min = min(global_min, depth_np.min())
            global_max = max(global_max, depth_np.max())
            
            # Update state
            prev_frame = frame_rgb.copy()
            prev_depth_stable = depth_stable.clone()
            
            frame_idx += 1
            pbar.update(1)
            
    finally:
        pbar.close()
        cap.release()
    
    print(f"📊 Scene cuts detected: {len(scene_cuts)}")
    print(f"📊 Alignment status breakdown:")
    for status, count in status_counts.items():
        if count > 0:
            print(f"   {status}: {count}")
    
    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'depths': np.array(all_depths),
        'fps': fps,
        'width': width,
        'height': height,
        'global_min': global_min,
        'global_max': global_max,
        'scene_cuts': np.array(scene_cuts),
        'source_video': str(video_path),
        'config_json': json.dumps({
            'version': 'v5_kabsch_umeyama',
            'model_encoder': args.encoder,
            'anchor_percent': args.anchor_percent,
            'max_scale_diff': args.max_scale_diff,
            'use_rotation': args.use_rotation,
            'fov_degrees': args.fov,
            'scene_threshold': args.scene_threshold,
            'input_size': args.input_size,
        })
    }
    
    print(f"\n💾 Saving to: {output_path}")
    np.savez_compressed(str(output_path), **save_dict)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {file_size_mb:.1f} MB")
    print(f"   Frames: {len(all_depths)}")
    print(f"   Depth range: {global_min:.4f} - {global_max:.4f}")
    print(f"   Scene cuts: {len(scene_cuts)}")
    
    # Print alignment statistics
    if alignment_stats:
        scales = [s['scale'] for s in alignment_stats if s.get('status') == 'aligned']
        anchors = [s['anchor_count'] for s in alignment_stats if s.get('status') == 'aligned']
        if scales:
            print(f"\n📊 Alignment Statistics:")
            print(f"   Scale: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}")
            print(f"   Anchors: mean={np.mean(anchors):.0f}, min={np.min(anchors):.0f}")
    
    print("\n✅ Extraction complete!")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Depth Extraction V5 - Kabsch-Umeyama 3D Alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction (scale-only, no rotation)
  python depth_extract_v5.py --video-path input.mp4 --output output.npz
  
  # Enable rotation recovery (for moving cameras)
  python depth_extract_v5.py --video-path input.mp4 --output output.npz --use-rotation
  
  # Custom FOV (default is 70°)
  python depth_extract_v5.py --video-path input.mp4 --output output.npz --fov 90
  
  # Adjust anchor percentage (20% slowest pixels)
  python depth_extract_v5.py --video-path input.mp4 --output output.npz --anchor-percent 0.2
        """
    )
    
    # Input/Output
    parser.add_argument('--video-path', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output .npz file')
    
    # Model settings
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Depth model encoder (default: vitl)')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input size for depth model (default: 518)')
    
    # Camera settings
    parser.add_argument('--fov', type=float, default=70.0,
                        help='Estimated horizontal field of view in degrees (default: 70)')
    
    # V5 Kabsch alignment parameters
    parser.add_argument('--use-rotation', action='store_true',
                        help='Enable rotation recovery (default: scale-only)')
    parser.add_argument('--anchor-percent', type=float, default=0.15,
                        help='Percentage of slowest-moving pixels to use as anchors (default: 0.15)')
    parser.add_argument('--max-scale-diff', type=float, default=0.1,
                        help='Max scale deviation from 1.0 (default: 0.1)')
    
    # Alignment parameters
    parser.add_argument('--min-valid-pixels', type=int, default=100,
                        help='Minimum valid anchor pixels for alignment')
    
    # Scene detection
    parser.add_argument('--scene-threshold', type=float, default=30.0,
                        help='Scene cut detection threshold (0 to disable)')
    
    args = parser.parse_args()
    extract_depth(args)


if __name__ == "__main__":
    main()
