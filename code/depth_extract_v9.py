"""
Depth Extraction Script V8 - Semantic Kabsch Alignment (YOLOv8 Integrated)

Key Improvement:
This version integrates Semantic Segmentation using YOLOv8 to solve the 
"Dynamic Anchor Problem" that affects V5, V6, and V7.

================================================================================
THE PROBLEM IN V5/V6/V7
================================================================================

Optical flow alone cannot distinguish between:
1. Truly Static Objects (walls, floors, buildings)
2. Temporarily Static Objects (person standing still)

When a person stands still, optical flow magnitude is near zero, causing the
alignment algorithm to treat them as valid anchors. When the person moves,
the alignment suddenly breaks, causing:
- Sudden scale jumps
- Depth flickering
- Temporal inconsistency

Real-World Example:
Frame 50: Person standing → low flow → selected as anchor
Frame 51: Person walks → alignment locked to moving target → FAILS

================================================================================
THE V8 SOLUTION: SEMANTIC FILTERING
================================================================================

V8 combines TWO filters:
1. Motion Filter (from V5): Select pixels with low optical flow
2. Semantic Filter (NEW): Exclude dynamic object classes

Valid Anchor = (Low Motion) AND (NOT Person) AND (NOT Car) AND ...

Process:
1. Run YOLOv8 Segmentation on every frame
2. Create a "Semantic Safe Mask" where:
   - 1 = Background pixels (walls, floor, static objects)
   - 0 = Dynamic object pixels (people, vehicles)
3. Combine with low-motion mask from optical flow
4. Use only the intersection for Kabsch alignment

Result:
The algorithm ignores all people/vehicles entirely, locking ONLY onto
the true static background. Even if a person stands perfectly still,
they are excluded from the anchor set.

================================================================================
YOLO CLASS IDS (COCO Dataset)
================================================================================

Common dynamic objects to exclude:
0  = Person
1  = Bicycle
2  = Car
3  = Motorcycle
5  = Bus
7  = Truck

You can customize the blacklist based on your scene.

================================================================================
PERFORMANCE CONSIDERATIONS
================================================================================

YOLOv8 Model Sizes:
- yolov8n-seg.pt: Nano   (Fastest, ~3ms/frame on GPU)
- yolov8s-seg.pt: Small  (Fast, ~5ms/frame)
- yolov8m-seg.pt: Medium (Balanced, ~10ms/frame)
- yolov8l-seg.pt: Large  (Best accuracy, ~20ms/frame)

Recommendation: Start with Nano (n). Only upgrade if segmentation quality
is insufficient for your specific video.

================================================================================
USAGE
================================================================================

# Basic V8 extraction (Auto-downloads YOLO model on first run)
python depth_extract_v8.py --video-path input.mp4 --output output.npz

# With rotation recovery (for moving cameras)
python depth_extract_v8.py --video-path input.mp4 --output output.npz --use-rotation

# Custom YOLO model size
python depth_extract_v8.py --video-path input.mp4 --output output.npz --yolo-model yolov8s-seg.pt

# Exclude multiple classes (Person + Car + Bicycle)
python depth_extract_v8.py --video-path input.mp4 --output output.npz --blacklist 0 2 1

# Low anchor percent (more selective, fewer but better anchors)
python depth_extract_v8.py --video-path input.mp4 --output output.npz --anchor-percent 0.01

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
    
    # Remove 'module.' prefix if present (from DataParallel training)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v 
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model = model.to(device).eval()
    print("✅ RAFT model loaded")
    return model


def load_yolo_model(model_name='yolov8n-seg.pt', device='cuda'):
    """
    Loads YOLOv8 Segmentation model.
    
    The model will be automatically downloaded on first use if not present.
    
    Args:
        model_name: YOLO model variant (n/s/m/l)
        device: Device for inference
        
    Returns:
        Loaded YOLO model
        
    Performance:
        - yolov8n-seg: ~3ms/frame (1920x1080) on RTX 3090
        - yolov8s-seg: ~5ms/frame
        - yolov8m-seg: ~10ms/frame
        - yolov8l-seg: ~20ms/frame
    """
    print(f"Loading YOLOv8 Segmentation model ({model_name})...")
    print("   (Will auto-download on first run)")
    
    # YOLO handles device placement internally
    model = YOLO(model_name)
    model.to(device)
    print("✅ YOLO model loaded")
    return model


# =============================================================================
# SEMANTIC MASK GENERATION (NEW IN V8)
# =============================================================================

def get_semantic_exclude_mask(frame_rgb, yolo_model, blacklist_classes=[0], 
                               confidence_threshold=0.5, device='cuda'):
    """
    Runs YOLO inference and creates a boolean mask where:
    True  = Safe Background (pixels to KEEP for alignment)
    False = Blacklisted Object (pixels to EXCLUDE from alignment)
    
    Algorithm:
    1. Run YOLOv8 segmentation on frame
    2. For each detected object:
       - If class is in blacklist AND confidence > threshold:
         → Mark those pixels as False (exclude)
    3. All other pixels remain True (safe for anchors)
    
    Args:
        frame_rgb: RGB frame [H, W, 3]
        yolo_model: Loaded Ultralytics YOLO model
        blacklist_classes: List of COCO class IDs to exclude (0=Person)
        confidence_threshold: Minimum confidence to exclude object
        device: Torch device
        
    Returns:
        safe_mask: Boolean tensor [1, 1, H, W] where True=background
        
    Example:
        Frame contains:
        - Person (class 0, conf=0.92) → Excluded
        - Car (class 2, conf=0.85) → Excluded if 2 in blacklist
        - Wall (no detection) → Kept
        - Chair (class 56, conf=0.45) → Kept (below threshold or not blacklisted)
    """
    H, W = frame_rgb.shape[:2]
    
    # Run inference
    # verbose=False: Suppress per-frame console output
    # retina_masks=True: High-quality mask upsampling
    results = yolo_model(frame_rgb, verbose=False, retina_masks=True)
    
    # Start with all pixels marked as "safe" (assume everything is background)
    safe_mask = torch.ones((H, W), dtype=torch.bool, device=device)
    
    result = results[0]
    
    # If no objects detected, return full safe mask
    if result.masks is None:
        return safe_mask.unsqueeze(0).unsqueeze(0)
    
    # Extract detection info
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    masks_data = result.masks.data  # Tensor on GPU [N, H', W']
    
    # Ensure mask resolution matches frame resolution
    # YOLO may output lower-res masks for speed
    if masks_data.shape[1:] != (H, W):
        masks_data = F.interpolate(
            masks_data.unsqueeze(1), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        # Re-binarize after interpolation
        masks_data = masks_data > 0.5
    
    # Iterate through each detected object
    for i, (cls_id, conf) in enumerate(zip(classes, confidences)):
        # Check if object should be excluded
        if int(cls_id) in blacklist_classes and conf >= confidence_threshold:
            # masks_data[i] is True where object pixels are
            object_mask = masks_data[i]
            
            # Remove these pixels from safe_mask
            # safe_mask &= ~object_mask means "keep safe only where object is NOT"
            safe_mask = safe_mask & (~object_mask)
    
    return safe_mask.unsqueeze(0).unsqueeze(0)


# =============================================================================
# CORE ALGORITHMS: FLOW, WARPING, ALIGNMENT
# =============================================================================

def compute_optical_flow(raft_model, frame1, frame2, device='cuda', iters=20):
    """
    Computes optical flow from frame2 to frame1 using RAFT.
    
    Args:
        raft_model: Loaded RAFT model
        frame1: Current frame RGB [H, W, 3]
        frame2: Previous frame RGB [H, W, 3]
        device: Torch device
        iters: RAFT refinement iterations (higher = more accurate but slower)
        
    Returns:
        flow: Optical flow tensor [1, 2, H, W]
    """
    with torch.no_grad():
        # Convert to torch tensors [1, 3, H, W]
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        # Pad to multiple of 8 (RAFT requirement)
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        
        # Run RAFT
        _, flow = raft_model(img1, img2, iters=iters, test_mode=True)
        
        # Remove padding
        return padder.unpad(flow)


def warp_depth_gpu(depth, flow):
    """
    Warps previous depth map to current frame using optical flow.
    
    Uses bilinear sampling to handle non-integer pixel coordinates.
    
    Args:
        depth: Depth map [1, 1, H, W]
        flow: Optical flow [1, 2, H, W]
        
    Returns:
        warped_depth: Warped depth map [1, 1, H, W]
        valid_mask: Mask of valid warped pixels [1, 1, H, W]
    """
    B, C, H, W = depth.shape
    
    # Create sampling grid
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device), 
        torch.arange(W, device=depth.device), 
        indexing='ij'
    )
    coords = torch.stack([x, y], dim=0).float().unsqueeze(0)
    
    # Add flow to get new coordinates
    sample_coords = coords + flow
    
    # Normalize to [-1, 1] for grid_sample
    sample_coords[:, 0, :, :] = 2.0 * sample_coords[:, 0, :, :] / (W - 1) - 1.0
    sample_coords[:, 1, :, :] = 2.0 * sample_coords[:, 1, :, :] / (H - 1) - 1.0
    sample_coords = sample_coords.permute(0, 2, 3, 1)
    
    # Sample depth at warped locations
    warped_depth = F.grid_sample(
        depth, sample_coords, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    
    # Create validity mask (1 if pixel stayed in bounds, 0 otherwise)
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
    
    Assumes:
    - Principal point at image center
    - Square pixels (fx = fy)
    - No skew
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_degrees: Horizontal field of view in degrees
        
    Returns:
        K: 3x3 camera intrinsic matrix
    """
    fov_rad = np.radians(fov_degrees)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2
    
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)


def depth_to_point_cloud_torch(depth, K, device='cuda'):
    """
    Converts depth map to 3D point cloud using camera intrinsics.
    
    Pinhole camera model:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
    
    Args:
        depth: Depth map [H, W] or [1, 1, H, W]
        K: Camera intrinsic matrix [3, 3]
        device: Torch device
        
    Returns:
        points: Point cloud [H, W, 3] where last dim is (X, Y, Z)
    """
    # Ensure depth is 2D
    if depth.dim() == 4:
        depth = depth.squeeze(0).squeeze(0)
    elif depth.dim() == 3:
        depth = depth.squeeze(0)
    
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    # Create pixel coordinate grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    
    # Back-project to 3D
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return torch.stack([X, Y, Z], dim=-1)


def solve_kabsch_umeyama_torch(P, Q, compute_rotation=True):
    """
    Solves for optimal similarity transform (scale + rotation + translation)
    that aligns point set P to point set Q using Kabsch-Umeyama algorithm.
    
    Minimizes: ||s*R*P + t - Q||^2
    
    Args:
        P: Source points [N, 3]
        Q: Target points [N, 3]
        compute_rotation: If False, only compute scale (faster)
        
    Returns:
        s: Scale factor (float)
        R: Rotation matrix [3, 3]
        t: Translation vector [3]
        
    Algorithm:
    1. Center both point sets at origin
    2. Compute variance of P
    3. If rotation enabled:
       - Compute cross-covariance matrix H = P^T * Q
       - SVD decomposition to find optimal rotation R
       - Scale s from trace(R*H) / var(P)
    4. If rotation disabled:
       - Scale s from var(Q) / var(P)
    5. Translation t = centroid_Q - s*R*centroid_P
    """
    device = P.device
    
    # Safety check
    if P.shape[0] < 3:
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)
    
    # Center point sets
    centroid_P = P.mean(dim=0)
    centroid_Q = Q.mean(dim=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute variance of P
    var_P = (P_centered ** 2).sum()
    
    # Scale-only mode (faster, no rotation)
    if not compute_rotation:
        var_Q = (Q_centered ** 2).sum()
        s = torch.sqrt(var_Q / (var_P + 1e-8))
        R = torch.eye(3, device=device)
        t = centroid_Q - s * centroid_P
        return s.item(), R, t
    
    # Full similarity transform (scale + rotation + translation)
    # Compute cross-covariance matrix
    H = torch.mm(P_centered.T, Q_centered)
    
    # SVD decomposition
    U, S, Vt = torch.linalg.svd(H)
    
    # Compute rotation matrix
    R = torch.mm(Vt.T, U.T)
    
    # Ensure proper rotation (det(R) = 1)
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = torch.mm(Vt.T, U.T)
    
    # Compute scale
    s = torch.trace(torch.mm(R, H)) / (var_P + 1e-8)
    
    # Compute translation
    t = centroid_Q - s * torch.mv(R, centroid_P)
    
    return s.item(), R, t


def detect_scene_cut(frame1, frame2, threshold=30.0):
    """
    Detects scene cuts using color histogram comparison.
    
    Computes 8x8x8 RGB histograms and compares using Chi-Square distance.
    Large distance indicates different scenes (e.g., camera cut, fade).
    
    Args:
        frame1: First frame RGB
        frame2: Second frame RGB
        threshold: Chi-square threshold for cut detection
        
    Returns:
        is_cut: True if scene cut detected
        
    Typical thresholds:
    - < 10: Very similar (same scene, small motion)
    - 10-30: Similar scene, larger motion
    - > 30: Likely different scene (cut)
    """
    # Compute 3D color histograms
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], 
                          [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], 
                          [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Compare using Chi-Square distance
    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    return distance > threshold


# =============================================================================
# V8: SEMANTIC KABSCH ALIGNMENT
# =============================================================================

def align_depth_kabsch_semantic(curr_depth, prev_depth_warped, flow, valid_mask, 
                                semantic_mask, K,
                                anchor_percent=0.15, max_scale_diff=0.1,
                                min_valid_pixels=100, use_rotation=False,
                                device='cuda'):
    """
    V8 Semantic Alignment: Combines motion filtering with semantic filtering.
    
    Three-Stage Filtering:
    1. Motion Filter: Select pixels with lowest optical flow (V5 logic)
    2. Semantic Filter: Exclude dynamic object classes (V8 NEW)
    3. Geometric Filter: Valid depth, in-bounds warping
    
    Final Valid Anchors = Motion ∩ Semantic ∩ Geometric
    
    This ensures anchors are:
    - Low motion (stable)
    - On background (not person/car)
    - Geometrically valid (depth > 0, in bounds)
    
    Args:
        curr_depth: Current frame depth [1, 1, H, W]
        prev_depth_warped: Previous depth warped to current frame [1, 1, H, W]
        flow: Optical flow [1, 2, H, W]
        valid_mask: Warp validity mask [1, 1, H, W]
        semantic_mask: YOLO semantic mask [1, 1, H, W] (True=background)
        K: Camera intrinsics [3, 3]
        anchor_percent: Percentage of pixels to select by low motion
        max_scale_diff: Maximum allowed scale change (prevents outliers)
        min_valid_pixels: Minimum anchors required for alignment
        use_rotation: Enable rotation recovery
        device: Torch device
        
    Returns:
        aligned_depth: Aligned depth map [1, 1, H, W]
        info: Dictionary with alignment statistics
    """
    
    # 1. Calculate Motion Magnitude
    flow_mag = torch.norm(flow, dim=1, keepdim=True)
    
    # Safety: Static Scene Protection
    if flow_mag.max() < 0.5:
        return curr_depth, {
            'scale': 1.0, 
            'status': 'static_scene', 
            'anchor_count': 0,
            'semantic_filter_reduction': 0
        }
    
    # 2. Motion Filter (V5 Logic)
    # Select top anchor_percent pixels with lowest flow
    flat_flow = flow_mag.view(-1)
    k_elements = int(flat_flow.numel() * anchor_percent)
    _, indices = torch.topk(flat_flow, k=k_elements, largest=False)
    motion_mask = torch.zeros_like(flat_flow, dtype=torch.bool)
    motion_mask[indices] = True
    motion_mask = motion_mask.view_as(flow_mag)
    
    # Count anchors before semantic filtering
    anchors_before_semantic = (
        motion_mask & 
        (valid_mask > 0.5) & 
        (prev_depth_warped > 0.001) & 
        (curr_depth > 0.001)
    ).sum().item()
    
    # 3. COMBINE ALL FILTERS (V8 Logic)
    # Valid = Motion AND Semantic AND Geometric
    final_valid_mask = (
        motion_mask &                    # Low motion (V5)
        (semantic_mask > 0.5) &          # Background only (V8 NEW)
        (valid_mask > 0.5) &             # In-bounds warp
        (prev_depth_warped > 0.001) &    # Valid previous depth
        (curr_depth > 0.001)             # Valid current depth
    )
    
    anchors_after_semantic = final_valid_mask.sum().item()
    semantic_reduction = anchors_before_semantic - anchors_after_semantic
    
    # Check if enough anchors remain
    if anchors_after_semantic < min_valid_pixels:
        return curr_depth, {
            'status': 'insufficient_anchors_after_semantic_filter', 
            'anchor_count': anchors_after_semantic,
            'semantic_filter_reduction': semantic_reduction
        }
    
    # 4. Extract 3D Point Clouds
    curr_points_3d = depth_to_point_cloud_torch(curr_depth, K, device)
    prev_points_3d = depth_to_point_cloud_torch(prev_depth_warped, K, device)
    
    # Select anchor points
    final_valid_2d = final_valid_mask.squeeze()
    P_anchors = curr_points_3d[final_valid_2d]  # Current frame anchor points
    Q_anchors = prev_points_3d[final_valid_2d]  # Previous frame anchor points
    
    # 5. Solve Kabsch-Umeyama
    s_raw, R, t = solve_kabsch_umeyama_torch(
        P_anchors, Q_anchors, 
        compute_rotation=use_rotation
    )
    
    # Clamp scale to prevent outliers
    s = np.clip(s_raw, 1.0 - max_scale_diff, 1.0 + max_scale_diff)
    
    # 6. Apply Transform
    H, W = curr_depth.shape[2], curr_depth.shape[3]
    
    if use_rotation:
        # Full 3D similarity transform: aligned = s*R*curr + t
        curr_flat = curr_points_3d.view(-1, 3)
        curr_rotated = torch.mm(curr_flat, R.T)
        curr_transformed = s * curr_rotated + t.unsqueeze(0)
        aligned_depth = curr_transformed[:, 2].view(1, 1, H, W)
        
        # Extract rotation angles for logging
        R_np = R.cpu().numpy()
        rot_y = np.arcsin(-R_np[2, 0]) * 180 / np.pi
        rot_x = np.arctan2(R_np[2, 1], R_np[2, 2]) * 180 / np.pi
        rot_z = np.arctan2(R_np[1, 0], R_np[0, 0]) * 180 / np.pi
        rot_info = f"({rot_x:.1f}°, {rot_y:.1f}°, {rot_z:.1f}°)"
    else:
        # Scale + Z-translation only (faster, suitable for fixed cameras)
        aligned_depth = s * curr_depth + t[2].item()
        rot_info = "disabled"
    
    return aligned_depth, {
        'scale': s,
        'rotation': rot_info,
        'translation': t.cpu().numpy().tolist(),
        'anchor_count': anchors_after_semantic,
        'semantic_filter_reduction': semantic_reduction,
        'status': 'aligned'
    }


# =============================================================================
# MAIN EXTRACTION LOOP
# =============================================================================

def extract_depth_v8(args):
    """
    Main V8 depth extraction with semantic filtering.
    
    Process Flow:
    1. Load models (Depth, RAFT, YOLO)
    2. For each frame:
       a. Infer raw depth
       b. Detect scene cuts
       c. Compute optical flow
       d. Run YOLO segmentation
       e. Align using semantic Kabsch
       f. Save aligned depth
    3. Save results to .npz
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load all three models
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
    
    # Estimate camera intrinsics
    K = estimate_camera_intrinsics(width, height, fov_degrees=args.fov)
    K_tensor = torch.from_numpy(K).float().to(device)
    
    # Print configuration
    print(f"\n🎥 Video: {video_name}")
    print(f"📊 Video: {width}x{height} @ {fps}fps, {frame_count} frames")
    print(f"📐 Camera intrinsics (estimated, FOV={args.fov}°):")
    print(f"   fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"🎯 V8 Semantic Kabsch-Umeyama Alignment:")
    print(f"   Blacklist Classes: {args.blacklist}")
    print(f"   Anchor Percent: {args.anchor_percent}")
    print(f"   Max Scale Diff: {args.max_scale_diff}")
    print(f"   Rotation Recovery: {'ENABLED' if args.use_rotation else 'DISABLED'}")
    
    # Storage
    all_depths = []
    prev_frame_rgb = None
    prev_depth_stable = None
    
    # Statistics tracking
    status_counts = {
        'aligned': 0,
        'static_scene': 0,
        'insufficient_anchors_after_semantic_filter': 0,
        'first_frame': 0,
        'scene_cut': 0
    }
    anchor_history = []
    semantic_reduction_history = []
    scale_history = []
    
    # Main processing loop
    pbar = tqdm(total=frame_count, desc="Extracting Depth (V8 Semantic)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Scene Cut Detection
        is_scene_cut = False
        if prev_frame_rgb is not None and args.scene_threshold > 0:
            is_scene_cut = detect_scene_cut(prev_frame_rgb, frame_rgb, args.scene_threshold)
            if is_scene_cut:
                status_counts['scene_cut'] += 1
        
        # 2. Raw Depth Inference
        with torch.no_grad():
            depth_raw = depth_model.infer_image(frame_rgb, args.input_size)
            depth_raw = torch.from_numpy(depth_raw).float().to(device).unsqueeze(0).unsqueeze(0)
        
        # 3. Alignment (if not first frame or scene cut)
        if prev_frame_rgb is not None and prev_depth_stable is not None and not is_scene_cut:
            
            # A. Compute Optical Flow
            flow = compute_optical_flow(raft_model, frame_rgb, prev_frame_rgb, device)
            prev_warped, valid_mask = warp_depth_gpu(prev_depth_stable, flow)
            
            # B. Get Semantic Mask (NEW in V8)
            # This excludes dynamic objects (people, cars, etc.)
            semantic_mask = get_semantic_exclude_mask(
                frame_rgb, yolo_model, 
                blacklist_classes=args.blacklist,
                confidence_threshold=args.yolo_confidence,
                device=device
            )
            
            # C. Semantic Kabsch Alignment
            curr_aligned, info = align_depth_kabsch_semantic(
                depth_raw, prev_warped, flow, valid_mask, semantic_mask, K_tensor,
                anchor_percent=args.anchor_percent,
                max_scale_diff=args.max_scale_diff,
                use_rotation=args.use_rotation,
                device=device
            )
            
            depth_stable = curr_aligned
            
            # Track statistics
            status = info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if 'anchor_count' in info:
                anchor_history.append(info['anchor_count'])
            if 'semantic_filter_reduction' in info:
                semantic_reduction_history.append(info['semantic_filter_reduction'])
            if 'scale' in info:
                scale_history.append(info['scale'])
        
        else:
            # First frame or scene cut: No alignment
            depth_stable = depth_raw
            if prev_frame_rgb is None:
                status_counts['first_frame'] += 1
        
        # 4. Store Result
        all_depths.append(depth_stable.squeeze().cpu().numpy().astype(np.float16))
        
        # 5. Update State
        prev_frame_rgb = frame_rgb.copy()
        prev_depth_stable = depth_stable.clone()
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Convert to numpy array
    all_depths = np.array(all_depths)
    
    # Calculate global statistics
    global_min = all_depths.min()
    global_max = all_depths.max()
    
    # Save Output
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
    
    # Print Summary
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
        print(f"\n📊 Anchor Statistics:")
        print(f"   Mean: {int(np.mean(anchor_history))}")
        print(f"   Min: {int(np.min(anchor_history))}")
        print(f"   Max: {int(np.max(anchor_history))}")
    
    if semantic_reduction_history:
        print(f"\n🧠 Semantic Filter Impact:")
        print(f"   Avg pixels excluded by YOLO: {int(np.mean(semantic_reduction_history))}")
        print(f"   Max exclusion: {int(np.max(semantic_reduction_history))}")
    
    if scale_history:
        print(f"\n📏 Scale Statistics:")
        print(f"   Mean: {np.mean(scale_history):.4f}")
        print(f"   Std: {np.std(scale_history):.4f}")
        print(f"   Range: [{np.min(scale_history):.4f}, {np.max(scale_history):.4f}]")
    
    print(f"\n✅ Extraction complete!")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V8 Depth Extraction with Semantic Filtering (YOLOv8)"
    )
    
    # Required arguments
    parser.add_argument('--video-path', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output .npz file')
    
    # YOLO settings (NEW in V8)
    parser.add_argument('--yolo-model', type=str, default='yolov8n-seg.pt',
                        help='YOLO model variant (n/s/m/l). Default: yolov8n-seg.pt')
    parser.add_argument('--blacklist', type=int, nargs='+', default=[0],
                        help='COCO class IDs to exclude. Default: 0 (Person)')
    parser.add_argument('--yolo-confidence', type=float, default=0.5,
                        help='Minimum YOLO confidence for exclusion. Default: 0.5')
    
    # Depth model settings
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Depth model size. Default: vitl')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Depth model input resolution. Default: 518')
    
    # Alignment settings
    parser.add_argument('--anchor-percent', type=float, default=0.15,
                        help='Percentage of low-motion pixels to consider. Default: 0.15')
    parser.add_argument('--max-scale-diff', type=float, default=0.1,
                        help='Maximum allowed scale change. Default: 0.1')
    parser.add_argument('--use-rotation', action='store_true',
                        help='Enable rotation recovery (for moving cameras)')
    
    # Camera and scene settings
    parser.add_argument('--fov', type=float, default=70.0,
                        help='Camera horizontal FOV in degrees. Default: 70.0')
    parser.add_argument('--scene-threshold', type=float, default=30.0,
                        help='Scene cut detection threshold. Default: 30.0')
    
    args = parser.parse_args()
    
    extract_depth_v8(args)
