"""
Depth Extraction V2 Final - Production Ready

Complete solution with:
- Hybrid fusion (background vs foreground)
- Aggressive hole filling with multiple strategies
- Bilateral smoothing
- Inpainting for remaining holes
- Optimized for minimal artifacts

================================================================================
USAGE
================================================================================

# Recommended settings for minimal holes
python depth_extract_v2_final.py \
  --video-path input.mp4 \
  --output output.npz \
  --photo-threshold 0.25 \
  --fb-threshold 4.0 \
  --fg-percentile 60 \
  --fg-blend 0.15 \
  --fill-fg-holes \
  --fg-hole-threshold 0.7 \
  --bilateral-smooth \
  --inpaint-holes

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

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2

# Add RAFT to path
RAFT_PATH = PROJECT_ROOT / "raft_repo"
sys.path.insert(0, str(RAFT_PATH / "core"))

from raft import RAFT
from utils.utils import InputPadder


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_depth_model(encoder='vitl', device='cuda'):
    """Load the Depth Anything V2 model."""
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
    model.load_state_dict(torch.load(checkpoint_paths[encoder], map_location='cpu'))
    model = model.to(device).eval()
    print("✅ Depth model loaded")
    return model


def load_raft_model(raft_checkpoint=None, device='cuda'):
    """Load the RAFT optical flow model."""
    if raft_checkpoint is None:
        raft_checkpoint = PROJECT_ROOT / 'raft_models' / 'raft-things.pth'
    
    print("Loading RAFT optical flow model...")
    
    args = argparse.Namespace()
    args.model = str(raft_checkpoint)
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    
    model = RAFT(args)
    
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
# OPTICAL FLOW
# =============================================================================

def compute_optical_flow(raft_model, frame1, frame2, device='cuda', iters=20):
    """Compute optical flow from frame1 to frame2."""
    with torch.no_grad():
        img1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        img2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        
        _, flow = raft_model(img1, img2, iters=iters, test_mode=True)
        flow = padder.unpad(flow)
        
        return flow


# =============================================================================
# WARPING
# =============================================================================

def warp_depth_gpu(depth, flow):
    """Warp depth map using optical flow."""
    B, C, H, W = depth.shape
    
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), 
                          torch.arange(W, device=depth.device), indexing='ij')
    coords = torch.stack([x, y], dim=0).float().unsqueeze(0)
    
    sample_coords = coords + flow
    sample_coords[:, 0, :, :] = 2.0 * sample_coords[:, 0, :, :] / (W - 1) - 1.0
    sample_coords[:, 1, :, :] = 2.0 * sample_coords[:, 1, :, :] / (H - 1) - 1.0
    sample_coords = sample_coords.permute(0, 2, 3, 1)
    
    warped_depth = F.grid_sample(depth, sample_coords, mode='bilinear', 
                                  padding_mode='zeros', align_corners=True)
    
    valid_mask = torch.ones_like(depth)
    valid_mask = F.grid_sample(valid_mask, sample_coords, mode='nearest',
                                padding_mode='zeros', align_corners=True)
    
    return warped_depth, valid_mask


def warp_rgb_gpu(rgb, flow):
    """Warp RGB image using optical flow."""
    B, C, H, W = rgb.shape
    
    y, x = torch.meshgrid(torch.arange(H, device=rgb.device), 
                          torch.arange(W, device=rgb.device), indexing='ij')
    coords = torch.stack([x, y], dim=0).float().unsqueeze(0)
    
    sample_coords = coords + flow
    sample_coords[:, 0, :, :] = 2.0 * sample_coords[:, 0, :, :] / (W - 1) - 1.0
    sample_coords[:, 1, :, :] = 2.0 * sample_coords[:, 1, :, :] / (H - 1) - 1.0
    sample_coords = sample_coords.permute(0, 2, 3, 1)
    
    warped_rgb = F.grid_sample(rgb, sample_coords, mode='bilinear', 
                                 padding_mode='zeros', align_corners=True)
    
    valid_mask = torch.ones_like(rgb[:, 0:1, :, :])
    valid_mask = F.grid_sample(valid_mask, sample_coords, mode='nearest',
                                padding_mode='zeros', align_corners=True)
    
    return warped_rgb, valid_mask


# =============================================================================
# RELIABILITY
# =============================================================================

def compute_photometric_reliability(curr_rgb, prev_rgb_warped, threshold=0.25):
    """Compute photometric reliability based on RGB L1 error."""
    if curr_rgb.max() > 1.1:
        curr_rgb = curr_rgb / 255.0
    if prev_rgb_warped.max() > 1.1:
        prev_rgb_warped = prev_rgb_warped / 255.0
    
    error = torch.abs(curr_rgb - prev_rgb_warped).mean(dim=1, keepdim=True)
    R_photo = (error < threshold).float()
    
    return R_photo


def compute_fb_consistency(flow_bwd, flow_fwd, threshold=4.0):
    """Compute forward-backward flow consistency."""
    flow_fwd_warped, _ = warp_depth_gpu(flow_fwd, flow_bwd)
    flow_consistency = torch.sqrt(torch.sum((flow_bwd + flow_fwd_warped)**2, dim=1, keepdim=True))
    R_fb = (flow_consistency < threshold).float()
    
    return R_fb


def combine_reliability_mask(valid_mask, R_photo, R_fb):
    """Combine reliability masks with AND operation."""
    R_raw = valid_mask * R_photo * R_fb
    R_raw = (R_raw > 0.5).float()
    return R_raw


def apply_opening(R_raw, kernel_size=3):
    """Apply opening (erosion → dilation) to remove speckles."""
    R = (R_raw > 0.5).float()
    
    if kernel_size <= 0:
        return R
    
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=R.device)
    
    # Erosion
    eroded_sum = F.conv2d(
        F.pad(R, (pad, pad, pad, pad), mode='replicate'),
        kernel, padding=0
    )
    R_eroded = (eroded_sum >= (kernel_size * kernel_size - 0.5)).float()
    
    # Dilation
    dilated_sum = F.conv2d(
        F.pad(R_eroded, (pad, pad, pad, pad), mode='replicate'),
        kernel, padding=0
    )
    R_open = (dilated_sum > 0.5).float()
    
    return R_open


def apply_light_dilation(R, kernel_size=3):
    """Apply light dilation to expand reliable regions slightly."""
    if kernel_size <= 0:
        return R
    
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=R.device)
    
    dilated_sum = F.conv2d(
        F.pad(R, (pad, pad, pad, pad), mode='replicate'),
        kernel, padding=0
    )
    R_dilated = (dilated_sum > 0.5).float()
    
    return R_dilated


# =============================================================================
# ALIGNMENT
# =============================================================================

def align_depth_robust(curr_depth, prev_depth_warped, reliability_mask, 
                       max_scale_diff=0.1, max_shift_diff=0.1, min_valid_pixels=100):
    """Align current frame to history using median-based scale & shift."""
    
    reliable_pixels = (reliability_mask > 0.5) & (prev_depth_warped > 1e-6) & torch.isfinite(prev_depth_warped)
    
    if reliable_pixels.sum() < min_valid_pixels:
        return curr_depth, {'scale': 1.0, 'shift': 0.0, 'status': 'fallback'}

    curr_vals = curr_depth[reliable_pixels]
    prev_vals = prev_depth_warped[reliable_pixels]

    curr_median = torch.median(curr_vals)
    prev_median = torch.median(prev_vals)
    
    curr_dev = torch.abs(curr_vals - curr_median)
    prev_dev = torch.abs(prev_vals - prev_median)
    
    eps = 1e-6
    s = (torch.median(prev_dev) + eps) / (torch.median(curr_dev) + eps)
    
    s_raw = s.item()
    s = torch.clamp(s, 1.0 - max_scale_diff, 1.0 + max_scale_diff)
    
    t = prev_median - s * curr_median
    
    scene_range = torch.max(curr_vals) - torch.min(curr_vals) + eps
    t_raw = t.item()
    t = torch.clamp(t, -max_shift_diff * scene_range, max_shift_diff * scene_range)

    curr_aligned = s * curr_depth + t
    
    alignment_info = {
        'scale': s.item(),
        'scale_raw': s_raw,
        'shift': t.item(),
        'shift_raw': t_raw,
        'scene_range': scene_range.item(),
        'reliable_pixels': reliable_pixels.sum().item(),
        'status': 'aligned'
    }
    
    return curr_aligned, alignment_info


# =============================================================================
# FOREGROUND DETECTION
# =============================================================================

def detect_foreground_regions(depth, percentile=60):
    """Detect foreground (people) vs background using depth."""
    depth_flat = depth.flatten()
    depth_threshold = torch.quantile(depth_flat, percentile / 100.0)
    
    fg_mask = (depth < depth_threshold).float()
    
    # Dilate foreground mask
    pad = 5
    kernel = torch.ones(1, 1, 11, 11, device=fg_mask.device)
    
    fg_dilated_sum = F.conv2d(
        F.pad(fg_mask, (pad,)*4, mode='replicate'),
        kernel, padding=0
    )
    fg_mask_dilated = (fg_dilated_sum > 0.5).float()
    
    return fg_mask_dilated


# =============================================================================
# FUSION
# =============================================================================

def fuse_depths_hybrid(depth_raw_aligned, prev_depth_warped, reliability_mask,
                       fg_mask, kernel_size=15, fg_blend=0.2):
    """Hybrid fusion: heavy history for background, prefer raw for foreground."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    pad = kernel_size // 2
    w_base = F.avg_pool2d(
        F.pad(reliability_mask.float(), (pad,)*4, mode='replicate'),
        kernel_size=kernel_size, stride=1
    )
    w_base = torch.clamp(w_base, 0.0, 1.0)
    
    # Modulate by foreground: bg uses full weight, fg uses reduced weight
    w_final = w_base * (1.0 - fg_mask) + (w_base * fg_blend) * fg_mask
    w_final = torch.clamp(w_final, 0.0, 1.0)
    
    depth_fused = w_final * prev_depth_warped + (1.0 - w_final) * depth_raw_aligned
    
    # NEW: Clamp final fusion to ensure valid depth
    depth_fused = torch.clamp(depth_fused, min=0.0)
    
    return depth_fused, w_final


def fill_foreground_holes_aggressive(depth_fused, depth_raw_aligned, prev_depth_warped,
                                     reliability_mask, fg_mask, fg_hole_threshold=0.7):
    """
    Aggressively fill holes in foreground regions.
    
    Key change: Use MUCH larger threshold to catch more holes.
    """
    # Compute soft reliability weight with LARGER kernel
    kernel_size = 15  # Increased from 11
    pad = kernel_size // 2
    w = F.avg_pool2d(
        F.pad(reliability_mask.float(), (pad,)*4, mode='replicate'),
        kernel_size=kernel_size, stride=1
    )
    
    # Detect foreground holes with HIGHER threshold (more aggressive)
    fg_holes = (fg_mask > 0.5) & (w < fg_hole_threshold)
    fg_holes = fg_holes.float()
    
    # Dilate holes MORE aggressively
    pad_dilate = 2  # 5x5 instead of 3x3
    kernel_dilate = torch.ones(1, 1, 5, 5, device=fg_holes.device)
    fg_holes_dilated_sum = F.conv2d(
        F.pad(fg_holes, (pad_dilate,)*4, mode='replicate'),
        kernel_dilate, padding=0
    )
    fg_holes_dilated = (fg_holes_dilated_sum > 0.5).float()
    
    # Smooth hole mask edges with LARGER kernel
    hole_weight = F.avg_pool2d(
        F.pad(fg_holes_dilated, (3,)*4, mode='replicate'),
        kernel_size=7, stride=1
    )
    hole_weight = torch.clamp(hole_weight, 0.0, 1.0)
    
    # Fill with raw depth (NOT warped - raw is better for visible foreground)
    depth_filled = depth_fused * (1.0 - hole_weight) + depth_raw_aligned * hole_weight
    
    return depth_filled


def bilateral_smooth_depth(depth, sigma_spatial=5, sigma_intensity=0.1):
    """Apply bilateral filtering to smooth depth while preserving edges."""
    depth_np = depth.squeeze().cpu().numpy()
    
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-6)
    
    depth_smooth = cv2.bilateralFilter(
        depth_norm.astype(np.float32), 
        d=9,
        sigmaColor=sigma_intensity, 
        sigmaSpace=sigma_spatial
    )
    
    depth_smooth = depth_smooth * (d_max - d_min) + d_min
    
    return torch.from_numpy(depth_smooth).unsqueeze(0).unsqueeze(0).to(depth.device)


def inpaint_small_holes(depth, max_hole_size=200):
    """Inpaint very small holes using OpenCV inpainting."""
    depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
    
    # Find zero/invalid regions
    mask = (np.abs(depth_np) < 1e-4).astype(np.uint8)
    
    if mask.sum() == 0:
        return depth
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Filter small holes
    small_holes_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_hole_size:
            small_holes_mask[labels == i] = 1
    
    # Inpaint
    if small_holes_mask.sum() > 0:
        depth_inpainted = cv2.inpaint(depth_np, small_holes_mask, 5, cv2.INPAINT_TELEA)
    else:
        depth_inpainted = depth_np
    
    return torch.from_numpy(depth_inpainted).unsqueeze(0).unsqueeze(0).to(depth.device)


def fill_holes_by_interpolation(depth, max_hole_size=500):
    """
    Fill holes by interpolating from surrounding valid depth.
    
    This is the nuclear option when both history and raw depth fail.
    """
    depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
    
    # Create mask of holes (invalid regions)
    # Consider both zero and extremely low values as holes
    mask = ((np.abs(depth_np) < 0.01) | (~np.isfinite(depth_np))).astype(np.uint8)
    
    if mask.sum() == 0:
        return depth
    
    # Find hole regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Process each hole
    result = depth_np.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area > max_hole_size:
            # Hole too big, skip (likely scene boundary)
            continue
        
        # Get hole mask for this component
        hole_mask = (labels == i).astype(np.uint8)
        
        # Dilate to get surrounding context
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        dilated = cv2.dilate(hole_mask, kernel)
        
        # Context = dilated region minus hole
        context_mask = dilated - hole_mask
        
        if context_mask.sum() < 10:
            # Not enough context, skip
            continue
        
        # Get depth values in context
        context_depths = depth_np[context_mask > 0]
        
        if len(context_depths) == 0:
            continue
        
        # Use median depth of surrounding context
        fill_value = np.median(context_depths)
        
        # Fill hole with this value
        result[hole_mask > 0] = fill_value
    
    # Final inpainting to smooth transitions
    final_mask = ((np.abs(result) < 0.01) | (~np.isfinite(result))).astype(np.uint8)
    if final_mask.sum() > 0:
        result = cv2.inpaint(result, final_mask, 7, cv2.INPAINT_TELEA)
    
    return torch.from_numpy(result).unsqueeze(0).unsqueeze(0).to(depth.device)


def detect_large_foreground_holes(depth, fg_mask, area_threshold=1000, total_threshold=600):
    """
    Detect catastrophic failures using TWO criteria:
    1. Largest single hole > area_threshold (e.g., 300 pixels)
    2. Total holes in foreground > total_threshold (e.g., 600 pixels)
    
    Returns: (is_catastrophic, total_holes, largest_hole)
    """
    fg_region = (fg_mask > 0.5).float()
    
    if fg_region.sum() < 100:
        return False, 0, 0  # Return (is_catastrophic, total_holes, largest_hole)
    
    depth_np = depth.squeeze().cpu().numpy()
    fg_np = fg_region.squeeze().cpu().numpy()
    
    holes_in_fg = ((np.abs(depth_np) < 0.01) | (~np.isfinite(depth_np))) & (fg_np > 0.5)
    
    total_hole_pixels = int(holes_in_fg.sum())
    
    if total_hole_pixels == 0:
        return False, 0, 0
    
    # Find largest hole
    holes_mask = holes_in_fg.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_mask, connectivity=8)
    
    largest_hole = 0
    if num_labels > 1:
        largest_hole = max([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)])
    
    # Check both criteria
    is_catastrophic = (total_hole_pixels > total_threshold) or (largest_hole > area_threshold)
    
    return is_catastrophic, total_hole_pixels, largest_hole


def temporal_gap_filling(all_depths, gap_indices, window=3):
    """
    Fill catastrophic gaps using temporal interpolation from neighboring frames.
    
    Args:
        all_depths: List of depth maps [H, W]
        gap_indices: List of frame indices with catastrophic failures
        window: Number of frames before/after to use for interpolation
        
    Returns:
        all_depths: List with gaps filled
    """
    if len(gap_indices) == 0:
        return all_depths
    
    print(f"\n🔧 Filling {len(gap_indices)} catastrophic gaps: {gap_indices}")
    
    for gap_idx in gap_indices:
        # Find valid neighboring frames
        before_frames = []
        after_frames = []
        
        # Look before
        for i in range(max(0, gap_idx - window), gap_idx):
            if i not in gap_indices:
                before_frames.append(i)
        
        # Look after
        for i in range(gap_idx + 1, min(len(all_depths), gap_idx + window + 1)):
            if i not in gap_indices:
                after_frames.append(i)
        
        if len(before_frames) == 0 and len(after_frames) == 0:
            print(f"   ⚠️ Frame {gap_idx}: No valid neighbors, skipping")
            continue
        
        # Collect valid depths
        valid_depths = []
        for i in before_frames + after_frames:
            valid_depths.append(all_depths[i])
        
        if len(valid_depths) == 0:
            continue
        
        # Use median of valid neighbors
        valid_stack = np.stack(valid_depths, axis=0)  # [N, H, W]
        filled_depth = np.median(valid_stack, axis=0)  # [H, W]
        
        # Replace gap frame
        all_depths[gap_idx] = filled_depth.astype(np.float16)
        print(f"   ✅ Frame {gap_idx}: Filled from {len(valid_depths)} neighbors")
    
    return all_depths


# =============================================================================
# SCENE CUT DETECTION
# =============================================================================

def detect_scene_cut(frame1, frame2, threshold=30.0):
    """Detect scene cuts using histogram difference."""
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
    """Main depth extraction function."""
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
    
    print(f"\n🎥 Extracting depth from: {video_path.stem}")
    print(f"📊 Video: {width}x{height} @ {fps}fps, {frame_count} frames")
    print(f"🔧 Config: photo_thresh={args.photo_threshold}, fb_thresh={args.fb_threshold}")
    print(f"🔧 Foreground: percentile={args.fg_percentile}, blend={args.fg_blend}")
    
    # Storage
    all_depths = []
    scene_cuts = []
    alignment_stats = []
    catastrophic_gaps = []  # Track frames with total failures
    
    # State
    prev_frame = None
    prev_depth_stable = None
    prev_frame_tensor = None
    
    # Global tracking
    global_min = float('inf')
    global_max = float('-inf')
    
    pbar = tqdm(total=frame_count, desc="Extracting Depth")
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Scene cut detection
            is_scene_cut = False
            if prev_frame is not None and args.scene_threshold > 0:
                is_scene_cut = detect_scene_cut(prev_frame, frame_rgb, args.scene_threshold)
                if is_scene_cut:
                    scene_cuts.append(frame_idx)
            
            # Get raw depth
            with torch.no_grad():
                depth_raw = depth_model.infer_image(frame_rgb, args.input_size)
                depth_raw = torch.from_numpy(depth_raw).float().to(device)
                depth_raw = depth_raw.unsqueeze(0).unsqueeze(0)
            
            # Convert frame to tensor
            curr_frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(device)
            curr_frame_tensor = curr_frame_tensor.unsqueeze(0) / 255.0
            
            # Temporal processing
            if prev_frame is not None and prev_depth_stable is not None and prev_frame_tensor is not None and not is_scene_cut:
                # Compute optical flow
                flow_bwd = compute_optical_flow(raft_model, frame_rgb, prev_frame, device, iters=args.flow_iters)
                flow_fwd = compute_optical_flow(raft_model, prev_frame, frame_rgb, device, iters=args.flow_iters)
                
                # Warp previous depth and RGB
                prev_depth_warped, valid_mask = warp_depth_gpu(prev_depth_stable, flow_bwd)
                prev_rgb_warped, _ = warp_rgb_gpu(prev_frame_tensor, flow_bwd)
                
                # Compute reliability
                R_photo = compute_photometric_reliability(
                    curr_frame_tensor, prev_rgb_warped, 
                    threshold=args.photo_threshold
                )
                
                R_fb = compute_fb_consistency(flow_bwd, flow_fwd, threshold=args.fb_threshold)
                
                # Valid warp check
                invalid_warp_guard = (prev_depth_warped > args.min_depth_value).float()
                invalid_warp_guard = invalid_warp_guard * torch.isfinite(prev_depth_warped).float()
                valid_mask = valid_mask * invalid_warp_guard
                
                # Combine reliability
                R_raw = combine_reliability_mask(valid_mask, R_photo, R_fb)
                
                # Morphology
                if args.morph_kernel_size > 0:
                    R_smooth = apply_opening(R_raw, kernel_size=args.morph_kernel_size)
                    
                    if args.apply_light_dilation:
                        R_smooth = apply_light_dilation(R_smooth, kernel_size=args.morph_kernel_size)
                    
                    R_smooth = R_smooth * valid_mask * invalid_warp_guard
                    R_smooth = (R_smooth > 0.5).float()
                else:
                    R_smooth = R_raw
                
                # Alignment
                curr_aligned, align_info = align_depth_robust(
                    depth_raw, prev_depth_warped, R_smooth,
                    max_scale_diff=args.max_scale_diff,
                    max_shift_diff=args.max_shift_diff,
                    min_valid_pixels=args.min_valid_pixels
                )
                alignment_stats.append(align_info)
                
                # Foreground detection
                fg_mask = detect_foreground_regions(
                    curr_aligned, 
                    percentile=args.fg_percentile
                )
                
                # Hybrid fusion
                depth_stable, fusion_weight = fuse_depths_hybrid(
                    curr_aligned, prev_depth_warped, R_smooth,
                    fg_mask,
                    kernel_size=args.fusion_kernel_size,
                    fg_blend=args.fg_blend
                )
                
                # NEW: Interpolate small holes FIRST (uses surrounding valid depth)
                # This fills small holes with correct surrounding values before aggressive filling
                if args.interpolate_holes:
                    depth_stable = fill_holes_by_interpolation(
                        depth_stable, 
                        max_hole_size=200  # Only fill very small holes here
                    )
                
                # THEN detect gaps (for truly large failures)
                if args.detect_gaps:
                    is_catastrophic, total_holes, largest_hole = detect_large_foreground_holes(
                        depth_stable, fg_mask, 
                        area_threshold=args.gap_area_threshold,
                        total_threshold=args.gap_total_threshold
                    )
                    
                    # LOG EVERY FRAME (not just ones with holes > 50)
                    if total_holes > 0:
                        status = "CATASTROPHIC" if is_catastrophic else "ok"
                        print(f"Frame {frame_idx:3d}: {total_holes:4d} total, largest={largest_hole:3d} [{status}]")
                    
                    if is_catastrophic:
                        catastrophic_gaps.append(frame_idx)
                        # Don't update prev_depth_stable
                    else:
                        prev_depth_stable = depth_stable.clone()
                else:
                    prev_depth_stable = depth_stable.clone()
                
                # NOW do hole filling (after gap detection)
                if args.fill_fg_holes:
                    depth_stable = fill_foreground_holes_aggressive(
                        depth_stable, curr_aligned, prev_depth_warped,
                        R_smooth, fg_mask,
                        fg_hole_threshold=args.fg_hole_threshold
                    )
                
                # Then smoothing
                if args.bilateral_smooth:
                    depth_stable = bilateral_smooth_depth(
                        depth_stable, 
                        sigma_spatial=args.bilateral_spatial,
                        sigma_intensity=args.bilateral_intensity
                    )
                
                # Inpaint remaining tiny holes
                if args.inpaint_holes:
                    depth_stable = inpaint_small_holes(depth_stable, max_hole_size=args.max_hole_size)
                
            else:
                # First frame or scene cut
                depth_stable = depth_raw
                prev_depth_stable = None  # Reset for scene cut, will be set below for first frame
                if frame_idx > 0:
                    alignment_stats.append({'status': 'scene_cut' if is_scene_cut else 'first_frame'})
            
            # DON'T call interpolate again at the end (already done early in pipeline)
            
            # Store depth
            depth_np = depth_stable.squeeze().cpu().numpy()
            all_depths.append(depth_np.astype(np.float16))
            
            # Update global stats
            global_min = min(global_min, depth_np.min())
            global_max = max(global_max, depth_np.max())
            
            # Update state
            prev_frame = frame_rgb.copy()
            # prev_depth_stable is updated in the temporal processing section above
            # (or set to None for first frame/scene cut)
            if prev_depth_stable is None:
                prev_depth_stable = depth_stable.clone()
            prev_frame_tensor = curr_frame_tensor.clone()
            
            frame_idx += 1
            pbar.update(1)
            
    finally:
        pbar.close()
        cap.release()
    
    print(f"📊 Scene cuts detected: {len(scene_cuts)}")
    
    # Post-process to fill catastrophic gaps using temporal interpolation
    if args.fill_gaps and len(catastrophic_gaps) > 0:
        all_depths = temporal_gap_filling(
            all_depths, catastrophic_gaps, 
            window=args.gap_window
        )
    
    # Save output
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
            'version': 'v2_final_production',
            'model_encoder': args.encoder,
            'photo_threshold': args.photo_threshold,
            'fb_threshold': args.fb_threshold,
            'fg_percentile': args.fg_percentile,
            'fg_blend': args.fg_blend,
            'fg_hole_threshold': args.fg_hole_threshold,
            'fill_fg_holes': args.fill_fg_holes,
            'bilateral_smooth': args.bilateral_smooth,
            'inpaint_holes': args.inpaint_holes,
            'max_hole_size': args.max_hole_size,
            'interpolate_holes': args.interpolate_holes,
            'max_interpolate_hole_size': args.max_interpolate_hole_size,
            'detect_gaps': args.detect_gaps,
            'gap_area_threshold': args.gap_area_threshold,
            'gap_total_threshold': args.gap_total_threshold,
            'fill_gaps': args.fill_gaps,
            'gap_window': args.gap_window,
        })
    }
    
    print(f"\n💾 Saving to: {output_path}")
    np.savez_compressed(str(output_path), **save_dict)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved! File size: {file_size_mb:.1f} MB")
    print(f"   Frames: {len(all_depths)}")
    print(f"   Depth range: {global_min:.4f} - {global_max:.4f}")
    
    # Print alignment statistics
    if alignment_stats:
        scales = [s['scale'] for s in alignment_stats if s.get('status') == 'aligned']
        shifts = [s['shift'] for s in alignment_stats if s.get('status') == 'aligned']
        if scales:
            print(f"\n📊 Alignment Statistics:")
            print(f"   Scale: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}")
            print(f"   Shift: mean={np.mean(shifts):.2f}, std={np.std(shifts):.2f}")
    
    print("\n✅ Extraction complete!")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Depth Extraction V2 Final - Production Ready",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    # Model settings
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--input-size', type=int, default=518)
    
    # Reliability parameters
    parser.add_argument('--photo-threshold', type=float, default=0.25,
                        help='Photometric error threshold (default: 0.25, looser)')
    parser.add_argument('--fb-threshold', type=float, default=4.0,
                        help='Forward-backward consistency threshold (default: 4.0, looser)')
    
    # Alignment parameters
    parser.add_argument('--max-scale-diff', type=float, default=0.1)
    parser.add_argument('--max-shift-diff', type=float, default=0.1)
    parser.add_argument('--min-valid-pixels', type=int, default=100)
    
    # Morphology
    parser.add_argument('--morph-kernel-size', type=int, default=7,
                        help='Morphological kernel size (default: 7, larger)')
    parser.add_argument('--apply-light-dilation', action='store_true')
    
    # Foreground/Fusion
    parser.add_argument('--fg-percentile', type=float, default=60,
                        help='Foreground depth percentile (default: 60)')
    parser.add_argument('--fg-blend', type=float, default=0.15,
                        help='History blend for foreground (default: 0.15 = 15%%)')
    parser.add_argument('--fusion-kernel-size', type=int, default=15)
    
    # Hole filling
    parser.add_argument('--fill-fg-holes', action='store_true',
                        help='Fill foreground holes aggressively')
    parser.add_argument('--fg-hole-threshold', type=float, default=0.7,
                        help='Threshold for detecting foreground holes (default: 0.7, very aggressive)')
    
    # Bilateral smoothing
    parser.add_argument('--bilateral-smooth', action='store_true')
    parser.add_argument('--bilateral-spatial', type=float, default=5)
    parser.add_argument('--bilateral-intensity', type=float, default=0.08)
    
    # Inpainting
    parser.add_argument('--inpaint-holes', action='store_true',
                        help='Inpaint remaining small holes')
    parser.add_argument('--max-hole-size', type=int, default=200,
                        help='Maximum hole size for inpainting (default: 200 pixels)')
    
    # Interpolation (nuclear option)
    parser.add_argument('--interpolate-holes', action='store_true',
                        help='Interpolate holes from surrounding valid depth (nuclear option)')
    parser.add_argument('--max-interpolate-hole-size', type=int, default=500,
                        help='Maximum hole size for interpolation (default: 500 pixels)')
    
    # Gap detection and filling
    parser.add_argument('--detect-gaps', action='store_true',
                        help='Detect catastrophic failures (entire person missing)')
    parser.add_argument('--gap-area-threshold', type=int, default=1000,
                        help='Minimum hole area to consider catastrophic (default: 1000 pixels)')
    parser.add_argument('--gap-total-threshold', type=int, default=600,
                        help='Total hole pixels to consider catastrophic (default: 600)')
    parser.add_argument('--fill-gaps', action='store_true',
                        help='Fill catastrophic gaps using temporal interpolation')
    parser.add_argument('--gap-window', type=int, default=3,
                        help='Number of frames before/after to use for gap filling (default: 3)')
    
    # Temporal
    parser.add_argument('--flow-iters', type=int, default=20)
    parser.add_argument('--min-depth-value', type=float, default=1e-6)
    parser.add_argument('--scene-threshold', type=float, default=30.0)
    
    args = parser.parse_args()
    extract_depth(args)


if __name__ == "__main__":
    main()
