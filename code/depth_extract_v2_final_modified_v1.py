"""
Depth Analysis Tool (Part 3 of 3)

This script performs statistical analysis on the extracted depth data (.npz).
It helps quantify "flickering" and validates the temporal consistency algorithm.

================================================================================
FEATURES
================================================================================

1. Chaos Map: A heatmap showing temporal standard deviation per pixel.
   - Dark = Stable pixel (good)
   - Bright = Flickering pixel (bad)

2. Pixel Seismograph: A graph showing depth values of a single pixel over time.
   - Overlays trust factor if available
   - Marks scene cuts with vertical lines

3. Summary Statistics: Quantitative metrics for overall stability.
   - Mean/Max/P95 standard deviation
   - Frame-to-frame delta analysis
   - Worst pixel identification

4. Multi-Model Comparison: Compare up to 5 different .npz files on the same plot.
   - Useful for comparing naive vs temporal, or different parameter settings
   - Each model shown in different color

================================================================================
USAGE
================================================================================

# Basic analysis (center pixel)
python depth_analyze.py --npz depth_data.npz

# Analyze specific pixel
python depth_analyze.py --npz depth_data.npz -x 500 -y 300

# Save plots instead of displaying
python depth_analyze.py --npz depth_data.npz --save analysis_output.png

# Analyze the most unstable pixel automatically
python depth_analyze.py --npz depth_data.npz --worst-pixel

# Compare up to 5 different depth maps (e.g., naive vs temporal)
python depth_analyze.py --compare-models naive.npz temporal.npz -x 500 -y 300

================================================================================
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def print_summary_stats(depths, std_map, scene_cuts):
    """Print quantitative stability metrics."""
    print("\n" + "=" * 60)
    print("📊 STABILITY SUMMARY")
    print("=" * 60)
    
    # Std dev stats
    mean_std = np.mean(std_map)
    max_std = np.max(std_map)
    min_std = np.min(std_map)
    p50_std = np.percentile(std_map, 50)
    p95_std = np.percentile(std_map, 95)
    p99_std = np.percentile(std_map, 99)
    
    print(f"\n🎯 Temporal Standard Deviation (lower = more stable):")
    print(f"   Mean:  {mean_std:.4f}")
    print(f"   Min:   {min_std:.4f}")
    print(f"   P50:   {p50_std:.4f}")
    print(f"   P95:   {p95_std:.4f}")
    print(f"   P99:   {p99_std:.4f}")
    print(f"   Max:   {max_std:.4f}")
    
    # Frame-to-frame delta
    depths_f32 = depths.astype(np.float32)
    frame_deltas = np.abs(np.diff(depths_f32, axis=0))
    mean_delta = np.mean(frame_deltas)
    max_delta = np.max(frame_deltas)
    p95_delta = np.percentile(frame_deltas, 95)
    
    print(f"\n📈 Frame-to-Frame Delta (depth change between consecutive frames):")
    print(f"   Mean:  {mean_delta:.4f}")
    print(f"   P95:   {p95_delta:.4f}")
    print(f"   Max:   {max_delta:.4f}")
    
    # Worst pixel
    worst_y, worst_x = np.unravel_index(np.argmax(std_map), std_map.shape)
    print(f"\n⚠️  Most Unstable Pixel: ({worst_x}, {worst_y}) with std={max_std:.4f}")
    
    # Scene cuts
    if scene_cuts is not None and scene_cuts.size > 0:
        print(f"\n🎬 Scene Cuts: {len(scene_cuts)} detected at frames {list(scene_cuts)}")
    else:
        print(f"\n🎬 Scene Cuts: None detected")
    
    print("=" * 60 + "\n")
    
    return worst_x, worst_y


def analyze_stability(npz_path, x=None, y=None, save_path=None, use_worst_pixel=False):
    """
    Analyze depth stability from extracted .npz file.
    
    Args:
        npz_path: Path to the .npz file
        x, y: Pixel coordinates to analyze (default: center)
        save_path: If provided, save plots to this path instead of displaying
        use_worst_pixel: If True, analyze the most unstable pixel
    """
    npz_path = Path(npz_path)
    print(f"🔍 Loading data from: {npz_path.name}...")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Load main depth data
    depths = data['depths']  # Shape: (Frames, Height, Width)
    frames, h, w = depths.shape
    print(f"📊 Data loaded: {frames} frames, {w}x{h} resolution")

    # Load optional data fields
    trust = None
    if 'trust_maps' in data:
        trust = data['trust_maps']
        print("✅ Trust maps found. Will overlay on graph.")
    else:
        print("ℹ️  No trust maps found (graph will show depth only).")

    scene_cuts = None
    if 'scene_cuts' in data:
        scene_cuts = data['scene_cuts']

    # Load config if available
    if 'config_json' in data:
        config = json.loads(str(data['config_json']))
        print(f"⚙️  Config: encoder={config.get('model_encoder', 'N/A')}, "
              f"flow_thresh={config.get('flow_threshold', 'N/A')}")

    # =========================================================================
    # ANALYSIS 1: Temporal Standard Deviation ("The Chaos Map")
    # =========================================================================
    print("\n📉 Calculating temporal stability map...")
    
    # Calculate Std Dev across time axis (Axis 0).
    # High Value = Unstable/Flickering pixel.
    # Low Value = Stable/Static pixel.
    # We cast to float32 to avoid overflow if input is float16.
    std_map = np.std(depths.astype(np.float32), axis=0)
    
    # Print summary statistics and get worst pixel
    worst_x, worst_y = print_summary_stats(depths, std_map, scene_cuts)
    
    # =========================================================================
    # DETERMINE PIXEL TO ANALYZE
    # =========================================================================
    if use_worst_pixel:
        x, y = worst_x, worst_y
        print(f"🎯 Analyzing worst pixel: ({x}, {y})")
    else:
        # Default to center pixel if coordinates not provided
        if x is None:
            x = w // 2
        if y is None:
            y = h // 2
        print(f"🎯 Analyzing pixel: ({x}, {y})")

    # Validate coordinates
    if not (0 <= x < w and 0 <= y < h):
        print(f"❌ Error: Coordinates ({x},{y}) are out of bounds.")
        return

    # =========================================================================
    # ANALYSIS 2: Single Pixel Trace ("The Seismograph")
    # =========================================================================
    print(f"📈 Extracting trace for pixel ({x}, {y})...")
    
    pixel_depth_trace = depths[:, y, x].astype(np.float32)
    pixel_trust_trace = trust[:, y, x].astype(np.float32) if trust is not None else None
    pixel_std = std_map[y, x]
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax1, ax2 = axes
    
    fig.suptitle(f'Depth Stability Analysis - {npz_path.name}', fontsize=14, fontweight='bold')

    # --- Plot 1: The Chaos Map ---
    # We use 'magma' colormap: Black/Purple = Stable, Orange/White = Unstable
    im = ax1.imshow(std_map, cmap='magma', interpolation='nearest')
    ax1.set_title('Temporal Instability Map (Std Dev)\nDarker = More Stable', fontsize=12)
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    
    # Mark the selected pixel with crosshairs
    ax1.axhline(y=y, color='cyan', linewidth=0.5, alpha=0.7)
    ax1.axvline(x=x, color='cyan', linewidth=0.5, alpha=0.7)
    ax1.plot(x, y, 'c+', markersize=15, markeredgewidth=2)
    ax1.plot(x, y, 'co', markersize=8, markerfacecolor='none', markeredgewidth=1.5)
    
    # Add annotation
    ax1.annotate(f'({x},{y})\nσ={pixel_std:.2f}', 
                 xy=(x, y), xytext=(x + 50, y + 50),
                 fontsize=9, color='white',
                 arrowprops=dict(arrowstyle='->', color='white', lw=1),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Depth Std Dev', shrink=0.8)
    
    # --- Plot 2: The Seismograph ---
    time_axis = np.arange(frames)
    ax2.set_title(f'Pixel ({x},{y}) Depth Over Time  |  σ = {pixel_std:.4f}', fontsize=12)
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Depth Value', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    
    # Plot Depth Line (Orange)
    line1, = ax2.plot(time_axis, pixel_depth_trace, color='#ff7f0e', 
                      label='Depth Value', linewidth=1.5, alpha=0.9)
    
    lines = [line1]
    
    # Overlay Scene Cuts (Vertical Red Lines)
    if scene_cuts is not None and scene_cuts.size > 0:
        for cut_frame in scene_cuts:
            if 0 <= cut_frame < frames:
                ax2.axvline(x=cut_frame, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        # Add dummy line for legend
        line_cut = ax2.axvline(x=-10, color='red', linestyle='--', label='Scene Cut', alpha=0.7)
        lines.append(line_cut)

    # Overlay Trust (Cyan Line on Secondary Axis)
    if pixel_trust_trace is not None:
        ax2_twin = ax2.twinx()
        line2, = ax2_twin.plot(time_axis, pixel_trust_trace, color='#17becf', 
                               label='Trust Factor', alpha=0.6, linestyle='-', linewidth=1)
        
        ax2_twin.set_ylabel('Trust (1.0=Raw, 0.0=History)', color='#17becf')
        ax2_twin.tick_params(axis='y', labelcolor='#17becf')
        ax2_twin.set_ylim(-0.05, 1.05)  # Slight padding
        
        lines.append(line2)

    # Unified Legend
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=9)
    
    # Grid for easier reading
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_xlim(0, frames - 1)

    plt.tight_layout()
    
    # Save or display
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Plot saved to: {save_path}")
    else:
        print("✅ Displaying plots... (close window to exit)")
        plt.show()
    
    plt.close()
    
    return {
        'std_map': std_map,
        'pixel_trace': pixel_depth_trace,
        'pixel_std': pixel_std,
        'worst_pixel': (worst_x, worst_y),
    }


def compare_pixels(npz_path, pixels, save_path=None):
    """
    Compare multiple pixels on the same graph.
    
    Args:
        npz_path: Path to .npz file
        pixels: List of (x, y) tuples
        save_path: Optional path to save the plot
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    depths = data['depths'].astype(np.float32)
    frames, h, w = depths.shape
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f'Multi-Pixel Comparison - {npz_path.name}', fontsize=14)
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Depth Value')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(pixels)))
    
    for i, (x, y) in enumerate(pixels):
        if 0 <= x < w and 0 <= y < h:
            trace = depths[:, y, x]
            std = np.std(trace)
            ax.plot(trace, color=colors[i], label=f'({x},{y}) σ={std:.2f}', alpha=0.8)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def parse_label_from_filename(npz_path, config=None):
    """
    Parse informative label from .npz filename.
    
    Examples:
        100_years_technion_d_v7_m200_d10_scaleOnly_a20_s10.npz -> "V7 m200 d10 so a20 s10"
        100_years_technion_a_v6_r1_a60_s10.npz -> "V6 r1 a60 s10"
        100_years_technion_a_v5_r1_a6_s10.npz -> "V5 r1 a6 s10"
        100_years_technion_a_depth_naive.npz -> "Naive"
    
    Args:
        npz_path: Path to the .npz file
        config: Optional config dict from the .npz file
    
    Returns:
        Informative label string
    """
    import re
    
    stem = Path(npz_path).stem
    
    # Check for naive
    if 'naive' in stem.lower():
        return "Naive"
    
    # Try to parse V7 pattern: _v7_m{maxpts}_d{dist}_{mode}_a{anchor}_s{scale}
    # Pattern: _v7_m200_d10_scaleOnly_a20_s10 or _v7_m500_d20_r0_a20_s10
    v7_pattern = r'_v7_m(\w+)_d(\d+)(?:_(scaleOnly|r[01]))_a(\d+)_s(\d+)'
    v7_match = re.search(v7_pattern, stem)
    
    if v7_match:
        max_points = v7_match.group(1)
        min_dist = v7_match.group(2)
        mode = v7_match.group(3)
        anchor = v7_match.group(4)
        scale = v7_match.group(5)
        
        # Format mode: "scaleOnly" -> "so", "r0" -> "", "r1" -> "r1"
        if mode == "scaleOnly":
            mode_str = "so"
        elif mode == "r1":
            mode_str = "r1"
        else:
            mode_str = ""
        
        # Build label
        parts = ["V7", f"m{max_points}", f"d{min_dist}"]
        if mode_str:
            parts.append(mode_str)
        parts.extend([f"a{anchor}", f"s{scale}"])
        return " ".join(parts)
    
    # Try to parse V9 pattern: _v9_r{0|1}_s{scale} or _v9_r1_s10_t{thresh} or _v9_r1_s10_mp{N} or _v9_a{anchor_pct}
    v9_pattern = r'_v9_r(\d)_s(\d+)(?:_t(\d+))?(?:_mp(\d+))?(?:_a(\d+))?'
    v9_match = re.search(v9_pattern, stem)
    if v9_match:
        rot = v9_match.group(1)
        scale = v9_match.group(2)
        thresh = v9_match.group(3)
        mp = v9_match.group(4)
        anchor = v9_match.group(5)
        label = f"V9 r{rot} s{scale}"
        if thresh:
            label += f" t{thresh}"
        if mp:
            label += f" mp{mp}"
        if anchor:
            label += f" a{anchor}"
        return label
    # V9 with anchor-percent only (e.g. _v9_a20)
    v9_simple = re.search(r'_v9_a(\d+)', stem)
    if v9_simple:
        return f"V9 a{v9_simple.group(1)}"
    
    # Try to parse V5/V6 pattern: _v{N}_r{0|1}_a{anchor}_s{scale}
    # Pattern: _v6_r1_a60_s10 or _v5_r0_a20_s15
    version_pattern = r'_v(\d+)_r(\d)_a(\d+)_s(\d+)'
    match = re.search(version_pattern, stem)
    
    if match:
        version = match.group(1)
        rotation = match.group(2)
        anchor = match.group(3)
        scale = match.group(4)
        return f"V{version} r{rotation} a{anchor} s{scale}"
    
    # Try simpler version pattern: just _v{N}
    simple_version = re.search(r'_v(\d+)', stem)
    if simple_version:
        version = simple_version.group(1)
        # Try to get params from config
        if config:
            parts = [f"V{version}"]
            if config.get('use_rotation'):
                parts.append("r1")
            anchor_pct = config.get('anchor_percent')
            if anchor_pct:
                parts.append(f"a{int(anchor_pct * 100)}")
            max_scale = config.get('max_scale_diff')
            if max_scale:
                parts.append(f"s{int(max_scale * 100)}")
            if len(parts) > 1:
                return " ".join(parts)
        return f"V{version}"
    
    # Fallback: use stem (truncated)
    return stem[:30] if len(stem) > 30 else stem


def compare_models(npz_paths, x=None, y=None, save_path=None, labels=None, 
                   show_delta=False, zoom_range=None):
    """
    Compare depth traces from multiple .npz files on the same plot.
    
    Focused on a clean visualization of pixel depth over time.
    
    Args:
        npz_paths: List of paths to .npz files (max 5)
        x, y: Pixel coordinates to analyze (default: center)
        save_path: If provided, save plots to this path instead of displaying
        labels: Optional list of labels for each model
        show_delta: If True, also show frame-to-frame delta plot
        zoom_range: Optional tuple (start_frame, end_frame) for zoomed view
    """
    if len(npz_paths) > 10:
        print(f"⚠️  Maximum 10 models supported. Using first 10.")
        npz_paths = npz_paths[:10]
    
    if len(npz_paths) < 2:
        print(f"❌ Need at least 2 .npz files to compare.")
        return
    
    # Define distinct colors for each model (up to 10)
    model_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue  
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#999999',  # Gray
        '#17becf',  # Cyan
        '#000000',  # Black
    ]
    
    # Load all data
    datasets = []
    for i, npz_path in enumerate(npz_paths):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            print(f"❌ File not found: {npz_path}")
            return
        
        print(f"🔍 Loading [{i+1}/{len(npz_paths)}]: {npz_path.name}...")
        data = np.load(npz_path, allow_pickle=True)
        
        # Support both 'depths' and 'depth' keys
        if 'depths' in data:
            depths = data['depths'].astype(np.float32)
        elif 'depth' in data:
            depths = data['depth'].astype(np.float32)
        else:
            print(f"❌ No depth data found in {npz_path.name}")
            return
        
        # Load config
        config = {}
        if 'config_json' in data:
            config = json.loads(str(data['config_json']))
        
        # Get label - prefer user-provided, then auto-parse from filename
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = parse_label_from_filename(npz_path, config)
        
        datasets.append({
            'path': npz_path,
            'depths': depths,
            'label': label,
            'color': model_colors[i],
            'config': config,
        })
    
    # Get dimensions from first file
    frames, h, w = datasets[0]['depths'].shape
    
    # Check all files have same dimensions
    for ds in datasets[1:]:
        f, dh, dw = ds['depths'].shape
        if (dh, dw) != (h, w):
            print(f"⚠️  Dimension mismatch: {ds['path'].name} is {dw}x{dh}, expected {w}x{h}")
        if f != frames:
            print(f"⚠️  Frame count mismatch: {ds['path'].name} has {f} frames, expected {frames}")
    
    # Default to center pixel
    if x is None:
        x = w // 2
    if y is None:
        y = h // 2
    
    # Validate coordinates
    if not (0 <= x < w and 0 <= y < h):
        print(f"❌ Error: Coordinates ({x},{y}) are out of bounds.")
        return
    
    print(f"\n🎯 Comparing models at pixel ({x}, {y})")
    
    # =========================================================================
    # CALCULATE STATS FOR EACH MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<35} {'Mean σ':<12} {'Pixel σ':<12} {'Mean Δ':<12}")
    print("-" * 70)
    
    for ds in datasets:
        depths = ds['depths']
        std_map = np.std(depths, axis=0)
        pixel_std = std_map[y, x]
        mean_std = np.mean(std_map)
        
        # Frame-to-frame delta
        frame_deltas = np.abs(np.diff(depths, axis=0))
        mean_delta = np.mean(frame_deltas)
        
        ds['std_map'] = std_map
        ds['pixel_std'] = pixel_std
        ds['mean_std'] = mean_std
        ds['mean_delta'] = mean_delta
        ds['trace'] = depths[:, y, x]
        
        print(f"{ds['label']:<35} {mean_std:<12.4f} {pixel_std:<12.4f} {mean_delta:<12.4f}")
    
    print("=" * 70)
    
    # =========================================================================
    # PLOTTING - Clean pixel depth over time visualization
    # =========================================================================
    
    # Determine number of rows based on options
    num_rows = 1
    if show_delta:
        num_rows += 1
    if zoom_range:
        num_rows += 1
    
    # Square aspect ratio with extra space for legend on the right
    fig, axes = plt.subplots(num_rows, 1, figsize=(12, 8 * num_rows))
    if num_rows == 1:
        axes = [axes]  # Make iterable
    
    fig.suptitle(f'Pixel ({x}, {y}) Depth Over Time', fontsize=14, fontweight='bold')
    
    # Distinct colors for each model (random/varied palette)
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#17becf',  # Cyan
        '#bcbd22',  # Olive
        '#7f7f7f',  # Gray
    ]
    
    def get_style_for_label(label, index):
        """Get color and linestyle based on model index."""
        label_lower = label.lower()
        
        # Naive is dark gray solid
        if 'naive' in label_lower:
            return '#404040', '-'
        
        # Determine linestyle from rotation
        linestyle = ':' if 'r0' in label_lower else '-'
        
        # Use index-based color
        color = distinct_colors[index % len(distinct_colors)]
        
        return color, linestyle
    
    time_axis = np.arange(frames)
    ax_idx = 0
    
    # --- Main Plot: Pixel Depth Traces ---
    ax1 = axes[ax_idx]
    ax1.set_title('Depth Value', fontsize=12)
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Depth Value')
    
    for i, ds in enumerate(datasets):
        color, linestyle = get_style_for_label(ds['label'], i)
        ax1.plot(time_axis, ds['trace'], color=color, 
                 linestyle=linestyle,
                 label=f"{ds['label']} (σ={ds['pixel_std']:.2f})", 
                 linewidth=1.0, alpha=0.9)
    
    # Legend outside plot on the right
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, frames - 1)
    ax_idx += 1
    
    # --- Optional: Frame-to-Frame Delta Plot ---
    if show_delta:
        ax_delta = axes[ax_idx]
        ax_delta.set_title('Frame-to-Frame Delta (Flickering Magnitude)', fontsize=12)
        ax_delta.set_xlabel('Frame Number')
        ax_delta.set_ylabel('|Δ Depth|')
        
        for i, ds in enumerate(datasets):
            delta = np.abs(np.diff(ds['trace']))
            mean_d = np.mean(delta)
            color, linestyle = get_style_for_label(ds['label'], i)
            ax_delta.plot(time_axis[1:], delta, color=color, 
                         linestyle=linestyle,
                         label=f"{ds['label']} (mean Δ={mean_d:.2f})", 
                         linewidth=0.8, alpha=0.8)
        
        ax_delta.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
        ax_delta.grid(True, alpha=0.3)
        ax_delta.set_xlim(0, frames - 1)
        ax_idx += 1
    
    # --- Optional: Zoomed Window ---
    if zoom_range:
        start_frame, end_frame = zoom_range
        start_frame = max(0, start_frame)
        end_frame = min(frames, end_frame)
        
        ax_zoom = axes[ax_idx]
        ax_zoom.set_title(f'Zoomed View: Frames {start_frame}-{end_frame}', fontsize=12)
        ax_zoom.set_xlabel('Frame Number')
        ax_zoom.set_ylabel('Depth Value')
        
        zoom_time = time_axis[start_frame:end_frame]
        for i, ds in enumerate(datasets):
            zoom_trace = ds['trace'][start_frame:end_frame]
            color, linestyle = get_style_for_label(ds['label'], i)
            ax_zoom.plot(zoom_time, zoom_trace, color=color, 
                        linestyle=linestyle,
                        label=ds['label'], linewidth=1.0, alpha=0.9, marker='o', markersize=2)
        
        ax_zoom.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_xlim(start_frame, end_frame - 1)
        ax_idx += 1
    
    # Adjust layout to make room for legend on the right
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Auto-generate save path if not provided
    if save_path is None:
        # Build clean filename from labels: V6_r1_a60_s10_vs_V5_r1_a6_s10_vs_Naive_x1000_y100.png
        def sanitize_label(label):
            """Convert label to filename-safe format."""
            return label.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        label_names = [sanitize_label(ds['label']) for ds in datasets]
        filename = f"{'_vs_'.join(label_names)}_x{x}_y{y}.png"
        
        # Save to data/depth_analysis/ directory (relative to project root)
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent  # code/scripts -> code -> project_root
        save_dir = project_root / "data" / "depth_analysis"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Comparison plot saved to: {save_path}")
    
    plt.close()
    
    return datasets


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze depth stability from .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  python depth_analyze.py --npz depth_data.npz
  python depth_analyze.py --npz depth_data.npz -x 500 -y 300
  python depth_analyze.py --npz depth_data.npz --worst-pixel
  python depth_analyze.py --npz depth_data.npz --save analysis.png
  
  # Compare multiple pixels in one file
  python depth_analyze.py --npz depth_data.npz --compare-pixels 100,200 500,300 960,540
  
  # Compare multiple models (up to 5 .npz files)
  python depth_analyze.py --compare-models naive.npz temporal.npz -x 500 -y 300
  python depth_analyze.py --compare-models model1.npz model2.npz model3.npz --save comparison.png
  python depth_analyze.py --compare-models naive.npz temporal.npz --labels "Naive" "Temporal"
        """
    )
    
    # Single file analysis
    parser.add_argument('--npz', type=str, default=None,
                        help='Path to the input .npz file from depth_extract.py')
    
    # Multi-model comparison (up to 5 files)
    parser.add_argument('--compare-models', nargs='+', type=str, default=None,
                        help='Compare up to 5 different .npz files on same plot')
    
    parser.add_argument('--labels', nargs='+', type=str, default=None,
                        help='Custom labels for each model in --compare-models')
    
    # Pixel coordinates
    parser.add_argument('-x', type=int, default=None, 
                        help='X coordinate of the pixel to analyze (default: center)')
    
    parser.add_argument('-y', type=int, default=None, 
                        help='Y coordinate of the pixel to analyze (default: center)')
    
    parser.add_argument('--worst-pixel', action='store_true',
                        help='Automatically analyze the most unstable pixel')
    
    parser.add_argument('--save', type=str, default=None,
                        help='Save plot to this path instead of displaying')
    
    parser.add_argument('--compare-pixels', nargs='+', type=str, default=None,
                        help='Compare multiple pixels in one file. Format: x1,y1 x2,y2 ...')
    
    parser.add_argument('--show-delta', action='store_true',
                        help='Show frame-to-frame delta plot (flickering magnitude)')
    
    parser.add_argument('--zoom', nargs=2, type=int, metavar=('START', 'END'),
                        help='Add zoomed view for frame range (e.g., --zoom 50 100)')
    
    args = parser.parse_args()
    
    # Mode 1: Multi-model comparison
    if args.compare_models:
        if len(args.compare_models) < 2:
            print("❌ Error: --compare-models requires at least 2 .npz files")
            return
        if len(args.compare_models) > 10:
            print("⚠️  Maximum 10 models supported. Using first 10.")
            args.compare_models = args.compare_models[:10]
        
        zoom_range = tuple(args.zoom) if args.zoom else None
        
        compare_models(
            args.compare_models,
            x=args.x,
            y=args.y,
            save_path=args.save,
            labels=args.labels,
            show_delta=args.show_delta,
            zoom_range=zoom_range
        )
        return
    
    # Require --npz for other modes
    if args.npz is None:
        print("❌ Error: --npz is required (or use --compare-models)")
        return
    
    # Validate input file
    if not Path(args.npz).exists():
        print(f"❌ Error: File not found: {args.npz}")
        return
    
    # Mode 2: Multi-pixel comparison (single file)
    if args.compare_pixels:
        pixels = []
        for coord in args.compare_pixels:
            try:
                x, y = map(int, coord.split(','))
                pixels.append((x, y))
            except ValueError:
                print(f"⚠️  Invalid coordinate format: {coord}. Use x,y format.")
        
        if pixels:
            compare_pixels(args.npz, pixels, args.save)
        return
    
    # Mode 3: Single pixel analysis
    analyze_stability(
        args.npz, 
        x=args.x, 
        y=args.y, 
        save_path=args.save,
        use_worst_pixel=args.worst_pixel
    )


if __name__ == "__main__":
    main()

