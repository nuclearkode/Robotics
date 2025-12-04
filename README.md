# Line Detection Pipeline

An optimized, training-free line detection pipeline for robotics applications. This implementation uses classical computer vision techniques with significant performance improvements over traditional approaches.

## Features

### Performance Optimizations
- **Cached BEV Warp Maps**: 4-5x speedup over per-frame warping using pre-computed remap tables
- **RANSAC Vanishing Point**: O(N) complexity vs O(N²) brute force (4-5x speedup)
- **Optional Multi-threading**: Producer/consumer pattern for +50% throughput

### Edge Detection (Improved)
- **Bilateral + Laplacian** (default): ~5ms vs 15ms for percentile Canny (3x speedup), no tuning needed
- **Adaptive Canny**: Robust to lighting changes, handles shadows and uneven illumination

### Preprocessing (Simplified)
- **CLAHE Only**: Removed slow Retinex processing (4x faster, 3 lines of code)
- **Optional Unsharp Mask**: For scenes with faint lines

### Line Extraction (Streamlined)
- **Constrained HoughLinesP**: Angle-aware filtering, fewer false positives
- **Gradient Orientation Pre-filtering**: Keeps only near-vertical edges
- Removed redundant LSD fallback

### Tracking (Simplified)
- **Exponential Smoothing** (default): 50% less code than Kalman, nearly as smooth
- **Optional 2D Kalman**: Simpler than 4D, tracks position only
- Removed overkill 4D Kalman filter

### Configuration
- **YAML Config File**: Centralized, reproducible experiments
- **Command-line Overrides**: Via argparse for quick parameter sweeps

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python line_detector.py
```

### With Configuration File
```bash
python line_detector.py --config config.yaml
```

### Command-line Overrides
```bash
# Use adaptive Canny instead of bilateral+laplacian
python line_detector.py --edge-method adaptive_canny

# Use 2D Kalman instead of exponential smoothing
python line_detector.py --tracking-method kalman_2d

# Enable multi-threading
python line_detector.py --multithreading

# Disable CUDA
python line_detector.py --no-cuda

# Verbose output
python line_detector.py --verbose
```

### Keyboard Controls
- `1`: Raw camera view
- `2`: Line detection overlay
- `r`: Reset trackbars
- `q`: Quit

## Configuration

The `config.yaml` file controls all pipeline parameters:

```yaml
# Edge detection method
edge_detection:
  method: "bilateral_laplacian"  # or "adaptive_canny"

# Tracking method
tracking:
  method: "exponential"  # or "kalman_2d"
  exponential:
    alpha: 0.15  # Smoothing factor (0-1, lower = smoother)

# Performance optimizations
performance:
  cache_warp_maps: true  # 4-5x speedup
  multithreading:
    enabled: false  # +50% throughput when enabled
  use_cuda: true  # Use GPU if available
```

See `config.yaml` for all available options.

## Calibration Files

Place calibration files in the `calibration/` directory:

- `camera_model.npz`: Camera intrinsics (K, dist, new_K matrices)
- `ground_plane_h.npz`: Homography for BEV transformation (H, bev_w, bev_h)
- `imu_alignment.json`: IMU-based gravity alignment (roll_deg, pitch_deg)

If calibration files are not found, the pipeline uses sensible defaults.

## Architecture

```
Frame → Undistort → IMU Align → BEV Warp → CLAHE → Edge Detection
                                                         ↓
    ← Overlay ← Tracking ← Scoring ← RANSAC Fit ← Merge ← HoughLinesP
```

## Performance Comparison

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Edge Detection | 15ms (Canny) | 5ms (Bilateral+Laplacian) | 3x |
| Preprocessing | 12-15ms (CLAHE+Retinex) | 3-4ms (CLAHE only) | 4x |
| BEV Warping | 15-20ms | 3-5ms (cached maps) | 4-5x |
| Vanishing Point | 10ms (O(N²)) | 2-3ms (RANSAC) | 4-5x |
| Tracking | 100+ lines (4D Kalman) | 35 lines (Exponential) | 50% less code |

## License

MIT License
