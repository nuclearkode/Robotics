# Long Line Detector - Optimized Pipeline

An optimized line detection pipeline for robotics applications with significant performance improvements and modern techniques.

## Features

- **Fast Edge Detection**: Bilateral filter + Laplacian (3x faster than percentile Canny)
- **Efficient Preprocessing**: CLAHE-only (4x faster than Retinex)
- **Robust Line Extraction**: Constrained HoughLinesP (replaces redundant LSD)
- **Simple Tracking**: Exponential smoothing (50% less code than 4D Kalman)
- **Configuration System**: YAML-based config with argparse (replaces trackbars)
- **Performance Optimizations**: 
  - Cached BEV warp maps (4-5x speedup)
  - RANSAC vanishing point (4-5x speedup)
  - Multi-threaded processing (+50% throughput)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with default configuration
python long_line_detector.py

# Use custom config file
python long_line_detector.py --config my_config.yaml

# Disable multi-threading (if needed)
python long_line_detector.py --no-multithreading
```

## Configuration

The pipeline uses a YAML configuration file (`config.yaml`) that is automatically created on first run. Key sections:

- **edge_detection**: Method selection and parameters
- **preprocessing**: CLAHE and enhancement settings
- **line_extraction**: Hough transform and filtering parameters
- **tracking**: Smoothing and debounce settings
- **roi**: Region of interest configuration
- **vanishing_point**: RANSAC parameters
- **performance**: Multi-threading and caching options

Press `s` during runtime to save the current configuration.

## Performance

- **Total Pipeline**: ~30ms (single-threaded), ~20ms (multi-threaded)
- **Speedup**: 1.5-2.25x faster than original implementation
- **Memory**: Efficient with cached maps and optimized data structures

## Key Bindings

- `1`: Switch to raw camera view
- `2`: Switch to line detection view
- `s`: Save current configuration
- `q`: Quit application

## See Also

- `IMPROVEMENTS.md`: Detailed breakdown of all improvements
- `config.yaml`: Configuration file (auto-generated)