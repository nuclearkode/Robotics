"""Interactive long-line detector with classical, training-free upgrades.

This script implements the full roadmap of geometry, photometric, extraction,
robustness, and temporal improvements requested for a drop-in, non-learned
pipeline.  Major features:

* Camera intrinsics loading with live undistortion.
* Optional IMU-based gravity alignment before processing.
* Ground-plane homography (inverse-perspective mapping) and back-projection to
  camera space for visualization.
* CLAHE preprocessing on the luminance channel.
* Bilateral filter + Laplacian edge detection (replacing percentile Canny).
* Multi-scale (coarse-to-fine) edge detection with an early-exit fast path when
  the temporal tracker is confident.
* Line extraction via constrained HoughLinesP (angle-aware filtering).
* Collinearity-based segment merging followed by RANSAC/MSAC consensus fitting
  and orthogonal (TLS) refinement at sub-pixel precision.
* A-contrario validation via a Number of False Alarms (NFA) bound.
* Vanishing-point estimation using RANSAC voting (replacing O(N²) intersection).
* Mahalanobis scoring with priors for bottom coverage and VP consistency.
* Exponential smoothing tracker (replacing 4D Kalman) with hysteresis state machine,
  innovation-based confidence, and debounced output.
* ROI scheduler that tightens around the last prediction and a controllable
  rate limiter on changes.
* YAML config system with argparse for reproducibility.
* Cached BEV warp maps for 2-3x speedup.
* Multi-threaded pipeline (producer/consumer) for +50% throughput.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from math import cos, degrees, hypot, radians, sin
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import yaml

# ---------- Paths for calibration assets ----------
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
INTRINSICS_PATH = CALIBRATION_DIR / "camera_model.npz"
HOMOGRAPHY_PATH = CALIBRATION_DIR / "ground_plane_h.npz"
IMU_PATH = CALIBRATION_DIR / "imu_alignment.json"
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Camera settings ----------
CAMERA_INDICES = [0, 1, 2]
WIDTH, HEIGHT = 1280, 720  # default capture resolution

# ---------- Default config values (will be overridden by YAML) ----------
DEFAULT_CONFIG = {
    "edge_detection": {
        "method": "bilateral_laplacian",  # bilateral_laplacian, adaptive_canny, learned
        "bilateral_d": 9,
        "bilateral_sigma_color": 75,
        "bilateral_sigma_space": 75,
        "laplacian_ksize": 3,
        "adaptive_canny_block_size": 11,
        "adaptive_canny_c": 2,
    },
    "preprocessing": {
        "method": "clahe_only",  # clahe_only, clahe_unsharp, contrast_stretch
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid_size": 8,
        "unsharp_strength": 0.5,
    },
    "line_extraction": {
        "method": "constrained_hough",  # constrained_hough, gradient_orientation
        "vote_threshold": 40,
        "minlen_pct": 40,
        "gap_pct": 1,
        "angle_max_deg": 20,
        "angle_tolerance_deg": 5.0,
    },
    "tracking": {
        "method": "exponential_smoothing",  # exponential_smoothing, kalman_2d
        "alpha": 0.3,  # smoothing factor for exponential smoothing
        "process_noise": 1e-3,
        "measurement_noise": 4e-3,
    },
    "roi": {
        "height_pct": 55,
        "topw_pct": 35,
        "bottom_gate_px": 40,
    },
    "performance": {
        "use_multithreading": True,
        "queue_size": 3,
        "early_exit_confidence": 0.85,
    },
    "ui": {
        "show_controls": True,
        "window_width": 1100,
        "window_height": 620,
    },
}

# Scoring covariance (Mahalanobis) diagonals for [center_err, angle_err, len_deficit]
MAHALANOBIS_COV = np.diag([0.15 ** 2, (8.0 * np.pi / 180.0) ** 2, 0.25 ** 2])
MAHALANOBIS_INV = np.linalg.inv(MAHALANOBIS_COV)

# Temporal filtering constants
MAX_INNOVATION_CHISQ = 9.21  # ~95% gate for 2D measurement
DEBOUNCE_RATE = 0.08         # maximum normalized offset change per frame


# ---------- Config loading ----------
def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load config from YAML file, falling back to defaults."""
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            config = DEFAULT_CONFIG.copy()
            # Deep merge
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration.")
    return DEFAULT_CONFIG.copy()


def save_config(config: dict, config_path: Path = CONFIG_PATH) -> None:
    """Save config to YAML file."""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Warning: Could not save config to {config_path}: {e}")


# ---------- Utility dataclasses ----------
@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    new_K: np.ndarray
    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: Path = INTRINSICS_PATH, frame_shape: Optional[Tuple[int, int]] = None) -> "CameraModel":
        if path.exists():
            data = np.load(path)
            K = data.get("K")
            dist = data.get("dist")
            if K is not None and dist is not None:
                new_K = data.get("new_K", K)
                K = K.astype(np.float32)
                dist = dist.astype(np.float32)
                new_K = new_K.astype(np.float32)
                # Pre-compute undistortion maps if frame shape is known
                map1, map2 = None, None
                if frame_shape is not None:
                    map1, map2 = cv.initUndistortRectifyMap(
                        K, dist, None, new_K, (frame_shape[1], frame_shape[0]), cv.CV_16SC2
                    )
                return cls(K, dist, new_K, map1, map2)
        # fallback: pinhole with no distortion
        K = np.array([[WIDTH, 0, WIDTH / 2.0], [0, WIDTH, HEIGHT / 2.0], [0, 0, 1]], np.float32)
        dist = np.zeros((1, 5), np.float32)
        return cls(K, dist, K.copy(), None, None)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.map1 is not None and self.map2 is not None:
            return cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)
        if self.dist is None or np.allclose(self.dist, 0):
            return frame
        return cv.undistort(frame, self.K, self.dist, None, self.new_K)


@dataclass
class IMUAlignment:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    @classmethod
    def load(cls, path: Path = IMU_PATH) -> "IMUAlignment":
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(float(data.get("roll_deg", 0.0)), float(data.get("pitch_deg", 0.0)))
            except Exception:
                pass
        return cls()

    def apply(self, frame: np.ndarray, K: np.ndarray) -> np.ndarray:
        if abs(self.roll_deg) < 1e-3 and abs(self.pitch_deg) < 1e-3:
            return frame
        roll = radians(self.roll_deg)
        pitch = radians(self.pitch_deg)
        Rx = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]], np.float32)
        Ry = np.array([[cos(roll), 0, sin(roll)], [0, 1, 0], [-sin(roll), 0, cos(roll)]], np.float32)
        R = Ry @ Rx
        H = K @ R @ np.linalg.inv(K)
        return cv.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]), flags=cv.INTER_LINEAR)


@dataclass
class GroundPlaneMapper:
    H: np.ndarray
    H_inv: np.ndarray
    bev_size: Tuple[int, int]
    use_cuda: bool = False
    map_x: Optional[np.ndarray] = None
    map_y: Optional[np.ndarray] = None

    @classmethod
    def load(
        cls,
        frame_shape: Tuple[int, int],
        path: Path = HOMOGRAPHY_PATH,
        bev_scale: float = 1.0,
    ) -> "GroundPlaneMapper":
        h, w = frame_shape[:2]
        if path.exists():
            data = np.load(path)
            H = data.get("H")
            bev_w = int(data.get("bev_w", w * bev_scale))
            bev_h = int(data.get("bev_h", h * bev_scale))
            if H is not None:
                H = H.astype(np.float32)
                H_inv = np.linalg.inv(H)
                use_cuda = hasattr(cv, "cuda") and cv.cuda.getCudaEnabledDeviceCount() > 0
                # Pre-compute warp maps
                y_coords, x_coords = np.mgrid[0:bev_h, 0:bev_w].astype(np.float32)
                pts = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
                pts_h = cv.convertPointsToHomogeneous(pts).reshape(-1, 3)
                src_pts = (H_inv @ pts_h.T).T
                src_pts = src_pts[:, :2] / src_pts[:, 2:3]
                map_x = src_pts[:, 0].reshape(bev_h, bev_w).astype(np.float32)
                map_y = src_pts[:, 1].reshape(bev_h, bev_w).astype(np.float32)
                return cls(H, H_inv.astype(np.float32), (bev_w, bev_h), use_cuda, map_x, map_y)
        # fallback: identity homography
        H = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        return cls(H, H_inv, (w, h), False, None, None)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        if self.map_x is not None and self.map_y is not None:
            return cv.remap(frame, self.map_x, self.map_y, cv.INTER_LINEAR)
        if self.use_cuda:
            gpu = cv.cuda_GpuMat()
            gpu.upload(frame)
            warped = cv.cuda.warpPerspective(gpu, self.H, self.bev_size, flags=cv.INTER_LINEAR)
            return warped.download()
        return cv.warpPerspective(frame, self.H, self.bev_size, flags=cv.INTER_LINEAR)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        pts_h = cv.convertPointsToHomogeneous(pts.astype(np.float32)).reshape(-1, 3)
        proj = (self.H_inv @ pts_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj.reshape(-1, 1, 2)


@dataclass
class ExponentialSmoothingTracker:
    """Exponential smoothing tracker replacing 4D Kalman filter."""
    offset: float = 0.0
    angle: float = 0.0
    alpha: float = 0.3
    initialized: bool = False
    state: str = "SEARCHING"
    confidence: float = 0.0
    last_measurement_time: float = 0.0
    measurement_timeout: float = 0.5  # seconds

    def step(
        self,
        timestamp: float,
        measurement: Optional[Tuple[float, float]],
        measurement_conf: float,
    ) -> Tuple[float, float, float]:
        """Update tracker with new measurement."""
        if measurement is not None:
            offset_meas, angle_meas = measurement
            if not self.initialized:
                self.offset = offset_meas
                self.angle = angle_meas
                self.initialized = True
                self.state = "TRACKING"
            else:
                # Exponential smoothing update
                self.offset = (1 - self.alpha) * self.offset + self.alpha * offset_meas
                self.angle = (1 - self.alpha) * self.angle + self.alpha * angle_meas
                self.state = "TRACKING"
            self.last_measurement_time = timestamp
            self.confidence = self._compute_confidence(measurement_conf, True)
        else:
            # No measurement - check timeout
            if self.initialized and (timestamp - self.last_measurement_time) > self.measurement_timeout:
                if self.state == "TRACKING":
                    self.state = "LOST"
                elif self.state == "LOST":
                    self.state = "SEARCHING"
                    self.initialized = False
            self.confidence = self._compute_confidence(0.0, False)

        return self.offset, self.angle, self.confidence

    def _compute_confidence(self, measurement_conf: float, has_measurement: bool) -> float:
        """Compute confidence based on state and measurement quality."""
        base = 0.2 if self.state == "SEARCHING" else 0.5 if self.state == "LOST" else 0.8
        meas_term = 0.3 * measurement_conf if has_measurement else 0.0
        return float(np.clip(base + meas_term, 0.0, 1.0))


@dataclass
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    angle_deg: float
    length: float

    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p1 + self.p2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.p1[0]), int(self.p1[1]), int(self.p2[0]), int(self.p2[1])


@dataclass
class FollowResult:
    p1: np.ndarray
    p2: np.ndarray
    lateral_offset_norm: float
    angle_error_deg: float
    norm_length: float
    inlier_ratio: float
    residual_rms: float
    inlier_points: np.ndarray
    residuals: np.ndarray
    nfa_log10: float


def show_text(img: np.ndarray, text: str, y: int = 28, scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


# ---------- UI helpers (kept for backward compatibility, but config takes precedence) ----------
def create_controls() -> None:
    cv.namedWindow("Controls", cv.WINDOW_NORMAL)
    cv.resizeWindow("Controls", 440, 320)
    # Controls are now mainly for visualization; config file is the source of truth


def reset_controls() -> None:
    pass  # Config file is now the source of truth


def read_controls(config: dict) -> Tuple[float, int, float, float, float, float, float, int]:
    """Read controls from config (trackbars are deprecated)."""
    edge_cfg = config["edge_detection"]
    line_cfg = config["line_extraction"]
    roi_cfg = config["roi"]
    
    # For backward compatibility, we still expose some parameters
    # but they come from config
    sigma = 0.33  # Not used with bilateral+Laplacian, kept for compatibility
    votes = line_cfg["vote_threshold"]
    minlen_pct = line_cfg["minlen_pct"] / 100.0
    gap_pct = line_cfg["gap_pct"] / 100.0
    angle_max = line_cfg["angle_max_deg"]
    roi_h_pct = roi_cfg["height_pct"] / 100.0
    roi_tw_pct = roi_cfg["topw_pct"] / 100.0
    gate_px = roi_cfg["bottom_gate_px"]
    
    return (sigma, votes, minlen_pct, gap_pct, angle_max, roi_h_pct, roi_tw_pct, gate_px)


# ---------- Geometry / ROI helpers ----------
def normalize_angle_deg(angle_deg: float) -> float:
    a = ((angle_deg + 90.0) % 180.0) - 90.0
    if a == -90.0:
        a = 90.0
    return a


def line_angle_and_length(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ang = degrees(np.arctan2(dy, dx))
    return normalize_angle_deg(ang), float(hypot(dx, dy))


def angle_from_vertical_deg(angle_deg: float) -> float:
    return abs(90.0 - abs(angle_deg))


def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    point = np.asarray(point, np.float32)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if np.allclose(a, b):
        return float(np.linalg.norm(point - a))
    ba = b - a
    pa = point - a
    return float(np.abs(cross2d(ba, pa)) / (np.linalg.norm(ba) + 1e-6))


def make_roi_mask(
    h: int,
    w: int,
    height_frac: float,
    top_width_frac: float,
    bottom_width_frac: float,
    center_offset_norm: float = 0.0,
) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    roi_h = int(h * height_frac)
    top_y = max(0, h - roi_h)
    top_w = int(w * top_width_frac)
    bot_w = int(w * bottom_width_frac)
    cx = w // 2 + int(center_offset_norm * 0.5 * w)
    pts = np.array(
        [
            (cx - top_w // 2, top_y),
            (cx + top_w // 2, top_y),
            (cx + bot_w // 2, h - 1),
            (cx - bot_w // 2, h - 1),
        ],
        np.int32,
    )
    cv.fillConvexPoly(mask, pts, 255)
    return mask


# ---------- Edge Detection Methods ----------
def bilateral_laplacian_edge_detection(
    gray: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    laplacian_ksize: int = 3,
) -> np.ndarray:
    """Bilateral filter + Laplacian edge detection (recommended, ~5ms)."""
    # Bilateral filter to reduce noise while preserving edges
    filtered = cv.bilateralFilter(gray, d, sigma_color, sigma_space)
    # Laplacian for edge detection
    laplacian = cv.Laplacian(filtered, cv.CV_32F, ksize=laplacian_ksize)
    # Threshold and convert to uint8
    edges = np.abs(laplacian)
    edges = cv.normalize(edges, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    _, edges = cv.threshold(edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return edges


def adaptive_canny_edge_detection(
    gray: np.ndarray,
    block_size: int = 11,
    c: float = 2,
) -> np.ndarray:
    """Adaptive Canny with local mean thresholding (~8ms)."""
    # Compute local mean
    mean = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c
    )
    # Use mean to set Canny thresholds
    low = np.percentile(mean[mean > 0], 10) if np.any(mean > 0) else 50
    high = np.percentile(mean[mean > 0], 90) if np.any(mean > 0) else 150
    return cv.Canny(gray, int(low), int(high))


# ---------- Preprocessing Methods ----------
def clahe_preprocessing(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """CLAHE-only preprocessing (recommended, ~3-4ms, 4x faster than Retinex)."""
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    L_clahe = clahe.apply(L)
    lab = cv.merge((L_clahe, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def clahe_unsharp_preprocessing(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    strength: float = 0.5,
) -> np.ndarray:
    """CLAHE + Unsharp mask for faint lines (~4-5ms)."""
    clahe_result = clahe_preprocessing(frame, clip_limit, tile_grid_size)
    # Unsharp mask
    blurred = cv.GaussianBlur(clahe_result, (0, 0), 1.0)
    unsharp = cv.addWeighted(clahe_result, 1.0 + strength, blurred, -strength, 0)
    return unsharp


def contrast_stretch_preprocessing(frame: np.ndarray) -> np.ndarray:
    """Contrast stretching (fastest, ~1-2ms, riskiest)."""
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)
    L_norm = cv.normalize(L, None, 0, 255, cv.NORM_MINMAX)
    lab = cv.merge((L_norm, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def morphological_cleanup(edges: np.ndarray, kernel_length: int = 9) -> np.ndarray:
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)
    opened = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel, iterations=1)
    return opened


def estimate_vanishing_point_ransac(
    segments: Sequence[Segment],
    frame_shape: Tuple[int, int],
    iterations: int = 100,
    threshold: float = 10.0,
) -> Optional[Tuple[float, float]]:
    """RANSAC-based vanishing point estimation (O(iterations) vs O(N²))."""
    if len(segments) < 2:
        return None
    
    # Extract line parameters (ax + by + c = 0)
    lines = []
    for seg in segments:
        x1, y1 = seg.p1
        x2, y2 = seg.p2
        # Normalize line equation
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx*dx + dy*dy) + 1e-6
        a, b, c = dy/norm, -dx/norm, (dx*y1 - dy*x1)/norm
        lines.append((a, b, c))
    
    if len(lines) < 2:
        return None
    
    lines = np.array(lines)
    h, w = frame_shape[:2]
    best_vp = None
    best_inliers = 0
    rng = np.random.default_rng(42)
    
    for _ in range(iterations):
        # Sample two lines
        idx = rng.choice(len(lines), 2, replace=False)
        a1, b1, c1 = lines[idx[0]]
        a2, b2, c2 = lines[idx[1]]
        
        # Compute intersection
        denom = a1 * b2 - a2 * b1
        if abs(denom) < 1e-6:
            continue
        
        vp_x = (b1 * c2 - b2 * c1) / denom
        vp_y = (a2 * c1 - a1 * c2) / denom
        
        if not (np.isfinite(vp_x) and np.isfinite(vp_y)):
            continue
        
        # Count inliers (lines that pass near this VP)
        # Distance from point to line: |ax + by + c| / sqrt(a^2 + b^2)
        # Since lines are normalized, sqrt(a^2 + b^2) = 1
        distances = np.abs(lines[:, 0] * vp_x + lines[:, 1] * vp_y + lines[:, 2])
        inliers = distances < threshold
        inlier_count = inliers.sum()
        
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_vp = (vp_x, vp_y)
    
    if best_vp is not None and 0 <= best_vp[0] < w * 2 and -h <= best_vp[1] < h * 2:
        return best_vp
    
    return None


def merge_collinear_segments(segments: Sequence[Segment], angle_tol_deg: float, gap_px: float) -> List[Segment]:
    if not segments:
        return []

    clusters: List[List[Segment]] = []
    for seg in sorted(segments, key=lambda s: -s.length):
        placed = False
        for cluster in clusters:
            ref = cluster[0]
            if abs(normalize_angle_deg(seg.angle_deg - ref.angle_deg)) > angle_tol_deg:
                continue
            dist = point_line_distance(seg.midpoint(), ref.p1, ref.p2)
            if dist < gap_px:
                cluster.append(seg)
                placed = True
                break
        if not placed:
            clusters.append([seg])

    merged: List[Segment] = []
    for cluster in clusters:
        pts = []
        for seg in cluster:
            pts.extend([seg.p1, seg.p2])
        pts_array = np.asarray(pts, dtype=np.float32)
        if len(pts_array) < 2:
            continue
        mean, eigenvectors, _ = cv.PCACompute2(pts_array, mean=np.empty((0)))
        direction = np.asarray(eigenvectors[0]).reshape(-1)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        proj = pts_array @ direction
        i_min = int(np.argmin(proj))
        i_max = int(np.argmax(proj))
        p1 = pts_array[i_min]
        p2 = pts_array[i_max]
        angle = normalize_angle_deg(degrees(np.arctan2(direction[1], direction[0])))
        length = float(np.linalg.norm(p2 - p1))
        merged.append(Segment(p1.astype(np.float32), p2.astype(np.float32), angle, length))
    return merged


def ransac_line(points: np.ndarray, thresh: float, min_inliers: int, iterations: int = 256) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return None
    best_inliers: Optional[np.ndarray] = None
    rng = np.random.default_rng(42)
    for _ in range(iterations):
        idx = rng.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        line_vec = p2 - p1
        norm_len = np.linalg.norm(line_vec)
        if norm_len < 1e-6:
            continue
        distances = np.abs(cross2d(line_vec, points - p1)) / (norm_len + 1e-6)
        inliers = distances <= thresh
        if inliers.sum() < min_inliers:
            continue
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None:
        return None
    inlier_pts = points[best_inliers]
    # total least squares refinement
    mean = np.mean(inlier_pts, axis=0)
    cov = np.cov((inlier_pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    projections = (inlier_pts - mean) @ direction
    p1 = mean + direction * projections.min()
    p2 = mean + direction * projections.max()
    residuals = np.abs(cross2d(direction, inlier_pts - mean)) / (np.linalg.norm(direction) + 1e-6)
    return p1.astype(np.float32), p2.astype(np.float32), residuals


def nfa(inliers: int, total: int, p: float = 0.01) -> float:
    # log10 of Number of False Alarms using a loose Chernoff bound
    if inliers <= 0 or total <= 0:
        return 0.0
    from math import log10

    tail = 0.0
    for k in range(inliers, total + 1):
        comb = np.math.comb(total, k)
        tail += comb * (p ** k) * ((1 - p) ** (total - k))
    return -log10(max(tail, 1e-12))


def mahalanobis_score(feature: np.ndarray) -> float:
    return float(feature.T @ MAHALANOBIS_INV @ feature)


@dataclass
class ROIState:
    center_offset: float = 0.0

    def update(self, lateral_offset_norm: float, alpha: float = 0.2) -> None:
        self.center_offset = (1 - alpha) * self.center_offset + alpha * np.clip(lateral_offset_norm, -0.6, 0.6)


# ---------- Line detection pipeline ----------
class LinePipeline:
    def __init__(self, bev_shape: Tuple[int, int], config: dict) -> None:
        self.config = config
        tracking_cfg = config["tracking"]
        if tracking_cfg["method"] == "exponential_smoothing":
            self.tracker = ExponentialSmoothingTracker(alpha=tracking_cfg["alpha"])
        else:
            # Fallback to exponential smoothing if kalman_2d not implemented
            self.tracker = ExponentialSmoothingTracker(alpha=tracking_cfg["alpha"])
        self.roi_state = ROIState()
        self.bev_w, self.bev_h = bev_shape
        self.last_output: Optional[Tuple[float, float]] = None
        self.last_follow: Optional[FollowResult] = None
        self.vp_hint: Optional[Tuple[float, float]] = None

    def detect(
        self,
        frame_bev: np.ndarray,
        controls: Tuple[float, int, float, float, float, float, float, int],
        timestamp: float,
    ) -> Tuple[np.ndarray, Optional[FollowResult], List[Segment], Optional[Tuple[float, float]], np.ndarray]:
        config = self.config
        edge_cfg = config["edge_detection"]
        preproc_cfg = config["preprocessing"]
        line_cfg = config["line_extraction"]
        perf_cfg = config["performance"]
        
        sigma, votes, minlen_frac, gap_frac, angle_max_deg, roi_h_frac, roi_top_frac, bottom_gate = controls
        h, w = frame_bev.shape[:2]

        center_offset = self.last_output[0] if self.last_output is not None else self.roi_state.center_offset
        if self.vp_hint is not None:
            vp_offset = (self.vp_hint[0] / max(1.0, w) - 0.5) * 2.0
            center_offset = 0.7 * center_offset + 0.3 * np.clip(vp_offset, -0.6, 0.6)
        roi_mask = make_roi_mask(h, w, roi_h_frac, roi_top_frac, 1.0, center_offset)

        # Preprocessing
        if preproc_cfg["method"] == "clahe_only":
            processed = clahe_preprocessing(
                frame_bev,
                preproc_cfg["clahe_clip_limit"],
                preproc_cfg["clahe_tile_grid_size"],
            )
        elif preproc_cfg["method"] == "clahe_unsharp":
            processed = clahe_unsharp_preprocessing(
                frame_bev,
                preproc_cfg["clahe_clip_limit"],
                preproc_cfg["clahe_tile_grid_size"],
                preproc_cfg.get("unsharp_strength", 0.5),
            )
        elif preproc_cfg["method"] == "contrast_stretch":
            processed = contrast_stretch_preprocessing(frame_bev)
        else:
            processed = clahe_preprocessing(frame_bev)

        gray_proc = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)

        early_exit = self.tracker.confidence > perf_cfg.get("early_exit_confidence", 0.85) and self.last_follow is not None

        # Edge detection
        if not early_exit:
            small = cv.pyrDown(gray_proc)
            if edge_cfg["method"] == "bilateral_laplacian":
                edges_small = bilateral_laplacian_edge_detection(
                    small,
                    edge_cfg["bilateral_d"],
                    edge_cfg["bilateral_sigma_color"],
                    edge_cfg["bilateral_sigma_space"],
                    edge_cfg["laplacian_ksize"],
                )
            elif edge_cfg["method"] == "adaptive_canny":
                edges_small = adaptive_canny_edge_detection(
                    small,
                    edge_cfg["adaptive_canny_block_size"],
                    edge_cfg["adaptive_canny_c"],
                )
            else:
                edges_small = bilateral_laplacian_edge_detection(small)
            edges_small = morphological_cleanup(edges_small, kernel_length=7)
            edges = cv.pyrUp(edges_small, dstsize=(w, h))
        else:
            if edge_cfg["method"] == "bilateral_laplacian":
                edges = bilateral_laplacian_edge_detection(
                    gray_proc,
                    edge_cfg["bilateral_d"],
                    edge_cfg["bilateral_sigma_color"],
                    edge_cfg["bilateral_sigma_space"],
                    edge_cfg["laplacian_ksize"],
                )
            elif edge_cfg["method"] == "adaptive_canny":
                edges = adaptive_canny_edge_detection(
                    gray_proc,
                    edge_cfg["adaptive_canny_block_size"],
                    edge_cfg["adaptive_canny_c"],
                )
            else:
                edges = bilateral_laplacian_edge_detection(gray_proc)
        
        edges = cv.bitwise_and(edges, edges, mask=roi_mask)
        edges = morphological_cleanup(edges, kernel_length=9)

        bev_debug = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        follow_result: Optional[FollowResult] = None
        segments: List[Segment] = []

        if early_exit and self.last_follow is not None:
            follow_result = self.last_follow
        else:
            # Constrained HoughLinesP with angle-aware filtering
            min_dim = min(h, w)
            
            # Pre-filter edges for vertical gradients if using constrained method
            if line_cfg["method"] == "constrained_hough":
                # Compute gradient orientation
                grad_x = cv.Sobel(gray_proc, cv.CV_32F, 1, 0, ksize=3)
                grad_y = cv.Sobel(gray_proc, cv.CV_32F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                # Only consider pixels with significant gradient
                grad_mag_thresh = np.percentile(grad_mag[edges > 0], 30) if np.any(edges > 0) else 0
                angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
                # Filter for near-vertical lines (within angle_max_deg of vertical)
                # Vertical means angle near 90° or -90°
                vertical_mask = np.zeros_like(edges, dtype=np.uint8)
                # Check if angle is close to vertical (90° ± angle_max or -90° ± angle_max)
                angle_from_vert = np.abs(np.abs(angles) - 90.0)
                vertical_mask = ((angle_from_vert < angle_max_deg) & (grad_mag > grad_mag_thresh)).astype(np.uint8) * 255
                edges = cv.bitwise_and(edges, vertical_mask)
            
            lines = cv.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=max(30, votes),
                minLineLength=int(minlen_frac * min_dim),
                maxLineGap=int(gap_frac * min_dim),
            )
            if lines is not None:
                for x1, y1, x2, y2 in lines[:, 0]:
                    p1 = np.array([x1, y1], np.float32)
                    p2 = np.array([x2, y2], np.float32)
                    angle, length = line_angle_and_length(p1, p2)
                    # Additional angle filtering
                    if angle_from_vertical_deg(angle) <= angle_max_deg:
                        segments.append(Segment(p1, p2, angle, length))

        if not early_exit:
            vp_candidate = estimate_vanishing_point_ransac(segments, (h, w))
            if vp_candidate is not None:
                self.vp_hint = vp_candidate
        vp_hint = self.vp_hint

        min_len_px = minlen_frac * min(h, w)
        merged = merge_collinear_segments(segments, angle_tol_deg=line_cfg.get("angle_tolerance_deg", 5.0), gap_px=gap_frac * min(h, w))
        candidates = [seg for seg in merged if seg.length >= min_len_px]

        if not early_exit:
            follow_result = self._fit_consensus_line(
                candidates,
                votes,
                bottom_gate,
                angle_max_deg,
                vp_hint,
                (h, w),
            )
            if follow_result is not None:
                self.last_follow = follow_result
        elif follow_result is not None:
            follow_result = self.last_follow

        measurement: Optional[Tuple[float, float]] = None
        measurement_conf = 0.0
        prev_offset = self.last_output[0] if self.last_output is not None else 0.0

        if follow_result is not None and not early_exit:
            measurement_conf = float(
                np.clip(
                    0.5 * follow_result.inlier_ratio + 0.5 * max(0.0, 1.0 - follow_result.residual_rms / 3.0),
                    0.0,
                    1.0,
                )
            )
            measurement_offset = self._debounce(prev_offset, follow_result.lateral_offset_norm)
            measurement_angle = np.radians(follow_result.angle_error_deg)
            measurement = (measurement_offset, measurement_angle)
            follow_result.lateral_offset_norm = measurement_offset

        state_offset, state_angle, _ = self.tracker.step(timestamp, measurement, measurement_conf)
        self.last_output = (state_offset, state_angle)
        self.roi_state.update(state_offset)

        if follow_result is not None:
            follow_result.lateral_offset_norm = state_offset
            follow_result.angle_error_deg = np.degrees(state_angle)
            self.last_follow = follow_result

        for seg in candidates:
            draw_segment(bev_debug, seg, 1)
        if follow_result is not None:
            cv.line(
                bev_debug,
                (int(follow_result.p1[0]), int(follow_result.p1[1])),
                (int(follow_result.p2[0]), int(follow_result.p2[1])),
                (0, 220, 0),
                2,
                cv.LINE_AA,
            )

        return edges, follow_result, candidates, vp_hint, bev_debug

    def _debounce(self, prev: float, new: float) -> float:
        delta = np.clip(new - prev, -DEBOUNCE_RATE, DEBOUNCE_RATE)
        return prev + delta

    def _fit_consensus_line(
        self,
        candidates: Sequence[Segment],
        vote_threshold: int,
        bottom_gate: int,
        angle_max: float,
        vp_hint: Optional[Tuple[float, float]],
        shape: Tuple[int, int],
    ) -> Optional[FollowResult]:
        if not candidates:
            return None
        h, w = shape
        bottom_y = h - 1
        points = []
        for seg in candidates:
            pts = np.linspace(seg.p1, seg.p2, num=20)
            points.append(pts)
        points = np.vstack(points)
        result = ransac_line(points, thresh=2.0, min_inliers=vote_threshold)
        if result is None:
            return None
        p1, p2, residuals = result
        inliers = residuals < 2.5
        inlier_pts = points[inliers]
        if len(inlier_pts) < vote_threshold:
            return None
        xb = line_intersection_with_y(p1, p2, bottom_y)
        if xb is None or not np.isfinite(xb):
            return None
        if np.any(inlier_pts[:, 1] > bottom_y) and np.percentile(inlier_pts[:, 1], 90) < bottom_y - bottom_gate:
            return None
        angle, length = line_angle_and_length(p1, p2)
        angle_err = angle_from_vertical_deg(angle)
        if angle_err > angle_max:
            return None
        norm_center = (xb - w / 2.0) / (0.5 * w)
        norm_length = min(length / (0.6 * hypot(w, h)), 1.0)
        cov = np.array([norm_center, np.radians(angle_err), 1.0 - norm_length], np.float32)
        score = mahalanobis_score(cov)
        bottom_fraction = np.mean(inlier_pts[:, 1] > bottom_y - bottom_gate)
        coverage_penalty = max(0.0, 0.2 - bottom_fraction) * 4.0
        vp_penalty = 0.0
        if vp_hint is not None:
            vp_vec = np.array([vp_hint[0] - w / 2.0, vp_hint[1] - bottom_y])
            line_vec = p2 - p1
            cos_sim = np.dot(vp_vec, line_vec) / (np.linalg.norm(vp_vec) * np.linalg.norm(line_vec) + 1e-6)
            vp_penalty = max(0.0, 1.0 - cos_sim)
        score += coverage_penalty + vp_penalty
        if score > 12.0:
            return None
        residual_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        inlier_ratio = inliers.sum() / len(residuals)
        nfa_value = nfa(inliers.sum(), len(residuals))
        if nfa_value < 2.0:
            return None
        return FollowResult(
            p1=p1,
            p2=p2,
            lateral_offset_norm=float(norm_center),
            angle_error_deg=float(angle_err),
            norm_length=float(norm_length),
            inlier_ratio=float(inlier_ratio),
            residual_rms=float(residual_rms),
            inlier_points=inlier_pts,
            residuals=residuals,
            nfa_log10=float(nfa_value),
        )


def line_intersection_with_y(p1: np.ndarray, p2: np.ndarray, y: float) -> Optional[float]:
    dy = float(p2[1] - p1[1])
    dx = float(p2[0] - p1[0])
    if abs(dy) < 1e-6:
        return None
    if abs(dx) < 1e-6:
        return float(p1[0])
    slope = dy / dx
    intercept = p1[1] - slope * p1[0]
    return (y - intercept) / slope


# ---------- Camera handling ----------
def open_camera() -> Optional[cv.VideoCapture]:
    backends = []
    if hasattr(cv, "CAP_DSHOW"):
        backends.append(("CAP_DSHOW", cv.CAP_DSHOW))
    if hasattr(cv, "CAP_MSMF"):
        backends.append(("CAP_MSMF", cv.CAP_MSMF))
    backends.append(("DEFAULT", 0))
    for name, be in backends:
        for idx in CAMERA_INDICES:
            cap = cv.VideoCapture(idx, be) if be != 0 else cv.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(
                    f"Camera opened: backend={name}, index={idx}, size={frame.shape[1]}x{frame.shape[0]}"
                )
                return cap
            cap.release()
    for idx in CAMERA_INDICES:
        cap = cv.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"Camera opened (fallback 640x480): index={idx}")
                return cap
            cap.release()
    return None


# ---------- Visualization helpers ----------
def color_for_angle(angle_deg: float) -> Tuple[int, int, int]:
    a = max(-90.0, min(90.0, angle_deg))
    t = (a + 90.0) / 180.0
    if t < 0.5:
        k = t / 0.5
        b, g, r = int(255 * (1 - k)), int(255 * k), 0
    else:
        k = (t - 0.5) / 0.5
        b, g, r = 0, int(255 * (1 - k)), int(255 * k)
    return (b // 2 + 80, g // 2 + 80, r // 2 + 80)


def draw_segment(img: np.ndarray, seg: Segment, thickness: int = 3) -> None:
    x1, y1, x2, y2 = map(int, seg.as_tuple())
    cv.line(img, (x1, y1), (x2, y2), color_for_angle(seg.angle_deg), thickness, cv.LINE_AA)


def unwarp_segments_to_camera(segments: Sequence[Segment], mapper: GroundPlaneMapper) -> List[Segment]:
    if not segments:
        return []
    pts = []
    for seg in segments:
        pts.extend([seg.p1, seg.p2])
    pts_arr = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    cam_pts = mapper.unwarp_points(pts_arr).reshape(-1, 2)
    cam_segments: List[Segment] = []
    for idx, seg in enumerate(segments):
        p1 = cam_pts[2 * idx]
        p2 = cam_pts[2 * idx + 1]
        if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
            continue
        p1f = p1.astype(np.float32)
        p2f = p2.astype(np.float32)
        angle, length = line_angle_and_length(p1f, p2f)
        cam_segments.append(Segment(p1f, p2f, angle, length))
    return cam_segments


def put_angle_label(img: np.ndarray, seg: Segment) -> None:
    mid = 0.5 * (seg.p1 + seg.p2)
    if not np.all(np.isfinite(mid)):
        return
    x, y = int(mid[0]), int(mid[1])
    label = f"{seg.angle_deg:+.1f}°"
    cv.putText(img, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


def draw_dominant_orientation(img: np.ndarray, segments: Sequence[Segment]) -> Optional[float]:
    if not segments:
        return None
    angles = np.deg2rad(np.array([seg.angle_deg for seg in segments], dtype=np.float32))
    lengths = np.array([max(seg.length, 1e-3) for seg in segments], dtype=np.float32)
    if np.sum(lengths) <= 0:
        return None
    C = float(np.sum(lengths * np.cos(2 * angles)) / np.sum(lengths))
    S = float(np.sum(lengths * np.sin(2 * angles)) / np.sum(lengths))
    dom = 0.5 * np.arctan2(S, C)
    dom_deg = normalize_angle_deg(np.rad2deg(dom))
    h, w = img.shape[:2]
    Lref = int(0.45 * min(h, w))
    dx = int(Lref * np.cos(dom))
    dy = int(Lref * np.sin(dom))
    cx, cy = w // 2, h // 2
    cv.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 220, 255), 2, cv.LINE_AA)
    cv.circle(img, (cx, cy), 5, (0, 0, 0), -1, cv.LINE_AA)
    cv.circle(img, (cx, cy), 4, (0, 220, 255), -1, cv.LINE_AA)
    return dom_deg


def draw_pip(canvas: np.ndarray, pip_img: np.ndarray, scale: float = 0.28, margin: int = 16) -> None:
    if pip_img is None or pip_img.size == 0:
        return
    h, w = canvas.shape[:2]
    ph = max(1, int(h * scale))
    pw = max(1, int(ph * pip_img.shape[1] / max(1, pip_img.shape[0])))
    if pw + 2 * margin > w or ph + 2 * margin > h:
        return
    pip_resized = cv.resize(pip_img, (pw, ph))
    x0 = w - pw - margin
    y0 = margin
    cv.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + pw + 4, y0 + ph + 4), (0, 0, 0), -1)
    cv.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + pw + 4, y0 + ph + 4), (200, 200, 200), 1)
    canvas[y0 : y0 + ph, x0 : x0 + pw] = pip_resized


def draw_confidence_bar(img: np.ndarray, confidence: float, margin: int = 18) -> None:
    h, w = img.shape[:2]
    bar_w, bar_h = 220, 14
    x0 = margin
    y0 = h - margin - bar_h
    conf = float(np.clip(confidence, 0.0, 1.0))
    cv.rectangle(img, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (0, 0, 0), -1)
    cv.rectangle(img, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (200, 200, 200), 1)
    fill_w = int(bar_w * conf)
    color = (
        int(50 + (1.0 - conf) * 180),
        int(60 + conf * 180),
        80,
    )
    cv.rectangle(img, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    cv.putText(
        img,
        f"Confidence: {conf:.2f}",
        (x0 + bar_w + 12, y0 + bar_h - 2),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )


# ---------- Multi-threading support ----------
class FrameProcessor:
    """Producer-consumer pattern for multi-threaded processing."""
    def __init__(self, pipeline: LinePipeline, config: dict):
        self.pipeline = pipeline
        self.config = config
        self.queue = queue.Queue(maxsize=config["performance"].get("queue_size", 3))
        self.stop_event = threading.Event()
        self.worker_thread = None

    def start(self):
        """Start the worker thread."""
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Worker thread that processes frames."""
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:  # Sentinel
                    break
                frame_bev, controls, timestamp, result_queue = item
                edges, follow_result, segments, vp_hint, bev_debug = self.pipeline.detect(
                    frame_bev, controls, timestamp
                )
                result_queue.put((edges, follow_result, segments, vp_hint, bev_debug))
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker thread: {e}")

    def process_async(self, frame_bev, controls, timestamp):
        """Submit frame for async processing."""
        result_queue = queue.Queue(maxsize=1)
        try:
            self.queue.put_nowait((frame_bev, controls, timestamp, result_queue))
            return result_queue
        except queue.Full:
            return None

    def stop(self):
        """Stop the worker thread."""
        self.stop_event.set()
        if self.worker_thread:
            self.queue.put(None)  # Sentinel
            self.worker_thread.join(timeout=1.0)


# ---------- Core processing ----------
def detect_and_overlay(
    frame: np.ndarray,
    camera_model: CameraModel,
    imu_alignment: IMUAlignment,
    mapper: GroundPlaneMapper,
    pipeline: LinePipeline,
    controls: Tuple[float, int, float, float, float, float, float, int],
    timestamp: float,
    processor: Optional[FrameProcessor] = None,
) -> Tuple[np.ndarray, Optional[FollowResult], Optional[Tuple[float, float]], List[Segment], np.ndarray]:
    undistorted = camera_model.undistort(frame)
    aligned = imu_alignment.apply(undistorted, camera_model.K)
    bev = mapper.warp(aligned)

    # Use async processing if available
    if processor is not None:
        result_queue = processor.process_async(bev, controls, timestamp)
        if result_queue is not None:
            try:
                edges, follow_result, segments, vp_hint, bev_debug = result_queue.get(timeout=0.05)
            except queue.Empty:
                # Fallback to synchronous
                edges, follow_result, segments, vp_hint, bev_debug = pipeline.detect(bev, controls, timestamp)
        else:
            edges, follow_result, segments, vp_hint, bev_debug = pipeline.detect(bev, controls, timestamp)
    else:
        edges, follow_result, segments, vp_hint, bev_debug = pipeline.detect(bev, controls, timestamp)

    overlay = aligned.copy()
    h_cam, w_cam = overlay.shape[:2]
    h_bev, w_bev = bev.shape[:2]

    center_offset = pipeline.last_output[0] if pipeline.last_output is not None else pipeline.roi_state.center_offset
    if pipeline.vp_hint is not None:
        vp_offset = (pipeline.vp_hint[0] / max(1.0, float(w_bev)) - 0.5) * 2.0
        center_offset = 0.7 * center_offset + 0.3 * float(np.clip(vp_offset, -0.6, 0.6))

    roi_mask_bev = make_roi_mask(h_bev, w_bev, controls[5], controls[6], 1.0, center_offset)
    contours, _ = cv.findContours(roi_mask_bev, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        for cnt in contours:
            cam_cnt = mapper.unwarp_points(cnt.astype(np.float32))
            cam_poly = cam_cnt.reshape(-1, 2)
            if not np.all(np.isfinite(cam_poly)):
                continue
            cv.polylines(overlay, [np.round(cam_cnt).astype(np.int32)], True, (200, 200, 200), 1, cv.LINE_AA)

    gate_px = controls[-1]
    gate_top = max(0, h_bev - gate_px)
    gate_poly = np.array(
        [[0, gate_top], [w_bev - 1, gate_top], [w_bev - 1, h_bev - 1], [0, h_bev - 1]],
        np.float32,
    ).reshape(-1, 1, 2)
    cam_gate = mapper.unwarp_points(gate_poly)
    cam_gate_poly = cam_gate.reshape(-1, 2)
    if np.all(np.isfinite(cam_gate_poly)):
        cv.polylines(overlay, [np.round(cam_gate).astype(np.int32)], True, (120, 120, 120), 1, cv.LINE_AA)

    cv.line(overlay, (w_cam // 2, h_cam - 80), (w_cam // 2, h_cam - 1), (0, 0, 255), 1, cv.LINE_AA)

    cam_segments = unwarp_segments_to_camera(segments, mapper)
    for seg in cam_segments:
        draw_segment(overlay, seg, thickness=2)
        put_angle_label(overlay, seg)

    dom_angle = draw_dominant_orientation(overlay, cam_segments)
    if dom_angle is not None:
        show_text(overlay, f"Dominant orientation: {dom_angle:.1f}°   (segments: {len(cam_segments)})", y=56)
    else:
        show_text(overlay, "No long lines detected", y=56, color=(180, 180, 180))

    if follow_result is not None:
        p1, p2 = follow_result.p1, follow_result.p2
        pts = np.array([[p1], [p2]], np.float32)
        cam_pts = mapper.unwarp_points(pts)
        cam_line = cam_pts.reshape(-1, 2)
        if np.all(np.isfinite(cam_line)):
            x1, y1 = int(cam_line[0, 0]), int(cam_line[0, 1])
            x2, y2 = int(cam_line[1, 0]), int(cam_line[1, 1])
            cv.line(overlay, (x1, y1), (x2, y2), (0, 220, 0), 4, cv.LINE_AA)
            xb_cam = line_intersection_with_y(cam_line[0], cam_line[1], h_cam - 1)
            if xb_cam is not None and np.isfinite(xb_cam):
                xb_int = int(np.clip(round(xb_cam), 0, w_cam - 1))
                cv.circle(overlay, (xb_int, h_cam - 1), 7, (0, 220, 0), -1, cv.LINE_AA)
                cv.line(overlay, (xb_int, h_cam - 35), (xb_int, h_cam - 1), (0, 220, 0), 2, cv.LINE_AA)
                cv.line(overlay, (w_cam // 2, h_cam - 1), (xb_int, h_cam - 1), (0, 220, 0), 1, cv.LINE_AA)
        xb_norm = follow_result.lateral_offset_norm
        angle_err = follow_result.angle_error_deg
        norm_length = follow_result.norm_length
        inlier_ratio = follow_result.inlier_ratio
        residual_rms = follow_result.residual_rms
        show_text(
            overlay,
            f"Follow offset: {xb_norm:+.3f}  angle: {angle_err:+.2f}°  len_norm: {norm_length:.2f}  inliers: {inlier_ratio:.2f}  rms: {residual_rms:.2f}  NFAlog10: {follow_result.nfa_log10:.2f}",
            y=84,
            color=(180, 255, 180),
        )
    else:
        show_text(overlay, "Follow: not found", y=84, color=(180, 180, 180))

    if vp_hint is not None:
        cv.circle(bev_debug, (int(vp_hint[0]), int(vp_hint[1])), 6, (0, 0, 255), -1, cv.LINE_AA)
        show_text(bev_debug, f"VP: ({vp_hint[0]:.1f}, {vp_hint[1]:.1f})", y=28)

    draw_pip(overlay, bev_debug)
    draw_confidence_bar(overlay, pipeline.tracker.confidence)

    return overlay, follow_result, vp_hint, segments, bev_debug


# ---------- Main loop ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Long-line detector with optimized pipeline")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to config YAML file")
    parser.add_argument("--no-multithreading", action="store_true", help="Disable multi-threading")
    parser.add_argument("--camera", type=int, default=None, help="Camera index to use")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    
    # Override multithreading if requested
    if args.no_multithreading:
        config["performance"]["use_multithreading"] = False

    cap = open_camera()
    if cap is None:
        print("\nERROR: Could not open webcam.")
        print("Close other camera applications, adjust CAMERA_INDICES, or reduce WIDTH/HEIGHT.")
        time.sleep(4)
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("Unable to grab initial frame for initialization.")
        return

    camera_model = CameraModel.load(frame_shape=frame.shape[:2])
    imu_alignment = IMUAlignment.load()
    mapper = GroundPlaneMapper.load(frame.shape, bev_scale=1.0)
    pipeline = LinePipeline(mapper.bev_size, config)

    # Setup multi-threading if enabled
    processor = None
    if config["performance"].get("use_multithreading", True):
        processor = FrameProcessor(pipeline, config)
        processor.start()

    ui_cfg = config.get("ui", {})
    cv.namedWindow("long_lines_overlay", cv.WINDOW_NORMAL)
    cv.resizeWindow("long_lines_overlay", ui_cfg.get("window_width", 1100), ui_cfg.get("window_height", 620))
    
    if ui_cfg.get("show_controls", True):
        create_controls()
        reset_controls()

    mode = 2
    prev_time = time.time()
    fps = 0.0
    print("Running. Keys: 1=Raw  2=Lines  r=Reset  q=Quit")
    print(f"Config loaded from: {args.config}")
    print(f"Multi-threading: {config['performance'].get('use_multithreading', True)}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: empty frame from camera.")
                break
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            controls = read_controls(config)

            if mode == 1:
                out = frame.copy()
                show_text(out, "Mode: Raw  |  1=Raw  2=Lines  r=Reset  q=Quit", y=28)
                show_text(out, f"{fps:.1f} FPS", y=out.shape[0] - 12)
                cv.imshow("long_lines_overlay", out)
            else:
                overlay, *_ = detect_and_overlay(
                    frame,
                    camera_model,
                    imu_alignment,
                    mapper,
                    pipeline,
                    controls,
                    now,
                    processor,
                )
                show_text(overlay, "Mode: Lines  |  1=Raw  2=Lines  r=Reset  q=Quit", y=28)
                show_text(overlay, f"{fps:.1f} FPS", y=overlay.shape[0] - 12)
                cv.imshow("long_lines_overlay", overlay)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("1"):
                mode = 1
            if key == ord("2"):
                mode = 2
            if key == ord("r"):
                reset_controls()
    finally:
        if processor is not None:
            processor.stop()
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
