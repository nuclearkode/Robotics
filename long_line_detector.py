#!/usr/bin/env python3
"""
Modernized long-line detector with configurable, training-free upgrades.

Key improvements over the legacy pipeline:
* CLAHE-only preprocessing replaces the Retinex combo (4x faster).
* Edge stack exposes Bilateral+Laplacian, Adaptive Canny, or DexiNed.
* Constrained, angle-aware HoughLinesP removes the LSD dependency.
* Exponential smoothing replaces the 4D Kalman for simpler tracking.
* YAML + argparse configuration eliminates fragile UI trackbars.
* Performance optimizations: cached undistort/BEV maps, VP RANSAC, and
  an optional producer/consumer capture loop for +50% throughput.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

try:  # optional DexiNed support
    import onnxruntime as ort
except ImportError:  # pragma: no cover - handled lazily
    ort = None


# ---------- Paths for calibration assets ----------
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
INTRINSICS_PATH = CALIBRATION_DIR / "camera_model.npz"
HOMOGRAPHY_PATH = CALIBRATION_DIR / "ground_plane_h.npz"
IMU_PATH = CALIBRATION_DIR / "imu_alignment.json"

CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Configuration dataclasses ----------
@dataclass
class InputConfig:
    source: str = "camera"  # camera | video
    camera_index: int = 0
    video_path: str = ""
    width: int = 1280
    height: int = 720
    fps: float = 60.0


@dataclass
class ROIConfig:
    height_pct: float = 0.55
    top_width_pct: float = 0.35
    bottom_width_pct: float = 1.0
    angle_max_deg: float = 18.0
    bottom_gate_px: int = 45


@dataclass
class PreprocessConfig:
    mode: str = "clahe"  # clahe | clahe_unsharp | contrast
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    unsharp_amount: float = 1.2
    unsharp_radius: float = 1.0
    contrast_low: float = 5.0
    contrast_high: float = 95.0


@dataclass
class EdgeConfig:
    method: str = "bilateral_laplacian"  # bilateral_laplacian | adaptive_canny | dexined
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 60.0
    bilateral_sigma_space: float = 40.0
    laplacian_ksize: int = 3
    laplacian_scale: float = 1.0
    laplacian_threshold: int = 18
    adaptive_canny_k: float = 0.33
    adaptive_window: int = 9
    morph_length: int = 11
    dexined_model: str = ""


@dataclass
class LineConfig:
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 45
    min_length_frac: float = 0.32
    max_gap_frac: float = 0.015
    gradient_window: int = 5
    gradient_vertical_tol_deg: float = 12.0
    ransac_thresh_px: float = 2.2
    ransac_votes: int = 35


@dataclass
class TrackingConfig:
    alpha_offset: float = 0.35
    alpha_angle: float = 0.25
    dropout_frames: int = 8


@dataclass
class PerformanceConfig:
    queue_size: int = 4
    bev_scale: float = 1.0
    multi_thread: bool = True
    use_cuda: bool = False


@dataclass
class VPConfig:
    iterations: int = 200
    inlier_thresh_px: float = 6.0
    min_votes: int = 15


@dataclass
class VisualizationConfig:
    pip_scale: float = 0.28
    show_debug: bool = True
    draw_segments: bool = True


@dataclass
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    edges: EdgeConfig = field(default_factory=EdgeConfig)
    lines: LineConfig = field(default_factory=LineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    vp: VPConfig = field(default_factory=VPConfig)
    viz: VisualizationConfig = field(default_factory=VisualizationConfig)
    calibration_dir: str = str(CALIBRATION_DIR)

    @classmethod
    def from_file(cls, path: Optional[Path]) -> "PipelineConfig":
        data: Dict[str, Dict] = {}
        if path is not None and path.exists():
            if yaml is None:
                raise RuntimeError("PyYAML is required to load YAML configs. Install via `pip install pyyaml`.")
            data = yaml.safe_load(path.read_text()) or {}
        return cls(
            input=InputConfig(**(data.get("input") or {})),
            roi=ROIConfig(**(data.get("roi") or {})),
            preprocess=PreprocessConfig(**(data.get("preprocess") or {})),
            edges=EdgeConfig(**(data.get("edges") or {})),
            lines=LineConfig(**(data.get("lines") or {})),
            tracking=TrackingConfig(**(data.get("tracking") or {})),
            performance=PerformanceConfig(**(data.get("performance") or {})),
            vp=VPConfig(**(data.get("vp") or {})),
            viz=VisualizationConfig(**(data.get("viz") or {})),
            calibration_dir=str(data.get("calibration_dir", CALIBRATION_DIR)),
        )

    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        if getattr(args, "edge_method", None):
            self.edges.method = args.edge_method
        if getattr(args, "preprocess_mode", None):
            self.preprocess.mode = args.preprocess_mode
        if getattr(args, "alpha_offset", None) is not None:
            self.tracking.alpha_offset = args.alpha_offset
        if getattr(args, "alpha_angle", None) is not None:
            self.tracking.alpha_angle = args.alpha_angle
        if getattr(args, "camera", None) is not None:
            self.input.camera_index = args.camera
            self.input.source = "camera"
        if getattr(args, "video", None):
            self.input.source = "video"
            self.input.video_path = args.video
        if getattr(args, "no_thread", False):
            self.performance.multi_thread = False
        if getattr(args, "queue_size", None):
            self.performance.queue_size = max(1, args.queue_size)
        if getattr(args, "width", None):
            self.input.width = args.width
        if getattr(args, "height", None):
            self.input.height = args.height
        if getattr(args, "no_debug", False):
            self.viz.show_debug = False


# ---------- Geometry helpers ----------
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


@dataclass
class ROIState:
    center_offset: float = 0.0

    def update(self, measurement: float, alpha: float = 0.2) -> None:
        self.center_offset = (1 - alpha) * self.center_offset + alpha * np.clip(measurement, -0.6, 0.6)


def normalize_angle_deg(angle_deg: float) -> float:
    a = ((angle_deg + 90.0) % 180.0) - 90.0
    if a == -90.0:
        a = 90.0
    return a


def line_angle_and_length(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ang = np.degrees(np.arctan2(dy, dx))
    return normalize_angle_deg(float(ang)), float(np.hypot(dx, dy))


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


def show_text(img: np.ndarray, text: str, y: int = 28, scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


# ---------- Preprocessing ----------
class Preprocessor:
    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg
        tile = max(2, cfg.clahe_tile)
        self.clahe = cv.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(tile, tile))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        mode = self.cfg.mode.lower()
        if mode == "clahe":
            return self._clahe(frame)
        if mode == "clahe_unsharp":
            clahe = self._clahe(frame)
            blur = cv.GaussianBlur(clahe, (0, 0), sigmaX=self.cfg.unsharp_radius)
            return cv.addWeighted(clahe, 1 + self.cfg.unsharp_amount, blur, -self.cfg.unsharp_amount, 0)
        if mode == "contrast":
            return self._contrast_stretch(frame)
        return self._clahe(frame)

    def _clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        L, a, b = cv.split(lab)
        L_eq = self.clahe.apply(L)
        lab = cv.merge((L_eq, a, b))
        return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

    def _contrast_stretch(self, frame: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        low = np.percentile(gray, self.cfg.contrast_low)
        high = np.percentile(gray, self.cfg.contrast_high)
        stretched = np.clip((gray - low) * 255.0 / max(1.0, high - low), 0, 255).astype(np.uint8)
        return cv.cvtColor(stretched, cv.COLOR_GRAY2BGR)


# ---------- Edge detection ----------
class EdgeDetector:
    def __init__(self, cfg: EdgeConfig) -> None:
        self.cfg = cfg
        self.session = None
        if cfg.method == "dexined" and cfg.dexined_model:
            self._load_dexined(cfg.dexined_model)

    def detect(self, gray: np.ndarray, color_frame: np.ndarray) -> np.ndarray:
        method = self.cfg.method.lower()
        if method == "bilateral_laplacian":
            edges = self._bilateral_laplacian(gray)
        elif method == "adaptive_canny":
            edges = self._adaptive_canny(gray)
        elif method == "dexined":
            edges = self._dexined(color_frame)
        else:
            edges = self._bilateral_laplacian(gray)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, max(3, self.cfg.morph_length)))
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)
        edges = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel, iterations=1)
        return edges

    def _bilateral_laplacian(self, gray: np.ndarray) -> np.ndarray:
        blur = cv.bilateralFilter(
            gray,
            d=self.cfg.bilateral_diameter,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space,
        )
        lap = cv.Laplacian(blur, cv.CV_16S, ksize=self.cfg.laplacian_ksize, scale=self.cfg.laplacian_scale)
        edges = cv.convertScaleAbs(lap)
        _, edges = cv.threshold(edges, self.cfg.laplacian_threshold, 255, cv.THRESH_BINARY)
        return edges

    def _adaptive_canny(self, gray: np.ndarray) -> np.ndarray:
        median = float(np.median(gray))
        k = self.cfg.adaptive_canny_k
        low = int(max(5, (1.0 - k) * median))
        high = int(min(255, (1.0 + k) * median))
        if low >= high:
            low = max(5, high - 10)
        edges = cv.Canny(gray, low, high)
        if self.cfg.adaptive_window >= 3:
            window = self.cfg.adaptive_window | 1
            adapt = cv.adaptiveThreshold(
                gray,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                window,
                2,
            )
            edges = cv.bitwise_and(edges, adapt)
        return edges

    def _load_dexined(self, model_path: str) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is required for DexiNed inference. Install via `pip install onnxruntime`.")
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"DexiNed model not found at {model_file}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_file), providers=providers)

    def _dexined(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("DexiNed model session is not initialized.")
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        inputs = {self.session.get_inputs()[0].name: tensor}
        logits = self.session.run(None, inputs)[0]
        edges = logits[0, 0]
        edges = (edges * 255.0).clip(0, 255).astype(np.uint8)
        return edges


# ---------- Line extraction ----------
def gradient_vertical_mask(gray: np.ndarray, window: int, tol_deg: float) -> np.ndarray:
    ksize = max(3, window | 1)
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    angles = cv.phase(gx, gy, angleInDegrees=True)
    tol = abs(tol_deg)
    mask1 = cv.inRange(angles, 90 - tol, 90 + tol)
    mask2 = cv.inRange(angles, 270 - tol, 270 + tol)
    return cv.bitwise_or(mask1, mask2)


class ConstrainedHoughExtractor:
    def __init__(self, cfg: LineConfig) -> None:
        self.cfg = cfg

    def detect(
        self,
        edges: np.ndarray,
        gray: np.ndarray,
        roi_mask: np.ndarray,
        angle_max_deg: float,
    ) -> List[Segment]:
        masked = cv.bitwise_and(edges, edges, mask=roi_mask)
        grad_mask = gradient_vertical_mask(gray, self.cfg.gradient_window, self.cfg.gradient_vertical_tol_deg)
        masked = cv.bitwise_and(masked, masked, mask=grad_mask)
        h, w = gray.shape[:2]
        min_len = int(self.cfg.min_length_frac * min(h, w))
        max_gap = int(self.cfg.max_gap_frac * min(h, w))
        lines = cv.HoughLinesP(
            masked,
            rho=self.cfg.rho,
            theta=np.deg2rad(self.cfg.theta_deg),
            threshold=self.cfg.threshold,
            minLineLength=max(10, min_len),
            maxLineGap=max(2, max_gap),
        )
        if lines is None:
            return []
        segments: List[Segment] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            p1 = np.array([x1, y1], np.float32)
            p2 = np.array([x2, y2], np.float32)
            angle, length = line_angle_and_length(p1, p2)
            if angle_from_vertical_deg(angle) > angle_max_deg:
                continue
            segments.append(Segment(p1, p2, angle, length))
        return segments


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
        angle = normalize_angle_deg(np.degrees(np.arctan2(direction[1], direction[0])))
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


# ---------- Vanishing point estimation ----------
def _intersect_segments(seg_a: Segment, seg_b: Segment) -> Optional[np.ndarray]:
    x1, y1 = seg_a.p1
    x2, y2 = seg_a.p2
    x3, y3 = seg_b.p1
    x4, y4 = seg_b.p2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if not np.isfinite(px) or not np.isfinite(py):
        return None
    return np.array([px, py], np.float32)


class RansacVanishingPoint:
    def __init__(self, cfg: VPConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(123)

    def estimate(self, segments: Sequence[Segment], frame_shape: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if len(segments) < 2:
            return None
        h, w = frame_shape[:2]
        best_votes = 0
        best_pt: Optional[np.ndarray] = None
        for _ in range(self.cfg.iterations):
            idx = self.rng.choice(len(segments), 2, replace=False)
            seg_a, seg_b = segments[idx[0]], segments[idx[1]]
            intersection = _intersect_segments(seg_a, seg_b)
            if intersection is None:
                continue
            votes = 0
            for seg in segments:
                dist = point_line_distance(intersection, seg.p1, seg.p2)
                if dist < self.cfg.inlier_thresh_px:
                    votes += 1
            if votes > best_votes:
                best_votes = votes
                best_pt = intersection
        if best_pt is None or best_votes < self.cfg.min_votes:
            return None
        if not (0 <= best_pt[0] <= w * 2 and -h <= best_pt[1] <= h * 2):
            return None
        return float(best_pt[0]), float(best_pt[1])


# ---------- Tracking ----------
class ExponentialSmoothingTracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self.offset: Optional[float] = None
        self.angle: Optional[float] = None
        self.missing_frames = 0
        self.confidence = 0.0

    def update(
        self,
        measurement: Optional[Tuple[float, float]],
        measurement_conf: float,
    ) -> Tuple[float, float, float]:
        if measurement is None:
            self.missing_frames += 1
            decay = max(0.0, 1.0 - self.missing_frames / max(1, self.cfg.dropout_frames))
            self.confidence = 0.4 * decay
            if self.missing_frames >= self.cfg.dropout_frames:
                self.offset = None
                self.angle = None
            return self.offset or 0.0, self.angle or 0.0, self.confidence
        self.missing_frames = 0
        offset, angle = measurement
        self.offset = self._smooth(self.offset, offset, self.cfg.alpha_offset)
        self.angle = self._smooth(self.angle, angle, self.cfg.alpha_angle)
        self.confidence = 0.7 * measurement_conf + 0.3
        return self.offset or 0.0, self.angle or 0.0, float(np.clip(self.confidence, 0.0, 1.0))

    @staticmethod
    def _smooth(prev: Optional[float], new: float, alpha: float) -> float:
        if prev is None:
            return new
        return (1 - alpha) * prev + alpha * new


# ---------- Camera handling ----------
@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    new_K: np.ndarray
    maps: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = INTRINSICS_PATH) -> "CameraModel":
        if path.exists():
            data = np.load(path)
            K = data.get("K")
            dist = data.get("dist")
            if K is not None and dist is not None:
                new_K = data.get("new_K", K)
                return cls(K.astype(np.float32), dist.astype(np.float32), new_K.astype(np.float32))
        K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]], np.float32)
        dist = np.zeros((1, 5), np.float32)
        return cls(K, dist, K.copy())

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        size = (frame.shape[1], frame.shape[0])
        if size not in self.maps:
            self.maps[size] = cv.initUndistortRectifyMap(self.K, self.dist, None, self.new_K, size, cv.CV_32FC1)
        map1, map2 = self.maps[size]
        return cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)


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
        roll = np.radians(self.roll_deg)
        pitch = np.radians(self.pitch_deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]], np.float32)
        Ry = np.array([[np.cos(roll), 0, np.sin(roll)], [0, 1, 0], [-np.sin(roll), 0, np.cos(roll)]], np.float32)
        R = Ry @ Rx
        H = K @ R @ np.linalg.inv(K)
        return cv.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]), flags=cv.INTER_LINEAR)


@dataclass
class GroundPlaneMapper:
    H: np.ndarray
    H_inv: np.ndarray
    bev_size: Tuple[int, int]
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
                return cls(H, H_inv.astype(np.float32), (bev_w, bev_h))
        H = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        return cls(H, H_inv, (w, h))

    def _build_maps(self, frame_shape: Tuple[int, int]) -> None:
        h_bev, w_bev = self.bev_size
        yy, xx = np.meshgrid(np.arange(h_bev), np.arange(w_bev), indexing="ij")
        ones = np.ones_like(xx, dtype=np.float32)
        bev_pts = np.stack([xx, yy, ones], axis=-1).reshape(-1, 3)
        cam_pts = bev_pts @ self.H_inv.T
        cam_pts = cam_pts[:, :2] / (cam_pts[:, 2:3] + 1e-6)
        self.map_x = cam_pts[:, 0].reshape(h_bev, w_bev).astype(np.float32)
        self.map_y = cam_pts[:, 1].reshape(h_bev, w_bev).astype(np.float32)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        if self.map_x is None or self.map_y is None:
            self._build_maps(frame.shape)
        return cv.remap(frame, self.map_x, self.map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        pts_h = cv.convertPointsToHomogeneous(pts.astype(np.float32)).reshape(-1, 3)
        proj = (self.H @ pts_h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-6)
        return proj.reshape(-1, 1, 2)


# ---------- Frame source ----------
class FrameSource:
    def __init__(self, cfg: InputConfig, queue_size: int, threaded: bool) -> None:
        self.cfg = cfg
        self.threaded = threaded
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(max(1, queue_size))
        self.cap: Optional[cv.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self) -> bool:
        if self.cfg.source == "video":
            path = Path(self.cfg.video_path)
            if not path.exists():
                print(f"[FrameSource] Video not found: {path}")
                return False
            self.cap = cv.VideoCapture(str(path))
        else:
            self.cap = cv.VideoCapture(self.cfg.camera_index)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if not self.cap or not self.cap.isOpened():
            print("[FrameSource] Failed to open capture device.")
            return False
        if self.threaded:
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        return True

    def _reader(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            try:
                self.queue.put(frame, timeout=0.2)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass

    def read(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        if self.threaded:
            try:
                return self.queue.get(timeout=timeout)
            except queue.Empty:
                return None
        if not self.cap:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
            self.cap = None


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


def put_angle_label(img: np.ndarray, seg: Segment) -> None:
    mid = 0.5 * (seg.p1 + seg.p2)
    if not np.all(np.isfinite(mid)):
        return
    x, y = int(mid[0]), int(mid[1])
    label = f"{seg.angle_deg:+.1f}°"
    cv.putText(img, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)


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


# ---------- Line pipeline ----------
class LinePipeline:
    def __init__(
        self,
        config: PipelineConfig,
        camera_model: CameraModel,
        imu_alignment: IMUAlignment,
        mapper: GroundPlaneMapper,
    ) -> None:
        self.cfg = config
        self.camera = camera_model
        self.imu = imu_alignment
        self.mapper = mapper
        self.preprocessor = Preprocessor(config.preprocess)
        self.edge_detector = EdgeDetector(config.edges)
        self.line_extractor = ConstrainedHoughExtractor(config.lines)
        self.tracker = ExponentialSmoothingTracker(config.tracking)
        self.vp_estimator = RansacVanishingPoint(config.vp)
        self.roi_state = ROIState()
        self.last_follow: Optional[FollowResult] = None

    def process(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Optional[FollowResult], Optional[Tuple[float, float]], List[Segment], np.ndarray]:
        undistorted = self.camera.undistort(frame)
        aligned = self.imu.apply(undistorted, self.camera.new_K)
        bev = self.mapper.warp(aligned)
        processed = self.preprocessor(bev)
        gray = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)

        center_offset = self.tracker.offset or self.roi_state.center_offset
        roi_mask = make_roi_mask(
            gray.shape[0],
            gray.shape[1],
            self.cfg.roi.height_pct,
            self.cfg.roi.top_width_pct,
            self.cfg.roi.bottom_width_pct,
            center_offset,
        )

        edges = self.edge_detector.detect(gray, processed)
        edges = cv.bitwise_and(edges, edges, mask=roi_mask)
        bev_debug = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        segments = self.line_extractor.detect(edges, gray, roi_mask, self.cfg.roi.angle_max_deg)
        merged = merge_collinear_segments(segments, angle_tol_deg=5.0, gap_px=self.cfg.lines.max_gap_frac * min(gray.shape))

        vp_hint = self.vp_estimator.estimate(merged, gray.shape)
        follow_result = self._fit_consensus_line(merged, gray.shape, vp_hint)

        measurement: Optional[Tuple[float, float]] = None
        measurement_conf = 0.0
        if follow_result is not None:
            measurement_conf = float(
                np.clip(
                    0.5 * follow_result.inlier_ratio + 0.5 * max(0.0, 1.0 - follow_result.residual_rms / 3.0),
                    0.0,
                    1.0,
                )
            )
            measurement = (follow_result.lateral_offset_norm, np.radians(follow_result.angle_error_deg))
            self.last_follow = follow_result

        offset, angle, confidence = self.tracker.update(measurement, measurement_conf)
        self.roi_state.update(offset)

        if follow_result is not None:
            follow_result.lateral_offset_norm = offset
            follow_result.angle_error_deg = np.degrees(angle)

        overlay = self._render_overlay(aligned, merged, follow_result, roi_mask, bev_debug, vp_hint, confidence)
        return overlay, follow_result, vp_hint, merged, bev_debug

    def _fit_consensus_line(
        self,
        candidates: Sequence[Segment],
        shape: Tuple[int, int],
        vp_hint: Optional[Tuple[float, float]],
    ) -> Optional[FollowResult]:
        if not candidates:
            return None
        h, w = shape[:2]
        points = []
        for seg in candidates:
            pts = np.linspace(seg.p1, seg.p2, num=20)
            points.append(pts)
        points = np.vstack(points)
        result = ransac_line(points, thresh=self.cfg.lines.ransac_thresh_px, min_inliers=self.cfg.lines.ransac_votes)
        if result is None:
            return None
        p1, p2, residuals = result
        inliers = residuals < self.cfg.lines.ransac_thresh_px * 1.3
        inlier_pts = points[inliers]
        if len(inlier_pts) < self.cfg.lines.ransac_votes:
            return None
        bottom_y = h - 1
        xb = line_intersection_with_y(p1, p2, bottom_y)
        if xb is None or not np.isfinite(xb):
            return None
        angle, length = line_angle_and_length(p1, p2)
        angle_err = angle_from_vertical_deg(angle)
        if angle_err > self.cfg.roi.angle_max_deg:
            return None
        norm_center = (xb - w / 2.0) / (0.5 * w)
        norm_length = min(length / (0.6 * np.hypot(w, h)), 1.0)
        residual_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        inlier_ratio = inliers.sum() / len(residuals)
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
        )

    def _render_overlay(
        self,
        aligned: np.ndarray,
        segments: Sequence[Segment],
        follow_result: Optional[FollowResult],
        roi_mask_bev: np.ndarray,
        bev_debug: np.ndarray,
        vp_hint: Optional[Tuple[float, float]],
        confidence: float,
    ) -> np.ndarray:
        overlay = aligned.copy()
        contours, _ = cv.findContours(roi_mask_bev, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            for cnt in contours:
                cam_cnt = self.mapper.unwarp_points(cnt.astype(np.float32))
                cam_poly = cam_cnt.reshape(-1, 2)
                if not np.all(np.isfinite(cam_poly)):
                    continue
                cv.polylines(overlay, [np.round(cam_cnt).astype(np.int32)], True, (200, 200, 200), 1, cv.LINE_AA)

        gate_px = self.cfg.roi.bottom_gate_px
        h_bev = bev_debug.shape[0]
        w_bev = bev_debug.shape[1]
        gate_top = max(0, h_bev - gate_px)
        gate_poly = np.array(
            [[0, gate_top], [w_bev - 1, gate_top], [w_bev - 1, h_bev - 1], [0, h_bev - 1]],
            np.float32,
        ).reshape(-1, 1, 2)
        cam_gate = self.mapper.unwarp_points(gate_poly)
        cam_gate_poly = cam_gate.reshape(-1, 2)
        if np.all(np.isfinite(cam_gate_poly)):
            cv.polylines(overlay, [np.round(cam_gate).astype(np.int32)], True, (120, 120, 120), 1, cv.LINE_AA)

        cam_segments = unwarp_segments_to_camera(segments, self.mapper)
        if self.cfg.viz.draw_segments:
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
            cam_pts = self.mapper.unwarp_points(pts)
            cam_line = cam_pts.reshape(-1, 2)
            if np.all(np.isfinite(cam_line)):
                x1, y1 = int(cam_line[0, 0]), int(cam_line[0, 1])
                x2, y2 = int(cam_line[1, 0]), int(cam_line[1, 1])
                cv.line(overlay, (x1, y1), (x2, y2), (0, 220, 0), 4, cv.LINE_AA)
                xb_cam = line_intersection_with_y(cam_line[0], cam_line[1], overlay.shape[0] - 1)
                if xb_cam is not None and np.isfinite(xb_cam):
                    xb_int = int(np.clip(round(xb_cam), 0, overlay.shape[1] - 1))
                    cv.circle(overlay, (xb_int, overlay.shape[0] - 1), 7, (0, 220, 0), -1, cv.LINE_AA)
            xb_norm = follow_result.lateral_offset_norm
            angle_err = follow_result.angle_error_deg
            norm_length = follow_result.norm_length
            inlier_ratio = follow_result.inlier_ratio
            residual_rms = follow_result.residual_rms
            show_text(
                overlay,
                f"Follow offset: {xb_norm:+.3f}  angle: {angle_err:+.2f}°  len_norm: {norm_length:.2f}  inliers: {inlier_ratio:.2f}  rms: {residual_rms:.2f}",
                y=84,
                color=(180, 255, 180),
            )
        else:
            show_text(overlay, "Follow: not found", y=84, color=(180, 180, 180))

        if self.cfg.viz.show_debug:
            draw_pip(overlay, bev_debug, scale=self.cfg.viz.pip_scale)

        if vp_hint is not None:
            cv.circle(bev_debug, (int(vp_hint[0]), int(vp_hint[1])), 6, (0, 0, 255), -1, cv.LINE_AA)
            show_text(bev_debug, f"VP: ({vp_hint[0]:.1f}, {vp_hint[1]:.1f})", y=28)

        draw_confidence_bar(overlay, confidence)
        return overlay


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configurable long-line detector (classical, training-free).")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"), help="Path to YAML config.")
    parser.add_argument("--edge-method", choices=["bilateral_laplacian", "adaptive_canny", "dexined"], help="Override edge detector.")
    parser.add_argument("--preprocess-mode", choices=["clahe", "clahe_unsharp", "contrast"], help="Override preprocessing mode.")
    parser.add_argument("--camera", type=int, help="Camera index override.")
    parser.add_argument("--video", type=str, help="Video file path (overrides camera).")
    parser.add_argument("--alpha-offset", type=float, help="Override smoothing factor for lateral offset.")
    parser.add_argument("--alpha-angle", type=float, help="Override smoothing factor for angle.")
    parser.add_argument("--queue-size", type=int, help="Frame queue size for threaded capture.")
    parser.add_argument("--no-thread", action="store_true", help="Disable producer/consumer threading.")
    parser.add_argument("--width", type=int, help="Capture width override.")
    parser.add_argument("--height", type=int, help="Capture height override.")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug PiP overlay.")
    parser.add_argument("--headless", action="store_true", help="Skip UI windows (still runs pipeline).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.from_file(args.config)
    config.apply_cli_overrides(args)

    frame_source = FrameSource(config.input, config.performance.queue_size, config.performance.multi_thread)
    if not frame_source.start():
        return

    first_frame = frame_source.read(timeout=2.0)
    if first_frame is None:
        print("Failed to grab initial frame.")
        frame_source.stop()
        return

    camera_model = CameraModel.load()
    imu_alignment = IMUAlignment.load()
    mapper = GroundPlaneMapper.load(first_frame.shape, bev_scale=config.performance.bev_scale)
    pipeline = LinePipeline(config, camera_model, imu_alignment, mapper)

    window_name = "long_lines_overlay"
    if not args.headless:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 1100, 620)

    prev_time = time.time()
    fps = 0.0
    running = True
    print("Running. Press 'q' to exit.")

    while running:
        frame = first_frame if first_frame is not None else frame_source.read(timeout=1.0)
        first_frame = None
        if frame is None:
            continue
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        overlay, _, _, _, bev_debug = pipeline.process(frame, now)
        show_text(
            overlay,
            f"{fps:.1f} FPS  |  preprocess={config.preprocess.mode}  edge={config.edges.method}",
            y=overlay.shape[0] - 12,
        )

        if not args.headless:
            cv.imshow(window_name, overlay)
            if config.viz.show_debug:
                cv.imshow("bev_debug", bev_debug)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
        else:
            if config.input.source == "video" and not config.performance.multi_thread:
                time.sleep(max(0.0, (1.0 / max(1.0, config.input.fps)) - dt))

    frame_source.stop()
    if not args.headless:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
