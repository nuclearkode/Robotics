#!/usr/bin/env python3
"""
Interactive long-line detector with modernized classical upgrades.

This implementation addresses the requested remediation list:

1. Percentile Canny → selectable edge frontends:
   * Bilateral + Laplacian (default, tuneless and fast)
   * Adaptive Canny with local mean normalization
   * DexiNed ONNX fallback for bullet-proof edges
2. Retinex → CLAHE-only preprocessing (plus optional presets).
3. LSD → constrained, orientation-aware HoughLinesP extractor.
4. 4D Kalman → exponential smoothing (alpha configurable) with an optional
   2D Kalman variant when extra stability is required.
5. Hard-coded trackbars → reproducible YAML + argparse configuration file.

In addition, the following performance optimizations are built-in:
* Cached BEV remap maps via initUndistortRectifyMap-style precomputation.
* Multi-threaded frame grabbing (producer/consumer).
* RANSAC-style vanishing-point voting instead of O(N²) enumeration.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - hard failure if PyYAML missing
    raise SystemExit("PyYAML is required (pip install pyyaml).") from exc


# --------------------------------------------------------------------------- #
# Configuration handling                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    bev_scale: float = 1.0
    preprocess: str = "clahe"
    preprocess_kwargs: Dict[str, float] = field(default_factory=dict)
    edge_method: str = "bilateral_laplacian"
    edge_kwargs: Dict[str, float] = field(default_factory=dict)
    roi_height_pct: float = 0.55
    roi_top_width_pct: float = 0.35
    roi_bottom_width_pct: float = 1.0
    min_line_length_px: int = 120
    max_line_gap_px: int = 16
    hough_threshold: int = 60
    max_hough_lines: int = 96
    angle_window_deg: float = 18.0
    bottom_gate_px: int = 42
    smoothing_alpha: float = 0.25
    smoothing_beta: float = 0.85
    tracking_mode: str = "smooth"  # smooth | kalman2d
    gradient_prefilter: bool = True
    bev_homography_path: str = "calibration/ground_plane_h.npz"
    intrinsics_path: str = "calibration/camera_model.npz"
    dexined_path: Optional[str] = None
    queue_size: int = 4
    headless: bool = False
    display_debug: bool = True
    ransac_iterations: int = 200
    ransac_tolerance_px: float = 12.0
    vp_consistency_weight: float = 0.15
    orientation_weight: float = 0.35
    length_weight: float = 0.25
    coverage_weight: float = 0.25
    config_out: Optional[str] = None


def _load_yaml_config(path: Optional[Path]) -> Dict:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping.")
    return data


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    yaml_cfg = _load_yaml_config(args.config)
    for key, value in yaml_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cli_overrides = {
        "camera_index": args.camera_index,
        "edge_method": args.edge_method,
        "preprocess": args.preprocess,
        "tracking_mode": args.tracking_mode,
        "dexined_path": args.dexined_path,
        "headless": args.headless,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.width:
        cfg.width = args.width
    if args.height:
        cfg.height = args.height
    if args.config_out:
        cfg.config_out = args.config_out
    if cfg.config_out:
        out_path = Path(cfg.config_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(cfg), f, indent=2, sort_keys=True)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modernized long-line detector")
    parser.add_argument("--config", type=Path, help="YAML config path")
    parser.add_argument("--config-out", dest="config_out", type=Path, help="Dump resolved config")
    parser.add_argument("--camera-index", type=int, dest="camera_index", help="Camera index override")
    parser.add_argument("--width", type=int, help="Capture width override")
    parser.add_argument("--height", type=int, help="Capture height override")
    parser.add_argument("--edge-method", choices=["bilateral_laplacian", "adaptive_canny", "dexined"])
    parser.add_argument("--preprocess", choices=["clahe", "clahe_unsharp", "contrast_stretch"])
    parser.add_argument("--tracking-mode", choices=["smooth", "kalman2d"])
    parser.add_argument("--dexined-path", type=str, help="Path to DexiNed ONNX model")
    parser.add_argument("--headless", action="store_true", help="Disable OpenCV windows")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Camera + warping helpers                                                    #
# --------------------------------------------------------------------------- #


def _default_intrinsics(width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = fy = width
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    dist = np.zeros((1, 5), np.float32)
    return K, dist, K.copy()


class BirdsEyeWarper:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._load_intrinsics()
        self._load_homography()

    def _load_intrinsics(self) -> None:
        path = Path(self.cfg.intrinsics_path)
        if path.exists():
            data = np.load(path)
            self.K = data.get("K")
            self.dist = data.get("dist")
            self.new_K = data.get("new_K", self.K)
            if self.K is None or self.dist is None:
                self.K, self.dist, self.new_K = _default_intrinsics(self.cfg.width, self.cfg.height)
        else:
            self.K, self.dist, self.new_K = _default_intrinsics(self.cfg.width, self.cfg.height)
        self.K = self.K.astype(np.float32)
        self.dist = self.dist.astype(np.float32)
        self.new_K = self.new_K.astype(np.float32)

    def _load_homography(self) -> None:
        path = Path(self.cfg.bev_homography_path)
        if path.exists():
            data = np.load(path)
            H = data.get("H")
            if H is None:
                H = np.eye(3, dtype=np.float32)
        else:
            H = np.eye(3, dtype=np.float32)
        self.H = H.astype(np.float32)
        self.H_inv = np.linalg.inv(self.H)
        bev_w = int(self.cfg.width * self.cfg.bev_scale)
        bev_h = int(self.cfg.height * self.cfg.bev_scale)
        self.bev_size = (bev_w, bev_h)
        self._precompute_maps()

    def _precompute_maps(self) -> None:
        bev_w, bev_h = self.bev_size
        grid_x, grid_y = np.meshgrid(np.arange(bev_w), np.arange(bev_h))
        ones = np.ones_like(grid_x, dtype=np.float32)
        pts = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3).T
        src = self.H_inv @ pts
        src /= src[2:3]
        map_x = src[0].reshape(bev_h, bev_w).astype(np.float32)
        map_y = src[1].reshape(bev_h, bev_w).astype(np.float32)
        self.map_x = map_x
        self.map_y = map_y

    def warp(self, frame: np.ndarray) -> np.ndarray:
        undistorted = cv.undistort(frame, self.K, self.dist, None, self.new_K)
        return cv.remap(undistorted, self.map_x, self.map_y, interpolation=cv.INTER_LINEAR)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        pts_h = cv.convertPointsToHomogeneous(pts.astype(np.float32)).reshape(-1, 3)
        proj = (self.H_inv @ pts_h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-9)
        return proj.reshape(-1, 1, 2)


class FrameProducer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.cap = cv.VideoCapture(cfg.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera index {cfg.camera_index} could not be opened.")
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=cfg.queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            try:
                self.queue.put(frame, timeout=0.01)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.queue.put(frame, timeout=0.01)
                except queue.Full:
                    pass

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


# --------------------------------------------------------------------------- #
# Preprocessing                                                               #
# --------------------------------------------------------------------------- #


def clahe_only(frame: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    L = clahe.apply(L)
    lab = cv.merge((L, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def clahe_unsharp(frame: np.ndarray, strength: float = 1.3, radius: float = 1.2, **kwargs) -> np.ndarray:
    clahe_img = clahe_only(frame, **kwargs)
    blur = cv.GaussianBlur(clahe_img, (0, 0), radius)
    sharpened = cv.addWeighted(clahe_img, strength, blur, -(strength - 1.0), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def contrast_stretch(frame: np.ndarray) -> np.ndarray:
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)
    Lf = L.astype(np.float32)
    Lf = (Lf - Lf.min()) / (Lf.ptp() + 1e-6)
    stretched = np.clip(Lf * 255.0, 0, 255).astype(np.uint8)
    lab = cv.merge((stretched, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def build_preprocessor(cfg: PipelineConfig):
    if cfg.preprocess == "clahe":
        return lambda frame: clahe_only(frame, **cfg.preprocess_kwargs)
    if cfg.preprocess == "clahe_unsharp":
        return lambda frame: clahe_unsharp(frame, **cfg.preprocess_kwargs)
    if cfg.preprocess == "contrast_stretch":
        return contrast_stretch
    raise ValueError(f"Unknown preprocess mode: {cfg.preprocess}")


# --------------------------------------------------------------------------- #
# Edge detection                                                              #
# --------------------------------------------------------------------------- #


class BilateralLaplacianEdgeDetector:
    def __init__(self, diameter: int = 5, sigma_color: float = 90.0, sigma_space: float = 90.0, percentile: float = 85.0):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.percentile = percentile

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        smooth = cv.bilateralFilter(gray, self.diameter, self.sigma_color, self.sigma_space)
        lap = cv.Laplacian(smooth, cv.CV_32F, ksize=3)
        lap = np.abs(lap)
        thresh = np.percentile(lap, self.percentile)
        _, binary = cv.threshold(lap, thresh, 255, cv.THRESH_BINARY)
        return binary.astype(np.uint8)


class AdaptiveCannyEdgeDetector:
    def __init__(self, window: int = 25, offset: float = 4.0, low_pct: float = 35.0, high_pct: float = 85.0):
        window = window if window % 2 == 1 else window + 1
        self.window = window
        self.offset = offset
        self.low_pct = low_pct
        self.high_pct = high_pct

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        blur = cv.GaussianBlur(gray, (0, 0), 1.2)
        adaptive = cv.adaptiveThreshold(
            blur,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            self.window,
            self.offset,
        )
        weights = blur[adaptive > 0]
        if weights.size == 0:
            weights = blur.reshape(-1)
        low = np.percentile(weights, self.low_pct)
        high = np.percentile(weights, self.high_pct)
        edges = cv.Canny(blur, int(max(1, low * 0.66)), int(max(low + 1, high)))
        return edges


class DexiNedEdgeDetector:
    def __init__(self, model_path: str, threshold: float = 0.3, device: str = "cpu"):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("onnxruntime is required for DexiNed mode.") from exc
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        preds = self.session.run(None, {self.input_name: tensor})[0]
        pred = preds[0]
        if pred.ndim == 3:
            pred = pred[0]
        pred = cv.resize(pred, (frame_bgr.shape[1], frame_bgr.shape[0]))
        edges = (pred > self.threshold).astype(np.uint8) * 255
        return edges


def build_edge_detector(cfg: PipelineConfig):
    if cfg.edge_method == "bilateral_laplacian":
        return BilateralLaplacianEdgeDetector(**cfg.edge_kwargs)
    if cfg.edge_method == "adaptive_canny":
        return AdaptiveCannyEdgeDetector(**cfg.edge_kwargs)
    if cfg.edge_method == "dexined":
        if not cfg.dexined_path:
            raise ValueError("dexined edge method selected but dexined_path not provided.")
        return DexiNedEdgeDetector(cfg.dexined_path, **cfg.edge_kwargs)
    raise ValueError(f"Unknown edge method: {cfg.edge_method}")


# --------------------------------------------------------------------------- #
# Line extraction                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    angle_deg: float
    length: float

    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.p1 + self.p2)


def _segment_from_points(p1: np.ndarray, p2: np.ndarray) -> Segment:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    # normalize around vertical
    angle = ((angle + 90.0) % 180.0) - 90.0
    length = float(np.hypot(dx, dy))
    return Segment(p1.astype(np.float32), p2.astype(np.float32), angle, length)


class ConstrainedHoughExtractor:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def __call__(self, edges: np.ndarray, gray: np.ndarray) -> List[Segment]:
        filtered = edges
        if self.cfg.gradient_prefilter:
            grad_x = cv.Scharr(gray, cv.CV_32F, 1, 0)
            grad_y = cv.Scharr(gray, cv.CV_32F, 0, 1)
            angle = cv.phase(grad_x, grad_y, angleInDegrees=True)
            deviation = np.abs(90.0 - np.abs(angle))
            mask = (deviation <= self.cfg.angle_window_deg).astype(np.uint8) * 255
            filtered = cv.bitwise_and(edges, edges, mask=mask)
        lines = cv.HoughLinesP(
            filtered,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.cfg.hough_threshold,
            minLineLength=self.cfg.min_line_length_px,
            maxLineGap=self.cfg.max_line_gap_px,
        )
        segments: List[Segment] = []
        if lines is None:
            return segments
        for x1, y1, x2, y2 in lines[: self.cfg.max_hough_lines, 0]:
            seg = _segment_from_points(np.array([x1, y1], np.float32), np.array([x2, y2], np.float32))
            segments.append(seg)
        return segments


# --------------------------------------------------------------------------- #
# Tracking                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class LineObservation:
    offset_norm: float
    angle_deg: float
    length_norm: float
    bottom_x: float


class ExponentialSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def update(self, obs: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = obs.copy()
        else:
            self.state = self.alpha * obs + (1.0 - self.alpha) * self.state
        return self.state


class SimpleKalman2D:
    def __init__(self, beta: float = 0.85) -> None:
        self.beta = beta
        self.state = np.zeros(2, np.float32)
        self.cov = np.eye(2, dtype=np.float32)
        self.q = np.diag([0.01, 0.01]).astype(np.float32)
        self.r = np.diag([0.1, 0.1]).astype(np.float32)
        self.initialized = False

    def update(self, obs: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.state = obs.copy()
            self.initialized = True
            return self.state
        A = np.eye(2, dtype=np.float32)
        self.state = A @ self.state
        self.cov = A @ self.cov @ A.T + self.q
        K = self.cov @ np.linalg.inv(self.cov + self.r)
        self.state = self.state + K @ (obs - self.state)
        self.cov = (np.eye(2, dtype=np.float32) - K) @ self.cov
        return self.state


class Tracker:
    def __init__(self, cfg: PipelineConfig) -> None:
        if cfg.tracking_mode == "kalman2d":
            self.impl = SimpleKalman2D(beta=cfg.smoothing_beta)
        else:
            self.impl = ExponentialSmoother(alpha=cfg.smoothing_alpha)

    def update(self, obs: LineObservation) -> LineObservation:
        payload = np.array([obs.offset_norm, obs.angle_deg], np.float32)
        smoothed = self.impl.update(payload)
        return LineObservation(
            offset_norm=float(smoothed[0]),
            angle_deg=float(smoothed[1]),
            length_norm=obs.length_norm,
            bottom_x=obs.bottom_x,
        )


# --------------------------------------------------------------------------- #
# Geometry helpers                                                            #
# --------------------------------------------------------------------------- #


def make_roi_mask(
    h: int,
    w: int,
    height_frac: float,
    top_width_frac: float,
    bottom_width_frac: float,
    center_offset_norm: float,
) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    roi_h = int(h * height_frac)
    top_y = max(0, h - roi_h)
    top_w = int(w * top_width_frac)
    bot_w = int(w * bottom_width_frac)
    cx = int(w / 2 + center_offset_norm * w * 0.5)
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


def intersection_with_bottom(seg: Segment, h: int) -> Optional[float]:
    dy = seg.p2[1] - seg.p1[1]
    dx = seg.p2[0] - seg.p1[0]
    if abs(dy) < 1e-4:
        return None
    slope = dy / dx if abs(dx) > 1e-5 else np.inf
    if np.isinf(slope):
        return float(seg.p1[0])
    intercept = seg.p1[1] - slope * seg.p1[0]
    xb = (h - 1 - intercept) / slope
    return float(xb)


def ransac_vanishing_point(
    segments: Sequence[Segment],
    iterations: int,
    tolerance: float,
) -> Optional[Tuple[float, float]]:
    if len(segments) < 2:
        return None
    rng = np.random.default_rng(42)
    best_score = 0
    best_pt = None
    for _ in range(iterations):
        idx = rng.choice(len(segments), 2, replace=False)
        seg_a, seg_b = segments[idx[0]], segments[idx[1]]
        denom = (
            (seg_a.p1[0] - seg_a.p2[0]) * (seg_b.p1[1] - seg_b.p2[1])
            - (seg_a.p1[1] - seg_a.p2[1]) * (seg_b.p1[0] - seg_b.p2[0])
        )
        if abs(denom) < 1e-6:
            continue
        px = (
            (seg_a.p1[0] * seg_a.p2[1] - seg_a.p1[1] * seg_a.p2[0]) * (seg_b.p1[0] - seg_b.p2[0])
            - (seg_a.p1[0] - seg_a.p2[0]) * (seg_b.p1[0] * seg_b.p2[1] - seg_b.p1[1] * seg_b.p2[0])
        ) / denom
        py = (
            (seg_a.p1[0] * seg_a.p2[1] - seg_a.p1[1] * seg_a.p2[0]) * (seg_b.p1[1] - seg_b.p2[1])
            - (seg_a.p1[1] - seg_a.p2[1]) * (seg_b.p1[0] * seg_b.p2[1] - seg_b.p1[1] * seg_b.p2[0])
        ) / denom
        if not np.isfinite(px) or not np.isfinite(py):
            continue
        votes = 0
        for seg in segments:
            dist = point_line_distance(np.array([px, py], np.float32), seg.p1, seg.p2)
            if dist < tolerance:
                votes += 1
        if votes > best_score:
            best_score = votes
            best_pt = (float(px), float(py))
    return best_pt


def point_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return float(np.linalg.norm(point - a))
    ab = b - a
    ap = point - a
    return float(np.abs(ab[0] * ap[1] - ab[1] * ap[0]) / (np.linalg.norm(ab) + 1e-6))


# --------------------------------------------------------------------------- #
# Line detection pipeline                                                     #
# --------------------------------------------------------------------------- #


class LineDetectionPipeline:
    def __init__(self, cfg: PipelineConfig, warper: BirdsEyeWarper):
        self.cfg = cfg
        self.warper = warper
        self.preprocess = build_preprocessor(cfg)
        self.edge_detector = build_edge_detector(cfg)
        self.extractor = ConstrainedHoughExtractor(cfg)
        self.tracker = Tracker(cfg)
        self.roi_center = 0.0
        self.vanishing_point: Optional[Tuple[float, float]] = None

    def process(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        bev = self.warper.warp(frame)
        pre = self.preprocess(bev)
        gray = cv.cvtColor(pre, cv.COLOR_BGR2GRAY)
        mask = make_roi_mask(
            pre.shape[0],
            pre.shape[1],
            self.cfg.roi_height_pct,
            self.cfg.roi_top_width_pct,
            self.cfg.roi_bottom_width_pct,
            self.roi_center,
        )
        roi_edges = self._compute_edges(pre, gray, mask)
        segments = self.extractor(roi_edges, gray)
        vp = ransac_vanishing_point(segments, self.cfg.ransac_iterations, self.cfg.ransac_tolerance_px)
        if vp is not None:
            self.vanishing_point = vp
        best_seg = self._select_best_segment(segments, pre.shape)
        observation = self._build_observation(best_seg, pre.shape)
        smoothed = None
        if observation:
            smoothed = self.tracker.update(observation)
            self.roi_center = 0.7 * self.roi_center + 0.3 * smoothed.offset_norm
        overlay = self._draw_overlay(frame, bev, roi_edges, segments, smoothed, vp)
        debug = {"edges": roi_edges, "bev": bev}
        return overlay, debug

    def _compute_edges(self, pre: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        edges = (
            self.edge_detector(pre if self.cfg.edge_method == "dexined" else gray)
            if self.cfg.edge_method == "dexined"
            else self.edge_detector(gray)
        )
        edges = cv.bitwise_and(edges, edges, mask=mask)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return edges

    def _select_best_segment(self, segments: Sequence[Segment], shape: Tuple[int, int, int]) -> Optional[Segment]:
        if not segments:
            return None
        h, w = shape[:2]
        best_score = float("inf")
        best_seg = None
        for seg in segments:
            xb = intersection_with_bottom(seg, h)
            if xb is None or not np.isfinite(xb):
                continue
            offset_norm = (xb - w / 2.0) / (0.5 * w)
            angle_pen = abs(90.0 - abs(seg.angle_deg)) / max(1.0, self.cfg.angle_window_deg)
            length_pen = 1.0 - min(1.0, seg.length / (0.6 * np.hypot(h, w)))
            coverage_pen = 0.0
            if min(seg.p1[1], seg.p2[1]) < (h - self.cfg.bottom_gate_px):
                coverage_pen = 1.0
            vp_pen = 0.0
            if self.vanishing_point is not None:
                vp_vec = np.array(self.vanishing_point) - seg.midpoint()
                seg_vec = seg.p2 - seg.p1
                cos_sim = np.dot(vp_vec, seg_vec) / (np.linalg.norm(vp_vec) * np.linalg.norm(seg_vec) + 1e-6)
                vp_pen = 1.0 - max(0.0, cos_sim)
            score = (
                self.cfg.orientation_weight * angle_pen
                + self.cfg.length_weight * length_pen
                + self.cfg.coverage_weight * coverage_pen
                + self.cfg.vp_consistency_weight * vp_pen
                + 0.2 * abs(offset_norm)
            )
            if score < best_score:
                best_score = score
                best_seg = seg
        return best_seg

    def _build_observation(self, seg: Optional[Segment], shape: Tuple[int, int, int]) -> Optional[LineObservation]:
        if seg is None:
            return None
        h, w = shape[:2]
        xb = intersection_with_bottom(seg, h)
        if xb is None or not np.isfinite(xb):
            return None
        offset_norm = np.clip((xb - w / 2.0) / (0.5 * w), -1.0, 1.0)
        angle = float(seg.angle_deg)
        length_norm = min(seg.length / (0.7 * np.hypot(h, w)), 1.0)
        return LineObservation(offset_norm=offset_norm, angle_deg=angle, length_norm=length_norm, bottom_x=xb)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        bev: np.ndarray,
        edges: np.ndarray,
        segments_bev: Sequence[Segment],
        smoothed: Optional[LineObservation],
        vp: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        overlay = frame.copy()
        if smoothed is not None:
            xb = int(overlay.shape[1] * (0.5 + smoothed.offset_norm * 0.5))
            cv.line(overlay, (overlay.shape[1] // 2, overlay.shape[0] - 1), (xb, overlay.shape[0] - 1), (0, 220, 0), 2)
            cv.circle(overlay, (xb, overlay.shape[0] - 1), 6, (0, 220, 0), -1)
            text = f"offset={smoothed.offset_norm:+.3f} angle={smoothed.angle_deg:+.2f}° len={smoothed.length_norm:.2f}"
            cv.putText(overlay, text, (32, 48), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(overlay, text, (32, 48), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
        bev_debug = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        for seg in segments_bev:
            cv.line(
                bev_debug,
                (int(seg.p1[0]), int(seg.p1[1])),
                (int(seg.p2[0]), int(seg.p2[1])),
                (80, 220, 255),
                2,
                cv.LINE_AA,
            )
        if vp is not None:
            cv.circle(bev_debug, (int(vp[0]), int(vp[1])), 6, (0, 0, 255), -1)
            cv.putText(
                bev_debug,
                f"VP ({vp[0]:.0f},{vp[1]:.0f})",
                (16, 24),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
        if self.cfg.display_debug and not self.cfg.headless:
            h, w = overlay.shape[:2]
            pip_h = int(h * 0.28)
            pip_w = int(pip_h * bev_debug.shape[1] / bev_debug.shape[0])
            pip = cv.resize(bev_debug, (pip_w, pip_h))
            x0, y0 = w - pip_w - 20, 20
            overlay[y0 : y0 + pip_h, x0 : x0 + pip_w] = pip
        return overlay


# --------------------------------------------------------------------------- #
# Application entry point                                                     #
# --------------------------------------------------------------------------- #


def run() -> None:
    args = parse_args()
    cfg = build_config(args)
    warper = BirdsEyeWarper(cfg)
    producer = FrameProducer(cfg)
    producer.start()
    pipeline = LineDetectionPipeline(cfg, warper)
    fps = 0.0
    last_time = time.time()
    window_name = "long_line_detector_modern"
    if not cfg.headless:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_name, 1200, 720)
    try:
        while True:
            frame = producer.read(timeout=1.0)
            if frame is None:
                continue
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - last_time + 1e-6))
            last_time = now
            overlay, debug = pipeline.process(frame, now)
            if not cfg.headless:
                cv.putText(
                    overlay,
                    f"{fps:.1f} FPS  |  Edge: {cfg.edge_method}  Pre: {cfg.preprocess}",
                    (20, overlay.shape[0] - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    cv.LINE_AA,
                )
                cv.putText(
                    overlay,
                    f"{fps:.1f} FPS  |  Edge: {cfg.edge_method}  Pre: {cfg.preprocess}",
                    (20, overlay.shape[0] - 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )
                cv.imshow(window_name, overlay)
                if cfg.display_debug:
                    cv.imshow("edges", debug["edges"])
                key = cv.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                print(f"{fps:.1f} FPS - no display (headless)")
    finally:
        producer.stop()
        if not cfg.headless:
            cv.destroyAllWindows()


if __name__ == "__main__":
    run()

