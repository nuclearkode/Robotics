"""Modernized long-line detector with reproducible config, faster classical
blocks, and lightweight temporal smoothing.

This implementation folds in the requested upgrades:

* Configurable preprocessing (CLAHE presets) via YAML + argparse.
* Edge detection options: Bilateral+Laplacian (default), Adaptive Canny,
  and DexiNed ONNX for learned edges.
* Constrained, angle-aware HoughLinesP instead of LSD, with optional
  gradient-orientation pre-filtering.
* Exponential smoothing replaces the heavyweight 4D Kalman tracker.
* Cached BEV remap maps (initUndistortRectifyMap-style) remove per-frame
  homography recomputation; producer/consumer threading hides capture
  latency; vanishing-point RANSAC avoids O(N^2) intersections.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from math import degrees, hypot
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import yaml

# ---------- Config -----------------------------------------------------------------


def _list_default(value: Iterable[int]) -> List[int]:
    return list(value)


@dataclass
class PipelineConfig:
    camera_indices: List[int] = field(default_factory=lambda: _list_default([0, 1, 2]))
    width: int = 1280
    height: int = 720
    bev_scale: float = 1.0

    preprocess_mode: str = "clahe"  # clahe | clahe_unsharp | contrast_stretch
    edge_mode: str = "bilateral_laplacian"  # bilateral_laplacian | adaptive_canny | learned
    line_method: str = "constrained_hough"  # constrained_hough only for now

    clip_limit: float = 2.0
    tile_grid: int = 8
    unsharp_amount: float = 0.6

    hough_threshold: int = 60
    min_line_pct: float = 0.35
    max_gap_pct: float = 0.02
    orientation_tolerance_deg: float = 15.0
    angle_max_deg: float = 18.0

    roi_height_pct: float = 0.6
    roi_top_pct: float = 0.32
    bottom_gate_px: int = 40

    tracker_alpha: float = 0.25
    smoothing_rate: float = 0.08

    dexined_model: Optional[str] = None
    vp_iterations: int = 250
    vp_threshold_px: float = 24.0

    queue_size: int = 4
    multi_thread: bool = True
    use_cuda: bool = False

    config_path: Optional[Path] = None

    def update_from_mapping(self, data: Dict[str, Any]) -> None:
        for key, value in data.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, Path):
                    setattr(self, key, Path(value))
                else:
                    setattr(self, key, value)


def load_config(path: Optional[Path]) -> PipelineConfig:
    cfg = PipelineConfig()
    if path is None:
        return cfg
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must map keys to values")
    cfg.update_from_mapping(data)
    cfg.config_path = path
    return cfg


# ---------- Camera + Geometry ------------------------------------------------------

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
INTRINSICS_PATH = CALIBRATION_DIR / "camera_model.npz"
HOMOGRAPHY_PATH = CALIBRATION_DIR / "ground_plane_h.npz"
IMU_PATH = CALIBRATION_DIR / "imu_alignment.json"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    new_K: np.ndarray
    map_x: Optional[np.ndarray] = None
    map_y: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: Path = INTRINSICS_PATH) -> "CameraModel":
        if path.exists():
            data = np.load(path)
            K = data.get("K")
            dist = data.get("dist")
            new_K = data.get("new_K", K)
            if K is not None and dist is not None:
                return cls(K.astype(np.float32), dist.astype(np.float32), new_K.astype(np.float32))
        K = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1]], np.float32)
        dist = np.zeros((1, 5), np.float32)
        return cls(K, dist, K.copy())

    def _ensure_maps(self, frame_shape: Tuple[int, int, int]) -> None:
        if self.map_x is not None and self.map_y is not None:
            return
        h, w = frame_shape[:2]
        self.map_x, self.map_y = cv.initUndistortRectifyMap(
            self.K,
            self.dist,
            None,
            self.new_K,
            (w, h),
            cv.CV_32FC1,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_maps(frame.shape)
        if self.map_x is None or self.map_y is None:
            return frame
        return cv.remap(frame, self.map_x, self.map_y, cv.INTER_LINEAR)


@dataclass
class IMUAlignment:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    @classmethod
    def load(cls, path: Path = IMU_PATH) -> "IMUAlignment":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(float(data.get("roll_deg", 0.0)), float(data.get("pitch_deg", 0.0)))
        except Exception:
            return cls()

    def apply(self, frame: np.ndarray, K: np.ndarray) -> np.ndarray:
        if abs(self.roll_deg) < 1e-3 and abs(self.pitch_deg) < 1e-3:
            return frame
        roll = np.deg2rad(self.roll_deg)
        pitch = np.deg2rad(self.pitch_deg)
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
    map_x: np.ndarray
    map_y: np.ndarray

    @classmethod
    def load(cls, frame_shape: Tuple[int, int, int], path: Path = HOMOGRAPHY_PATH, bev_scale: float = 1.0) -> "GroundPlaneMapper":
        h, w = frame_shape[:2]
        if path.exists():
            data = np.load(path)
            H = data.get("H")
            bev_w = int(data.get("bev_w", w * bev_scale))
            bev_h = int(data.get("bev_h", h * bev_scale))
            if H is not None:
                H = H.astype(np.float32)
                H_inv = np.linalg.inv(H)
                map_x, map_y = cls._build_maps(H_inv, bev_w, bev_h)
                return cls(H, H_inv.astype(np.float32), (bev_w, bev_h), map_x, map_y)
        H = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        map_x, map_y = cls._build_maps(H_inv, int(w * bev_scale), int(h * bev_scale))
        return cls(H, H_inv, (int(w * bev_scale), int(h * bev_scale)), map_x, map_y)

    @staticmethod
    def _build_maps(H_inv: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        homog = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=-1).astype(np.float32)
        src = (H_inv @ homog.T).T
        src_xy = (src[:, :2] / np.clip(src[:, 2:3], 1e-6, None)).reshape(height, width, 2)
        map_x = src_xy[..., 0].astype(np.float32)
        map_y = src_xy[..., 1].astype(np.float32)
        return map_x, map_y

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv.remap(frame, self.map_x, self.map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        pts_h = cv.convertPointsToHomogeneous(pts.astype(np.float32)).reshape(-1, 3)
        proj = (self.H_inv @ pts_h.T).T
        proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
        return proj.reshape(-1, 1, 2)


# ---------- Preprocessing + Edges --------------------------------------------------


def clahe_only(frame: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    L_eq = clahe.apply(L)
    lab = cv.merge((L_eq, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def clahe_unsharp(frame: np.ndarray, clip_limit: float, tile_grid: int, amount: float) -> np.ndarray:
    clahe_img = clahe_only(frame, clip_limit, tile_grid)
    blur = cv.GaussianBlur(clahe_img, (0, 0), 1.2)
    return cv.addWeighted(clahe_img, 1 + amount, blur, -amount, 0)


def contrast_stretch(frame: np.ndarray) -> np.ndarray:
    f32 = frame.astype(np.float32)
    min_val = np.min(f32)
    max_val = np.max(f32)
    stretched = (f32 - min_val) / max(1e-3, max_val - min_val)
    return (stretched * 255).astype(np.uint8)


class Preprocessor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        mode = self.config.preprocess_mode.lower()
        if mode == "clahe_unsharp":
            return clahe_unsharp(frame, self.config.clip_limit, self.config.tile_grid, self.config.unsharp_amount)
        if mode == "contrast_stretch":
            return contrast_stretch(frame)
        return clahe_only(frame, self.config.clip_limit, self.config.tile_grid)


class EdgeDetector:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dexined = None
        if config.edge_mode == "learned" and config.dexined_model:
            model_path = Path(config.dexined_model)
            if model_path.exists():
                self.dexined = cv.dnn.readNetFromONNX(str(model_path))

    def detect(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grad_x = cv.Scharr(gray, cv.CV_32F, 1, 0)
        grad_y = cv.Scharr(gray, cv.CV_32F, 0, 1)
        mode = self.config.edge_mode.lower()
        if mode == "adaptive_canny":
            edges = self._adaptive_canny(gray)
        elif mode == "learned" and self.dexined is not None:
            edges = self._dexined(gray)
        else:
            edges = self._bilateral_laplacian(gray)
        return edges, grad_x, grad_y

    def _bilateral_laplacian(self, gray: np.ndarray) -> np.ndarray:
        filtered = cv.bilateralFilter(gray, 5, 55, 55)
        lap = cv.Laplacian(filtered, cv.CV_16S, ksize=3)
        edges = cv.convertScaleAbs(lap)
        _, otsu = cv.threshold(edges, 0, 255, cv.THRESH_OTSU)
        mask = cv.medianBlur(otsu, 5)
        return mask

    def _adaptive_canny(self, gray: np.ndarray) -> np.ndarray:
        mean = cv.blur(gray, (21, 21))
        residual = cv.absdiff(gray, mean)
        sigma = max(5.0, np.mean(residual))
        lower = max(10.0, 0.66 * sigma)
        upper = lower * 2.4
        edges = cv.Canny(gray, lower, upper, L2gradient=True)
        adaptive = cv.adaptiveThreshold(
            gray,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            19,
            -5,
        )
        return cv.bitwise_and(edges, adaptive)

    def _dexined(self, gray: np.ndarray) -> np.ndarray:
        if self.dexined is None:
            return self._bilateral_laplacian(gray)
        blob = cv.dnn.blobFromImage(gray, scalefactor=1.0 / 255.0, size=(512, 512), mean=0.5, swapRB=False, crop=False)
        self.dexined.setInput(blob)
        out = self.dexined.forward()
        edge_map = out[0, 0]
        edge_map = cv.resize(edge_map, (gray.shape[1], gray.shape[0]))
        edge_map = cv.normalize(edge_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        _, binary = cv.threshold(edge_map, 0, 255, cv.THRESH_OTSU)
        return binary


# ---------- Geometry helpers -------------------------------------------------------


def normalize_angle_deg(angle_deg: float) -> float:
    a = ((angle_deg + 90.0) % 180.0) - 90.0
    return 90.0 if a == -90.0 else a


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


# ---------- Segments & VP ----------------------------------------------------------


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


def gradient_orientation_mask(grad_x: np.ndarray, grad_y: np.ndarray, tolerance_deg: float) -> np.ndarray:
    angles = np.rad2deg(np.arctan2(grad_y, grad_x))
    mask = (np.abs(90.0 - np.abs(angles)) <= tolerance_deg).astype(np.uint8)
    return (mask * 255).astype(np.uint8)


def constrained_hough_segments(
    edges: np.ndarray,
    grad_mask: np.ndarray,
    min_length: float,
    max_gap: float,
    threshold: int,
) -> List[Segment]:
    masked = cv.bitwise_and(edges, grad_mask)
    lines = cv.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=int(min_length),
        maxLineGap=int(max_gap),
    )
    segments: List[Segment] = []
    if lines is None:
        return segments
    for x1, y1, x2, y2 in lines[:, 0]:
        p1 = np.array([x1, y1], np.float32)
        p2 = np.array([x2, y2], np.float32)
        angle, length = line_angle_and_length(p1, p2)
        segments.append(Segment(p1, p2, angle, length))
    return segments


def estimate_vanishing_point_ransac(
    segments: Sequence[Segment],
    frame_shape: Tuple[int, int],
    iterations: int,
    threshold_px: float,
) -> Optional[Tuple[float, float]]:
    if len(segments) < 3:
        return None
    h, w = frame_shape[:2]
    rng = np.random.default_rng(13)
    best_pt: Optional[np.ndarray] = None
    best_votes = 0

    def segment_to_line(seg: Segment) -> Tuple[np.ndarray, np.ndarray]:
        p1 = seg.p1
        p2 = seg.p2
        direction = p2 - p1
        normal = np.array([direction[1], -direction[0]])
        normal /= np.linalg.norm(normal) + 1e-9
        c = -np.dot(normal, p1)
        return normal, c

    lines = [segment_to_line(seg) for seg in segments]

    for _ in range(iterations):
        idx = rng.choice(len(lines), 2, replace=False)
        n1, c1 = lines[idx[0]]
        n2, c2 = lines[idx[1]]
        denom = n1[0] * n2[1] - n1[1] * n2[0]
        if abs(denom) < 1e-6:
            continue
        px = (c2 * n1[1] - c1 * n2[1]) / denom
        py = (c1 * n2[0] - c2 * n1[0]) / denom
        candidate = np.array([px, py], np.float32)
        votes = 0
        for normal, c in lines:
            dist = abs(np.dot(normal, candidate) + c)
            if dist <= threshold_px:
                votes += 1
        if votes > best_votes:
            best_votes = votes
            best_pt = candidate

    if best_pt is None:
        return None
    if -h <= best_pt[1] <= h * 2 and -w <= best_pt[0] <= w * 2:
        return float(best_pt[0]), float(best_pt[1])
    return None


# ---------- Temporal smoothing -----------------------------------------------------


class ExponentialSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.state: Optional[np.ndarray] = None
        self.confidence: float = 0.0

    def update(self, measurement: Optional[np.ndarray], quality: float) -> Optional[np.ndarray]:
        if measurement is None:
            self.confidence *= 0.9
            return self.state
        measurement = measurement.astype(np.float32)
        if self.state is None:
            self.state = measurement
        else:
            self.state = (1.0 - self.alpha) * self.state + self.alpha * measurement
        self.confidence = 0.6 * self.confidence + 0.4 * float(np.clip(quality, 0.0, 1.0))
        return self.state


@dataclass
class FollowResult:
    p1: np.ndarray
    p2: np.ndarray
    lateral_offset_norm: float
    angle_error_deg: float
    norm_length: float
    inlier_ratio: float
    residual_rms: float
    nfa_log10: float


@dataclass
class ROIState:
    center_offset: float = 0.0

    def update(self, measurement: float, alpha: float = 0.2) -> None:
        self.center_offset = float((1 - alpha) * self.center_offset + alpha * np.clip(measurement, -0.7, 0.7))


# ---------- Pipeline ----------------------------------------------------------------


class LinePipeline:
    def __init__(self, bev_shape: Tuple[int, int], config: PipelineConfig) -> None:
        self.config = config
        self.preprocessor = Preprocessor(config)
        self.edge_detector = EdgeDetector(config)
        self.roi_state = ROIState()
        self.tracker = ExponentialSmoother(config.tracker_alpha)
        self.bev_h, self.bev_w = bev_shape[1], bev_shape[0]
        self.vp_hint: Optional[Tuple[float, float]] = None
        self.last_follow: Optional[FollowResult] = None

    def detect(
        self,
        frame_bev: np.ndarray,
        timestamp: float,
    ) -> Tuple[np.ndarray, Optional[FollowResult], List[Segment], Optional[Tuple[float, float]], np.ndarray]:
        processed = self.preprocessor(frame_bev)
        gray = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
        edges, grad_x, grad_y = self.edge_detector.detect(gray)
        grad_mask = gradient_orientation_mask(grad_x, grad_y, self.config.orientation_tolerance_deg)

        center_offset = self.last_follow.lateral_offset_norm if self.last_follow is not None else self.roi_state.center_offset
        if self.vp_hint is not None:
            vp_offset = (self.vp_hint[0] / max(1.0, float(frame_bev.shape[1])) - 0.5) * 2.0
            center_offset = 0.7 * center_offset + 0.3 * float(np.clip(vp_offset, -0.6, 0.6))

        roi_mask = make_roi_mask(
            frame_bev.shape[0],
            frame_bev.shape[1],
            self.config.roi_height_pct,
            self.config.roi_top_pct,
            1.0,
            center_offset,
        )
        edges = cv.bitwise_and(edges, edges, mask=roi_mask)
        grad_mask = cv.bitwise_and(grad_mask, grad_mask, mask=roi_mask)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1, 9)))

        min_dim = min(frame_bev.shape[:2])
        min_len = self.config.min_line_pct * min_dim
        max_gap = self.config.max_gap_pct * min_dim
        segments = constrained_hough_segments(edges, grad_mask, min_len, max_gap, self.config.hough_threshold)

        if segments:
            self.vp_hint = estimate_vanishing_point_ransac(
                segments,
                frame_bev.shape[:2],
                self.config.vp_iterations,
                self.config.vp_threshold_px,
            )
        else:
            self.vp_hint = None

        follow = self._fit_consensus_line(segments, frame_bev.shape)
        if follow is not None:
            measurement = np.array(
                [follow.lateral_offset_norm, np.deg2rad(follow.angle_error_deg)],
                np.float32,
            )
            smoothed = self.tracker.update(measurement, follow.inlier_ratio)
            if smoothed is not None:
                follow.lateral_offset_norm = float(smoothed[0])
                follow.angle_error_deg = float(np.rad2deg(smoothed[1]))
            self.last_follow = follow
            self.roi_state.update(follow.lateral_offset_norm)
        else:
            self.tracker.update(None, 0.0)
            self.last_follow = None

        bev_debug = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        for seg in segments:
            draw_segment(bev_debug, seg, 1)
        if follow is not None:
            cv.line(
                bev_debug,
                (int(follow.p1[0]), int(follow.p1[1])),
                (int(follow.p2[0]), int(follow.p2[1])),
                (0, 220, 0),
                2,
                cv.LINE_AA,
            )

        return edges, follow, segments, self.vp_hint, bev_debug

    def _fit_consensus_line(self, segments: Sequence[Segment], shape: Tuple[int, int]) -> Optional[FollowResult]:
        if not segments:
            return None
        h, w = shape[:2]
        points: List[np.ndarray] = []
        for seg in segments:
            samples = np.linspace(seg.p1, seg.p2, num=32)
            points.append(samples)
        pts = np.vstack(points)
        result = ransac_line(pts, thresh=2.0, min_inliers=self.config.hough_threshold)
        if result is None:
            return None
        p1, p2, residuals = result
        inliers = residuals < 2.5
        if inliers.sum() < self.config.hough_threshold:
            return None
        bottom_y = h - 1
        xb = line_intersection_with_y(p1, p2, bottom_y)
        if xb is None or not np.isfinite(xb):
            return None
        angle, length = line_angle_and_length(p1, p2)
        angle_err = angle_from_vertical_deg(angle)
        if angle_err > self.config.angle_max_deg:
            return None
        norm_center = (xb - w / 2.0) / (0.5 * w)
        norm_length = min(length / (0.6 * np.hypot(w, h)), 1.0)
        residual_rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
        inlier_ratio = float(inliers.sum() / len(residuals))
        nfa_value = nfa(inliers.sum(), len(residuals))
        if nfa_value < 2.0:
            return None
        return FollowResult(
            p1=p1,
            p2=p2,
            lateral_offset_norm=float(norm_center),
            angle_error_deg=float(angle_err),
            norm_length=float(norm_length),
            inlier_ratio=inlier_ratio,
            residual_rms=residual_rms,
            nfa_log10=float(nfa_value),
        )


# ---------- RANSAC + scoring utilities --------------------------------------------


def ransac_line(
    points: np.ndarray,
    thresh: float,
    min_inliers: int,
    iterations: int = 256,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return None
    best_inliers: Optional[np.ndarray] = None
    rng = np.random.default_rng(42)
    for _ in range(iterations):
        idx = rng.choice(len(points), 2, replace=False)
        p1, p2 = points[idx]
        vec = p2 - p1
        norm_len = np.linalg.norm(vec)
        if norm_len < 1e-4:
            continue
        distances = np.abs(cross2d(vec, points - p1)) / (norm_len + 1e-6)
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
    direction /= np.linalg.norm(direction) + 1e-6
    projections = (inlier_pts - mean) @ direction
    p1 = mean + direction * projections.min()
    p2 = mean + direction * projections.max()
    residuals = np.abs(cross2d(direction, inlier_pts - mean)) / (np.linalg.norm(direction) + 1e-6)
    return p1.astype(np.float32), p2.astype(np.float32), residuals


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


def nfa(inliers: int, total: int, p: float = 0.01) -> float:
    if inliers <= 0 or total <= 0:
        return 0.0
    tail = 0.0
    for k in range(inliers, total + 1):
        comb = np.math.comb(total, k)
        tail += comb * (p ** k) * ((1 - p) ** (total - k))
    tail = max(tail, 1e-12)
    return -np.log10(tail)


# ---------- Visualization helpers --------------------------------------------------


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
    x, y = int(mid[0]), int(mid[1])
    label = f"{seg.angle_deg:+.1f}°"
    cv.putText(img, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(img, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)


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
        angle, length = line_angle_and_length(p1, p2)
        cam_segments.append(Segment(p1.astype(np.float32), p2.astype(np.float32), angle, length))
    return cam_segments


def show_text(img: np.ndarray, text: str, y: int = 28, scale: float = 0.7, color=(255, 255, 255)) -> None:
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


def draw_confidence_bar(img: np.ndarray, confidence: float, margin: int = 18) -> None:
    h, w = img.shape[:2]
    bar_w, bar_h = 220, 14
    x0 = margin
    y0 = h - margin - bar_h
    conf = float(np.clip(confidence, 0.0, 1.0))
    cv.rectangle(img, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (0, 0, 0), -1)
    cv.rectangle(img, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (200, 200, 200), 1)
    fill_w = int(bar_w * conf)
    color = (int(50 + (1.0 - conf) * 180), int(60 + conf * 180), 80)
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


# ---------- Frame grabbing ---------------------------------------------------------


class FrameGrabber(threading.Thread):
    def __init__(self, cap: cv.VideoCapture, queue_size: int) -> None:
        super().__init__(daemon=True)
        self.cap = cap
        self.frames: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            try:
                self.frames.put(frame, timeout=0.2)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.stop_event.set()


# ---------- Core processing --------------------------------------------------------


def detect_and_overlay(
    frame: np.ndarray,
    camera_model: CameraModel,
    imu_alignment: IMUAlignment,
    mapper: GroundPlaneMapper,
    pipeline: LinePipeline,
    timestamp: float,
) -> Tuple[np.ndarray, Optional[FollowResult], Optional[Tuple[float, float]], List[Segment], np.ndarray]:
    undistorted = camera_model.undistort(frame)
    aligned = imu_alignment.apply(undistorted, camera_model.K)
    bev = mapper.warp(aligned)

    edges, follow_result, segments, vp_hint, bev_debug = pipeline.detect(bev, timestamp)

    overlay = aligned.copy()
    h_cam, w_cam = overlay.shape[:2]
    gate_px = pipeline.config.bottom_gate_px
    h_bev, w_bev = bev.shape[:2]

    if vp_hint is not None:
        cv.circle(bev_debug, (int(vp_hint[0]), int(vp_hint[1])), 6, (0, 0, 255), -1, cv.LINE_AA)
        show_text(bev_debug, f"VP: ({vp_hint[0]:.1f}, {vp_hint[1]:.1f})", y=28)

    roi_mask_bev = make_roi_mask(
        h_bev,
        w_bev,
        pipeline.config.roi_height_pct,
        pipeline.config.roi_top_pct,
        1.0,
        pipeline.roi_state.center_offset,
    )
    contours, _ = cv.findContours(roi_mask_bev, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cam_cnt = mapper.unwarp_points(cnt.astype(np.float32))
        cam_poly = cam_cnt.reshape(-1, 2)
        if np.all(np.isfinite(cam_poly)):
            cv.polylines(overlay, [np.round(cam_cnt).astype(np.int32)], True, (200, 200, 200), 1, cv.LINE_AA)

    gate_top = max(0, h_bev - gate_px)
    gate_poly = np.array(
        [[0, gate_top], [w_bev - 1, gate_top], [w_bev - 1, h_bev - 1], [0, h_bev - 1]],
        np.float32,
    ).reshape(-1, 1, 2)
    cam_gate = mapper.unwarp_points(gate_poly)
    if np.all(np.isfinite(cam_gate)):
        cv.polylines(overlay, [np.round(cam_gate).astype(np.int32)], True, (120, 120, 120), 1, cv.LINE_AA)

    cam_segments = unwarp_segments_to_camera(segments, mapper)
    for seg in cam_segments:
        draw_segment(overlay, seg, thickness=2)
        put_angle_label(overlay, seg)

    if follow_result is not None:
        pts = np.array([[follow_result.p1], [follow_result.p2]], np.float32)
        cam_pts = mapper.unwarp_points(pts)
        cam_line = cam_pts.reshape(-1, 2)
        if np.all(np.isfinite(cam_line)):
            x1, y1 = map(int, cam_line[0])
            x2, y2 = map(int, cam_line[1])
            cv.line(overlay, (x1, y1), (x2, y2), (0, 220, 0), 4, cv.LINE_AA)
            xb_cam = line_intersection_with_y(cam_line[0], cam_line[1], h_cam - 1)
            if xb_cam is not None and np.isfinite(xb_cam):
                xb_int = int(np.clip(round(xb_cam), 0, w_cam - 1))
                cv.circle(overlay, (xb_int, h_cam - 1), 7, (0, 220, 0), -1, cv.LINE_AA)
                cv.line(overlay, (w_cam // 2, h_cam - 1), (xb_int, h_cam - 1), (0, 220, 0), 1, cv.LINE_AA)
        show_text(
            overlay,
            f"offset {follow_result.lateral_offset_norm:+.3f}  angle {follow_result.angle_error_deg:+.2f}°  "
            f"len {follow_result.norm_length:.2f}  inliers {follow_result.inlier_ratio:.2f}  "
            f"rms {follow_result.residual_rms:.2f}  NFA {follow_result.nfa_log10:.2f}",
            y=84,
            color=(180, 255, 180),
        )
    else:
        show_text(overlay, "Follow: not found", y=84, color=(180, 180, 180))

    draw_pip(overlay, bev_debug)
    draw_confidence_bar(overlay, pipeline.tracker.confidence)
    return overlay, follow_result, pipeline.vp_hint, segments, bev_debug


# ---------- Camera helpers ---------------------------------------------------------


def open_camera(indices: Sequence[int], width: int, height: int) -> Optional[cv.VideoCapture]:
    for idx in indices:
        cap = cv.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"Camera opened: index={idx} size={frame.shape[1]}x{frame.shape[0]}")
            return cap
        cap.release()
    return None


# ---------- CLI --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-line detector with classical upgrades")
    parser.add_argument("--config", type=Path, help="YAML config file", default=None)
    parser.add_argument("--edge-mode", choices=["bilateral_laplacian", "adaptive_canny", "learned"], default=None)
    parser.add_argument("--preprocess", choices=["clahe", "clahe_unsharp", "contrast_stretch"], default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Exponential smoothing alpha")
    parser.add_argument("--dexined", type=Path, default=None, help="Path to DexiNed ONNX model")
    parser.add_argument("--no-thread", action="store_true", help="Disable producer/consumer threading")
    return parser.parse_args()


def apply_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    if args.edge_mode:
        cfg.edge_mode = args.edge_mode
    if args.preprocess:
        cfg.preprocess_mode = args.preprocess
    if args.alpha is not None:
        cfg.tracker_alpha = args.alpha
    if args.dexined:
        cfg.dexined_model = str(args.dexined)
    if args.no_thread:
        cfg.multi_thread = False
    return cfg


# ---------- Main --------------------------------------------------------------------


def run() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    cap = open_camera(cfg.camera_indices, cfg.width, cfg.height)
    if cap is None:
        raise RuntimeError("Unable to open any camera index")

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Unable to read initial frame")

    camera_model = CameraModel.load()
    imu_alignment = IMUAlignment.load()
    mapper = GroundPlaneMapper.load(frame.shape, bev_scale=cfg.bev_scale)
    pipeline = LinePipeline(mapper.bev_size, cfg)

    grabber: Optional[FrameGrabber] = None
    if cfg.multi_thread:
        grabber = FrameGrabber(cap, cfg.queue_size)
        grabber.start()

    cv.namedWindow("long_lines_overlay", cv.WINDOW_NORMAL)
    cv.resizeWindow("long_lines_overlay", 1100, 620)

    prev_time = time.time()
    fps = 0.0
    print("Running. Press q to quit, space to dump config snapshot.")

    try:
        while True:
            if grabber is not None:
                frame = grabber.get()
                if frame is None:
                    continue
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("WARNING: empty frame")
                    break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            overlay, *_ = detect_and_overlay(frame, camera_model, imu_alignment, mapper, pipeline, now)
            show_text(overlay, f"{fps:.1f} FPS", y=overlay.shape[0] - 12)
            cv.imshow("long_lines_overlay", overlay)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                snapshot_path = Path("pipeline_snapshot.json")
                snapshot = asdict(cfg)
                snapshot.update({"tracker_confidence": pipeline.tracker.confidence})
                snapshot_path.write_text(json.dumps(snapshot, indent=2))
                print(f"Saved snapshot to {snapshot_path}")
    finally:
        if grabber is not None:
            grabber.stop()
            grabber.join(timeout=1.0)
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    run()
