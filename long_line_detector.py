"""Modernized interactive long-line detector.

This implementation replaces the legacy percentile-based Canny + Retinex + LSD
stack with the faster configuration outlined in the refactor brief:

* CLAHE-only preprocessing (Retinex removed) plus optional enhancement hooks.
* Three pluggable edge detectors (bilateral+Laplacian default, adaptive Canny,
  DexiNed ONNX fallback) with config-driven parameters.
* Constrained HoughLinesP with gradient-orientation pre-filtering replaces LSD.
* Exponential smoothing tracker (offset + angle) instead of a full 4D Kalman.
* YAML + argparse configuration instead of ad-hoc UI trackbars.
* Performance upgrades: cached BEV remap maps, RANSAC VP voting, optional
  producer/consumer threading for capture vs processing.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CALIBRATION_DIR = ROOT / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
INTRINSICS_PATH = CALIBRATION_DIR / "camera_model.npz"
HOMOGRAPHY_PATH = CALIBRATION_DIR / "ground_plane_h.npz"
IMU_PATH = CALIBRATION_DIR / "imu_alignment.json"
DEFAULT_CONFIG_PATH = ROOT / "config" / "line_detector.yaml"

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "intrinsics": str(INTRINSICS_PATH),
        "homography": str(HOMOGRAPHY_PATH),
        "imu_alignment": str(IMU_PATH),
        "bev_scale": 1.0,
    },
    "preprocess": {
        "mode": "clahe",
        "clahe": {"clip_limit": 2.0, "tile_grid": 8},
        "unsharp": {"enabled": False, "amount": 0.35, "sigma": 1.2},
        "contrast_stretch": {"enabled": False, "low_pct": 2.0, "high_pct": 98.0},
    },
    "edges": {
        "method": "bilateral_laplacian",
        "bilateral_laplacian": {
            "diameter": 7,
            "sigma_color": 45.0,
            "sigma_space": 7.5,
            "laplacian_ksize": 3,
            "auto_threshold": True,
            "manual_threshold": 18,
        },
        "adaptive_canny": {
            "block_size": 15,
            "offset": 4.0,
            "post_blur": 3,
        },
        "learned": {
            "model_path": "models/dexined.onnx",
            "input_size": [512, 512],
            "score_thresh": 0.25,
        },
        "vertical_kernel": 13,
        "orientation_tol_deg": 15.0,
    },
    "lines": {
        "vote_threshold": 55,
        "min_length_pct": 0.35,
        "max_gap_pct": 0.015,
        "angle_max_deg": 18.0,
        "roi": {
            "height_pct": 0.55,
            "top_width_pct": 0.35,
            "bottom_width_pct": 0.9,
            "bottom_gate_px": 40,
        },
        "ransac": {"distance_px": 2.5, "min_inliers": 45, "iterations": 256},
    },
    "tracking": {
        "alpha_offset": 0.35,
        "alpha_angle": 0.4,
        "dropout_decay": 0.9,
    },
    "performance": {
        "use_threads": True,
        "queue_size": 3,
        "vp_ransac_iters": 120,
        "vp_threshold_px": 24.0,
    },
    "visualization": {
        "pip_scale": 0.28,
        "confidence_margin": 22,
    },
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_by_dotted_key(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor: Any = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def parse_override(raw: str) -> Tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must look like section.key=value, got '{raw}'")
    key, value = raw.split("=", 1)
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse override value '{value}': {exc}") from exc
    return key.strip(), parsed


def load_config(config_path: Path, overrides: Sequence[str]) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy via json
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            user_cfg = yaml.safe_load(handle) or {}
        config = deep_update(config, user_cfg)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(DEFAULT_CONFIG, handle, sort_keys=False)
        print(f"Wrote default config to {config_path}")
    for raw in overrides:
        key, value = parse_override(raw)
        set_by_dotted_key(config, key, value)
    return config


# ---------------------------------------------------------------------------
# Camera, IMU, and mapping utilities
# ---------------------------------------------------------------------------


@dataclass
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    new_K: np.ndarray
    map1: Optional[np.ndarray] = None
    map2: Optional[np.ndarray] = None

    @classmethod
    def load(cls, path: Path, width: int, height: int) -> "CameraModel":
        if path.exists():
            data = np.load(path)
            K = data.get("K")
            dist = data.get("dist")
            new_K = data.get("new_K", K)
            if K is not None and dist is not None:
                return cls(K.astype(np.float32), dist.astype(np.float32), new_K.astype(np.float32))
        # fallback pinhole
        fx = float(width)
        fy = float(width)
        cx = width / 2.0
        cy = height / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        dist = np.zeros((1, 5), np.float32)
        return cls(K, dist, K.copy())

    def prepare_rectify_map(self, width: int, height: int) -> None:
        if np.allclose(self.dist, 0):
            self.map1 = None
            self.map2 = None
            return
        self.map1, self.map2 = cv.initUndistortRectifyMap(
            self.K, self.dist, np.eye(3, dtype=np.float32), self.new_K, (width, height), cv.CV_32FC1
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.map1 is None or self.map2 is None:
            return frame
        return cv.remap(frame, self.map1, self.map2, interpolation=cv.INTER_LINEAR)


@dataclass
class IMUAlignment:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    @classmethod
    def load(cls, path: Path) -> "IMUAlignment":
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
    map_x: np.ndarray
    map_y: np.ndarray

    @classmethod
    def load(cls, frame_shape: Tuple[int, int], path: Path, bev_scale: float) -> "GroundPlaneMapper":
        h, w = frame_shape[:2]
        if path.exists():
            data = np.load(path)
            H = data.get("H")
            bev_w = int(data.get("bev_w", w * bev_scale))
            bev_h = int(data.get("bev_h", h * bev_scale))
            if H is not None:
                H = H.astype(np.float32)
                H_inv = np.linalg.inv(H)
                map_x, map_y = cls._precompute_maps(H_inv, (bev_w, bev_h))
                return cls(H, H_inv.astype(np.float32), (bev_w, bev_h), map_x, map_y)
        H = np.eye(3, dtype=np.float32)
        H_inv = np.eye(3, dtype=np.float32)
        map_x, map_y = cls._precompute_maps(H_inv, (w, h))
        return cls(H, H_inv, (w, h), map_x, map_y)

    @staticmethod
    def _precompute_maps(H_inv: np.ndarray, bev_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        bev_w, bev_h = bev_size
        grid_x, grid_y = np.meshgrid(np.arange(bev_w), np.arange(bev_h))
        dst = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
        dst_flat = dst.reshape(-1, 1, 2)
        src = cv.perspectiveTransform(dst_flat, H_inv).reshape(bev_h, bev_w, 2)
        map_x = src[..., 0].astype(np.float32)
        map_y = src[..., 1].astype(np.float32)
        return map_x, map_y

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv.remap(frame, self.map_x, self.map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        pts_h = cv.convertPointsToHomogeneous(pts.astype(np.float32)).reshape(-1, 3)
        proj = (self.H_inv @ pts_h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-6)
        return proj.reshape(-1, 1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def normalize_angle_deg(angle_deg: float) -> float:
    a = ((angle_deg + 90.0) % 180.0) - 90.0
    if a == -90.0:
        a = 90.0
    return a


def line_angle_and_length(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return normalize_angle_deg(angle), float(np.hypot(dx, dy))


def angle_from_vertical_deg(angle_deg: float) -> float:
    return abs(90.0 - abs(angle_deg))


def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


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


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------


def preprocess_frame(frame: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    L, a, b = cv.split(lab)
    clahe_cfg = cfg.get("clahe", {})
    clahe = cv.createCLAHE(
        clipLimit=float(clahe_cfg.get("clip_limit", 2.0)),
        tileGridSize=(int(clahe_cfg.get("tile_grid", 8)), int(clahe_cfg.get("tile_grid", 8))),
    )
    L = clahe.apply(L)

    if cfg.get("contrast_stretch", {}).get("enabled", False):
        low = float(cfg["contrast_stretch"].get("low_pct", 2.0))
        high = float(cfg["contrast_stretch"].get("high_pct", 98.0))
        lo_v, hi_v = np.percentile(L, [low, high])
        L = np.clip((L - lo_v) * 255.0 / max(1.0, hi_v - lo_v), 0, 255).astype(np.uint8)

    if cfg.get("unsharp", {}).get("enabled", False):
        amount = float(cfg["unsharp"].get("amount", 0.35))
        sigma = float(cfg["unsharp"].get("sigma", 1.2))
        blur = cv.GaussianBlur(L, (0, 0), sigma)
        L = cv.addWeighted(L, 1 + amount, blur, -amount, 0)

    lab = cv.merge((L, a, b))
    enhanced = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    gray = cv.cvtColor(enhanced, cv.COLOR_BGR2GRAY)
    return enhanced, gray


def edges_bilateral_laplacian(gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    params = cfg.get("bilateral_laplacian", {})
    filtered = cv.bilateralFilter(
        gray,
        int(params.get("diameter", 7)),
        float(params.get("sigma_color", 45.0)),
        float(params.get("sigma_space", 7.5)),
    )
    lap = cv.Laplacian(filtered, cv.CV_16S, ksize=int(params.get("laplacian_ksize", 3)))
    edges = cv.convertScaleAbs(lap)
    if params.get("auto_threshold", True):
        _, binary = cv.threshold(edges, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    else:
        thr = int(params.get("manual_threshold", 18))
        _, binary = cv.threshold(edges, thr, 255, cv.THRESH_BINARY)
    return binary


def edges_adaptive(gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    params = cfg.get("adaptive_canny", {})
    block = max(3, int(params.get("block_size", 15)))
    if block % 2 == 0:
        block += 1
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    magnitude = cv.magnitude(gx, gy)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    local_mean = cv.blur(magnitude, (block, block))
    diff = cv.subtract(magnitude, local_mean)
    offset = float(params.get("offset", 4.0))
    _, binary = cv.threshold(diff, offset, 255, cv.THRESH_BINARY)
    blur_k = max(1, int(params.get("post_blur", 3)))
    if blur_k > 1:
        binary = cv.GaussianBlur(binary, (blur_k | 1, blur_k | 1), 0)
        _, binary = cv.threshold(binary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def edges_learned(gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for learned edge detection") from exc
    params = cfg.get("learned", {})
    model_path = Path(params.get("model_path", ""))
    if not model_path.exists():  # pragma: no cover
        raise RuntimeError(f"DexiNed model not found at {model_path}")
    input_h, input_w = params.get("input_size", [512, 512])
    resized = cv.resize(gray, (input_w, input_h))
    tensor = resized.astype(np.float32) / 255.0
    tensor = tensor[None, None, :, :]
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    scores = session.run([output_name], {input_name: tensor})[0]
    score = scores[0, 0]
    score = cv.resize(score, (gray.shape[1], gray.shape[0]))
    thresh = float(params.get("score_thresh", 0.25))
    binary = (score > thresh).astype(np.uint8) * 255
    return binary


EDGE_METHODS = {
    "bilateral_laplacian": edges_bilateral_laplacian,
    "adaptive_canny": edges_adaptive,
    "learned": edges_learned,
}


def detect_edges(gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    method = cfg.get("method", "bilateral_laplacian")
    detector = EDGE_METHODS.get(method)
    if detector is None:
        raise ValueError(f"Unknown edge detector '{method}'")
    try:
        edges = detector(gray, cfg)
    except RuntimeError as exc:
        print(f"[edges] {exc}; falling back to bilateral_laplacian")
        edges = edges_bilateral_laplacian(gray, cfg)
    kernel_len = max(3, int(cfg.get("vertical_kernel", 11)))
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_len))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)
    edges = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel, iterations=1)
    return edges


# ---------------------------------------------------------------------------
# Line extraction helpers
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    angle_deg: float
    length: float

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
    n_inliers: int


@dataclass
class ROIState:
    center_offset: float = 0.0

    def update(self, measurement: float, alpha: float = 0.2) -> None:
        self.center_offset = (1 - alpha) * self.center_offset + alpha * float(np.clip(measurement, -0.6, 0.6))


@dataclass
class ExponentialTracker:
    alpha_offset: float
    alpha_angle: float
    dropout_decay: float
    offset: Optional[float] = None
    angle: Optional[float] = None
    confidence: float = 0.0

    def step(self, measurement: Optional[Tuple[float, float]], measurement_conf: float) -> Tuple[float, float, float]:
        if measurement is None:
            self.confidence *= self.dropout_decay
            if self.offset is None:
                self.offset = 0.0
            if self.angle is None:
                self.angle = 0.0
            return self.offset, self.angle, self.confidence
        offset_meas, angle_meas = measurement
        self.offset = self._smooth(self.offset, offset_meas, self.alpha_offset)
        self.angle = self._smooth(self.angle, angle_meas, self.alpha_angle)
        self.confidence = 0.6 * measurement_conf + 0.4 * self.confidence
        return self.offset, self.angle, self.confidence

    @staticmethod
    def _smooth(current: Optional[float], measurement: float, alpha: float) -> float:
        if current is None:
            return measurement
        return (1 - alpha) * current + alpha * measurement


def gradient_orientation_mask(gray: np.ndarray, tol_deg: float) -> np.ndarray:
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    orientation = cv.phase(gx, gy, angleInDegrees=True)
    line_orientation = (orientation + 90.0) % 180.0 - 90.0
    mask = (np.abs(line_orientation) <= tol_deg).astype(np.uint8) * 255
    mask = cv.GaussianBlur(mask, (0, 0), tol_deg / 45.0 + 1e-3)
    _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return mask


def constrained_hough(
    edges: np.ndarray,
    gray: np.ndarray,
    cfg: Dict[str, Any],
    roi_mask: np.ndarray,
) -> List[Segment]:
    tol = float(cfg.get("orientation_tol_deg", 15.0))
    orientation_mask = gradient_orientation_mask(gray, tol)
    masked_edges = cv.bitwise_and(edges, edges, mask=cv.bitwise_and(roi_mask, orientation_mask))
    min_dim = min(edges.shape[:2])
    min_len = int(cfg.get("min_length_pct", 0.35) * min_dim)
    max_gap = int(cfg.get("max_gap_pct", 0.015) * min_dim)
    vote = int(cfg.get("vote_threshold", 55))
    lines = cv.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=vote,
        minLineLength=max(20, min_len),
        maxLineGap=max(5, max_gap),
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


def ransac_line(
    segments: Sequence[Segment],
    cfg: Dict[str, Any],
    frame_shape: Tuple[int, int],
) -> Optional[FollowResult]:
    if not segments:
        return None
    points = []
    for seg in segments:
        num = max(6, int(seg.length // 6))
        pts = np.linspace(seg.p1, seg.p2, num=num)
        points.append(pts)
    points_arr = np.vstack(points)
    params = cfg.get("ransac", {})
    thresh = float(params.get("distance_px", 2.5))
    min_inliers = int(params.get("min_inliers", 45))
    iters = int(params.get("iterations", 256))
    if len(points_arr) < min_inliers:
        return None
    rng = np.random.default_rng(7)
    best_inliers: Optional[np.ndarray] = None
    for _ in range(iters):
        idx = rng.choice(len(points_arr), 2, replace=False)
        p1, p2 = points_arr[idx]
        vec = p2 - p1
        norm = np.linalg.norm(vec)
        if norm < 1e-3:
            continue
        distances = np.abs(cross2d(vec, points_arr - p1)) / (norm + 1e-6)
        inliers = distances <= thresh
        if inliers.sum() < min_inliers:
            continue
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None:
        return None
    inlier_pts = points_arr[best_inliers]
    mean = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    projections = centered @ direction
    p1 = (mean + direction * projections.min()).astype(np.float32)
    p2 = (mean + direction * projections.max()).astype(np.float32)
    angle, length = line_angle_and_length(p1, p2)
    h, w = frame_shape[:2]
    angle_err = angle_from_vertical_deg(angle)
    if angle_err > cfg.get("angle_max_deg", 18.0):
        return None
    xb = line_intersection_with_y(p1, p2, h - 1)
    if xb is None or not np.isfinite(xb):
        return None
    norm_offset = (xb - w / 2.0) / (0.5 * w)
    norm_length = min(length / (0.6 * np.hypot(w, h)), 1.0)
    residuals = np.abs(cross2d(direction, centered))
    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
    inlier_ratio = float(best_inliers.sum() / len(points_arr))
    return FollowResult(
        p1=p1,
        p2=p2,
        lateral_offset_norm=float(norm_offset),
        angle_error_deg=float(angle_err),
        norm_length=float(norm_length),
        inlier_ratio=inlier_ratio,
        residual_rms=residual_rms,
        n_inliers=int(best_inliers.sum()),
    )


def ransac_vanishing_point(
    segments: Sequence[Segment],
    iterations: int,
    threshold_px: float,
) -> Optional[Tuple[float, float]]:
    if len(segments) < 2:
        return None
    rng = np.random.default_rng(11)
    best_support = 0
    best_point: Optional[np.ndarray] = None
    seg_pairs = min(iterations, len(segments) * (len(segments) - 1) // 2)
    for _ in range(seg_pairs):
        s1, s2 = rng.choice(segments, 2, replace=False)
        pt = intersect_segments(s1, s2)
        if pt is None:
            continue
        support = 0
        for seg in segments:
            dist = point_line_distance(pt, seg.p1, seg.p2)
            if dist < threshold_px:
                support += 1
        if support > best_support:
            best_support = support
            best_point = pt
    if best_point is None or best_support < 3:
        return None
    return float(best_point[0]), float(best_point[1])


def intersect_segments(seg_a: Segment, seg_b: Segment) -> Optional[np.ndarray]:
    x1, y1 = seg_a.p1
    x2, y2 = seg_a.p2
    x3, y3 = seg_b.p1
    x4, y4 = seg_b.p2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    return np.array([px, py], np.float32)


def point_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    point = np.asarray(point, np.float32)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    if np.allclose(a, b):
        return float(np.linalg.norm(point - a))
    ba = b - a
    pa = point - a
    return float(np.abs(cross2d(ba, pa)) / (np.linalg.norm(ba) + 1e-6))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


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


def draw_segment(img: np.ndarray, seg: Segment, thickness: int = 2) -> None:
    x1, y1, x2, y2 = seg.as_tuple()
    cv.line(img, (x1, y1), (x2, y2), color_for_angle(seg.angle_deg), thickness, cv.LINE_AA)


def put_text(img: np.ndarray, text: str, y: int, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    cv.putText(img, text, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(img, text, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA)


def draw_confidence_bar(img: np.ndarray, confidence: float, margin: int) -> None:
    h, w = img.shape[:2]
    bar_w, bar_h = 220, 16
    x0 = margin
    y0 = h - margin - bar_h
    conf = float(np.clip(confidence, 0.0, 1.0))
    cv.rectangle(img, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (0, 0, 0), -1)
    cv.rectangle(img, (x0 - 3, y0 - 3), (x0 + bar_w + 3, y0 + bar_h + 3), (120, 120, 120), 1)
    fill_w = int(bar_w * conf)
    color = (int(50 + (1.0 - conf) * 160), int(80 + conf * 140), 90)
    cv.rectangle(img, (x0, y0), (x0 + fill_w, y0 + bar_h), color, -1)
    cv.putText(img, f"Confidence {conf:.2f}", (x0 + bar_w + 12, y0 + bar_h - 2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_pip(canvas: np.ndarray, pip_img: np.ndarray, scale: float, margin: int = 16) -> None:
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
    cv.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + pw + 4, y0 + ph + 4), (220, 220, 220), 1)
    canvas[y0 : y0 + ph, x0 : x0 + pw] = pip_resized


# ---------------------------------------------------------------------------
# Line pipeline
# ---------------------------------------------------------------------------


class LinePipeline:
    def __init__(self, mapper: GroundPlaneMapper, config: Dict[str, Any]):
        self.mapper = mapper
        self.config = config
        trk_cfg = config["tracking"]
        self.tracker = ExponentialTracker(
            alpha_offset=float(trk_cfg.get("alpha_offset", 0.35)),
            alpha_angle=float(trk_cfg.get("alpha_angle", 0.4)),
            dropout_decay=float(trk_cfg.get("dropout_decay", 0.9)),
        )
        self.roi_state = ROIState()
        self.vp_hint: Optional[Tuple[float, float]] = None
        self.last_follow: Optional[FollowResult] = None

    def detect(self, bev_frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, Optional[FollowResult], List[Segment], Optional[Tuple[float, float]], np.ndarray]:
        pre_cfg = self.config["preprocess"]
        edges_cfg = self.config["edges"]
        lines_cfg = self.config["lines"]
        perf_cfg = self.config["performance"]

        processed, gray = preprocess_frame(bev_frame, pre_cfg)
        edges = detect_edges(gray, edges_cfg)

        roi_cfg = lines_cfg.get("roi", {})
        center_offset = self.roi_state.center_offset
        if self.vp_hint is not None:
            vp_offset = (self.vp_hint[0] / max(1.0, bev_frame.shape[1]) - 0.5) * 2.0
            center_offset = 0.75 * center_offset + 0.25 * float(np.clip(vp_offset, -0.6, 0.6))
        roi_mask = make_roi_mask(
            bev_frame.shape[0],
            bev_frame.shape[1],
            float(roi_cfg.get("height_pct", 0.55)),
            float(roi_cfg.get("top_width_pct", 0.35)),
            float(roi_cfg.get("bottom_width_pct", 0.9)),
            center_offset,
        )
        edges = cv.bitwise_and(edges, edges, mask=roi_mask)

        segments = constrained_hough(edges, gray, {**lines_cfg, **edges_cfg}, roi_mask)
        follow = ransac_line(segments, lines_cfg, bev_frame.shape)

        measurement: Optional[Tuple[float, float]] = None
        measurement_conf = 0.0
        if follow is not None:
            measurement = (follow.lateral_offset_norm, np.radians(follow.angle_error_deg))
            measurement_conf = float(np.clip(0.5 * follow.inlier_ratio + 0.5 * max(0.0, 1.0 - follow.residual_rms / 3.0), 0.0, 1.0))
            self.last_follow = follow
        offset, angle, conf = self.tracker.step(measurement, measurement_conf)
        self.roi_state.update(offset)
        if follow is not None:
            follow.lateral_offset_norm = offset
            follow.angle_error_deg = float(np.degrees(angle))

        vp = ransac_vanishing_point(
            segments,
            int(perf_cfg.get("vp_ransac_iters", 120)),
            float(perf_cfg.get("vp_threshold_px", 24.0)),
        )
        if vp is not None:
            self.vp_hint = vp

        bev_debug = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        for seg in segments:
            draw_segment(bev_debug, seg)
        if follow is not None:
            cv.line(
                bev_debug,
                (int(follow.p1[0]), int(follow.p1[1])),
                (int(follow.p2[0]), int(follow.p2[1])),
                (0, 220, 0),
                2,
                cv.LINE_AA,
            )
        if self.vp_hint is not None:
            cv.circle(bev_debug, (int(self.vp_hint[0]), int(self.vp_hint[1])), 6, (0, 0, 255), -1, cv.LINE_AA)
        return edges, follow, segments, self.vp_hint, bev_debug


# ---------------------------------------------------------------------------
# Producer-consumer capture helper
# ---------------------------------------------------------------------------


class FrameProducer:
    def __init__(self, cap: cv.VideoCapture, queue_size: int):
        self.cap = cap
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.running = threading.Event()

    def start(self) -> None:
        self.running.set()
        self.thread.start()

    def stop(self) -> None:
        self.running.clear()
        self.thread.join(timeout=1.0)

    def _worker(self) -> None:
        while self.running.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            try:
                self.queue.put(frame, timeout=0.01)
            except queue.Full:
                try:
                    _ = self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put(frame, timeout=0.01)

    def get(self, timeout: float = 0.2) -> Optional[np.ndarray]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Core processing & visualization
# ---------------------------------------------------------------------------


def detect_and_overlay(
    frame: np.ndarray,
    camera_model: CameraModel,
    imu_alignment: IMUAlignment,
    mapper: GroundPlaneMapper,
    pipeline: LinePipeline,
    config: Dict[str, Any],
    timestamp: float,
) -> Tuple[np.ndarray, Optional[FollowResult], Optional[Tuple[float, float]], List[Segment], np.ndarray]:
    undistorted = camera_model.undistort(frame)
    aligned = imu_alignment.apply(undistorted, camera_model.K)
    bev = mapper.warp(aligned)

    edges, follow, segments, vp_hint, bev_debug = pipeline.detect(bev, timestamp)

    overlay = aligned.copy()
    pip_scale = float(config["visualization"].get("pip_scale", 0.28))
    draw_pip(overlay, bev_debug, pip_scale)

    cam_segments = unwarp_segments_to_camera(segments, mapper)
    for seg in cam_segments:
        draw_segment(overlay, seg, thickness=2)

    if follow is not None:
        cam_line = mapper.unwarp_points(np.array([[follow.p1], [follow.p2]], np.float32)).reshape(-1, 2)
        if np.all(np.isfinite(cam_line)):
            x1, y1 = map(int, cam_line[0])
            x2, y2 = map(int, cam_line[1])
            cv.line(overlay, (x1, y1), (x2, y2), (0, 220, 0), 4, cv.LINE_AA)
            xb = line_intersection_with_y(cam_line[0], cam_line[1], overlay.shape[0] - 1)
            if xb is not None and np.isfinite(xb):
                xb_i = int(np.clip(round(xb), 0, overlay.shape[1] - 1))
                cv.circle(overlay, (xb_i, overlay.shape[0] - 1), 7, (0, 220, 0), -1, cv.LINE_AA)
        put_text(
            overlay,
            f"offset {follow.lateral_offset_norm:+.3f}  angle {follow.angle_error_deg:+.2f}°  len {follow.norm_length:.2f}  inliers {follow.n_inliers}",
            y=72,
            color=(180, 255, 180),
        )
    else:
        put_text(overlay, "Follow: not found", y=72, color=(200, 200, 200))

    if vp_hint is not None:
        put_text(overlay, f"VP ({vp_hint[0]:.1f}, {vp_hint[1]:.1f})", y=96, color=(200, 220, 255))

    draw_confidence_bar(overlay, pipeline.tracker.confidence, int(config["visualization"].get("confidence_margin", 22)))
    return overlay, follow, vp_hint, segments, bev_debug


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
        if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
            continue
        angle, length = line_angle_and_length(p1, p2)
        cam_segments.append(Segment(p1.astype(np.float32), p2.astype(np.float32), angle, length))
    return cam_segments


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def open_camera(camera_cfg: Dict[str, Any]) -> Optional[cv.VideoCapture]:
    index = int(camera_cfg.get("index", 0))
    width = int(camera_cfg.get("width", 1280))
    height = int(camera_cfg.get("height", 720))
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive long-line detector (modernized)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="section.key=value",
        help="Override a config entry (can be repeated)",
    )
    parser.add_argument("--no-thread", action="store_true", help="Disable producer/consumer threading")
    parser.add_argument("--dump-config", action="store_true", help="Print merged config and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.override)
    if args.no_thread:
        config["performance"]["use_threads"] = False
    if args.dump_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return

    cam_cfg = config["camera"]
    cap = open_camera(cam_cfg)
    if cap is None:
        raise RuntimeError(f"Unable to open camera index {cam_cfg.get('index', 0)}")

    width = int(cam_cfg.get("width", 1280))
    height = int(cam_cfg.get("height", 720))
    camera_model = CameraModel.load(Path(cam_cfg.get("intrinsics", INTRINSICS_PATH)), width, height)
    camera_model.prepare_rectify_map(width, height)
    imu_alignment = IMUAlignment.load(Path(cam_cfg.get("imu_alignment", IMU_PATH)))

    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to grab initial frame")
    mapper = GroundPlaneMapper.load(frame.shape, Path(cam_cfg.get("homography", HOMOGRAPHY_PATH)), float(cam_cfg.get("bev_scale", 1.0)))
    pipeline = LinePipeline(mapper, config)

    producer: Optional[FrameProducer] = None
    if config["performance"].get("use_threads", True):
        producer = FrameProducer(cap, int(config["performance"].get("queue_size", 3)))
        producer.start()

    cv.namedWindow("long_lines_overlay", cv.WINDOW_NORMAL)
    cv.resizeWindow("long_lines_overlay", 1100, 640)

    fps = 0.0
    last_time = time.time()
    print("Running. Press q to quit.")

    try:
        while True:
            if producer is not None:
                frame = producer.get()
                if frame is None:
                    continue
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - last_time))
            last_time = now

            overlay, *_ = detect_and_overlay(frame, camera_model, imu_alignment, mapper, pipeline, config, now)
            put_text(overlay, f"{fps:.1f} FPS", y=overlay.shape[0] - 18)
            cv.imshow("long_lines_overlay", overlay)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        if producer is not None:
            producer.stop()
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
