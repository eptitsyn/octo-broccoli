from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class TableDatasetConfig:
    image_size: Tuple[int, int] = (768, 768)  # (height, width)
    num_control_points: int = 16
    line_width: int = 3
    junction_radius: int = 4
    distance_sigma: float = 3.0
    separator_merge_distance: float = 3.0
    min_separator_visibility: float = 0.15
    normalize: bool = True
    include_text_mask: bool = True


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def _as_polygon_array(poly: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(poly, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
        raise ValueError(f"Invalid polygon shape: {arr.shape}")
    return arr


def _letterbox(
    image: np.ndarray,
    polygons: Dict[str, List[np.ndarray]],
    output_size: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, List[np.ndarray]], Dict[str, float]]:
    out_h, out_w = output_size
    src_h, src_w = image.shape[:2]
    scale = min(out_w / src_w, out_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pad_x = (out_w - new_w) // 2
    pad_y = (out_h - new_h) // 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    transformed: Dict[str, List[np.ndarray]] = {}
    for key, items in polygons.items():
        transformed[key] = []
        for poly in items:
            p = poly.copy()
            p[:, 0] = p[:, 0] * scale + pad_x
            p[:, 1] = p[:, 1] * scale + pad_y
            transformed[key].append(p)

    meta = {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "src_h": float(src_h),
        "src_w": float(src_w),
        "out_h": float(out_h),
        "out_w": float(out_w),
    }
    return canvas, transformed, meta


def _polygon_mask(poly: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _extract_envelopes(
    polygons: Sequence[np.ndarray],
    orientation: str,
    height: int,
    width: int,
    num_points: int,
) -> List[Dict[str, np.ndarray]]:
    """Extract top/bottom or left/right polygon envelopes as sampled polylines.

    Each output has:
      points: [K, 2] pixel coordinates
      visibility: [K] bool
    """
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(orientation)

    curves: List[Dict[str, np.ndarray]] = []
    if orientation == "horizontal":
        axis = np.linspace(0, width - 1, num_points)
    else:
        axis = np.linspace(0, height - 1, num_points)

    for poly in polygons:
        mask = _polygon_mask(poly, height, width)
        if not mask.any():
            continue

        if orientation == "horizontal":
            lo = np.full(width, np.nan, dtype=np.float32)
            hi = np.full(width, np.nan, dtype=np.float32)
            for x in np.flatnonzero(mask.any(axis=0)):
                ys = np.flatnonzero(mask[:, x])
                lo[x] = ys[0]
                hi[x] = ys[-1]
            envelopes = (lo, hi)
        else:
            lo = np.full(height, np.nan, dtype=np.float32)
            hi = np.full(height, np.nan, dtype=np.float32)
            for y in np.flatnonzero(mask.any(axis=1)):
                xs = np.flatnonzero(mask[y])
                lo[y] = xs[0]
                hi[y] = xs[-1]
            envelopes = (lo, hi)

        for values in envelopes:
            valid_idx = np.flatnonzero(np.isfinite(values))
            if valid_idx.size < 2:
                continue

            sampled_valid = (axis >= valid_idx[0]) & (axis <= valid_idx[-1])
            sampled_values = np.interp(axis, valid_idx, values[valid_idx])

            if orientation == "horizontal":
                points = np.stack([axis, sampled_values], axis=-1)
            else:
                points = np.stack([sampled_values, axis], axis=-1)

            curves.append(
                {
                    "points": points.astype(np.float32),
                    "visibility": sampled_valid.astype(np.float32),
                }
            )

    return curves


def _curve_distance(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    visible = (a["visibility"] > 0.5) & (b["visibility"] > 0.5)
    if visible.mean() < 0.10:
        return float("inf")
    delta = a["points"][visible] - b["points"][visible]
    return float(np.linalg.norm(delta, axis=1).mean())


def _merge_curves(
    curves: Sequence[Dict[str, np.ndarray]],
    max_distance: float,
    min_visibility: float,
) -> List[Dict[str, np.ndarray]]:
    groups: List[List[Dict[str, np.ndarray]]] = []
    for curve in curves:
        if curve["visibility"].mean() < min_visibility:
            continue
        best_group = None
        best_distance = float("inf")
        for group_idx, group in enumerate(groups):
            reference = group[0]
            distance = _curve_distance(curve, reference)
            if distance < best_distance:
                best_distance = distance
                best_group = group_idx
        if best_group is not None and best_distance <= max_distance:
            groups[best_group].append(curve)
        else:
            groups.append([curve])

    merged: List[Dict[str, np.ndarray]] = []
    for group in groups:
        k = group[0]["points"].shape[0]
        points = np.zeros((k, 2), dtype=np.float32)
        visibility = np.zeros(k, dtype=np.float32)
        for i in range(k):
            valid = [g["points"][i] for g in group if g["visibility"][i] > 0.5]
            if valid:
                points[i] = np.mean(valid, axis=0)
                visibility[i] = 1.0
            else:
                points[i] = group[0]["points"][i]
        merged.append({"points": points, "visibility": visibility})
    return merged


def _sort_curves(
    curves: List[Dict[str, np.ndarray]], orientation: str
) -> List[Dict[str, np.ndarray]]:
    coord = 1 if orientation == "horizontal" else 0

    def key(curve: Dict[str, np.ndarray]) -> float:
        visible = curve["visibility"] > 0.5
        if not visible.any():
            return float("inf")
        return float(np.median(curve["points"][visible, coord]))

    return sorted(curves, key=key)


def _normalize_curves(
    curves: Sequence[Dict[str, np.ndarray]], height: int, width: int, line_width: int
) -> Dict[str, Tensor]:
    if not curves:
        k = 0
        return {
            "control_points": torch.zeros((0, 0, 2), dtype=torch.float32),
            "visibility": torch.zeros((0, 0), dtype=torch.float32),
            "widths": torch.zeros((0, 0), dtype=torch.float32),
        }

    points = np.stack([c["points"] for c in curves]).astype(np.float32)
    visibility = np.stack([c["visibility"] for c in curves]).astype(np.float32)
    points[..., 0] /= max(width - 1, 1)
    points[..., 1] /= max(height - 1, 1)
    widths = np.full(
        visibility.shape, line_width / max(height, width), dtype=np.float32
    )
    return {
        "control_points": torch.from_numpy(points),
        "visibility": torch.from_numpy(visibility),
        "widths": torch.from_numpy(widths),
    }


def _draw_curves(
    curves: Sequence[Dict[str, np.ndarray]], height: int, width: int, thickness: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for curve in curves:
        points = curve["points"]
        visible = curve["visibility"] > 0.5
        for i in range(len(points) - 1):
            if not (visible[i] and visible[i + 1]):
                continue
            p1 = tuple(np.round(points[i]).astype(int))
            p2 = tuple(np.round(points[i + 1]).astype(int))
            cv2.line(mask, p1, p2, 1, thickness=thickness, lineType=cv2.LINE_AA)
    return (mask > 0).astype(np.float32)


def _segment_intersection(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> Optional[np.ndarray]:
    da = a2 - a1
    db = b2 - b1
    matrix = np.stack([da, -db], axis=1)
    det = np.linalg.det(matrix)
    if abs(det) < 1e-7:
        return None
    t, u = np.linalg.solve(matrix, b1 - a1)
    if -0.02 <= t <= 1.02 and -0.02 <= u <= 1.02:
        return a1 + t * da
    return None


def _curve_intersections(
    horizontal: Sequence[Dict[str, np.ndarray]],
    vertical: Sequence[Dict[str, np.ndarray]],
) -> List[Tuple[np.ndarray, int]]:
    """Returns (point, class_id), where class ids are endpoint/L/T/X = 0/1/2/3."""
    result: List[Tuple[np.ndarray, int]] = []
    for h in horizontal:
        for v in vertical:
            hp, hv = h["points"], h["visibility"] > 0.5
            vp, vv = v["points"], v["visibility"] > 0.5
            found = None
            h_seg_idx = v_seg_idx = -1
            for i in range(len(hp) - 1):
                if not (hv[i] and hv[i + 1]):
                    continue
                for j in range(len(vp) - 1):
                    if not (vv[j] and vv[j + 1]):
                        continue
                    p = _segment_intersection(hp[i], hp[i + 1], vp[j], vp[j + 1])
                    if p is not None:
                        found, h_seg_idx, v_seg_idx = p, i, j
                        break
                if found is not None:
                    break
            if found is None:
                continue

            h_end = h_seg_idx == 0 or h_seg_idx == len(hp) - 2
            v_end = v_seg_idx == 0 or v_seg_idx == len(vp) - 2
            if h_end and v_end:
                cls = 1  # L
            elif h_end or v_end:
                cls = 2  # T
            else:
                cls = 3  # X
            result.append((found, cls))
    return result


def _junction_masks(
    horizontal: Sequence[Dict[str, np.ndarray]],
    vertical: Sequence[Dict[str, np.ndarray]],
    height: int,
    width: int,
    radius: int,
) -> np.ndarray:
    masks = np.zeros((4, height, width), dtype=np.float32)
    intersections = _curve_intersections(horizontal, vertical)
    for point, cls in intersections:
        x, y = np.round(point).astype(int)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(masks[cls], (x, y), radius, 1.0, thickness=-1)

    # Visible free endpoints that are not close to an intersection.
    intersection_points = [p for p, _ in intersections]
    for curve in list(horizontal) + list(vertical):
        visible_idx = np.flatnonzero(curve["visibility"] > 0.5)
        if visible_idx.size == 0:
            continue
        for idx in (visible_idx[0], visible_idx[-1]):
            point = curve["points"][idx]
            if (
                intersection_points
                and min(np.linalg.norm(point - p) for p in intersection_points)
                <= radius * 2
            ):
                continue
            x, y = np.round(point).astype(int)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(masks[0], (x, y), radius, 1.0, thickness=-1)
    return masks


def _distance_map(mask: np.ndarray, sigma: float) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(mask, dtype=np.float32)
    distance = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    return np.exp(-(distance**2) / (2.0 * sigma**2)).astype(np.float32)


def _table_corners(
    polygons: Sequence[np.ndarray], height: int, width: int
) -> np.ndarray:
    if not polygons:
        return np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
    pts = np.concatenate(polygons, axis=0).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    sums = hull[:, 0] + hull[:, 1]
    diffs = hull[:, 0] - hull[:, 1]
    tl = hull[np.argmin(sums)]
    br = hull[np.argmax(sums)]
    tr = hull[np.argmax(diffs)]
    bl = hull[np.argmin(diffs)]
    corners = np.stack([tl, tr, br, bl]).astype(np.float32)
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    return corners


def _geometry_target(
    horizontal: Sequence[Dict[str, np.ndarray]],
    all_polygons: Sequence[np.ndarray],
    height: int,
    width: int,
) -> Dict[str, Tensor]:
    slopes: List[float] = []
    for curve in horizontal:
        valid = curve["visibility"] > 0.5
        pts = curve["points"][valid]
        if len(pts) >= 2:
            dx = pts[-1, 0] - pts[0, 0]
            dy = pts[-1, 1] - pts[0, 1]
            if abs(dx) > 1e-6:
                slopes.append(float(np.arctan2(dy, dx)))
    theta = float(np.median(slopes)) if slopes else 0.0
    angle_vector = torch.tensor([np.sin(theta), np.cos(theta)], dtype=torch.float32)

    corners_px = _table_corners(all_polygons, height, width)
    corners_norm = corners_px.copy()
    corners_norm[:, 0] /= max(width - 1, 1)
    corners_norm[:, 1] /= max(height - 1, 1)

    destination = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(
        corners_norm.astype(np.float32), destination
    )
    homography /= max(homography[2, 2], 1e-8)

    return {
        "angle_vector": angle_vector,
        "corners": torch.from_numpy(corners_norm.astype(np.float32)),
        "homography": torch.from_numpy(homography.astype(np.float32)),
    }


def _text_mask(polygons: Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for poly in polygons:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1.0)
    return mask[None]


class TableSeparatorDataset(Dataset):
    """Dataset for logical horizontal/vertical table separators.

    Expected annotation JSON keys:
      row: list of row-region polygons
      col: list of column-region polygons
      line: optional text-line polygons
      cell: optional cell polygons
      is_wireless: bool

    Images and JSON files are paired by filename stem.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_dir: str | Path,
        config: Optional[TableDatasetConfig] = None,
        image_extensions: Sequence[str] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".webp",
        ),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.config = config or TableDatasetConfig()
        self.image_extensions = {ext.lower() for ext in image_extensions}

        if not self.image_dir.is_dir():
            raise FileNotFoundError(self.image_dir)
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(self.annotation_dir)

        image_by_stem = {
            p.stem: p
            for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.image_extensions
        }
        annotation_by_stem = {p.stem: p for p in self.annotation_dir.rglob("*.json")}
        common = sorted(set(image_by_stem) & set(annotation_by_stem))
        if not common:
            raise RuntimeError(
                "No image/JSON pairs with matching filename stems were found"
            )
        self.samples = [(image_by_stem[s], annotation_by_stem[s]) for s in common]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Any]]:
        image_path, annotation_path = self.samples[index]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        with annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)

        polygons: Dict[str, List[np.ndarray]] = {}
        for key in ("row", "col", "line", "cell"):
            polygons[key] = [
                _as_polygon_array(poly) for poly in annotation.get(key, [])
            ]

        image, polygons, resize_meta = _letterbox(
            image, polygons, self.config.image_size
        )
        height, width = image.shape[:2]

        horizontal_candidates = _extract_envelopes(
            polygons["row"], "horizontal", height, width, self.config.num_control_points
        )
        vertical_candidates = _extract_envelopes(
            polygons["col"], "vertical", height, width, self.config.num_control_points
        )
        horizontal = _sort_curves(
            _merge_curves(
                horizontal_candidates,
                self.config.separator_merge_distance,
                self.config.min_separator_visibility,
            ),
            "horizontal",
        )
        vertical = _sort_curves(
            _merge_curves(
                vertical_candidates,
                self.config.separator_merge_distance,
                self.config.min_separator_visibility,
            ),
            "vertical",
        )

        horizontal_mask = _draw_curves(
            horizontal, height, width, self.config.line_width
        )
        vertical_mask = _draw_curves(vertical, height, width, self.config.line_width)
        line_masks = np.stack([horizontal_mask, vertical_mask]).astype(np.float32)
        junction_masks = _junction_masks(
            horizontal, vertical, height, width, self.config.junction_radius
        )
        distance_maps = np.stack(
            [
                _distance_map(horizontal_mask, self.config.distance_sigma),
                _distance_map(vertical_mask, self.config.distance_sigma),
            ]
        ).astype(np.float32)

        all_table_polygons = polygons["cell"] or (polygons["row"] + polygons["col"])
        geometry = _geometry_target(horizontal, all_table_polygons, height, width)

        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
            / 255.0
        )
        if self.config.normalize:
            image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        target: Dict[str, Any] = {
            "horizontal": _normalize_curves(
                horizontal, height, width, self.config.line_width
            ),
            "vertical": _normalize_curves(
                vertical, height, width, self.config.line_width
            ),
            "line_masks": torch.from_numpy(line_masks),
            "junction_masks": torch.from_numpy(junction_masks),
            "distance_maps": torch.from_numpy(distance_maps),
            "geometry": geometry,
            "is_wireless": torch.tensor(bool(annotation.get("is_wireless", False))),
            "image_size": torch.tensor([height, width], dtype=torch.long),
            "original_size": torch.tensor(
                [int(resize_meta["src_h"]), int(resize_meta["src_w"])], dtype=torch.long
            ),
            "resize_meta": resize_meta,
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
        }
        if self.config.include_text_mask:
            target["text_mask"] = torch.from_numpy(
                _text_mask(polygons["line"], height, width)
            )

        return image_tensor, target


def table_collate_fn(
    batch: Sequence[Tuple[Tensor, Dict[str, Any]]],
) -> Tuple[Tensor, List[Dict[str, Any]]]:
    """Stacks fixed-size images and preserves variable-length target dictionaries."""
    images, targets = zip(*batch)
    return torch.stack(list(images), dim=0), list(targets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect one TableSeparatorDataset sample"
    )
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("annotation_dir", type=Path)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    args = parser.parse_args()

    config = TableDatasetConfig(image_size=(args.height, args.width))
    dataset = TableSeparatorDataset(args.image_dir, args.annotation_dir, config)
    image, target = dataset[0]
    print("image:", tuple(image.shape))
    print("horizontal:", tuple(target["horizontal"]["control_points"].shape))
    print("vertical:", tuple(target["vertical"]["control_points"].shape))
    print("line_masks:", tuple(target["line_masks"].shape))
    print("junction_masks:", tuple(target["junction_masks"].shape))
    print("distance_maps:", tuple(target["distance_maps"].shape))
