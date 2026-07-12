"""Geometry and semantic helpers for IB subgraph preparation."""

from dataclasses import dataclass
import re

import numpy as np


_FLOOR_NOUNS = ("floor", "ground", "road", "walkway", "pavement")
_FLOOR_NOUN_PATTERN = "|".join(_FLOOR_NOUNS)
_RELATION_TO_FLOOR = re.compile(
    rf"\b(?:on|above|over|near|beside|next to)\s+(?:the\s+)?(?:{_FLOOR_NOUN_PATTERN})\b"
)
_FLOOR_AS_SUBJECT = re.compile(
    rf"^(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){{0,4}}(?:{_FLOOR_NOUN_PATTERN})\b"
)


@dataclass(frozen=True)
class FloorFilterConfig:
    """Controls semantic/geometric floor primitive filtering."""

    enabled: bool = True
    vertical_axis: str = "z"
    max_thickness: float = 0.35
    min_footprint_area: float = 4.0
    semantic_threshold: float = 0.5
    pointcloud_trim_quantile: float = 0.10
    pointcloud_min_points: int = 50
    extreme_thickness_ratio: float = 0.5
    extreme_area_multiplier: float = 2.5

    def __post_init__(self):
        if self.vertical_axis.lower() not in {"x", "y", "z"}:
            raise ValueError("vertical_axis must be one of: x, y, z")
        if self.max_thickness <= 0 or self.min_footprint_area <= 0:
            raise ValueError("Floor geometry thresholds must be positive.")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be within [0, 1].")
        if not 0.0 <= self.pointcloud_trim_quantile < 0.5:
            raise ValueError("pointcloud_trim_quantile must be within [0, 0.5).")
        if self.pointcloud_min_points < 1:
            raise ValueError("pointcloud_min_points must be positive.")


@dataclass(frozen=True)
class FloorDecision:
    """Diagnostic result for one primitive."""

    is_floor: bool
    semantic_score: float
    vertical_thickness: float
    bbox_vertical_thickness: float
    footprint_area: float
    reason: str


def axis_index(axis):
    """Convert x/y/z or an integer axis into an array index."""
    if isinstance(axis, int):
        if axis not in {0, 1, 2}:
            raise ValueError("Axis index must be 0, 1, or 2.")
        return axis
    try:
        return {"x": 0, "y": 1, "z": 2}[str(axis).lower()]
    except KeyError as exc:
        raise ValueError("Axis must be one of: x, y, z.") from exc


def bbox_extent(bbox):
    """Return an extent vector for either an Open3D AABB or OBB."""
    get_extent = getattr(bbox, "get_extent", None)
    extent = get_extent() if callable(get_extent) else bbox.extent
    return np.asarray(extent, dtype=float)


def pointcloud_vertical_thickness(obj, vertical_idx, trim_quantile, min_points):
    """Estimate thickness while ignoring a small fraction of attached/outlier points."""
    point_cloud = obj.get("pcd")
    points = getattr(point_cloud, "points", None)
    if points is None:
        return None

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < min_points:
        return None

    coordinates = points[:, vertical_idx]
    coordinates = coordinates[np.isfinite(coordinates)]
    if len(coordinates) < min_points:
        return None

    low, high = np.quantile(
        coordinates,
        [trim_quantile, 1.0 - trim_quantile],
    )
    return float(max(0.0, high - low))


def semantic_floor_score(obj):
    """Score captions that describe a floor, excluding 'object on floor' relations."""
    bg_class = str(obj.get("bg_class") or "").strip().lower()
    if bg_class in _FLOOR_NOUNS:
        return 1.0

    caption = obj.get("caption", "")
    if isinstance(caption, (list, tuple)):
        segments = [str(value).strip().lower() for value in caption]
    else:
        segments = [
            segment.strip().lower()
            for segment in re.split(r"[,;]", str(caption))
        ]

    floor_mentions = 0
    subject_mentions = 0
    for segment in segments:
        if not segment or not re.search(rf"\b(?:{_FLOOR_NOUN_PATTERN})\b", segment):
            continue
        floor_mentions += 1
        if _RELATION_TO_FLOOR.search(segment):
            continue
        if _FLOOR_AS_SUBJECT.search(segment):
            subject_mentions += 1

    return subject_mentions / floor_mentions if floor_mentions else 0.0


def classify_floor_primitive(obj, config):
    """Classify one primitive using axis-aware geometry and caption semantics."""
    extent = bbox_extent(obj["bbox"])
    vertical_idx = axis_index(config.vertical_axis)
    horizontal_idxs = [idx for idx in range(3) if idx != vertical_idx]

    bbox_thickness = float(extent[vertical_idx])
    robust_thickness = pointcloud_vertical_thickness(
        obj,
        vertical_idx,
        config.pointcloud_trim_quantile,
        config.pointcloud_min_points,
    )
    thickness = bbox_thickness if robust_thickness is None else robust_thickness
    footprint_area = float(np.prod(extent[horizontal_idxs]))
    semantic_score = semantic_floor_score(obj)

    moderately_flat = (
        thickness <= config.max_thickness
        and footprint_area >= config.min_footprint_area
    )
    extremely_flat = (
        thickness <= config.max_thickness * config.extreme_thickness_ratio
        and footprint_area
        >= config.min_footprint_area * config.extreme_area_multiplier
    )
    semantic_floor = semantic_score >= config.semantic_threshold

    if extremely_flat:
        reason = "extreme_geometry"
    elif moderately_flat and semantic_floor:
        reason = "semantic+geometry"
    elif moderately_flat:
        reason = "geometry_without_semantic_support"
    elif semantic_floor:
        reason = "semantic_without_geometry_support"
    else:
        reason = "not_floor"

    return FloorDecision(
        is_floor=extremely_flat or (moderately_flat and semantic_floor),
        semantic_score=semantic_score,
        vertical_thickness=thickness,
        bbox_vertical_thickness=bbox_thickness,
        footprint_area=footprint_area,
        reason=reason,
    )


def filter_floor_primitives(objects, object_ids, config):
    """Return retained object IDs and diagnostics for removed floor primitives."""
    if not config.enabled:
        return list(object_ids), []

    retained_ids = []
    removed = []
    for object_id in object_ids:
        if not 0 <= int(object_id) < len(objects):
            continue
        decision = classify_floor_primitive(objects[int(object_id)], config)
        if decision.is_floor:
            removed.append((int(object_id), decision))
        else:
            retained_ids.append(int(object_id))

    return retained_ids, removed
