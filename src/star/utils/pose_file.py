"""Timestamped robot poses from a TUM trajectory file (numpy and scipy only).

A TUM file has one pose per line: ``timestamp tx ty tz qx qy qz qw``, with the
timestamp in seconds and the pose of the robot base in the map frame. Ground-truth
poses from a dataset, another SLAM system or ``tools/loop_closure`` all use it.
"""

from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_NS = Decimal(1_000_000_000)


def seconds_to_ns(seconds) -> int:
    """Exact integer nanoseconds from a decimal string or number of seconds."""
    if isinstance(seconds, bool):
        raise ValueError("Timestamp must be a number of seconds")
    value = Decimal(str(seconds))
    if not value.is_finite() or value < 0:
        raise ValueError(f"Invalid timestamp: {seconds}")
    return int((value * _NS).to_integral_value())


def validate_transform(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Pose must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("Pose must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) <= 0:
        raise ValueError("Pose rotation must be orthonormal and right-handed")
    return matrix


def load_tum(path):
    """Read a TUM file into (timestamps_ns int64[N], poses float64[N, 4, 4]).

    Comments (#) and blank lines are skipped. Timestamps must be strictly
    increasing and quaternions unit length; nothing is sorted or repaired.
    """
    times, poses = [], []
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            try:
                if len(fields) != 8:
                    raise ValueError("expected 8 fields: timestamp tx ty tz qx qy qz qw")
                timestamp = seconds_to_ns(fields[0])
                values = np.asarray(fields[1:], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("non-finite pose")
                if not np.isclose(np.linalg.norm(values[3:]), 1.0, atol=1e-4, rtol=0):
                    raise ValueError("quaternion is not unit length")
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
                pose[:3, 3] = values[:3]
            except (ValueError, InvalidOperation) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if times and timestamp <= times[-1]:
                raise ValueError(f"{path}:{line_no}: timestamps must be strictly increasing")
            times.append(timestamp)
            poses.append(pose)
    if not times:
        raise ValueError(f"{path}: no poses")
    return np.asarray(times, dtype=np.int64), np.asarray(poses)


def write_tum(path, timestamps_ns, poses):
    """Write poses as TUM with exact nanosecond timestamps."""
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (len(timestamps_ns), 4, 4):
        raise ValueError("One 4x4 pose per timestamp is required")
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with Path(path).open("w", encoding="utf-8") as out:
        for t, pose, q in zip(timestamps_ns, poses, quaternions):
            seconds, nanoseconds = divmod(int(t), 1_000_000_000)
            x, y, z = pose[:3, 3]
            out.write(f"{seconds}.{nanoseconds:09d} {x:.9f} {y:.9f} {z:.9f} "
                      f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


class PoseTimeline:
    """Poses at strictly increasing timestamps, interpolated in between.

    Positions are interpolated linearly and rotations by slerp. Queries outside
    [first, last] raise; nothing is extrapolated. ``max_gap_s`` optionally refuses
    to interpolate across a longer gap between samples.
    """

    def __init__(self, timestamps_ns, poses, max_gap_s=None):
        times = np.asarray(timestamps_ns)
        if times.ndim != 1 or times.dtype.kind not in "iu" or not len(times):
            raise ValueError("Timestamps must be a non-empty 1D integer-nanosecond array")
        times = times.astype(np.int64)
        if np.any(times[1:] <= times[:-1]):
            raise ValueError("Timestamps must be strictly increasing")
        poses = np.array(poses, dtype=np.float64)
        if poses.shape != (len(times), 4, 4):
            raise ValueError("One 4x4 pose per timestamp is required")
        for pose in poses:
            validate_transform(pose)
        if max_gap_s is not None and (isinstance(max_gap_s, bool) or not max_gap_s > 0):
            raise ValueError("max_gap_s must be positive or None")
        self.timestamps_ns = times
        self.poses = poses
        self.max_gap_ns = None if max_gap_s is None else seconds_to_ns(max_gap_s)

    @classmethod
    def from_tum(cls, path, max_gap_s=None):
        return cls(*load_tum(path), max_gap_s=max_gap_s)

    def covers(self, timestamp_ns) -> bool:
        return bool(self.timestamps_ns[0] <= int(timestamp_ns) <= self.timestamps_ns[-1])

    def at(self, timestamp_ns) -> np.ndarray:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, np.integer)):
            raise ValueError("Query time must be integer nanoseconds")
        t = int(timestamp_ns)
        times = self.timestamps_ns
        if not self.covers(t):
            raise ValueError(f"Time {t} ns is outside the trajectory [{times[0]}, {times[-1]}]")
        upper = int(np.searchsorted(times, t))
        if times[upper] == t:
            return self.poses[upper].copy()
        lower = upper - 1
        gap = int(times[upper]) - int(times[lower])
        if self.max_gap_ns is not None and gap > self.max_gap_ns:
            raise ValueError(f"Gap of {gap / 1e9:.3f} s around {t} ns exceeds max_gap_s")
        fraction = (t - int(times[lower])) / gap
        pose = np.eye(4)
        pose[:3, 3] = (1 - fraction) * self.poses[lower, :3, 3] + fraction * self.poses[upper, :3, 3]
        rotations = Rotation.from_matrix(self.poses[[lower, upper], :3, :3])
        pose[:3, :3] = Slerp([0.0, 1.0], rotations)(fraction).as_matrix()
        return pose
