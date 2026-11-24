"""Acceptance gates for a loop-closure-corrected keyframe trajectory (numpy only).

Judge a corrected trajectory against the raw odometry it corrects, never by
correction magnitude or a rigid fit (a real, smooth vertical drift correction also
fits as a small "tilt").

    python3 tools/loop_closure/trajectory_gates.py KEYFRAMES_NPZ [THRESHOLDS_JSON]

prints a JSON report and exits 3 when any gate fails.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from loop_closure import DEFAULT_THRESHOLDS  # stdlib-only module

REVISIT_RADIUS_M = 2.0
REVISIT_MIN_GAP_S = 60.0


def _stats(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
            "max": float(values.max())}


def step_errors(raw, corrected):
    """Body-frame relative translation error per consecutive keyframe step (m)."""
    dr = np.einsum("nji,nj->ni", raw[:-1, :3, :3], raw[1:, :3, 3] - raw[:-1, :3, 3])
    dc = np.einsum("nji,nj->ni", corrected[:-1, :3, :3], corrected[1:, :3, 3] - corrected[:-1, :3, 3])
    return np.linalg.norm(dc - dr, axis=1)


def up_axis_deg(raw, corrected):
    """Angle between world up as seen in the body frame (R^T e_z), raw vs corrected.

    This is what the IMU constrains; a world-frame yaw correction leaves it unchanged
    even on sloped ground, unlike comparing body z in the world frame.
    """
    cos = np.clip((raw[:, 2, :3] * corrected[:, 2, :3]).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def revisit_pairs(positions, times_s):
    dxy = np.linalg.norm(positions[:, None, :2] - positions[None, :, :2], axis=2)
    gap = np.abs(times_s[:, None] - times_s[None, :])
    return np.argwhere(np.triu((dxy < REVISIT_RADIUS_M) & (gap > REVISIT_MIN_GAP_S), 1))


def evaluate(raw, corrected, timestamps_ns, thresholds=None):
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    unknown = set(limits) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise ValueError(f"Unknown gate thresholds: {sorted(unknown)}")
    if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in limits.values()):
        raise ValueError("Gate thresholds must be finite positive numbers")
    raw, corrected = np.asarray(raw, float), np.asarray(corrected, float)
    if raw.shape != corrected.shape or raw.ndim != 3 or raw.shape[1:] != (4, 4) or len(raw) < 2:
        raise ValueError("raw/corrected must be matching Nx4x4 keyframe poses, N >= 2")
    if not (np.isfinite(raw).all() and np.isfinite(corrected).all()):
        raise ValueError("Keyframe poses must be finite")
    timestamps_ns = np.asarray(timestamps_ns)
    if timestamps_ns.shape != (len(raw),):
        raise ValueError("timestamps_ns must be one timestamp per keyframe")
    times = timestamps_ns.astype(np.int64) / 1e9
    steps, up = step_errors(raw, corrected), up_axis_deg(raw, corrected)
    pairs = revisit_pairs(corrected[:, :3, 3], times)
    report = {"keyframes": int(len(raw)), "thresholds": limits,
              "step_error_m": _stats(steps), "up_axis_deg": _stats(up),
              "revisit_pairs": int(len(pairs))}
    gates = {
        "step_consistency": report["step_error_m"]["max"] <= limits["max_step_m"],
        "attitude": (report["up_axis_deg"]["median"] <= limits["up_median_deg"]
                     and report["up_axis_deg"]["max"] <= limits["up_max_deg"]),
    }
    if len(pairs):
        z = corrected[:, 2, 3]
        report["revisit_abs_dz_corrected_m"] = _stats(np.abs(z[pairs[:, 0]] - z[pairs[:, 1]]))
        zr = raw[:, 2, 3]
        report["revisit_abs_dz_raw_m"] = _stats(np.abs(zr[pairs[:, 0]] - zr[pairs[:, 1]]))
        dz = report["revisit_abs_dz_corrected_m"]
        gates["revisit_height"] = (dz["median"] <= limits["revisit_dz_median_m"]
                                   and dz["p90"] <= limits["revisit_dz_p90_m"])
    else:
        # No revisits means no loop closures could have mattered; nothing to judge.
        gates["revisit_height"] = True
    report["gates"] = gates
    report["passed"] = all(gates.values())
    return report


def evaluate_npz(path, thresholds=None):
    with np.load(path, allow_pickle=False) as data:
        return evaluate(data["raw"], data["corrected"], data["timestamps_ns"], thresholds)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        print(__doc__, file=sys.stderr)
        return 2
    thresholds = json.loads(argv[1]) if len(argv) == 2 else None
    report = evaluate_npz(argv[0], thresholds)
    print(json.dumps(report, indent=1, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
