#!/usr/bin/env python3
"""Keyframes, trajectory gates and the dense corrected trajectory. Runs inside the SLAM image.

    postprocess.py RAW_TUM CORRECTED_TUM BAG ODOMETRY_TOPIC OUT_DIR [THRESHOLDS_JSON]

RAW_TUM holds KISS's keyframes as recorded odometry (odom <- base_link) and
CORRECTED_TUM the re-solved keyframes (map <- base_link), at the same timestamps.
Writes OUT_DIR/keyframes.npz and OUT_DIR/postprocess.json. Only when every gate
passes does it also write OUT_DIR/trajectory.tum: each recorded odometry pose
T(t) inside the keyframe span, corrected by C(t) @ T(t), where the correction
C = T_corrected @ inverse(T_raw) is interpolated between keyframes. Exits 3
when a gate fails.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from star.utils.pose_file import PoseTimeline, load_tum, write_tum  # noqa: E402
import trajectory_gates  # noqa: E402


def read_odometry(bag, topic):
    """(timestamps_ns, poses, frames) of every message on `topic`; repeated or earlier stamps are dropped."""
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    if topic not in {t.name for t in reader.get_all_topics_and_types()}:
        raise SystemExit(f"POSTPROCESS_FAIL bag has no {topic}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    times, positions, quaternions, frames, dropped = [], [], [], set(), 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        message = deserialize_message(data, Odometry)
        t = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if times and t <= times[-1]:
            dropped += 1
            continue
        p, q = message.pose.pose.position, message.pose.pose.orientation
        times.append(t)
        positions.append([p.x, p.y, p.z])
        quaternions.append([q.x, q.y, q.z, q.w])
        frames.add((message.header.frame_id, message.child_frame_id))
    if not times:
        raise SystemExit(f"POSTPROCESS_FAIL no messages on {topic}")
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(np.asarray(quaternions)).as_matrix()
    poses[:, :3, 3] = positions
    return np.asarray(times, dtype=np.int64), poses, sorted(frames), dropped


def densify(keyframe_times, raw, corrected, odom_times, odom_poses):
    """C(t) @ T(t) for every odometry pose inside the keyframe span."""
    correction = PoseTimeline(keyframe_times, corrected @ np.linalg.inv(raw))
    inside = (odom_times >= keyframe_times[0]) & (odom_times <= keyframe_times[-1])
    times = odom_times[inside]
    poses = np.array([correction.at(int(t)) @ pose for t, pose in zip(times, odom_poses[inside])])
    return times, poses, int((odom_times < keyframe_times[0]).sum()), int((odom_times > keyframe_times[-1]).sum())


def main(argv):
    if len(argv) not in (5, 6):
        print(__doc__, file=sys.stderr)
        return 2
    raw_tum, corrected_tum, bag, topic, out = argv[:5]
    thresholds = json.loads(argv[5]) if len(argv) == 6 else None
    out = Path(out)
    raw_times, raw = load_tum(raw_tum)
    times, corrected = load_tum(corrected_tum)
    if not np.array_equal(raw_times, times):
        raise SystemExit("POSTPROCESS_FAIL raw and corrected keyframe timestamps differ")
    with open(out / "keyframes.npz", "wb") as stream:
        np.savez_compressed(stream, raw=raw, corrected=corrected, timestamps_ns=times)
    gates = trajectory_gates.evaluate(raw, corrected, times, thresholds)
    result = {"gates": gates}
    if gates["passed"]:
        odom_times, odom_poses, frames, dropped = read_odometry(bag, topic)
        dense_times, dense, before, after = densify(times, raw, corrected, odom_times, odom_poses)
        tmp = out / ".trajectory.tum.tmp"
        write_tum(tmp, dense_times, dense)
        os.replace(tmp, out / "trajectory.tum")
        result["trajectory"] = {
            "poses": int(len(dense_times)), "start_ns": int(dense_times[0]), "end_ns": int(dense_times[-1]),
            "odometry_messages": int(len(odom_times)), "odometry_dropped_repeated_stamps": dropped,
            "odometry_before_first_keyframe": before, "odometry_after_last_keyframe": after,
            "odometry_frames": [list(pair) for pair in frames],
        }
    (out / "postprocess.json").write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False))
    return 0 if gates["passed"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
