#!/usr/bin/env python3
"""Feed KISS-SAM from a recorded bag in one pass: recorded /odometry interpolated at
LiDAR stamps, plus /ouster/points placed in the odom frame via
T_odom_base(t) * T_base_lidar. No deskew. Reads the bag read-only; paced at most RATE x
recorded time. usage: feeder.py BAG [RATE] [SAVE_DIR] [MAX_SEC]"""
import sys, time, bisect
import numpy as np
import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R, Slerp

BAG = sys.argv[1]
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SAVE_DIR = sys.argv[3] if len(sys.argv) > 3 else ""
MAX_SEC = float(sys.argv[4]) if len(sys.argv) > 4 else 1e12
import os
LIDAR_TOPIC = os.environ.get("FEEDER_LIDAR_TOPIC", "/ouster/points")
ODOM_TOPIC = os.environ.get("FEEDER_ODOM_TOPIC", "/odometry")
BASE_FRAME = os.environ.get("FEEDER_BASE_FRAME")
LIDAR_FRAME = os.environ.get("FEEDER_LIDAR_FRAME")
# Trailbot base_link->os_lidar, used when FEEDER_BASE_FRAME/FEEDER_LIDAR_FRAME are unset.
T_BL = np.eye(4)
T_BL[:3, :3] = R.from_quat([0, 0, 1, 0]).as_matrix()
T_BL[:3, 3] = [-0.06, 0, 1.0644]


def static_transform(bag, base, lidar):
    """T_base_lidar composed from the bag's /tf_static; raises if the chain is absent."""
    from tf2_msgs.msg import TFMessage
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    if "/tf_static" not in {t.name for t in reader.get_all_topics_and_types()}:
        raise SystemExit("FEEDER_FAIL bag has no /tf_static")
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf_static"]))
    parent = {}
    while reader.has_next():
        _, data, _ = reader.read_next()
        for tf in deserialize_message(data, TFMessage).transforms:
            T = np.eye(4)
            q, t = tf.transform.rotation, tf.transform.translation
            T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = [t.x, t.y, t.z]
            parent[tf.child_frame_id] = (tf.header.frame_id, T)
        chain, frame = np.eye(4), lidar
        for _ in range(32):
            if frame == base:
                return chain
            if frame not in parent:
                break
            frame, T_parent_frame = parent[frame]
            chain = T_parent_frame @ chain
    raise SystemExit(f"FEEDER_FAIL no /tf_static chain {base} -> {lidar}")
FIELDS = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1)
          for i, k in enumerate(["x", "y", "z", "intensity"])]


def stamp(h):
    return h.stamp.sec + 1e-9 * h.stamp.nanosec


def main():
    global T_BL
    if BASE_FRAME and LIDAR_FRAME:
        T_BL = static_transform(BAG, BASE_FRAME, LIDAR_FRAME)
        print(f"T_{BASE_FRAME}_{LIDAR_FRAME} from /tf_static: t={T_BL[:3, 3].round(4).tolist()} "
              f"quat={R.from_matrix(T_BL[:3, :3]).as_quat().round(4).tolist()}", flush=True)
    rclpy.init()
    n = Node("bag_feeder")
    qos = rclpy.qos.QoSProfile(depth=200, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)
    po = n.create_publisher(Odometry, "/odometry", qos)
    pc = n.create_publisher(PointCloud2, "/cloud_registered", qos)
    ps = n.create_publisher(String, "/km_sam/save_dir", 10)
    time.sleep(3.0)

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="mcap"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    r.set_filter(rosbag2_py.StorageFilter(topics=[ODOM_TOPIC, LIDAR_TOPIC]))
    ts, P, Q, pend = [], [], [], []
    st = {"t0w": None, "t0b": None, "sent": 0, "skipped": 0, "first_odom_stamp": None}

    def emit(c, t):
        if st["t0b"] is None:
            st["t0w"], st["t0b"] = time.monotonic(), t
        lag = (t - st["t0b"]) / RATE - (time.monotonic() - st["t0w"])
        if lag > 0:
            time.sleep(lag)
        T = np.eye(4)
        if st["sent"] == 0:
            # First keyframe is stamped at the FIRST odometry message (pose P[0]) so no journal
            # observation (pose time = some odometry stamp) can precede KISS keyframe coverage.
            c.header.stamp = st["first_odom_stamp"]
            T[:3, 3], T[:3, :3] = P[0], R.from_quat(Q[0]).as_matrix()
        else:
            i = bisect.bisect_left(ts, t)
            a, b = i - 1, i
            w8 = (t - ts[a]) / (ts[b] - ts[a])
            T[:3, 3] = (1 - w8) * np.array(P[a]) + w8 * np.array(P[b])
            T[:3, :3] = Slerp([ts[a], ts[b]], R.from_quat([Q[a], Q[b]]))([t]).as_matrix()[0]
        off = {f.name: f.offset for f in c.fields}
        raw = np.frombuffer(bytes(c.data), dtype=np.uint8).reshape(-1, c.point_step)
        col = lambda k: raw[:, off[k]:off[k] + 4].copy().view(np.float32)[:, 0]
        xyz = np.stack([col("x"), col("y"), col("z")], 1)
        inten = col("intensity")
        ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.5)
        W = (T @ T_BL)[:3]
        w = xyz[ok] @ W[:, :3].T + W[:, 3]
        out = np.empty(len(w), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4")])
        out["x"], out["y"], out["z"], out["intensity"] = w[:, 0], w[:, 1], w[:, 2], inten[ok]

        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = c.header.stamp, "odom", "base_link"
        q = R.from_matrix(T[:3, :3]).as_quat()
        pp = od.pose.pose
        pp.position.x, pp.position.y, pp.position.z = T[:3, 3]
        pp.orientation.x, pp.orientation.y, pp.orientation.z, pp.orientation.w = q
        pc2 = PointCloud2()
        pc2.header.stamp, pc2.header.frame_id = c.header.stamp, "odom"
        pc2.height, pc2.width, pc2.fields = 1, len(out), FIELDS
        pc2.point_step, pc2.row_step, pc2.is_dense = 16, 16 * len(out), True
        pc2.data = out.tobytes()
        po.publish(od)
        pc.publish(pc2)
        st["sent"] += 1
        if st["sent"] % 600 == 0:
            n.get_logger().info(f"sent {st['sent']} scans, bag t+{t - st['t0b']:.0f}s, lag {-lag:.2f}s")

    done = False
    while r.has_next() and rclpy.ok() and not done:
        tp, d, _ = r.read_next()
        if tp == ODOM_TOPIC:
            m = deserialize_message(d, Odometry)
            t = stamp(m.header)
            if ts and t <= ts[-1]:
                continue
            if not ts:
                st["first_odom_stamp"] = m.header.stamp
            p = m.pose.pose
            ts.append(t)
            P.append([p.position.x, p.position.y, p.position.z])
            Q.append([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
            while pend and pend[0][1] < ts[-1]:
                c, tc = pend.pop(0)
                if tc < ts[0]:
                    st["skipped"] += 1
                    continue
                if st["t0b"] is not None and tc - st["t0b"] > MAX_SEC:
                    done = True
                    break
                emit(c, tc)
        else:
            c = deserialize_message(d, PointCloud2)
            pend.append((c, stamp(c.header)))
            if len(pend) > 50:
                pend.pop(0)
                st["skipped"] += 1
    st["skipped"] += len(pend)
    n.get_logger().info(f"done: sent={st['sent']} skipped={st['skipped']} odom={len(ts)}; waiting 60s for loop timers")
    time.sleep(60.0)
    if SAVE_DIR:
        ps.publish(String(data=SAVE_DIR))
        n.get_logger().info(f"save requested -> {SAVE_DIR}")
        time.sleep(30.0)
    print("FEEDER_DONE", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
