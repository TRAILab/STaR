"""Tests for pose files and tools/loop_closure; numpy and scipy only (no ROS, no Docker).

    python -m unittest discover -s tests
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/loop_closure"))

from star.utils.pose_file import PoseTimeline, load_tum, seconds_to_ns, write_tum  # noqa: E402
import loop_closure  # noqa: E402
import postprocess  # noqa: E402
import trajectory_gates  # noqa: E402

SECOND = 1_000_000_000


def pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0, roll_deg=0.0):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [roll_deg, 0, yaw_deg], degrees=True).as_matrix()
    matrix[:3, 3] = [x, y, z]
    return matrix


def loop_trajectory(n=200, seconds=400.0):
    """A 10 m square driven twice, so the second lap revisits the first."""
    times = (np.linspace(0, seconds, n) * SECOND).astype(np.int64) + 1_700_000_000 * SECOND
    s = np.linspace(0, 80, n) % 40
    side = np.minimum((s // 10).astype(int), 3)
    u = s - 10 * side
    xy = np.select([(side == k)[:, None] for k in range(3)], [np.c_[u, 0 * u], np.c_[10 + 0 * u, u], np.c_[10 - u, 10 + 0 * u]],
                   np.c_[0 * u, 10 - u])
    yaw = 90.0 * side
    return times, np.array([pose(x, y, 0, a) for (x, y), a in zip(xy, yaw)])


class PoseFileTests(unittest.TestCase):
    def test_round_trip_keeps_nanoseconds(self):
        times = np.array([1_700_000_000_123_456_789, 1_700_000_001_000_000_001], dtype=np.int64)
        poses = np.array([pose(1, 2, 3, 30), pose(4, 5, 6, -60, 5)])
        with tempfile.TemporaryDirectory() as d:
            write_tum(Path(d) / "t.tum", times, poses)
            loaded_times, loaded = load_tum(Path(d) / "t.tum")
        np.testing.assert_array_equal(loaded_times, times)
        np.testing.assert_allclose(loaded, poses, atol=1e-8)

    def test_interpolates_and_never_extrapolates(self):
        timeline = PoseTimeline(np.array([0, 2 * SECOND]), [pose(0, 0, 0, 0), pose(2, 0, 0, 90)])
        middle = timeline.at(SECOND)
        np.testing.assert_allclose(middle[:3, 3], [1, 0, 0])
        np.testing.assert_allclose(Rotation.from_matrix(middle[:3, :3]).as_euler("xyz", degrees=True), [0, 0, 45],
                                   atol=1e-9)
        np.testing.assert_array_equal(timeline.at(0), pose())
        with self.assertRaises(ValueError):
            timeline.at(2 * SECOND + 1)
        with self.assertRaises(ValueError):
            timeline.at(1.0)

    def test_max_gap(self):
        timeline = PoseTimeline(np.array([0, 2 * SECOND]), [pose(), pose(1)], max_gap_s=1.0)
        with self.assertRaises(ValueError):
            timeline.at(SECOND)
        self.assertTrue(timeline.covers(SECOND))

    def test_rejects_bad_files(self):
        bad = {
            "decreasing": "2.0 0 0 0 0 0 0 1\n1.0 0 0 0 0 0 0 1\n",
            "fields": "1.0 0 0 0 0 0 1\n",
            "quaternion": "1.0 0 0 0 0 0 0 2\n",
            "empty": "# nothing\n",
            "nan": "1.0 nan 0 0 0 0 0 1\n",
        }
        with tempfile.TemporaryDirectory() as d:
            for name, text in bad.items():
                path = Path(d) / f"{name}.tum"
                path.write_text(text)
                with self.subTest(name), self.assertRaises(ValueError):
                    load_tum(path)

    def test_seconds_to_ns_is_exact(self):
        self.assertEqual(seconds_to_ns("1700000000.123456789"), 1_700_000_000_123_456_789)
        with self.assertRaises(ValueError):
            seconds_to_ns(True)


class GateTests(unittest.TestCase):
    def test_rigid_correction_passes(self):
        times, raw = loop_trajectory()
        report = trajectory_gates.evaluate(raw, raw.copy(), times)
        self.assertTrue(report["passed"], report)
        self.assertGreater(report["revisit_pairs"], 0)

    def test_step_jump_fails(self):
        times, raw = loop_trajectory()
        corrected = raw.copy()
        corrected[100:, 0, 3] += 0.2
        report = trajectory_gates.evaluate(raw, corrected, times)
        self.assertFalse(report["gates"]["step_consistency"])
        self.assertFalse(report["passed"])

    def test_tilt_fails(self):
        times, raw = loop_trajectory()
        corrected = np.array([pose(roll_deg=0.5) @ p for p in raw])
        report = trajectory_gates.evaluate(raw, corrected, times)
        self.assertFalse(report["gates"]["attitude"])

    def test_yaw_correction_is_not_a_tilt(self):
        times, raw = loop_trajectory()
        sloped = np.array([p @ pose(roll_deg=8.0) for p in raw])
        corrected = np.array([pose(yaw_deg=2.0) @ p for p in sloped])
        report = trajectory_gates.evaluate(sloped, corrected, times)
        self.assertTrue(report["gates"]["attitude"], report["up_axis_deg"])

    def test_revisit_height_fails(self):
        times, raw = loop_trajectory()
        corrected = raw.copy()
        corrected[:, 2, 3] += np.linspace(0, 0.5, len(raw))
        report = trajectory_gates.evaluate(raw, corrected, times)
        self.assertFalse(report["gates"]["revisit_height"])

    def test_unknown_threshold(self):
        times, raw = loop_trajectory()
        with self.assertRaises(ValueError):
            trajectory_gates.evaluate(raw, raw, times, {"bogus": 1.0})


class DensifyTests(unittest.TestCase):
    def test_constant_correction_is_applied_to_every_pose(self):
        keyframe_times = np.array([10, 20, 30], dtype=np.int64) * SECOND
        raw = np.array([pose(0), pose(1), pose(2, yaw_deg=10)])
        offset = pose(5, -3, 0.2, yaw_deg=7)
        corrected = np.array([offset @ p for p in raw])
        odom_times = np.arange(5, 36, dtype=np.int64) * SECOND
        odom = np.array([pose(0.1 * i, 0.05 * i, 0, yaw_deg=i) for i in range(len(odom_times))])
        times, dense, before, after = postprocess.densify(keyframe_times, raw, corrected, odom_times, odom)
        np.testing.assert_array_equal(times, np.arange(10, 31) * SECOND)
        self.assertEqual((before, after), (5, 5))
        np.testing.assert_allclose(dense, [offset @ p for p in odom[5:26]], atol=1e-9)

    def test_correction_is_interpolated_between_keyframes(self):
        keyframe_times = np.array([0, 2 * SECOND])
        raw = np.array([pose(), pose(1)])
        corrected = np.array([pose(), pose(1.2)])
        times, dense, _, _ = postprocess.densify(keyframe_times, raw, corrected, np.array([SECOND]), np.array([pose(0.5)]))
        np.testing.assert_allclose(dense[0][:3, 3], [0.6, 0, 0])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "bag").mkdir()
        (self.root / "ws/bin").mkdir(parents=True)
        (self.root / "ws/bin/reopt").write_text("binary")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return loop_closure.parse_args(["--bag", str(self.root / "bag"), "--workspace", str(self.root / "ws"),
                                        "--out", str(self.root / "out"), *extra])

    def fake_docker(self, passed):
        times, raw = loop_trajectory()

        def run(args, mounts, command, *, log, name, cpus=None, extra=()):
            work = self.root / "out/work"
            with open(log, "a") as stream:
                stream.write(f"{name}\n")
            if name.endswith("-kiss"):
                export = work / "kiss/export/trailbot"
                export.mkdir(parents=True)
                write_tum(export / "raw_poses_tum.txt", times, raw)
                (export / "graph.g2o").write_text("EDGE_SE3:QUAT 0 1\n")
                (work / "kiss/kiss.log").write_text("local fix 4: wrote graph.g2o\nLC accepted\n")
                with open(log, "a") as stream:
                    stream.write("RECORDING_CLOSED_OK\n")
            elif name.endswith("-reopt"):
                write_tum(work / "corrected_keyframes.tum", times, raw)
            else:
                np.savez(work / "keyframes.npz", raw=raw)
                gates = {"passed": passed, "gates": {"step_consistency": passed}}
                trajectory = {"poses": len(times)}
                (work / "postprocess.json").write_text(json.dumps({"gates": gates, "trajectory": trajectory}))
                if passed:
                    write_tum(work / "trajectory.tum", times, raw)
                    return 0
                return 3
            return 0
        return run

    def test_pass_writes_trajectory_and_report(self):
        code = loop_closure.run(self.args(), docker_run=self.fake_docker(True), check=lambda a: None)
        self.assertEqual(code, 0)
        report = json.loads((self.root / "out/report.json").read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["kiss"]["loops_accepted"], 1)
        self.assertEqual(report["trajectory_sha256"], loop_closure.sha256(self.root / "out/trajectory.tum"))
        for name in ("keyframes.npz", "corrected_keyframes.tum", "graph.g2o", "raw_keyframes.tum"):
            self.assertTrue((self.root / "out" / name).is_file(), name)

    def test_failed_gate_writes_no_trajectory(self):
        code = loop_closure.run(self.args(), docker_run=self.fake_docker(False), check=lambda a: None)
        self.assertEqual(code, 3)
        self.assertFalse((self.root / "out/trajectory.tum").exists())
        self.assertFalse(json.loads((self.root / "out/report.json").read_text())["passed"])

    def test_rejects_bad_arguments(self):
        for extra in (["--gates", '{"bogus": 1}'], ["--gates", '{"max_step_m": -1}'], ["--gravity-deg", "0"],
                      ["--odometry-topic", "odometry"], ["--base-frame", "base link"]):
            with self.subTest(extra), self.assertRaises(SystemExit):
                self.args(*extra)

    def test_preflight_refuses_a_stale_reopt(self):
        args = self.args()
        for relative in ("install/setup.bash", "install/kiss_matcher_ros/lib/kiss_matcher_ros/kiss_matcher_sam"):
            (self.root / "ws" / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.root / "ws" / relative).write_text("")
        (self.root / "ws/bin/reopt.source_sha256").write_text("0" * 64 + "  reopt.cpp\n")
        inspected = mock.Mock(returncode=0)
        with mock.patch.object(loop_closure.shutil, "which", return_value="/usr/bin/docker"), \
                mock.patch.object(loop_closure.subprocess, "run", return_value=inspected), \
                self.assertRaisesRegex(SystemExit, "different reopt.cpp"):
            loop_closure.preflight(args)


if __name__ == "__main__":
    unittest.main()
