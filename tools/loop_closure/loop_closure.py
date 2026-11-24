#!/usr/bin/env python3
"""Gravity-anchored loop-closure poses from a recorded bag (Python standard library only).

    python3 tools/loop_closure/loop_closure.py --bag BAG --workspace WS --out OUT [options]

Runs on the host and starts throwaway containers from the SLAM image built by setup.sh:

    KISS-Matcher SAM (patched) on the bag -> pose graph dump (graph.g2o, raw keyframes)
    -> batch re-solve with a per-keyframe gravity prior (reopt)
    -> trajectory gates (fail closed) -> OUT/trajectory.tum

OUT/trajectory.tum is a TUM trajectory of base_link in the map frame at every recorded
odometry stamp; pass it to STaR capture as `pose_file`. When a gate fails, no trajectory
is written and the command exits 3; OUT/report.json says which gate failed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

DEFAULT_THRESHOLDS = {
    "max_step_m": 0.05,            # consecutive-keyframe odometry violation, max
    "up_median_deg": 0.10,         # corrected vs raw (IMU) up axis
    "up_max_deg": 0.30,
    "revisit_dz_median_m": 0.02,   # |dz| between revisits of the same place, corrected
    "revisit_dz_p90_m": 0.10,
}

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
PATCH = TOOLS / "kiss/kiss_matcher_8b58b11_local_fixes.patch"
REOPT_SOURCE = TOOLS / "reopt.cpp"
DEFAULT_KISS_CONFIG = TOOLS / "kiss/slam_config_trailbot.yaml"
KISS_SOURCE = Path("src/KISS-Matcher")
PGM = KISS_SOURCE / "ros/src/slam/pose_graph_manager.cpp"
# KISS-Matcher's own odometry/loop noise (deg, m), reused by the re-solve.
ODOM_ROT_DEG, ODOM_TRANS_M, LOOP_ROT_DEG, LOOP_TRANS_M = 0.573, 0.1, 0.573, 0.1
TOPIC = re.compile(r"(/[A-Za-z0-9_]+)+")
FRAME = re.compile(r"[A-Za-z0-9_/]+")
_NOISE = ("nvidia", "cuda version", "license", "container image")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True, type=Path, help="rosbag2 directory (MCAP) with odometry, LiDAR and /tf_static")
    p.add_argument("--workspace", required=True, type=Path, help="workspace prepared by setup.sh")
    p.add_argument("--out", required=True, type=Path, help="new output directory")
    p.add_argument("--image", default="star-loop-closure:jazzy", help="SLAM image built by setup.sh")
    p.add_argument("--odometry-topic", default="/odometry", help="nav_msgs/Odometry of base_link in odom")
    p.add_argument("--lidar-topic", default="/ouster/points", help="sensor_msgs/PointCloud2 in the LiDAR frame")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument("--lidar-frame", default="os_lidar", help="LiDAR frame in /tf_static")
    p.add_argument("--kiss-config", type=Path, default=DEFAULT_KISS_CONFIG)
    p.add_argument("--gravity-deg", type=float, default=0.1, help="sigma of the per-keyframe gravity prior")
    p.add_argument("--kernel", choices=("none", "huber", "cauchy"), default="cauchy",
                   help="robust kernel on loop factors in the re-solve")
    p.add_argument("--gates", default=None,
                   help=f"JSON overriding gate limits; keys {sorted(DEFAULT_THRESHOLDS)}")
    p.add_argument("--kiss-dump", type=Path, default=None,
                   help="re-solve an existing KISS export dir (graph.g2o, raw_poses_tum.txt) instead of running KISS")
    p.add_argument("--rate", type=float, default=1.0, help="bag playback rate into KISS")
    p.add_argument("--cpus", type=float, default=10, help="CPU limit of the KISS container")
    args = p.parse_args(argv)
    validate(args)
    return args


def validate(args):
    for name, low, high in (("gravity_deg", 1e-3, 10.0), ("rate", 0.05, 10.0), ("cpus", 1, 256)):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [{low}, {high}]")
    for name in ("odometry_topic", "lidar_topic"):
        if not TOPIC.fullmatch(getattr(args, name)):
            raise SystemExit(f"--{name.replace('_', '-')} must be an absolute ROS topic")
    for name in ("base_frame", "lidar_frame"):
        if not FRAME.fullmatch(getattr(args, name)):
            raise SystemExit(f"--{name.replace('_', '-')} must be a TF frame name")
    limits = dict(DEFAULT_THRESHOLDS)
    if args.gates:
        overrides = json.loads(args.gates)
        unknown = set(overrides) - set(DEFAULT_THRESHOLDS)
        if not isinstance(overrides, dict) or unknown:
            raise SystemExit(f"--gates has unknown keys {sorted(unknown)}")
        limits.update(overrides)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values()):
        raise SystemExit("Gate limits must be finite positive numbers")
    args.gate_limits = limits
    args.bag, args.workspace, args.out = args.bag.resolve(), args.workspace.resolve(), args.out.resolve()
    args.kiss_config = args.kiss_config.resolve()
    if args.kiss_dump:
        args.kiss_dump = args.kiss_dump.resolve()


def seq_name(kiss_config):
    """KISS writes its export to <out>/export/<seq_name>."""
    match = re.search(r"^\s*seq_name:\s*[\"']?([A-Za-z0-9_]+)", Path(kiss_config).read_text(), re.M)
    return match.group(1) if match else "trailbot"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(args):
    """Cheap checks before any container starts."""
    ws = args.workspace
    if shutil.which("docker") is None:
        raise SystemExit("Docker is required")
    if subprocess.run(["docker", "image", "inspect", args.image], capture_output=True).returncode:
        raise SystemExit(f"SLAM image {args.image} not found; run tools/loop_closure/setup.sh")
    for relative in ("bin/reopt", "bin/reopt.source_sha256", "install/setup.bash",
                     "install/kiss_matcher_ros/lib/kiss_matcher_ros/kiss_matcher_sam"):
        if not (ws / relative).exists():
            raise SystemExit(f"Workspace lacks {relative}; run tools/loop_closure/setup.sh {ws}")
    if (ws / "bin/reopt.source_sha256").read_text().split()[0] != sha256(REOPT_SOURCE):
        raise SystemExit("bin/reopt was built from a different reopt.cpp; rerun setup.sh")
    source = (ws / PGM).read_text() if (ws / PGM).is_file() else ""
    if "local fix 4" not in source:
        raise SystemExit(f"KISS-Matcher source lacks {PATCH.name}")
    if "local fix 2" in source:
        raise SystemExit("KISS-Matcher source has a live gravity factor (local fix 2); it breaks odometry")
    if not args.kiss_config.is_file():
        raise SystemExit(f"Missing KISS config {args.kiss_config}")
    metadata = args.bag / "metadata.yaml"
    if not metadata.is_file():
        raise SystemExit(f"{args.bag} is not a rosbag2 directory")
    text = metadata.read_text()
    topics = [args.odometry_topic] + ([] if args.kiss_dump else [args.lidar_topic, "/tf_static"])
    for topic in topics:
        if not re.search(rf"name:\s*{re.escape(topic)}\s*$", text, re.M):
            raise SystemExit(f"The bag has no {topic} topic")
    if not args.kiss_dump and not re.search(r"storage_identifier:\s*mcap\s*$", text, re.M):
        raise SystemExit("The KISS feeder reads MCAP bags; convert with `ros2 bag convert`")
    if args.kiss_dump:
        for name in ("graph.g2o", "raw_poses_tum.txt"):
            if not (args.kiss_dump / name).is_file():
                raise SystemExit(f"--kiss-dump lacks {name}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty; KISS is not resumable, so use a new output directory")


def git(ws, *argv):
    result = subprocess.run(["git", "-C", str(ws / KISS_SOURCE), *argv], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def docker(args, mounts, command, *, log, name, cpus=None, extra=()):
    argv = ["docker", "run", "--rm", "--read-only", "--tmpfs", "/tmp", "--log-driver", "none",
            "--network", "none", "--ulimit", "core=0", "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "--name", name, *extra]
    if cpus:
        argv += ["--cpus", str(cpus)]
    for host, inside, mode in mounts:
        argv += ["-v", f"{host}:{inside}{':ro' if mode == 'ro' else ''}"]
    argv += [args.image, "bash", "-c", command]
    with open(log, "a") as stream:
        stream.write(f"\n$ {shlex.join(argv)}\n")
        stream.flush()
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                if not any(word in line.lower() for word in _NOISE):
                    stream.write(line)
            return process.wait()
        except BaseException:
            # An interrupted run must not leave a detached container behind.
            subprocess.run(["docker", "kill", name], capture_output=True)
            process.kill()
            process.wait()
            raise


def step(name):
    print(f"[loop_closure {time.strftime('%H:%M:%S')}] {name}", flush=True)


def run(args, docker_run=docker, check=preflight):
    check(args)
    out, ws = args.out, args.workspace
    work = out / "work"
    work.mkdir(parents=True)
    log = out / "loop_closure.log"
    seq = seq_name(args.kiss_config)
    export = work / "kiss/export" / seq
    tag = hashlib.sha256(str(out).encode()).hexdigest()[:12]
    started = time.time()

    if args.kiss_dump:
        step(f"reusing KISS graph dump {args.kiss_dump}")
        export.mkdir(parents=True)
        for name in ("graph.g2o", "raw_poses_tum.txt"):
            shutil.copyfile(args.kiss_dump / name, export / name)
        kiss_log = args.kiss_dump.parents[1] / "kiss.log"
        text = kiss_log.read_text(errors="replace") if kiss_log.is_file() else ""
        kiss = {"reused_dump": str(args.kiss_dump)}
    else:
        step("KISS-Matcher on the bag (about the bag's duration at --rate 1)")
        env = {"KISS_CONFIG": f"/kiss_config/{args.kiss_config.name}", "FEEDER_LIDAR_TOPIC": args.lidar_topic,
               "FEEDER_ODOM_TOPIC": args.odometry_topic, "FEEDER_BASE_FRAME": args.base_frame,
               "FEEDER_LIDAR_FRAME": args.lidar_frame}
        mounts = [(ws, "/ws", "rw"), (TOOLS, "/tools", "ro"), (args.bag, "/media/bag", "ro"),
                  (args.kiss_config.parent, "/kiss_config", "ro"), (work, "/work", "rw")]
        code = docker_run(args, mounts, f"bash /tools/kiss/run_kiss.sh /work/kiss {args.rate} 100000",
                          log=log, name=f"star-lc-{tag}-kiss", cpus=args.cpus,
                          extra=("--tmpfs", "/run", "--shm-size", "4g",
                                 *[a for k, v in env.items() for a in ("-e", f"{k}={v}")]))
        kiss_log = work / "kiss/kiss.log"
        text = kiss_log.read_text(errors="replace") if kiss_log.is_file() else ""
        if code or "RECORDING_CLOSED_OK" not in log.read_text(errors="replace"):
            raise SystemExit(f"KISS run failed (exit {code}); see {log}")
        if "SIGSEGV" in text:
            raise SystemExit(f"KISS crashed (SIGSEGV); is {PATCH.name} applied?")
        if not ((export / "graph.g2o").is_file() and (export / "raw_poses_tum.txt").is_file()
                and "local fix 4: wrote graph.g2o" in text):
            raise SystemExit(f"KISS did not dump its pose graph; see {kiss_log}")
        kiss = {}
    kiss.update({"loops_accepted": text.count("LC accepted") if text else None,
                 "loops_rejected": text.count("LC rejected") if text else None,
                 "g2o_edges": (export / "graph.g2o").read_text().count("EDGE_SE3"),
                 "graph_sha256": sha256(export / "graph.g2o"),
                 "raw_keyframes_sha256": sha256(export / "raw_poses_tum.txt")})

    step("gravity re-solve")
    inside = f"/work/kiss/export/{seq}"
    solve = (f"{ODOM_ROT_DEG} {ODOM_TRANS_M} {LOOP_ROT_DEG} {LOOP_TRANS_M} {args.gravity_deg} "
             f"{args.kernel} raw")
    q = shlex.quote
    code = docker_run(args, [(ws, str(ws), "ro"), (work, "/work", "rw")],
                      # GTSAM's own dependencies (libmetis-gtsam) are not on reopt's runpath.
                      f"export LD_LIBRARY_PATH={q(str(ws / 'deps/install/lib'))}:$LD_LIBRARY_PATH; "
                      f"{q(str(ws / 'bin/reopt'))} {inside}/graph.g2o {inside}/raw_poses_tum.txt "
                      f"/work/corrected_keyframes.tum {solve}", log=log, name=f"star-lc-{tag}-reopt", cpus=4)
    if code or not (work / "corrected_keyframes.tum").is_file():
        raise SystemExit(f"Gravity re-solve failed (exit {code}); see {log}")

    step("trajectory gates and dense trajectory")
    command = (f"source /opt/ros/jazzy/setup.bash; python3 /repo/tools/loop_closure/postprocess.py "
               f"{inside}/raw_poses_tum.txt /work/corrected_keyframes.tum /media/bag {q(args.odometry_topic)} "
               f"/work {q(json.dumps(args.gate_limits))}")
    code = docker_run(args, [(REPO, "/repo", "ro"), (work, "/work", "rw"), (args.bag, "/media/bag", "ro")],
                      command, log=log, name=f"star-lc-{tag}-post")
    result_path = work / "postprocess.json"
    # postprocess.py exits 3 when a gate fails; anything else non-zero is a crash.
    if code not in (0, 3) or not result_path.is_file():
        raise SystemExit(f"Post-processing failed (exit {code}); see {log}")
    result = json.loads(result_path.read_text())
    gates = result["gates"]
    if bool(gates["passed"]) != (code == 0) or gates["passed"] != (work / "trajectory.tum").is_file():
        raise SystemExit(f"Gate report disagrees with its exit code {code}; see {log}")

    report = {
        "passed": bool(gates["passed"]), "gates": gates, "kiss": kiss, "trajectory": result.get("trajectory"),
        "gravity_deg": args.gravity_deg,
        "noise": {"odom_rot_deg": ODOM_ROT_DEG, "odom_trans_m": ODOM_TRANS_M, "loop_rot_deg": LOOP_ROT_DEG,
                  "loop_trans_m": LOOP_TRANS_M, "kernel": args.kernel, "init": "raw"},
        "kiss_commit": git(ws, "rev-parse", "HEAD"),
        "kiss_source_diff_sha256": hashlib.sha256((git(ws, "diff") or "").encode()).hexdigest(),
        "patch_sha256": sha256(PATCH), "kiss_config_sha256": sha256(args.kiss_config),
        "reopt_sha256": sha256(ws / "bin/reopt"), "reopt_source_sha256": sha256(REOPT_SOURCE),
        "image": args.image, "bag": str(args.bag), "odometry_topic": args.odometry_topic,
        "seconds": round(time.time() - started, 1),
    }
    for name in ("keyframes.npz", "corrected_keyframes.tum"):
        shutil.copyfile(work / name, out / name)
    shutil.copyfile(export / "graph.g2o", out / "graph.g2o")
    shutil.copyfile(export / "raw_poses_tum.txt", out / "raw_keyframes.tum")
    if report["passed"]:
        shutil.copyfile(work / "trajectory.tum", out / "trajectory.tum")
        report["trajectory_sha256"] = sha256(out / "trajectory.tum")
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    if not report["passed"]:
        failed = [name for name, ok in gates["gates"].items() if not ok]
        print(f"Loop-closure trajectory failed gates {failed}; no trajectory written. See {out / 'report.json'}",
              file=sys.stderr)
        return 3
    step(f"done: {out / 'trajectory.tum'} ({report['trajectory']['poses']} poses)")
    return 0


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
