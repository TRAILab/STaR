# Loop-closure poses for memory construction (optional)

STaR builds memory from each frame's robot pose. If your dataset has ground-truth poses, write them
as a TUM file and set `pose_file` (see [Memory Construction](../../docs/BuildMemory.md#optional-poses-from-a-file));
you do not need anything here.

If your bag only has drifting odometry, these tools turn it into a **gravity-anchored loop-closure
trajectory**. Revisits of the same place then line up, so memory merges an object seen on two visits
instead of storing it twice. The tools run offline on a recorded bag, inside Docker, and write the
same kind of TUM file:

```
recorded odometry + LiDAR
  -> KISS-Matcher SAM (loop closure, patched to dump its pose graph)
  -> batch re-solve of that graph with a per-keyframe gravity prior (GTSAM)
  -> trajectory gates: stop here, with a report, if the result is implausible
  -> trajectory.tum (base_link in the map frame at every odometry stamp)
```

## Requirements

- Docker, about 6 GB of disk for the workspace and network access for the one-time setup.
- A rosbag2 directory in MCAP format with:
  - `nav_msgs/Odometry` of `base_link` in an odometry frame (for example from Fast-LIO);
  - the LiDAR as `sensor_msgs/PointCloud2` in its sensor frame;
  - `/tf_static` with the `base_link` to LiDAR transform.

## Setup (once)

```bash
tools/loop_closure/setup.sh /data/loop_closure_ws
```

It builds the SLAM image `star-loop-closure:jazzy` from [`Dockerfile`](Dockerfile) (ROS 2 Jazzy). It
then clones [KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher) at `8b58b11`, applies
[`kiss/kiss_matcher_8b58b11_local_fixes.patch`](kiss/kiss_matcher_8b58b11_local_fixes.patch) and clones
[GTSAM](https://github.com/borglab/gtsam) 4.2. Finally it builds both inside the image and compiles
[`reopt.cpp`](reopt.cpp). Neither library is vendored here. The build takes tens of minutes;
`BUILD_JOBS` limits its parallel jobs. Rerunning skips the steps already done, so if a download fails,
run it again.

## Run

```bash
python3 tools/loop_closure/loop_closure.py --bag /data/my_bag --workspace /data/loop_closure_ws \
    --out /data/my_bag_loop_closure \
    --odometry-topic /odometry --lidar-topic /ouster/points --lidar-frame os_lidar
```

KISS replays the bag at recorded speed (`--rate` changes it), so a run takes about as long as the bag.
Then set the result in `configs/scenegraph/collection_docker_coda.yaml` and build memory as usual:

```yaml
pose_file: /data/my_bag_loop_closure/trajectory.tum
```

`python3 tools/loop_closure/loop_closure.py --help` lists every option. `--kiss-dump DIR` re-solves an
existing KISS export (`graph.g2o`, `raw_poses_tum.txt`) without running KISS again.

## Outputs

| File | Contents |
|---|---|
| `trajectory.tum` | Corrected `base_link` poses in the map frame at every odometry stamp between the first and last keyframe. Written only when every gate passes. |
| `report.json` | Gate results, loop closures accepted/rejected, solver settings, and hashes of the inputs, patch, config and `reopt` binary. |
| `keyframes.npz`, `raw_keyframes.tum`, `corrected_keyframes.tum`, `graph.g2o` | KISS keyframes before and after the re-solve, and the pose graph. |
| `loop_closure.log`, `work/` | Container commands and logs, including KISS's own. |

`trajectory.tum` applies the keyframe correction `C = T_corrected · T_raw⁻¹`, interpolated between
keyframes, to every recorded odometry pose. It keeps odometry's smooth local motion and takes the
global shape from the loop closures.

## Trajectory gates

A loop-closure solution can be wrong: a false loop bends the map. A run therefore fails, and writes no
trajectory, unless the corrected keyframes stay consistent with the odometry they correct:

| Gate | Limit (`--gates` overrides) | What it catches |
|---|---|---|
| Step error vs odometry, max | 5 cm (`max_step_m`) | Jumps between consecutive keyframes that odometry did not see |
| Up axis vs odometry, median / max | 0.10° / 0.30° (`up_median_deg`, `up_max_deg`) | Tilt; IMU-aided odometry knows where up is |
| Height mismatch at revisits, median / p90 | 2 cm / 10 cm (`revisit_dz_median_m`, `revisit_dz_p90_m`) | Places visited twice (keyframes within 2 m, more than 60 s apart) left at different heights |

The gates compare against the raw odometry, not the size of the correction, so a large but correct
fix of vertical drift still passes.

## Why the gravity re-solve

Stock KISS-Matcher SAM has no gravity term. Roll and pitch errors in its loop registrations bend
keyframe attitudes through its loose odometry rotation noise, and the corrected map tilts by
0.4-0.6° (median) on our test bags. Adding a gravity factor live inside KISS's incremental solver
broke its odometry. So KISS dumps its factor graph, and [`reopt.cpp`](reopt.cpp) re-solves it in batch
(Levenberg-Marquardt). The re-solve uses KISS's own noise models, a Cauchy kernel on loop factors and a
per-keyframe attitude prior (σ = `--gravity-deg`, 0.1° by default) that holds body z, in the world
frame, to the odometry's.

## Results and limits

Tested on Trailbot (Ouster LiDAR, Fast-LIO odometry) in a parking lot, a 20-minute bag with repeated
visits:

| | Raw odometry | Loop closure |
|---|---|---|
| Height mismatch at revisits, median / p90 | 87 / 90 cm | 0.60 / 5.76 cm |
| Step error vs odometry, max | - | 3.43 cm |
| Up axis vs odometry, median / max | - | 0.010° / 0.105° |

A second KISS run of the same bag accepted 49 loop closures instead of 51 and also passed every gate
(revisit height mismatch 0.77 / 3.8 cm, step error 3.12 cm, up axis 0.010° / 0.061°).

In STaR memory built on these poses, objects merged across visits rose from 104 to 162, and
cross-visit duplicates (pairs of objects from different visits within 1 m with caption similarity
≥ 0.8) fell from 115 to 87; at similarity ≥ 0.9 they fell from 69 to 40. In a warehouse bag, loop
closure did not reduce duplicates.

Limits:

- [`kiss/slam_config_trailbot.yaml`](kiss/slam_config_trailbot.yaml) is tuned for Trailbot's Ouster
  and 1.5 m keyframes; other sensors may need other KISS settings (`--kiss-config`).
- The attitude prior has only been tested on near-flat ground. On slopes it also pulls world-frame
  yaw toward the odometry's; the gates measure tilt correctly either way.
- KISS-Matcher's loop acceptance can differ between runs of the same bag; `report.json` records one run.
- Only offline use is supported. Nothing here runs on a robot.

## Files

| File | Role |
|---|---|
| `setup.sh`, `Dockerfile` | One-time image and workspace build |
| `loop_closure.py` | Host entry point (Python standard library only): runs the steps below in throwaway containers |
| `kiss/run_kiss.sh`, `kiss/feeder.py`, `kiss/slam_config_trailbot.yaml` | Run KISS-Matcher on a bag; the feeder replays recorded odometry and places LiDAR scans in the odometry frame (no re-run of the odometry) |
| `kiss/build_kiss.sh`, `kiss/kiss_matcher_8b58b11_local_fixes.patch` | Build KISS-Matcher and GTSAM. The patch fixes a crash on a first rejected loop (fix 1), exports the final estimate on save (fix 3) and dumps `graph.g2o` and the raw keyframes (fix 4). Fix 5 publishes graph snapshots for an online re-solve and does nothing without a subscriber. |
| `reopt.cpp` | Batch re-solve with the gravity prior |
| `postprocess.py`, `trajectory_gates.py` | Keyframes, gates and the dense trajectory |

Tests: `python -m unittest discover -s tests` (numpy and scipy; no ROS or Docker).
