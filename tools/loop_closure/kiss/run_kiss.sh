#!/bin/bash
# Run patched KISS-Matcher SAM on a recorded bag and record its outputs. Runs INSIDE the SLAM image
# with the KISS workspace at /ws, these tools at /tools (read-only) and the bag at /media/bag (read-only).
# usage: run_kiss.sh OUT_DIR [RATE] [MAX_SEC]
set -o pipefail
OUT=$1; RATE=${2:-1.0}; MAX=${3:-1e12}
CONFIG=${KISS_CONFIG:-/tools/kiss/slam_config_trailbot.yaml}
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/ws/home TMPDIR=/ws/tmp PYTHONDONTWRITEBYTECODE=1
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-88} ROS_LOG_DIR=$OUT/roslog RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
unset FASTRTPS_DEFAULT_PROFILES_FILE
source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash
export LD_LIBRARY_PATH=/ws/deps/install/lib:$LD_LIBRARY_PATH
mkdir -p $OUT /ws/home /ws/tmp
ros2 bag record -s mcap -o $OUT/kiss_out /km_sam/path/original /km_sam/path/corrected /km_sam/pose_stamped \
  /km_sam/loop_detection /tf > $OUT/record.log 2>&1 &
REC=$!
/ws/install/kiss_matcher_ros/lib/kiss_matcher_ros/kiss_matcher_sam --ros-args -r __ns:=/km_sam -r __node:=kiss_matcher_sam \
  -r /cloud:=/cloud_registered -r /odom:=/odometry -p map_frame:=map -p base_frame:=base_link --params-file $CONFIG \
  > $OUT/kiss.log 2>&1 &
KISS=$!
sleep 5
timeout --signal=TERM $(( ${MAX%.*} < 100000 ? ${MAX%.*} * 2 + 600 : 86400 )) python3 /tools/kiss/feeder.py /media/bag $RATE $OUT/export $MAX 2>&1 | tee $OUT/feeder.log
sleep 10
# Background jobs of a non-interactive shell ignore SIGINT, so stop with SIGTERM and wait for a
# clean exit; the recorder must close the MCAP (final chunk + metadata.yaml) after the save.
stop_wait() {  # pid name seconds
  pkill -TERM -P "$1" 2>/dev/null; kill -TERM "$1" 2>/dev/null
  for _ in $(seq 1 "$3"); do kill -0 "$1" 2>/dev/null || { echo "$2 stopped"; return 0; }; sleep 1; done
  echo "$2 did not stop in $3 s; killing"; pkill -KILL -P "$1" 2>/dev/null; kill -KILL "$1" 2>/dev/null
}
stop_wait $KISS kiss 60
stop_wait $REC recorder 60
wait
[ -f $OUT/kiss_out/metadata.yaml ] && echo "RECORDING_CLOSED_OK" || echo "RECORDING_NOT_CLOSED"
echo RUN_DONE
