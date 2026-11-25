#!/bin/bash
# One-time setup of the loop-closure workspace WS (put it on a data drive, ~6 GB):
#   1. build the SLAM image from ./Dockerfile
#   2. clone KISS-Matcher 8b58b11 and apply kiss/kiss_matcher_8b58b11_local_fixes.patch, clone GTSAM 4.2
#   3. build GTSAM and KISS-Matcher inside the image (kiss/build_kiss.sh; needs network the first time)
#   4. compile reopt.cpp into WS/bin/reopt, with the source hash that loop_closure.py checks
# Steps already done are skipped, except step 4, which always recompiles.
# usage: tools/loop_closure/setup.sh WS [IMAGE]       (IMAGE defaults to star-loop-closure:jazzy)
#        BUILD_JOBS=N limits parallel compile jobs (default 10).
set -euo pipefail
WS=${1:?usage: setup.sh WS [IMAGE]}
IMAGE=${2:-star-loop-closure:jazzy}
T=$(cd "$(dirname "$0")" && pwd -P)
mkdir -p "$WS"
WS=$(cd "$WS" && pwd -P)
log() { echo "[setup $(date +%T)] $*"; }

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "image $IMAGE exists"
else
  log "building image $IMAGE"
  docker build -t "$IMAGE" -f "$T/Dockerfile" "$T"
fi

KISS=$WS/src/KISS-Matcher
if [ ! -d "$KISS" ]; then
  log "cloning KISS-Matcher 8b58b11"
  rm -rf "$KISS.partial" && mkdir -p "$(dirname "$KISS")"
  git clone -q https://github.com/MIT-SPARK/KISS-Matcher "$KISS.partial"
  git -C "$KISS.partial" checkout -q 8b58b11
  mv "$KISS.partial" "$KISS"
fi
if grep -q "local fix 4" "$KISS/ros/src/slam/pose_graph_manager.cpp"; then
  log "KISS-Matcher patch already applied"
else
  [ "$(git -C "$KISS" rev-parse --short=7 HEAD)" = 8b58b11 ] || { echo "KISS-Matcher is not at 8b58b11"; exit 1; }
  git -C "$KISS" apply "$T/kiss/kiss_matcher_8b58b11_local_fixes.patch"
  log "applied the KISS-Matcher patch"
fi
if [ ! -d "$WS/deps/gtsam" ]; then
  log "cloning GTSAM 4.2"
  rm -rf "$WS/deps/gtsam.partial" && mkdir -p "$WS/deps"
  git clone -q --depth 1 -b 4.2 https://github.com/borglab/gtsam "$WS/deps/gtsam.partial"
  mv "$WS/deps/gtsam.partial" "$WS/deps/gtsam"
fi

if [ -x "$WS/install/kiss_matcher_ros/lib/kiss_matcher_ros/kiss_matcher_sam" ]; then
  log "KISS-Matcher already built (rebuild: rm -rf $WS/build/kiss_matcher_ros $WS/install/kiss_matcher_ros)"
else
  log "building GTSAM and KISS-Matcher (tens of minutes)"
  docker run --rm --log-driver none --user "$(id -u):$(id -g)" -e BUILD_JOBS="${BUILD_JOBS:-10}" \
    -v "$WS:/ws" -v "$T:/tools:ro" "$IMAGE" bash /tools/kiss/build_kiss.sh all
fi

log "compiling reopt"
mkdir -p "$WS/bin"
docker run --rm --log-driver none --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$WS:$WS" -v "$T:/tools:ro" "$IMAGE" \
  g++ -std=c++17 -O2 -I"$WS/deps/install/include" -I/usr/include/eigen3 /tools/reopt.cpp -L"$WS/deps/install/lib" \
    -lgtsam -lboost_serialization -lboost_system -lboost_filesystem -lboost_thread -lboost_timer -lboost_chrono \
    -Wl,-rpath,"$WS/deps/install/lib" -o "$WS/bin/reopt"
(cd "$T" && sha256sum reopt.cpp) > "$WS/bin/reopt.source_sha256"
log "SETUP_OK workspace=$WS image=$IMAGE"
