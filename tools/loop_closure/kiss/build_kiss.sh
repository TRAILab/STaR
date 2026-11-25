#!/bin/bash
# Build GTSAM 4.2 (once) and the patched KISS-Matcher ROS workspace. Runs INSIDE a ROS 2 Jazzy image with
# build tools (../Dockerfile) with the KISS workspace mounted at /ws; see ../README.md for the layout.
# usage: build_kiss.sh [all|kiss]   (kiss = rebuild only kiss_matcher_ros after a source change)
# KISS_OFFLINE=1 reuses already-fetched FetchContent sources (build/kiss_matcher_ros/_deps) so a rebuild works
# under --network none; a first build needs network to fetch them.
set -eo pipefail
WHAT=${1:-all}
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/ws/home TMPDIR=/ws/tmp CCACHE_DISABLE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p /ws/home /ws/tmp /ws/logs
source /opt/ros/jazzy/setup.bash
J=${BUILD_JOBS:-10}
if [ ! -f /ws/deps/install/lib/cmake/GTSAM/GTSAMConfig.cmake ]; then
  echo "[$(date +%T)] GTSAM configure"
  cmake -S /ws/deps/gtsam -B /ws/deps/gtsam_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/ws/deps/install \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_PYTHON=OFF
  cmake --build /ws/deps/gtsam_build -j$J && cmake --install /ws/deps/gtsam_build
  echo "[$(date +%T)] GTSAM done"
fi
export CMAKE_PREFIX_PATH=/ws/deps/install:$CMAKE_PREFIX_PATH LD_LIBRARY_PATH=/ws/deps/install/lib:$LD_LIBRARY_PATH
SELECT=(); EXTRA=()
[ "$WHAT" = kiss ] && SELECT=(--packages-select kiss_matcher_ros)
[ -n "${KISS_OFFLINE:-}" ] && EXTRA=(-DFETCHCONTENT_FULLY_DISCONNECTED=ON)
cd /ws && echo "[$(date +%T)] colcon build $WHAT"
colcon --log-base /ws/logs/colcon build --base-paths src --build-base /ws/build --install-base /ws/install \
  "${SELECT[@]}" --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release -DGTSAM_DIR=/ws/deps/install/lib/cmake/GTSAM "${EXTRA[@]}" \
  --event-handlers console_direct+ 2>&1 | tail -5
echo "[$(date +%T)] BUILD_OK"
