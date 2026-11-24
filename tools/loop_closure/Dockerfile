# SLAM image for tools/loop_closure: ROS 2 Jazzy plus the build dependencies of the patched
# KISS-Matcher SAM (8b58b11) and GTSAM 4.2. Sources and builds live in a mounted workspace,
# not in the image; see README.md. Built by setup.sh.
FROM ros:jazzy-ros-base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ros-jazzy-pcl-ros ros-jazzy-pcl-conversions ros-jazzy-tf2-eigen ros-jazzy-tf2-sensor-msgs \
    ros-jazzy-tf2-geometry-msgs ros-jazzy-rosbag2-storage-mcap ros-jazzy-rmw-fastrtps-cpp \
    libboost-all-dev libeigen3-dev libtbb-dev libomp-dev python3-scipy python3-numpy \
    python3-colcon-common-extensions build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
