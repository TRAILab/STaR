# Memory Construction (RGB-LiDAR Version)

This guide explains how to run `scripts/run_data_collection_lidar.py` to build robot multimodal memory in a task-agnostic way from synchronized RGB images, LiDAR point clouds, and robot poses. The memory includes 3D primitives, video captions, and key frames.

## 1. Enter The Docker Container

From the repository root on the host machine, enter the Docker folder:

```bash
cd docker/
```

Start the STAR Docker container:

```bash
./run_star.sh
```

After the container starts, go to the project directory inside the container:

```bash
cd star/
```

All following commands should be run inside this container unless otherwise noted.

## 2. Check The Active Config

The top-level Hydra config is:

```bash
configs/config.yaml
```

For CODa-style LiDAR collection, use:

```yaml
defaults:
  - inference: docker
  - scenegraph: collection_docker_coda
  - dataset: CODa_docker

enable_online_captioning: true
```

Important LiDAR parameters are in:

```bash
configs/scenegraph/collection_docker_coda.yaml
```

Before running, check these fields:

```yaml
rgb_cam_topic: /camera/image_raw
lidar_topic: /lidar/pointcloud

camera_intrinsic_matrix: ...
camera_projection_matrix: ...
lidar_to_camera_matrix: ...
base_link_to_lidar_matrix: ...

front_axis: "x"
bbox_mode: obb
```

Use `bbox_mode: obb` for tighter rotated boxes, or `bbox_mode: aabb` for axis-aligned boxes. This setting affects object merging and 3D visualization.

Also check `front_axis`. The pipeline filters out point cloud points on the backside before projection and memory construction, so this value must match your LiDAR coordinate convention. For CODa/KITTI-style data, the forward axis is usually `x`. 

## 3. Disable Blocking Visualization During Memory Construction

For normal memory construction, keep Open3D visualization effectively disabled by setting the visualization interval to a very large value:

```yaml
vis_interval: 1000000
```

The visualization window is blocking. If it opens during memory construction, the scene graph process may stop consuming observations while the ROS subscriber continues receiving data. In that case, the observation buffer can keep growing and RAM usage can increase.

For short debugging runs only, you can temporarily use a smaller value:

```yaml
vis_interval: 10
```

This shows the 3D visualization every 10 processed scene graph frames. Do not use a small `vis_interval` for long memory construction runs.

## 4. Start The ROS Environment

Open a terminal inside the container or workspace and source ROS:

```bash
source /opt/ros/humble/setup.bash
```

## 5. Run Memory Construction

From the repository root, run:

```bash
python scripts/run_data_collection_lidar.py
```

At startup, the script prints a summary with the active config files, subscribed topics, camera intrinsics, extrinsic calibration, output folders, bbox mode, model paths, and logging options.

Check this printout carefully before playing data.

## 6. Publish CODa Data As ROS Topics

Keep `run_data_collection_lidar.py` running in the first terminal. In a second terminal, enter the same Docker container again:

```bash
cd docker/
./run_star.sh
cd star/
```

Then publish the CODa dataset as ROS topics:

```bash
python scripts/write_coda_rosbag.py
```

This publishes:

```text
/camera/image_raw
/lidar/pointcloud
/tf
/tf_static
```

Make sure these topics match the scene graph YAML:

```yaml
rgb_cam_topic: /camera/image_raw
lidar_topic: /lidar/pointcloud
```

If the topic names do not match, either update the YAML config or remap the topics.

If your computer cannot process frames fast enough, reduce the CODa playback rate in:

```bash
configs/dataset/CODa_docker.yaml
```

For example:

```yaml
playback_rate: 0.5
```

A lower playback rate helps keep data publishing and memory construction balanced, so the observation buffer does not grow too fast and RAM usage stays under control.

If you are using your own rosbag or live robot sensors instead of CODa, start them in the second terminal and make sure they publish the configured RGB, LiDAR, and TF topics.

## 7. Monitor Progress

During memory construction, the script prints buffer status:

```text
[memory][buffer] emitted observation window | scenegraph_buffer=3 video_caption_buffer=2
```

The two important buffers are:

```text
scenegraph_buffer
video_caption_buffer
```

When both buffers are `0`, all queued observations have been processed.

If enable_online_captioning: true, the script generates video captions online while constructing the 3D primitives.

## 8. Finish Safely

After the rosbag finishes, wait until the script prints:

```text
[memory][complete] scenegraph_buffer=0 video_caption_buffer=0
It is safe to press Ctrl-C to terminate.
```

Then press:

```text
Ctrl-C
```

Wait for the final save message. The most important output is:

```text
Saved point cloud map to /workspace/results/<sequence>/pcd/full_pcd.pkl.gz
```

Do not close the terminal before this file is saved.

## 9. Check The Outputs

Common outputs are:

```bash
/workspace/results/<sequence>/pcd/full_pcd.pkl.gz
/workspace/results/<sequence>/annotated_rgb/
/workspace/results/<sequence>/caption/
```
After this step, you will have the three main components of the memory:

- `full_pcd.pkl.gz`: 3D primitives and their geometry/captions.
- `annotated_rgb/`: key frames with object indices overlaid.
- `caption/`: video captions for observation windows.
The annotated RGB images show object indices. These indices can be used later in the 3D visualization tool.

## 10. Visualize The Memory

After memory construction finishes, run:

```bash
python scripts/vis_3D.py --sequence_id <sequence>
```

Example:

```bash
python scripts/vis_3D.py --sequence_id 0
```

Inside the visualizer, use object indices from the annotated RGB images:

```text
1,5,8
```

You can also search by text:

```text
search car
search tree
search building
```

Press `q` to quit.
