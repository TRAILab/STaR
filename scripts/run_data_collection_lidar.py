import argparse
import os
import time
import multiprocessing
import threading
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

import traceback
import pickle

from queue import Queue
import tf_transformations
import numpy as np
import sys
from PIL import Image as PILImage
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).resolve().parent))
from star.utils.util import ListJitterBuffer, JitterBuffer

multiprocessing.set_start_method("spawn", force=True) # Avoid swap overflow
import hydra
import numpy
import numpy.core
import numpy.core.numeric
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros

import rclpy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from omegaconf import DictConfig, open_dict
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

from star.captioners.captioner import CaptionManager
from star.captioners.nvila_captioner import NVILACaptioner
from star.scenegraph.scenegraph_constructor_lidar import run_scenegraph_generation
from star.utils.data_collection_helpers import (
    buffers_are_empty,
    format_buffer_status,
    preprocess_position_matrix,
    process_cfg,
    shutdown_process,
    to_seconds,
)
from star.utils.lidar_collection_helpers import (
    load_base_link_to_lidar_matrix,
    print_startup_summary,
)

numpy._core = numpy.core
numpy._core.numeric = numpy.core.numeric

# Force into sys.modules
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric

MAX_RETRIES: int = 2
MAP_FRAME = "map"
BASE_FRAME = "base_link"     
    
class ObservationHub(Node):
    def __init__(
        self,
        cfg: DictConfig,
        observation_buffer: Queue,
        sync_queue_size: int = 1000,
        max_time_diff: float = 0.02
    ) -> None:
        super().__init__('captioner_node')
        self.cfg: DictConfig = cfg
        self.jitter_buffer = JitterBuffer(desired_timewindow=1.0)

        self.observation_buffer: List[Queue] = observation_buffer
        self.observation_ready: bool = False
   
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_lookup_mode = str(getattr(cfg, "tf_lookup_mode", "latest")).lower()
        self.tf_lookup_timeout_sec = float(getattr(cfg, "tf_lookup_timeout_sec", 0.05))
        self.tf_warning_interval_sec = float(getattr(cfg, "tf_warning_interval_sec", 2.0))
        self.verbose = bool(getattr(cfg, "verbose", False))
        self.last_tf_warning_time = 0.0
        self.print_buffer_status_every_frame = bool(getattr(cfg, "print_buffer_status_every_frame", True))
        print("\n[ros] Node ready")
        print(
            f"[ros] TF lookup: {MAP_FRAME} <- {BASE_FRAME}, "
            f"mode={self.tf_lookup_mode}, timeout={self.tf_lookup_timeout_sec:.3f}s"
        )

        # create subscribers
        print("[ros] Subscribing")
        print(f"  RGB camera              {self.cfg.rgb_cam_topic}")
        print(f"  LiDAR                   {self.cfg.lidar_topic}")
        print(f"  sync                    queue={sync_queue_size}, max_diff={max_time_diff:.3f}s")
        self.image_sub = message_filters.Subscriber(self, Image, self.cfg.rgb_cam_topic)
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, self.cfg.lidar_topic)

        ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            sync_queue_size, max_time_diff
        )

        ts.registerCallback(self.observation_callback)

        self.callback_times = 0
        self.cv_image = None

        # Start the data preprocessing thread
        self.livedata_preprocessing = threading.Thread(target=self.run_livedata_processing)
        self.livedata_preprocessing.start()


    def observation_callback(self, camera_msg: Image, lidar_msg: PointCloud2) -> None:
        """
        Callback function for processing camera and lidar messages and pose message.

        Args:
            camera_msg (sensor_msgs.msg.Image): Camera message.
            lidar_msg (sensor_msgs.msg.PointCloud2): Lidar message.

        """
        # print("Received synchronized messages!")
        timestamp_cam = camera_msg.header.stamp

        try:
            if self.tf_lookup_mode == "timestamp":
                lookup_time = rclpy.time.Time.from_msg(timestamp_cam)
            else:
                lookup_time = rclpy.time.Time()
            tf_msg: TransformStamped = self.tf_buffer.lookup_transform(
                MAP_FRAME,
                BASE_FRAME,
                lookup_time,
                timeout=Duration(seconds=self.tf_lookup_timeout_sec),
            )
        except Exception as e:
            now = time.monotonic()
            if now - self.last_tf_warning_time >= self.tf_warning_interval_sec:
                self.last_tf_warning_time = now
                frame_idx = self.callback_times
                message = (
                    f"[ros][tf-wait] frame={frame_idx} waiting for TF {MAP_FRAME} <- {BASE_FRAME}; "
                    f"skipping one frame near sensor_time={to_seconds(timestamp_cam):.6f}s "
                    f"(mode={self.tf_lookup_mode})."
                )
                if self.verbose:
                    message += f" detail={type(e).__name__}: {e}"
                print(message, flush=True)
            return

        cv_image = self.bridge.imgmsg_to_cv2(
            camera_msg, desired_encoding='rgb8') # rgb8

        # Deserialize PointCloud2 data into xyz points
        point_gen = pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
        # Filter out points with all 0s
        # self.points = np.array([[x, y, z, 1] for x, y, z in point_gen if any([i!=0 for i in [x,y,z]])])

        points = np.array([point_gen['x'], point_gen['y'], point_gen['z']]).T
        mask = ~np.all(points == 0, axis=1)
        # Apply the mask to filter out rows with all zero coordinates
        filtered_points = points[mask]

        # get the transformation matrix of the lidar
        translation = tf_msg.transform.translation
        rotation = tf_msg.transform.rotation
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = tf_transformations.quaternion_matrix(quaternion)
        translation = [translation.x, translation.y, translation.z]
        translation_matrix = tf_transformations.translation_matrix(translation)
        transformation_matrix = np.dot(translation_matrix, rotation_matrix)

        # with self.observation_lock:
        self.cv_image = cv_image
        self.timestamp_cam = timestamp_cam
        self.points = filtered_points
        self.transformation_matrix = transformation_matrix
        self.callback_times += 1
        
        self.observation_ready = True

    def run_livedata_processing(self):
        self.get_logger().info("Initialized live data processing...")
        first_transformation_matrix = None
        scenegraph_frame_idx = 0
        T_baselink_to_lidar = load_base_link_to_lidar_matrix(self.cfg)
        T_lidar_to_baselink = np.linalg.inv(T_baselink_to_lidar)
        print("\n[ros] Waiting for synchronized RGB + LiDAR observations...")
        
        while rclpy.ok():
            if not self.observation_ready:
                # print("observation_ready is not set, waiting for observations...")
                time.sleep(0.01)
                continue

            # Snapshot one fresh observation and mark it consumed.
            observation = [
                self.callback_times, # idx 0
                self.cv_image, # color 1
                self.points, # point_cloud 2
                self.transformation_matrix, # pose 3
                to_seconds(self.timestamp_cam), # timestamp 4
            ]

            if first_transformation_matrix is None:
                first_transformation_matrix = observation[3]
                first_transformation_matrix_inv = np.linalg.inv(first_transformation_matrix)
            base_link_2map_TF = np.dot(first_transformation_matrix_inv, observation[3])

            lidar_2map_TF = T_lidar_to_baselink @ base_link_2map_TF 
            observation[3] = lidar_2map_TF
            timestamp = observation[4] # - offset_time + self.timestamp

            # print("Got new observation with timestamp:", timestamp)
            data_in_timewindow = self.jitter_buffer.add(observation, timestamp)    
            if data_in_timewindow is None:
                continue

            #print(f"Data in window: {data_in_timewindow[0][4]} - {data_in_timewindow[-1][4]}, size: {len(data_in_timewindow)}")
            #print(f"Processing data with length {len(data_in_timewindow)}...")

            memory_buffer_sg = []
            memory_buffer_caption = []
            
            for obs in data_in_timewindow:
                # print(f"Processing observation with timestamp {obs[4]:.2f}...")
                memory_buffer_sg.append((
                    scenegraph_frame_idx,
                    obs[1],
                    obs[2],
                    obs[3],
                    obs[4],
                ))

                memory_buffer_caption.append((
                    scenegraph_frame_idx,
                    obs[1],
                    obs[4],
                    obs[3],
                ))

            #print(f"Times window {len(data_in_timewindow)}, timestamp range: {data_in_timewindow[0][4]:.2f} to {data_in_timewindow[-1][4]:.2f}")
    
            self.observation_buffer['scenegraph'].put(memory_buffer_sg)
            self.observation_buffer['captioner'].put(memory_buffer_caption)
            scenegraph_frame_idx += 1
            if self.print_buffer_status_every_frame:
                print(
                    "[memory][buffer] "
                    f"frame={observation[0]} "
                    f"timestamp={timestamp:.3f} "
                    f"window_frames={len(data_in_timewindow)} "
                    f"{format_buffer_status(self.observation_buffer)}",
                    flush=True,
                )

            # else:
            #     #print(f"observation_ready is {self.observation_ready.is_set()}, waiting for observations...")
            #     time.sleep(0.01)


def start_ros_node(observation_buffer, args, cfg):
    print("Starting ROS2 node...")
    rclpy.init(args=args)
    subscriber = ObservationHub(cfg, observation_buffer)
    subscriber.set_parameters([
        rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, False)
    ])

    try:
        while rclpy.ok():
            rclpy.spin_once(subscriber, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("Ctrl-C received, shutting down ros2 node...")
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

def save_observations(observation_buffer, cfg, loop_event):
    data_in_timewindow = {
        'position': [],
        'rotation': [],
        'timestamps': [],
        'images': []
    }
    while not loop_event.is_set():
        if not observation_buffer.empty():
            print("Accessing observation buffer...")
            data = observation_buffer.get()
            for observation in data:
                _, image, timestamp, pose = observation
                position, rotation = preprocess_position_matrix(pose)

                data_in_timewindow['position'].append(position)
                data_in_timewindow['rotation'].append(rotation)
                data_in_timewindow['timestamps'].append(timestamp)
                data_in_timewindow['images'].append(image)

            if not os.path.exists(cfg.save_video_path):
                os.makedirs(cfg.save_video_path)
            with open(f"{cfg.save_video_path}/{data_in_timewindow['timestamps'][0]}.pkl", 'wb') as f:
                pickle.dump(data_in_timewindow, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("Data saved successfully.")

            data_in_timewindow = {
                'position': [],
                'rotation': [],
                'timestamps': [],
                'images': []
            }
        else:
            time.sleep(0.1)
            continue

def run_online_captioning(observation_buffer, captioner, args, loop_event, observation_buffers=None):
    caption_manager = CaptionManager(args, captioner)
    jitter: ListJitterBuffer = ListJitterBuffer(desired_timewindow=args.caption_freq)
    if observation_buffers is None:
        observation_buffers = {
            "captioner": observation_buffer,
            "scenegraph": observation_buffer,
        }
    finish_idle_sec = float(getattr(args, "finish_idle_sec", 30.0))
    finish_status_interval_sec = float(getattr(args, "finish_status_interval_sec", 5.0))
    last_data_time = time.monotonic()
    last_status_time = 0.0
    saw_caption_data = False
    completion_notice_printed = False

    try:
        while not loop_event.is_set():
            if not observation_buffer.empty():
                #logging.info("Accessing observation buffer...")
                data = observation_buffer.get()
                last_data_time = time.monotonic()
                saw_caption_data = True
                completion_notice_printed = False
                data_in_timewindow = []

                for observation in data:
                    step_data: Dict[str, Any] = {}
                    frame_idx, image, timestamp, pose = observation

                    position, rotation = preprocess_position_matrix(pose)
                    image = PILImage.fromarray(image) # Convert to PIL Image

                    step_data['position'] = position
                    step_data['rotation'] = rotation
                    step_data['timestamps'] = timestamp
                    step_data['images'] = image
                    step_data['frame_indices'] = frame_idx
                    data_in_timewindow.append(step_data)

                jitter_frame: List[Dict[str, Any]] = jitter.add(data_in_timewindow) # Primitive jitter frame
                if jitter_frame:
                    frame: Dict[str, List[Any]] = {}
                    for data in jitter_frame:
                        for key in ['position', 'rotation', 'timestamps', 'images', 'frame_indices']:
                            if key not in frame:
                                frame[key] = []
                            frame[key].append(data[key])
                    frame['frame_indices'] = list(dict.fromkeys(frame['frame_indices']))
                    frame['file_start'] = str(frame['timestamps'][0])
                    frame['file_end'] = str(frame['timestamps'][-1])

                    caption_manager.caption_video(
                        frame,
                        query=args.query,
                        max_retries=MAX_RETRIES,
                        max_new_tokens=args.max_new_tokens,
                    )
            else:
                now = time.monotonic()
                idle_sec = now - last_data_time
                if idle_sec >= finish_status_interval_sec and (now - last_status_time) >= finish_status_interval_sec:
                    last_status_time = now
                    print(
                        "[memory][status] "
                        f"no new caption data for {idle_sec:.1f}s | "
                        f"{format_buffer_status(observation_buffers)}",
                        flush=True,
                    )
                if (
                    not completion_notice_printed
                    and saw_caption_data
                    and idle_sec >= finish_idle_sec
                    and buffers_are_empty(observation_buffers)
                ):
                    caption_manager.maybe_save_caption_data(force=True)
                    completion_notice_printed = True
                    print(
                        "[memory][complete] "
                        f"No new observations for {idle_sec:.1f}s and both buffers are empty. "
                        "Caption data has been saved. "
                        f"{format_buffer_status(observation_buffers)}. "
                        "It is safe to press Ctrl-C to terminate.",
                        flush=True,
                    )
                time.sleep(0.1)
                caption_manager.maybe_save_caption_data()
                continue

            caption_manager.maybe_save_caption_data()
    finally:
        caption_manager.maybe_save_caption_data(force=True)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg : DictConfig):
    sg_cfg = cfg['scenegraph']
    # The scene-graph worker receives only this config subtree. Preserve the
    # active dataset location and sequence for calibration_source: dataset.
    with open_dict(sg_cfg):
        sg_cfg.dataset_calibration_root = cfg.dataset.basedir
        sg_cfg.dataset_calibration_sequence = cfg.dataset.sequence

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_id", type=str, default='05')
    parser.add_argument("--out_path", type=str, default=f"./data/captions/{4}/captions")
    parser.add_argument("--data_path", type=str, default="./data/semantickitti_data") # coda_data
    parser.add_argument("--seconds_per_caption", type=int, default=3)

    parser.add_argument("--num-video-frames", type=int, default=15)
    parser.add_argument("--captioner_name", type=str, default="NVILA-Lite-2B")

    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/NVILA-Lite-2B")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--query", type=str, default=getattr(sg_cfg, "query", None))
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, default=None)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--json-schema", type=str, default=None)
    parser.add_argument("--save-every", dest="save_every", type=int, default=10)
    parser.add_argument("--save-interval-sec", dest="save_interval_sec", type=float, default=30.0)
    args = parser.parse_args()

    sg_cfg = process_cfg(sg_cfg)
    # Keep the loaded NVILA checkpoint consistent with the active Hydra config
    # and with the model name used for the caption output file.
    args.model_path = sg_cfg.model_path
    print_startup_summary(sg_cfg)
    buffer_names = ['captioner', 'scenegraph']
    if sg_cfg.observation_buffer_size > 0:
        observation_buffer = {name: multiprocessing.Queue(maxsize=sg_cfg.observation_buffer_size) for name in buffer_names}
    else:
        observation_buffer = {name: multiprocessing.Queue() for name in buffer_names}
    prepare_event = multiprocessing.Event()
    loop_event = multiprocessing.Event()

    sg_process = multiprocessing.Process(
        target=run_scenegraph_generation,
        args=(
            sg_cfg,
            observation_buffer['scenegraph'],
            prepare_event,
            loop_event
        )
    )
    sg_process.start()
    while not prepare_event.is_set():
        time.sleep(0.5)
    prepare_event.clear()

    ros_process = multiprocessing.Process(target=start_ros_node, args=(observation_buffer, None, sg_cfg))
    ros_process.start()
    print("Processes started")

    try:
        if cfg.enable_online_captioning:
            captioner: NVILACaptioner = NVILACaptioner(args)
            run_online_captioning(observation_buffer['captioner'], captioner, sg_cfg, loop_event, observation_buffer)
        else:
            save_observations(observation_buffer['captioner'], sg_cfg, loop_event)
    except Exception as e:
        traceback.print_exc()
        print(f"Error occurred: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down main thread...")
    finally:
        loop_event.set()
        shutdown_process(ros_process, "ROS process")
        shutdown_process(sg_process, "Scenegraph process")


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True) # Important!
    main()
