import time
import os
import sys
import traceback
from typing import List, Dict
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(sys.path[0]).resolve().parent))

import hydra
import rclpy
import numpy as np
import rosbag2_py
import tf_transformations
from rclpy.serialization import serialize_message
from sensor_msgs_py import point_cloud2
from omegaconf import DictConfig
from rclpy.qos import (
    QoSProfile, 
    DurabilityPolicy,
    ReliabilityPolicy
)
import warnings
warnings.filterwarnings("ignore")
from tf2_ros import TransformBroadcaster
from tf2_msgs.msg import TFMessage
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import (
    PointCloud2,
    PointField,
    Image
)

from star.some_class.datasets_class_CODa import CODaDataset
from star.scenegraph.helpers import iter_by_dataset


BASE_FRAME = "base_link"
LIDAR_FRAME = "os1"
CAMERA_FRAME = "camera_link"


def _short_path(path: str) -> str:
    return str(path) if path else "<none>"


def print_dataset_summary(dataset: CODaDataset, dataset_cfg: DictConfig, mode: str) -> None:
    first_image = dataset.color_paths[0] if dataset.color_paths else None
    first_lidar = dataset.pc_paths[0] if dataset.pc_paths else None
    image_dir = Path(first_image).parent if first_image else "<none>"
    lidar_dir = Path(first_lidar).parent if first_lidar else "<none>"
    timestamp_start = dataset.timestamps[0] if dataset.timestamps else None
    timestamp_end = dataset.timestamps[-1] if dataset.timestamps else None

    print("\n============ CODa Dataset Startup ============")
    print(f"Mode:                 {mode}")
    print(f"Dataset root:         {_short_path(dataset_cfg.basedir)}")
    print(f"Sequence:             {dataset_cfg.sequence}")
    print(f"Frames selected:      {len(dataset)}")
    print(f"Frame range/stride:   start={dataset.start}, end={dataset.end}, stride={dataset.stride}")
    if timestamp_start is not None and timestamp_end is not None:
        print(f"Timestamps:           {timestamp_start:.6f} -> {timestamp_end:.6f}")
    print("==== Pose ====")
    print(f"Pose file:            {dataset.pose_path}")
    print(f"Pose format:          {getattr(dataset, 'pose_format', '<unknown>')}")
    print("==== Calibration ====")
    print(f"Calibration source:   {getattr(dataset, 'calib_source', '<unknown>')}")
    print("==== Sensors ====")
    print(f"Image directory:      {image_dir}")
    print(f"Image frames:         {len(dataset.color_paths)}")
    print(f"LiDAR directory:      {lidar_dir}")
    print(f"LiDAR frames:         {len(dataset.pc_paths)}")
    print("==== ROS Output ====")
    print(f"Image topic/frame:    /camera/image_raw ({CAMERA_FRAME})")
    print(f"LiDAR topic/frame:    /lidar/pointcloud ({LIDAR_FRAME})")
    print(f"TF frames:            map -> {BASE_FRAME} -> {LIDAR_FRAME} -> {CAMERA_FRAME}")
    print("==============================================\n")


def _matrix_from_yaml_block(block: Dict) -> np.ndarray:
    rows = block["rows"]
    cols = block["cols"]
    data = block["data"]
    return np.array(data, dtype=np.float64).reshape(rows, cols)


def _transform_stamped_from_matrix(
    matrix: np.ndarray,
    parent_frame: str,
    child_frame: str,
) -> TransformStamped:
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = float(matrix[0, 3])
    tf_msg.transform.translation.y = float(matrix[1, 3])
    tf_msg.transform.translation.z = float(matrix[2, 3])

    q = tf_transformations.quaternion_from_matrix(matrix)
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def load_coda_static_transforms(dataset_root: str, sequence: str) -> List[TransformStamped]:
    calib_dir = Path(dataset_root) / "calibrations" / str(sequence)
    os1_to_base_path = calib_dir / "calib_os1_to_base.yaml"
    os1_to_cam0_path = calib_dir / "calib_os1_to_cam0.yaml"

    if not os1_to_base_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {os1_to_base_path}")
    if not os1_to_cam0_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {os1_to_cam0_path}")

    with open(os1_to_base_path, "r") as f:
        os1_to_base = _matrix_from_yaml_block(yaml.safe_load(f)["extrinsic_matrix"])

    with open(os1_to_cam0_path, "r") as f:
        os1_to_cam0 = _matrix_from_yaml_block(yaml.safe_load(f)["extrinsic_matrix"])

    # The CODa files are named by source->target. For TF we want:
    # base_link -> os1 and os1 -> camera_link.
    base_to_os1 = np.linalg.inv(os1_to_base)

    return [
        _transform_stamped_from_matrix(base_to_os1, BASE_FRAME, LIDAR_FRAME),
        _transform_stamped_from_matrix(os1_to_cam0, LIDAR_FRAME, CAMERA_FRAME),
    ]

class CodaRosBagWriter:
    def __init__(self, dataset_cfg: DictConfig) -> None:
        self.__cfg: DictConfig = dataset_cfg
        self.__writer = rosbag2_py.SequentialWriter()
        self.__bridge = CvBridge()
        self.__topic_infos: Dict[str, rosbag2_py.TopicMetadata] = {}
        self.__dataset = CODaDataset(
            dataset_cfg.basedir, 
            dataset_cfg.sequence, 
            stride=1, # Publish one by one
            start=dataset_cfg.start, 
            end=dataset_cfg.end
        )
        self.__data_iter = iter_by_dataset(self.__dataset)
        self.__static_transforms = load_coda_static_transforms(
            str(dataset_cfg.basedir),
            str(dataset_cfg.sequence),
        )
        print_dataset_summary(self.__dataset, dataset_cfg, "rosbag writer")

    def reset(self) -> None:
        path: str = self.__cfg.basedir + "rosbag/" + self.__cfg.sequence
        self.__is_created: bool = True if os.path.exists(path) else False
        # shutil.rmtree(path) # Remove existing bag if it exists

        if self.__is_created:
            print(f"Rosbag for sequence {self.__cfg.sequence} already exists at {path}. Skipping creation.")
            return

        storage_options = rosbag2_py.StorageOptions(
            uri=path,
            storage_id='sqlite3'
        )

        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        self.__writer.open(storage_options, converter_options)

        self.__topic_infos["/camera/image_raw"] = rosbag2_py.TopicMetadata(
            name='/camera/image_raw',
            type='sensor_msgs/msg/Image',
            serialization_format='cdr'
        )

        self.__topic_infos["/lidar/pointcloud"] = rosbag2_py.TopicMetadata(
            name='/lidar/pointcloud',
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr'
        )

        self.__topic_infos["/tf_static"] = rosbag2_py.TopicMetadata(
            name='/tf_static',
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr',
            offered_qos_profiles=(
                "history: keep_last\n"
                "depth: 1\n"
                "reliability: reliable\n"
                "durability: transient_local\n"
            )
        )

        self.__topic_infos["/tf"] = rosbag2_py.TopicMetadata(
            name='/tf',
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr'
        )

        for topic_info in self.__topic_infos.values():
            self.__writer.create_topic(topic_info)

    def write_messages(self) -> None:
        if self.__is_created:
            return

        try:
            started: bool = False
            frame_idx: int = 0

            while True:                
                data = next(self.__data_iter)

                color, pc, pose, timestamp = data[0],data[1],data[2],data[5]
                if not started:
                    tf_static_msg = self.__convert_to_tf_static()
                    self.__writer.write("/tf_static", serialize_message(tf_static_msg), int(timestamp * 1e9)) # Static transform can have timestamp 0 or any fixed value
                    started: bool = True

                img_msg = self.__convert_to_imgmsg(color, timestamp)
                pcd_msg = self.__convert_to_pointcloud2(pc, timestamp)
                pose_msg = self.__convert_to_tf_pose(pose, timestamp)

                self.__writer.write("/camera/image_raw", serialize_message(img_msg), int(timestamp * 1e9))
                self.__writer.write("/lidar/pointcloud", serialize_message(pcd_msg), int(timestamp * 1e9))
                self.__writer.write("/tf", serialize_message(TFMessage(transforms=[pose_msg])), int(timestamp * 1e9))
                frame_idx += 1
                progress = 100.0 * frame_idx / max(len(self.__dataset), 1)
                progress_msg = (
                    f"\r\033[KWriting {frame_idx}/{len(self.__dataset)} "
                    f"| {progress:5.1f}% | t={timestamp:.3f}"
                )
                print(progress_msg, end="", flush=True)
                time.sleep(0.1) # Sleep briefly to avoid overwhelming the writer

        except StopIteration:
            print(f"\nFinished writing all data to rosbag at {self.__cfg.basedir}rosbag/{self.__cfg.sequence}.")
        finally:
            self.__writer.close()

    def __convert_to_pointcloud2(self, points: np.ndarray, stamp: float) -> PointCloud2:
        header: Header = Header()

        header.frame_id = LIDAR_FRAME
        header.stamp.sec = int(stamp)
        header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        # header.stamp = self.get_clock().now().to_msg()
        
        fields: List[PointField] = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        pc2: PointCloud2 = point_cloud2.create_cloud(header, fields, points)

        return pc2

    def __convert_to_tf_pose(self, pose: np.ndarray, stamp: float) -> TransformStamped:
        t: TransformStamped = TransformStamped()
        t.header.frame_id = "map"
        t.child_frame_id = BASE_FRAME
        t.header.stamp.sec = int(stamp)
        t.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        # t.header.stamp = self.get_clock().now().to_msg()

        t.transform.translation.x = float(pose[0, 3])
        t.transform.translation.y = float(pose[1, 3])
        t.transform.translation.z = float(pose[2, 3])

        q = tf_transformations.quaternion_from_matrix(pose)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        return t

    def __convert_to_imgmsg(self, img: np.ndarray, stamp: float) -> Image:
        img_msg: Image = self.__bridge.cv2_to_imgmsg(img, encoding="bgr8")
        header: Header = Header()

        header.frame_id = CAMERA_FRAME
        header.stamp.sec = int(stamp)
        header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        # header.stamp = self.get_clock().now().to_msg()

        img_msg.header = header

        return img_msg
    
    def __convert_to_tf_static(self) -> TFMessage:
        msg: TFMessage = TFMessage()
        msg.transforms.extend(self.__static_transforms)
        return msg


class CodaPublisher(Node):
    def __init__(self, dataset_cfg: DictConfig) -> None:
        super().__init__("coda_dataset_node")
        self.__dataset = CODaDataset(
            dataset_cfg.basedir, 
            dataset_cfg.sequence, 
            stride=1, # Publish one by one
            start=dataset_cfg.start, 
            end=dataset_cfg.end
        )
        self.__bridge = CvBridge()
        self.__static_transforms = load_coda_static_transforms(
            str(dataset_cfg.basedir),
            str(dataset_cfg.sequence),
        )
        print_dataset_summary(self.__dataset, dataset_cfg, "ROS publisher")

        tf_qos = QoSProfile(depth=1)
        tf_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        tf_qos.reliability = ReliabilityPolicy.RELIABLE
        self.__tf_pub =self.create_publisher(TFMessage, "/tf_static", tf_qos)

        self.__image_pub = self.create_publisher(Image, "camera/image_raw", 10)
        self.__pcd_pub = self.create_publisher(PointCloud2, "lidar/pointcloud", 10)
        self.__pose_pub = TransformBroadcaster(self) # self.create_publisher(PoseStamped, "pose", 10)
        self.__playback_rate = float(getattr(dataset_cfg, "playback_rate", 1.0))
        if self.__playback_rate <= 0:
            raise ValueError(f"playback_rate must be positive, got {self.__playback_rate}.")
        self.__skip_on_error = bool(getattr(dataset_cfg, "skip_on_error", False))
        self.__base_publish_period = 0.1
        self.__timer_period = self.__base_publish_period / self.__playback_rate
        self.__frame_idx = 0
        self.__is_finished = False
        print(
            "Playback: "
            f"rate={self.__playback_rate:.2f}x, timer_period={self.__timer_period:.3f}s, "
            f"skip_on_error={self.__skip_on_error}"
        )
        self.__timer = self.create_timer(self.__timer_period, self.publish_data)
        self.__data_iter = iter_by_dataset(self.__dataset)
        self.publish_tf_static() 

    def publish_tf_static(self) -> None:
        msg: TFMessage = TFMessage()
        msg.transforms.extend(self.__static_transforms)
        self.__tf_pub.publish(msg)

    def publish_data(self) -> None: 
        if self.__is_finished:
            return

        dataset_index = self.__dataset.start + self.__frame_idx * self.__dataset.stride
        color_path = self.__dataset.color_paths[self.__frame_idx] if self.__frame_idx < len(self.__dataset.color_paths) else "<out-of-range>"
        pc_path = self.__dataset.pc_paths[self.__frame_idx] if self.__frame_idx < len(self.__dataset.pc_paths) else "<out-of-range>"

        try:
            data = next(self.__data_iter)

            color, pc, pose, timestamp = data[0],data[1],data[2],data[5]

            if color is None:
                raise ValueError(f"cv2.imread returned None for image: {color_path}")
            if pc is None:
                raise ValueError(f"Point cloud load returned None for file: {pc_path}")

            img_msg = self.__convert_to_imgmsg(color, timestamp)
            pcd_msg = self.__convert_to_pointcloud2(pc, timestamp)
            pose_msg = self.__convert_to_tf_pose(pose, timestamp)

            self.__pose_pub.sendTransform(pose_msg)
            self.__image_pub.publish(img_msg)
            self.__pcd_pub.publish(pcd_msg)

            self.__frame_idx += 1
            progress = 100.0 * self.__frame_idx / max(len(self.__dataset), 1)
            progress_msg = (
                f"\r\033[KPublishing {self.__frame_idx}/{len(self.__dataset)} "
                f"| {progress:5.1f}% | idx={dataset_index} | t={timestamp:.3f}"
            )
            #print(progress_msg, end="", flush=True)
        except StopIteration:
            print(f"\nFinished publishing all data after {self.__frame_idx} frames. Shutting down node.")
            self.__shutdown_publisher()
        except Exception as exc:
            self.get_logger().error(
                f"Failed on frame local_idx={self.__frame_idx} dataset_idx={dataset_index} "
                f"image={color_path} pointcloud={pc_path}: {exc}"
            )
            self.get_logger().error(traceback.format_exc())
            self.__frame_idx += 1
            if not self.__skip_on_error:
                self.__shutdown_publisher()
                raise

    def __shutdown_publisher(self) -> None:
        if self.__is_finished:
            return
        self.__is_finished = True
        if self.__timer is not None:
            self.__timer.cancel()

    def __convert_to_pointcloud2(self, points: np.ndarray, stamp: float) -> PointCloud2:
        header: Header = Header()

        header.frame_id = LIDAR_FRAME
        header.stamp.sec = int(stamp)
        header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        # header.stamp = self.get_clock().now().to_msg()

        front_mask: np.ndarray = points[:, 0] > 0
        points = points[front_mask]
        
        fields: List[PointField] = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        pc2: PointCloud2 = point_cloud2.create_cloud(header, fields, points)

        return pc2

    def __convert_to_tf_pose(self, pose: np.ndarray, stamp: float) -> TransformStamped:
        t: TransformStamped = TransformStamped()
        t.header.frame_id = "map"
        t.child_frame_id = BASE_FRAME
        t.header.stamp.sec = int(stamp)
        t.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        # t.header.stamp = self.get_clock().now().to_msg()

        t.transform.translation.x = float(pose[0, 3])
        t.transform.translation.y = float(pose[1, 3])
        t.transform.translation.z = float(pose[2, 3])

        q = tf_transformations.quaternion_from_matrix(pose)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        return t

    def __convert_to_imgmsg(self, img: np.ndarray, stamp: float) -> Image:
        img_msg: Image = self.__bridge.cv2_to_imgmsg(img, encoding="bgr8")
        header: Header = Header()

        header.frame_id = CAMERA_FRAME
        header.stamp.sec = int(stamp)
        header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        # header.stamp = self.get_clock().now().to_msg()

        img_msg.header = header

        return img_msg
    

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def publish_by_node(cfg: DictConfig) -> None:
    rclpy.init()
    node: CodaPublisher = CodaPublisher(cfg.dataset)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        # node.publish_tf_static() # Publish static transform once at the beginning
        executor.spin()
        # rclpy.spin(node)
    except KeyboardInterrupt:
        print("Data publishing interrupted by user. Shutting down node.")
    except StopIteration:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    writer: CodaRosBagWriter = CodaRosBagWriter(cfg.dataset)
    writer.reset()
    writer.write_messages()  


if __name__ == "__main__":
    # main()
    publish_by_node()
    
