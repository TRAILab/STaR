"""CODa dataset loader with video-caption scheduling and multi-frame point-cloud support."""

import abc
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from natsort import natsorted

import cv2
import numpy as np

import cv2
import numpy as np
from PIL import Image as PILImage

class VideoCaptionScheduler:
    def __init__(self, frame_interval=10, caption_every_n_sec=3, fps=10, color_paths=None):
        self.frame_interval = frame_interval
        self.caption_every_n_sec = caption_every_n_sec
        self.fps = fps
        self.segment_frame_count = caption_every_n_sec * fps
        self.caption_frames_needed = 6
        self.color_paths = color_paths

        self.frame_idx_buffer = []
        self.last_caption_segment_idx = -1

    def add(self, idx: int, new_object_detected: bool):
        # Actual frame range for the current segment; for example, idx=3 represents [21, 22, ..., 30].
        start_frame = idx * self.fps - self.fps + 1
        end_frame = idx * self.fps
        new_frames = list(range(start_frame, end_frame + 1))

        # Add these frame indices to the buffer.
        self.frame_idx_buffer.extend(new_frames)

        # Limit the buffer length.
        if len(self.frame_idx_buffer) > self.segment_frame_count:
            self.frame_idx_buffer = self.frame_idx_buffer[-self.segment_frame_count:]

        # Determine whether captioning should be triggered.
        if (
            new_object_detected and
            idx % self.caption_every_n_sec == 0 and
            idx != self.last_caption_segment_id
        ):
            self.last_caption_segment_idx = idx

            if len(self.frame_idx_buffer) >= self.segment_frame_count:
                sampled_idxs = self._sample_evenly(self.frame_idx_buffer, self.caption_frames_needed)
                image_paths = [self.color_paths[i] for i in sampled_idxs]
                print(f"Triggered captioning at segment {idx}, frames {sampled_idxs}, paths {image_paths}")

                # Read the images and convert them to PIL.Image objects in RGB uint8 format.
                images = [
                    PILImage.fromarray(
                        cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype('uint8').copy(), 'RGB'
                    )
                    for p in image_paths
                ]

                return True, images, image_paths

        return False, None, None

    def _sample_evenly(self, buffer, num_samples):
        # Get the actual frame index range from the buffer.
        min_frame = buffer[0]
        max_frame = buffer[-1]

        # Sample frame indices evenly from max_frame - duration + delta through max_frame.
        sampled_frames = np.linspace(
            max_frame - self.segment_frame_count + self.fps // 2,
            max_frame,
            num_samples
        ).astype(int)

        # Ensure the result contains only frames in the buffer, since some systems may skip frames.
        sampled_frames = [f for f in sampled_frames if f in buffer]
        return sampled_frames



class CODaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        **kwargs,
    ):
        self.timestamp_path = os.path.join(basedir, f"timestamps/{sequence}.txt")
        self.timestamps = self.load_timestamps()

        self.calib_path = os.path.join(basedir, f"calibrations/{sequence}/calib.txt")
        self.calib_source = None
        self.calib = self.load_calib()
        # Prefer the converted KITTI-style pose file, and keep the original CODa
        # timestamp/quaternion pose file as a fallback.
        self.pose_path = os.path.join(basedir, f"poses/dense_global/pose_{sequence}_.txt")
        if not os.path.exists(self.pose_path):
            missing_pose_path = self.pose_path
            self.pose_path = os.path.join(basedir, f"poses/dense_global/{sequence}.txt")
            print(f"Warning: pose file {missing_pose_path} not found. Falling back to {self.pose_path}.")
        self.pose_format = None
        self.poses = self.load_poses()
        # Load image files from the rectified cam0 directory.
        #
        self.color_paths = natsorted(glob.glob(f"{basedir}/2d_rect/cam0/{sequence}/2d_rect_cam0_{sequence}_*.png"))
        self.orig_color_paths = self.color_paths
        # Load point-cloud files from the compensated os1 directory.

        self.pc_paths = natsorted(glob.glob(f"{basedir}/3d_comp/os1/{sequence}/3d_comp_os1_{sequence}_*.bin"))
        # Start and end indices; use the complete dataset when no end index is provided.
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )
        # Verify that each image has a corresponding point cloud.
        if len(self.color_paths) != len(self.pc_paths):
            raise ValueError("Number of color and depth images must be the same.")
        self.num_imgs = len(self.color_paths)
        if self.end == -1:
            self.end = self.num_imgs
        # Preserve all poses and point-cloud paths for multi-frame overlapping projections.
        self.all_pc_paths = self.pc_paths
        self.all_poses = self.poses
        # Read data at intervals specified by stride.
        self.stride = stride

        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.pc_paths = self.pc_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        # Number of files after applying the selected range and stride.
        self.num_imgs = len(self.color_paths)
        print("\n Congratulations! The CODa dataset is loaded and ready for any use! \n")
        super().__init__()

    def load_timestamps(self):
        timestamps = []
        with open(self.timestamp_path,'r') as f:
            lines = f.readlines()
            timestamps = [float(line.strip()) for line in lines]
        return timestamps

    def load_poses(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        def pose_from_original_row(values):
            _, x, y, z, qw, qx, qy, qz = values
            quat_norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if quat_norm == 0:
                raise ValueError(f"Invalid zero-norm quaternion in pose file: {self.pose_path}")
            qx, qy, qz, qw = qx / quat_norm, qy / quat_norm, qz / quat_norm, qw / quat_norm

            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=np.float64,
            )
            pose[:3, 3] = [x, y, z]
            return pose

        poses = []
        with open(self.pose_path, "r") as f:
            pose_rows = [
                list(map(float, line.strip().split()))
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if not pose_rows:
                raise ValueError(f"No poses found in pose file: {self.pose_path}")

            row_len = len(pose_rows[0])
            if any(len(row) != row_len for row in pose_rows):
                raise ValueError(f"Inconsistent pose row lengths in pose file: {self.pose_path}")

            if row_len == 12:
                self.pose_format = "KITTI 3x4 matrix (12 values per row)"
                poses = np.array(pose_rows, dtype=np.float64).reshape(-1,3,4)
                ones_column = np.zeros((poses.shape[0], 1, 4))
                ones_column[:, :, -1] = 1.0
                poses = np.append(poses, ones_column, axis=1)
            elif row_len == 16:
                self.pose_format = "4x4 matrix (16 values per row)"
                poses = np.array(pose_rows, dtype=np.float64).reshape(-1, 4, 4)
            elif row_len == 8:
                self.pose_format = "Original CODa time x y z qw qx qy qz -> raw_relative"
                poses = np.array([pose_from_original_row(row) for row in pose_rows])
                first_pose_inv = np.linalg.inv(poses[0])
                poses = first_pose_inv @ poses
            else:
                raise ValueError(
                    "Unsupported CODa pose format in "
                    f"{self.pose_path}. Expected 12 KITTI values, 16 matrix values, "
                    "or original CODa rows as time x y z qw qx qy qz."
                )
        return poses



    def load_calib(self):
        calib = {}

        if os.path.exists(self.calib_path):
            self.calib_source = str(self.calib_path)
            with open(self.calib_path, "r") as calib_file:
                calib_lines = calib_file.readlines()
                # Load the camera intrinsic matrix.
                P_rect_line = calib_lines[2]
                P_rect_02 = np.array(list(map(float, P_rect_line.strip().split()[1:]))).reshape(3, 4)
                calib["P_rect_20"] = P_rect_02
                # Load the camera extrinsic matrix.
                Tr_line = calib_lines[4]
                Tr = np.array(list(map(float, Tr_line.strip().split()[1:]))).reshape(3, 4)
                Tr = np.vstack([Tr, [0, 0, 0, 1]])
                calib['T_cam2_velo'] = Tr

            return calib

        fallback_calib_dir = Path(self.calib_path).parent
        intrinsics_path = fallback_calib_dir / "calib_cam0_intrinsics.yaml"
        extrinsics_path = fallback_calib_dir / "calib_os1_to_cam0.yaml"

        if not intrinsics_path.exists() or not extrinsics_path.exists():
            raise FileNotFoundError(
                "Could not load CODa calibration. Missing calib.txt and/or fallback YAML files: "
                f"{self.calib_path}, {intrinsics_path}, {extrinsics_path}"
            )

        def matrix_from_yaml_block(block):
            return np.array(block["data"], dtype=np.float64).reshape(block["rows"], block["cols"])

        with open(intrinsics_path, "r") as intrinsics_file:
            intrinsics = yaml.safe_load(intrinsics_file)
        with open(extrinsics_path, "r") as extrinsics_file:
            extrinsics = yaml.safe_load(extrinsics_file)

        self.calib_source = f"{intrinsics_path}, {extrinsics_path}"
        calib["P_rect_20"] = matrix_from_yaml_block(intrinsics["projection_matrix"])
        calib["T_cam2_velo"] = matrix_from_yaml_block(extrinsics["extrinsic_matrix"])

        return calib

    def load_velo_scan(self, velo_filename):
        scan = np.fromfile(velo_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        pc_path = self.pc_paths[index]
        timestamp = self.timestamps[index]
        # Load the image in OpenCV format.
        color = cv2.imread(color_path)
        # Load the point cloud as a NumPy array.
        pointCloud = self.load_velo_scan(pc_path)  # Read the raw lidar data.
        pose = self.poses[index]
        # Historical frames used for multi-frame overlapping projections.
        his_pointCloud = []
        his_pose = []
        if index > 0:
            for i in range(self.stride-1):
                his_index = self.start+index*self.stride-i-1
                his_pointCloud.append(self.load_velo_scan(self.all_pc_paths[his_index]))
                his_pose.append(self.all_poses[his_index])
        return (
            color,
            pointCloud,
            pose,
            his_pointCloud,
            his_pose,
            timestamp
            # self.orig_color_paths
        )
