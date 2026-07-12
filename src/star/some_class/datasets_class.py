"""SemanticKITTI dataset loader with multi-frame point-cloud support."""

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


class SemanticKittiDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        **kwargs,
    ):
        # Base directory for the sequence.
        self.input_folder = os.path.join(basedir, sequence)
        # Load the calibration file directly.
        self.calib_path = os.path.join(self.input_folder, "calib.txt")
        self.calib = self.load_calib()
        # Load all poses directly from the pose file.
        self.pose_path = os.path.join(self.input_folder, "poses.txt")
        self.poses = self.load_poses()
        # Load image files from the image_2 directory.
        #
        self.color_paths = natsorted(glob.glob(f"{self.input_folder}/image_2/*.png"))
        self.orig_color_paths = self.color_paths
        ## Load point-cloud files from the velodyne directory.
        self.pc_paths = natsorted(glob.glob(f"{self.input_folder}/velodyne/*.bin"))
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
        print("\n Congratulations! Dataset is loaded and ready for any use! \n")
        super().__init__()

    def load_poses(self):
        '''
        Load poses.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
            poses = np.array([list(map(float, line.strip().split())) for line in lines])
            poses = poses.reshape(-1,3,4)
            ones_column = np.zeros((poses.shape[0], 1, 4))
            ones_column[:, :, -1] = 1.0
            poses = np.append(poses, ones_column, axis=1)
            # Transform poses into the lidar coordinate frame.
            T_baselink_to_lidar = np.array([
                            [ 0, -1,  0,  0],  # Flip the X-axis
                            [ 1,  0,  0,  0],  # Flip the Y-axis
                            [ 0,  0,  1,  0],  # Z-axis remains the same
                            [ 0,  0,  0,  1]   # Homogeneous coordinate
                        ])
            T_lidar_to_baselink = np.linalg.inv(T_baselink_to_lidar)
            
            poses = poses @ self.calib['T_cam2_velo']

            # # Plot the poses in 3D

            # # Plot trajectory (x, y, z)
            # # ax.plot(cam_pose_new[:300, 0], cam_pose_new[:300, 1], cam_pose_new[:300, 2], label='Trajectory', marker='.', color='r')

            # # Set labels

            # # Optional: Set axis limits for better visualization

            # # Show plot
        return poses
    
   

    def load_calib(self):
        '''
        Load the calibration file.
        '''
        calib = {}
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
    
    def load_velo_scan(self, velo_filename):
        '''
        Load lidar data.
        '''
        scan = np.fromfile(velo_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan
    
    def __len__(self):
        '''
        Return the number of items in the dataset.
        '''
        return self.num_imgs
    
    def __getitem__(self, index):
        '''
        Retrieve an image, point cloud, and pose by index.
        '''
        color_path = self.color_paths[index]
        pc_path = self.pc_paths[index]
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
            his_pose
        )
