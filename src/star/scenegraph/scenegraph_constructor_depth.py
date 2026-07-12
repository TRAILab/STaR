import time
import sys
import os
import copy
import random

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import numpy as np

import torch
from pathlib import Path
import os
import gzip
import pickle

from datetime import datetime

from tqdm import trange

import open3d as o3d
import hydra
from omegaconf import DictConfig
import cv2
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrow
from PIL import Image, ImageDraw, ImageFont
import io
import string
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation, center_of_mass
from scipy.interpolate import griddata
sys.path.append("/home/trailbot/RAG/4DMOS/src/mos4d")
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import mode

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from os.path import join

import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter

from collections import defaultdict
import logging
import multiprocessing

import cv2
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from star.some_class.map_class import MapObjectList
from utils.world_map import (
    ObjectVisualizer,
    visualize_objects,
    visualize_objects_org
)
from utils.utils import (
    load_models,
    prepare_labeled_contours_merge,
    gobs_to_detection_list_depth,
    denoise_objects,
    visualize_depth_on_rgb,
    load_calib,
)
from utils.merge import (
    compute_spatial_similarities,
    compute_caption_similarities,
    compute_ft_similarities,
    aggregate_similarities,
    merge_obj2_into_obj1,
    merge_detections_to_objects_2
)
from .helpers import (
    init_scenegraph,
    prepare_mos_model,
    load_background_objects,
    save_scene_graph,
    iter_by_event_depth
)


BG_CAPTIONS = [
    "warehouse floor",
    "concrete floor",
    "gray floor",
    "empty floor",
    "ground surface",
    "a smooth floor",
    "aisle floor",
    "loading area floor",
    "industrial floor",
    "painted floor",
    "warehouse wall",
    "plain wall",
    "concrete wall",
    "large wall",
    "white wall",
    "metal wall",
    "side wall",
    "back wall",
    "warehouse partition",
    "ceiling",
    "warehouse ceiling",
    "ceiling panel",
    "overhead ceiling",
    "ceiling structure",
    "warehouse aisle",
    "empty aisle",
    "open warehouse space",
    "warehouse background",
]

BG_CAPTIONS_Pro = [
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "ceiling",
    "ceiling",
    "ceiling",
    "ceiling",
    "ceiling",
    "a scene",
    "a scene",
    "a scene",
    "a scene",
]

BG_CAPTIONS_Pro_Sim = ["floor", "wall", "ceiling", "a scene"]

DEFAULT_DEPTH_TO_RGB: np.ndarray = np.array([
    [1, 0, 0, -0.0115], # 0.04750],#0.04150   -0.03650
    [0, 1, 0, 0], #-0.04750
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


def load_depth_to_rgb_matrix(cfg) -> np.ndarray:
    matrix = cfg.get("depth_to_rgb_matrix", DEFAULT_DEPTH_TO_RGB)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"depth_to_rgb_matrix must be 4x4, got shape {matrix.shape}.")
    return matrix

# cos_a -sin_a
# sin_a  cos_a
# 0      0        1

# 634.086240   0.000000   640.000000
#    0.000000   566.489981   360.000000
#    0.000000   0.000000   1.000000
#for CODa
# fx = 381.03681045751637
# fy = 381.18300653594775
# cx = 327.3319215686275
# cy = 250.3970196078432

DEFAULT_RGB_CAMERA_MATRIX: np.ndarray = np.array([
    [634.086240, 0.0, 640.0],
    [0.0, 566.489981, 360.0],
    [0.0, 0.0, 1.0]
])

DEFAULT_DEPTH_CAMERA_MATRIX: np.ndarray = np.array([
    [634.086240, 0.0, 640.0],
    [0.0, 566.489981, 360.0],
    [0.0, 0.0, 1.0]
])


def load_3x3_matrix_from_cfg(cfg, key: str, default: np.ndarray) -> np.ndarray:
    matrix = cfg.get(key, default)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{key} must be 3x3, got shape {matrix.shape}.")
    return matrix

D: np.ndarray = np.array([0, 0, 0.00245, 0, 0, 0]) # fisheye distortion coefficients

import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Scene Graph class
class SceneGraph:
    def __init__(self):
        self.nodes = {}  # {node_id: {"objects": [indices], "summary": str}}
        self.edges = []  # [(node_id1, node_id2, "relationship")]

    def add_node(self, node_id, objects, summary=None):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"objects": objects, "summary": summary}

    def update_summary(self, node_id, summary):
        if node_id in self.nodes:
            self.nodes[node_id]["summary"] = summary

    def add_edge(self, node_id1, node_id2, relationship):
        self.edges.append((node_id1, node_id2, relationship))

    def visualize(self):
        print("Nodes:")
        for node_id, node_data in self.nodes.items():
            print(f"  Node {node_id}: Objects={node_data['objects']}, Summary={node_data['summary']}")
        print("Edges:")
        for edge in self.edges:
            print(f"  {edge[0]} --({edge[2]})--> {edge[1]}")

def process_cfg(cfg: DictConfig):
    '''
    Preprocess the configuration.
    '''
    cfg.basedir = Path(cfg.basedir)
    cfg.save_vis_path = Path(cfg.save_vis_path)
    cfg.save_cap_path = Path(cfg.save_cap_path)
    cfg.save_pcd_path = Path(cfg.save_pcd_path)
    return cfg

def world_to_map(points, resolution, map_origin):
    """
    Convert world coordinates of points to map indices.
    :param points: Nx3 array of 3D points
    :param resolution: Resolution of the grid in meters
    :param map_origin: Origin of the map in world coordinates (x, y)
    :return: Nx2 array of map indices
    """

    raw_points = np.asarray(points.points)  # NumPy array of shape (N, 3)
    # Project to 2D (XY plane)
    points_2d = raw_points[:, :3]  # Keep only x and y coordinates
    # Convert to grid indices
    map_coords = (points_2d - map_origin) // resolution
    return map_coords.astype(int)

def build_terrain_map(map, points, map_origin, resolution, height_threshold=0.09):
    points = np.asarray(points)
    tree = KDTree(points[:,[0, 2]])

    for pcd in points:
        _, idx = tree.query([pcd[0], pcd[2]], k=5)

        neihbor_highs = points[idx][:, 1]
        height_var = abs(np.max(neihbor_highs) - np.min(neihbor_highs))
        pcd_indices = np.floor(([pcd[0], pcd[2]] - map_origin) / resolution).astype(int)

        map[pcd_indices[0], pcd_indices[1]] = 5 if height_var > height_threshold else 0
        # if height_var > height_threshold:
        #     print(f"---------{pcd_indices[0], pcd_indices[1]]}---{height_var}----------------")

    return map


def invert_transformation_matrix(matrix):
    R = matrix[:3, :3]  # Extract the rotation part
    t = matrix[:3, 3]   # Extract the translation part

    # Compute the inverse
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)

    # Construct the inverse transformation matrix
    inv_matrix = np.eye(4)
    inv_matrix[:3, :3] = R_inv
    inv_matrix[:3, 3] = t_inv
    return inv_matrix

def backward_projection_RGB(road_pcd_list, object_pcd_list, trans_pose, calib):
    road_pcd_homogeneous = np.hstack((road_pcd_list, np.ones((len(road_pcd_list), 1)))).T
    object_pcd_homogeneous = np.hstack((object_pcd_list, np.ones((len(object_pcd_list), 1)))).T

    trans_pose_inv  = invert_transformation_matrix(trans_pose)
    road_pcd_homo_inv = trans_pose_inv @ road_pcd_homogeneous
    object_pcd_homo_inv = trans_pose_inv @ object_pcd_homogeneous

    road_cam = calib['P_rect_20'].dot(calib['T_cam2_velo']).dot(road_pcd_homo_inv)
    object_cam = calib['P_rect_20'].dot(calib['T_cam2_velo']).dot(object_pcd_homo_inv)
    road_cam[:2, :] /= road_cam[2, :]
    object_cam[:2, :] /= object_cam[2, :]
    u,v,z  = road_cam
    pixels = np.dstack((v,u)).squeeze()
    return pixels

def backward_projection_RGB1(sematic_info, trans_pose, calib):
    for i, info in enumerate(sematic_info):
        road_pcd_list = info['SoM']
        if len(road_pcd_list) >0:
            road_pcd_homogeneous = np.hstack((road_pcd_list, np.ones((len(road_pcd_list), 1)))).T

            trans_pose_inv  = invert_transformation_matrix(trans_pose)
            road_pcd_homo_inv = trans_pose_inv @ road_pcd_homogeneous

            road_cam = calib['P_rect_20'].dot(calib['T_cam2_velo']).dot(road_pcd_homo_inv)
            road_cam[:2, :] /= road_cam[2, :]
            #object_cam[:2, :] /= object_cam[2, :]
            u,v,z  = road_cam
            pixels = np.dstack((v,u)).squeeze()
        else:
            pixels = np.array([])
        sematic_info[i]["pixels"] = pixels

    return sematic_info


# def backward_projection_to_rgb(occupancy_map_points, pcd, pose, resolution = 0.1, offsets = np.array([0, 0]), object_indices_dict=None):
def backward_projection_to_rgb(sematic_info, road_interval, object_interval, sematic_map, pose, resolution, map_origin, object_indices_dict, calib, robot_state):
    for j, info in enumerate(sematic_info):

        if info["cap"] == "driveable area":
            road_index = info["img_bbox"]
            road_grid_points = get_semantic_area_grid_points(sematic_map, road_index, road_interval, robot_state)
            global_points = road_grid_points * resolution + map_origin

            indices_all = object_indices_dict[road_index][0]
            points_all = object_indices_dict[road_index][1]
            indices_tree = KDTree(indices_all)
            idx_results = []
            for i, data in enumerate(road_grid_points):
                _, idx = indices_tree.query(data, k=1)
                idx_results.append(idx)
            org_points = points_all[idx_results]
            avg_value = np.mean(org_points[:, 1])
            updated_data = np.zeros((org_points.shape[0], 3))
            updated_data[:, 0] = global_points[:, 0]
            updated_data[:, 1] = avg_value
            updated_data[:, 2] = global_points[:, 1]
            if road_grid_points.shape[0] > 0:
                sematic_info[j]["SoM"] = updated_data
                sematic_info[j]["Occ_idx"] = road_grid_points
            else:
                sematic_info[j]["SoM"] = []
                sematic_info[j]["Occ_idx"] = []
        else:
            object_index = info["img_bbox"]
            object_grid_points = get_semantic_area_grid_points(sematic_map, object_index, object_interval)
            global_points = object_grid_points * resolution + map_origin
            indices_all = object_indices_dict[object_index][0]
            points_all = object_indices_dict[object_index][1]
            indices_tree = KDTree(indices_all)
            idx_results = []
            for i, data in enumerate(object_grid_points):
                _, idx = indices_tree.query(data, k=1)
                idx_results.append(idx)
            org_points = points_all[idx_results]
            updated_data = np.zeros((org_points.shape[0], 3))
            updated_data[:, 0] = global_points[:, 0]
            updated_data[:, 1] = org_points[:, 1]
            updated_data[:, 2] = global_points[:, 1]

            if object_grid_points.shape[0] > 0:
                sematic_info[j]["SoM"] = updated_data
                sematic_info[j]["Occ_idx"] = object_grid_points
            else:
                sematic_info[j]["SoM"] = []
                sematic_info[j]["Occ_idx"] = []

        road_pcd_list = info['SoM']
        if len(road_pcd_list) >0:
            road_pcd_homogeneous = np.hstack((road_pcd_list, np.ones((len(road_pcd_list), 1)))).T

            trans_pose_inv  = invert_transformation_matrix(pose)
            road_pcd_homo_inv = trans_pose_inv @ road_pcd_homogeneous

            road_cam = calib['P_rect_20'].dot(calib['T_cam2_velo']).dot(road_pcd_homo_inv)
            road_cam[:2, :] /= road_cam[2, :]
            #object_cam[:2, :] /= object_cam[2, :]
            u,v,z  = road_cam
            pixels = np.dstack((v,u)).squeeze()
        else:
            pixels = np.array([])
        sematic_info[j]["pixels"] = pixels
    return sematic_info



def visualize_backward_projection(rgb, pixels, indice_list, color_list):
    print("Visualizing backward projection...")
    for i, indices in enumerate(indice_list):
        for index in indices:
            color = color_list[i] if i < len(color_list) else [255, 0, 0]
            rgb[pixels[index][1], pixels[index][0]] = color
            print([pixels[index][1], pixels[index][0]])
    cv2.imwrite("backward_projection.png", rgb)

def create_2d_occupancy_map_obbox(detected_objects, bg_list, resolution=0.1):
    """
    Create a 2D occupancy map based on oriented 3D bounding boxes (from PointClouds),
    projecting onto the X-Z plane.

    Args:
        detected_objects (list of dict): List of detected objects, each containing 'pcd' (PointCloud).
        bg_list (list of dict): List of background objects, each containing 'pcd' (PointCloud).
        resolution (float): The resolution of the map in meters per grid cell.

    Returns:
        occupancy_map (np.ndarray): A 2D occupancy map (X-Z plane).
        map_origin (np.ndarray): Origin of the map in world coordinates (X, Z).
    """
    all_points = []

    # Collect corner points of oriented bounding boxes
    for i, obj in enumerate(detected_objects + bg_list):
        pcd = obj['pcd']
        obbox = pcd.get_oriented_bounding_box(robust=True)
        corners = np.asarray(obbox.get_box_points())  # (8, 3) array

        # Project to Z-X plane
        all_points.append(corners[:, [0, 2]])  # Keep X and Z coordinates

    # Combine all points into one array
    all_points = np.vstack(all_points)

    # Determine global bounds in X-Z plane
    min_bound = np.min(all_points, axis=0)  # [min_x, min_z]
    max_bound = np.max(all_points, axis=0)  # [max_x, max_z]

    # Calculate occupancy map size
    map_size = np.ceil((max_bound - min_bound) / resolution).astype(int)
    occupancy_map = np.full(map_size, -1, dtype=np.int8)  # Initialize with unknown (-1)
    semantic_map = np.full(map_size, -1, dtype=np.int8)

    # Store the map origin (X, Z)
    map_origin = min_bound
    for obj in bg_list:
        pcd = obj['pcd']
        obbox = pcd.get_oriented_bounding_box(robust=True)
        corners = np.asarray(obbox.get_box_points())[:, [0, 2]]

        min_xz = np.min(corners, axis=0)
        max_xz = np.max(corners, axis=0)

        x_indices = np.arange(
            int((min_xz[0] - map_origin[0]) / resolution),
            int((max_xz[0] - map_origin[0]) / resolution) + 1
        )
        z_indices = np.arange(
            int((min_xz[1] - map_origin[1]) / resolution),
            int((max_xz[1] - map_origin[1]) / resolution) + 1
        )

        # Mark free cells
        for x in x_indices:
            for z in z_indices:
                if 0 <= x < map_size[0] and 0 <= z < map_size[1]:
                    occupancy_map[x, z] = 0
                    semantic_map[x, z] = i
    # Mark occupied and free cells
    #for obj in detected_objects:
    for i, obj in enumerate(detected_objects):
        pcd = obj['pcd']
        obbox = pcd.get_oriented_bounding_box(robust=True)
        corners = np.asarray(obbox.get_box_points())[:, [0, 2]]  # Project to X-Z

        # Compute grid bounds
        min_xz = np.min(corners, axis=0)
        max_xz = np.max(corners, axis=0)

        x_indices = np.arange(
            int((min_xz[0] - map_origin[0]) / resolution),
            int((max_xz[0] - map_origin[0]) / resolution) + 1
        )
        z_indices = np.arange(
            int((min_xz[1] - map_origin[1]) / resolution),
            int((max_xz[1] - map_origin[1]) / resolution) + 1
        )

        # Mark occupied cells
        for x in x_indices:
            for z in z_indices:
                if 0 <= x < map_size[0] and 0 <= z < map_size[1]:
                    occupancy_map[x, z] = 1
                    semantic_map[x, z] = i

    return occupancy_map, semantic_map, map_origin

def get_semantic_area_grid_points(semantic_map, semantic_index, grid_interval=10, robot_state=None):
    copied_map = copy.deepcopy(semantic_map)
    object_mask = (copied_map == semantic_index)
    x_indices = np.arange(copied_map.shape[0])
    y_indices = np.arange(copied_map.shape[1])
    # if robot_state["robot_state"]
    grid_point_mask1 = ((x_indices[:, None] % (grid_interval) == 0) & ((x_indices[:, None] // (grid_interval)) %2 == 0)) & ((y_indices % (grid_interval) == 0) & ((y_indices // (grid_interval)) %2 == 0))
    grid_point_mask2 = ((x_indices[:, None] % (grid_interval) == 0) & ((x_indices[:, None] // (grid_interval)) %2 == 1)) & ((y_indices % (grid_interval) == 0) & ((y_indices // (grid_interval)) %2 == 1))
    grid_point_mask = grid_point_mask1 | grid_point_mask2
    return np.argwhere(object_mask & grid_point_mask)#[:, ::-1]

def get_contour_sampling_points(semantic_map, ignore_index, threshold=0, sampling_interval=20):
    copied_map = copy.deepcopy(semantic_map)
    copied_map[copied_map == ignore_index] = -1
    edges = np.zeros_like(copied_map, dtype=np.uint8)

    diff_x_forward = np.diff(copied_map, axis=1)
    diff_x_backward = np.diff(copied_map[:, ::-1], axis=1)[:, ::-1]
    edges[:, 1:] |= (np.abs(diff_x_forward) > threshold)
    edges[:, :-1] |= (np.abs(diff_x_backward) > threshold)

    diff_y_forward = np.diff(copied_map, axis=0)
    diff_y_backward = np.diff(copied_map[::-1, :], axis=0)[::-1, :]
    edges[1:, :] |= (np.abs(diff_y_forward) > threshold)
    edges[:-1, :] |= (np.abs(diff_y_backward) > threshold)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    keypoints = []
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            continue # skip the external contour
        for i in range(0, len(contour), sampling_interval):
            keypoints.append(contour[i][0])
    return np.array(keypoints)

def find_semantic_index(semantic_info, caption):
    for info in semantic_info:
        if info["cap"] == caption:
            return info["img_bbox"]
    return 0

def get_semantic_indices(semantic_info):
    indices = []
    for info in semantic_info:
        indices.append(info["img_bbox"])
    return sorted(indices)

def generate_random_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.random()  # generate random float between 0 to 1
        g = random.random()
        b = random.random()
        colors.append((r, g, b))
    return colors


def is_overlapping(new_point, existing_points, min_distance=10):
    """
    Check if the new_point is too close to any of the existing points.
    If so, return True; otherwise, return False.
    """
    if len(existing_points) == 0:
        return False
    distances = cdist([new_point], existing_points)
    return np.any(distances < min_distance)

def visualize_map(map, map_origin, resolution, idx, robot_state, map_type="occ", sematic_info= None, zoom =False, drivable_indices=None, interval = 20): # sematic
    """
    Visualize the 2D occupancy map on the Z-X plane.

    Args:
        occupancy_map (np.ndarray): The 2D occupancy map.
        map_origin (np.ndarray): Origin of the map in world coordinates.
        resolution (float): Resolution of the map.
    """

    if map_type == "occ":
        # Define the colors for each value
        colors = ['gray', 'white', 'black']  # -1: gray (unknown), 0: white (free), 1: black (occupied)
        cmap = ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5]  # Boundaries for values (-1, 0, 1)
        norm = BoundaryNorm(bounds, cmap.N)

        # Create grid axes
        z_extent = map_origin[0] + np.array([0, map.shape[0]]) * resolution
        x_extent = map_origin[1] + np.array([0, map.shape[1]]) * resolution

        # Plot the occupancy map
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(
            map.T,
            origin="lower",
            cmap=cmap,
            norm=norm
        )
        cbar = plt.colorbar(ticks=[-1, 0, 1], label="Occupancy")  # Tick labels for legend
        cbar.set_label("Object Index", fontsize=10)  # Set the colorbar label
        cbar.ax.tick_params(labelsize=10)  # Set the colorbar tick label size
        plt.title("2D Occupancy Map")
        plt.xlabel("X-axis")
        plt.ylabel("Z-axis")

        plt.grid(True, linestyle="--", alpha=0.5)
    if map_type == "sematic":
        # Plot with colorbar
        # Generate a large colormap with distinct colors
        # Create grid axes
        z_extent = map_origin[0] + np.array([0, map.shape[0]]) * resolution
        x_extent = map_origin[1] + np.array([0, map.shape[1]]) * resolution


        num_classes = len(sematic_info)
        semantic_indices = get_semantic_indices(sematic_info)
        semantic_indices = [-1] + semantic_indices
        base_colormap = plt.cm.get_cmap("tab20c", num_classes-1)  # Use "tab20c" or "viridis", etc.

        custom_colors = base_colormap(np.linspace(0, 1, len(semantic_indices) - 1))

        # Add a specific color for -1 (e.g., gray for "Unknown Area")
        custom_colors = np.vstack(([1.0, 1.0, 1.0, 1], custom_colors))  # Add gray at the start
        # TODO: Delete the for loop if do not want to highlight the drivable area

        road_color  = [(0.75, 0.85, 0.85, 1), (0.85, 0.85, 0.85, 1), (0.95, 0.85, 0.85, 1), (0.85, 0.75, 0.85, 1), (0.85, 0.95, 0.85, 1), (0.85, 0.85, 0.95, 1)]
        for idx, road_index in enumerate(drivable_indices):
            # if idx large than the number of road color, staart from the beginning
            # if idx >= len(road_color):
            idx = idx % len(road_color)
        # for road_index in drivable_indices:
            custom_colors[semantic_indices.index(road_index)] = to_rgba(road_color[idx])  # Change the color of the drivable area

        cmap = mcolors.ListedColormap(custom_colors)

        boundaries = np.array(semantic_indices + [max(semantic_indices) + 1])# - 0.5

        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        map_x_min, map_x_max = 0, map.shape[0]
        map_z_min, map_z_max = 0, map.shape[1]
        x_min, x_max = max(-5, map_x_min), min(320, map_x_max)
        z_min, z_max = max(-5, map_z_min), min(320, map_z_max)

        # Plot with colorbar
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the semantic map
        im = ax.imshow(map.T,
                       origin="lower",
                       cmap=cmap,
                       norm=norm)


        # Add a colorbar with custom labels
        cbar = plt.colorbar(im, ax=ax, ticks=np.asarray(semantic_indices))
        sorted_sematic_info = sorted(sematic_info, key=lambda x: int(x['img_bbox']))

        road_grid_points = np.array([])
        for free_index in drivable_indices:
            grid_points = get_semantic_area_grid_points(map, free_index, interval)
            road_grid_points = np.vstack((road_grid_points, grid_points)) if road_grid_points.size else grid_points

        dot_flag = False
        if dot_flag:
            marks = 'dot'  # 'dot' or 'pad'
            labels_string =list(string.ascii_lowercase)
            for i, point in enumerate(road_grid_points):
                label_x = point[0]//interval
                label_y = point[1]//interval
                if marks =='dot':
                    plt.text(point[0], point[1]-3, f'{labels_string[label_x]+labels_string[label_y]}', fontsize=10, weight="bold", color='red', ha='center', va='top')
                else:
                    plt.text(point[0], point[1], f'{labels_string[label_x]+labels_string[label_y]}',
                fontsize=10, color='white', ha='center', va='center',
                bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.15'))
            if marks == 'dot':
                plt.scatter(road_grid_points[:, 0], road_grid_points[:, 1], color='red', s=6, marker='o')
        used_positions = []
        for label in sorted_sematic_info:  # Loop through each object class
            object_mask = (map == label["img_bbox"])  # Create a mask for the current label
            if np.any(object_mask):  # If the object exists in the map
                # Compute the centroid of the object
                centroid = center_of_mass(object_mask)
                x= centroid[0]
                y = centroid[1]
                if is_overlapping(centroid, used_positions, min_distance=8):
                    x += 8  # Offset x-coordinate
                    y += 8  # Offset y-coordinate
                # Add the numerical label at the centroid
                used_positions.append([x,y])
                if zoom:
                    if centroid[0] < 300 and centroid[1] < 300:
                        ax.text(
                            x,  # X-coordinate (column)
                            y,  # Y-coordinate (row)
                            f"{label['img_bbox']}",  # Label text
                            color="black", ha="center", va="center", fontsize=8, weight="bold"
                        )
                        plt.xlim(x_min, x_max)
                        plt.ylim(z_min, z_max)

                else:
                    ax.text(
                            x,  # X-coordinate (column)
                            y,  # Y-coordinate (row)
                            f"{label['img_bbox']}",  # Label text
                            color="black", ha="center", va="center", fontsize=8, weight="bold"
                        )
        sorted_sematic_info = [{"cap": "Unknown Area", "img_bbox": -1}] + sorted(sematic_info, key=lambda x: int(x['img_bbox']))
        # Set custom labels matching colors
        labels = [f"{info['img_bbox']} - {info['cap']}" for info in sorted_sematic_info]

        cbar.ax.set_yticklabels(labels)  # Set the custom labels
        cbar.set_label("Object Index", fontsize=10)  # Set the colorbar label
        cbar.ax.tick_params(labelsize=10)  # Set the colorbar tick label size


        # Add labels and title
        plt.title("2D Semantic Map")
        plt.xlabel(f"X-axis\n drivable area: {drivable_indices}")
        plt.ylabel("Z-axis")

        plt.gca().xaxis.set_major_locator(MultipleLocator(50)) # main locator
        plt.gca().yaxis.set_major_locator(MultipleLocator(100))
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator(5)) # minor locator
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator(10))

        plt.gca().xaxis.set_tick_params(which='minor', labelbottom=False)
        plt.gca().yaxis.set_tick_params(which='minor', labelleft=False)

        plt.grid(True, which='major', linestyle="--", alpha=0.5)
        plt.grid(True, which='minor', linestyle="--", alpha=0.25)

        arrow_length = 8
        circle_radius = 10
        robot_indices = robot_state["robot_indices"]
        robot_theta = robot_state["robot_state"][1]
        dx = arrow_length * np.cos(robot_theta[0])
        dz = arrow_length * np.sin(robot_theta[0])

        # Plot the robot position and heading
        plot_robot = False
        if plot_robot:
            plt.plot(robot_indices[0], robot_indices[1], 'o', color='red', markersize=circle_radius, label='Robot Position', zorder=1)
            offset_x = 0.5  # Adjust the horizontal offset
            offset_y = 0.5  # Adjust the vertical offset
            plt.text(
                robot_indices[0]+5,  # X-coordinate with offset
                robot_indices[1]+5,  # Y-coordinate with offset
                'Robot',  # The text label
                fontsize=10, weight = "bold", color='black', ha='left', va='bottom'
            )
            plt.arrow(robot_indices[0], robot_indices[1], dx, dz, head_width=4, head_length=4, fc='black', ec='black', label='_nolegend_', linewidth=2, zorder=2)
        # Create custom arrow for legend
        arrow_legend = FancyArrow(0, 0, 0.5, 0, width=0.2, length_includes_head=True, head_width=0.4, head_length=0.3, color='black')

        # Add the legend with custom arrow

    if map_type == "terrain":
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(map.T, cmap='terrain', origin='lower')
        plt.colorbar(im, label='Height Variation')
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("2D Terrain Map")
        plt.show()
    plt.tight_layout()

    return fig


def fill_holes_in_occupancy_map(occupancy_map, mode="morphological", gap_closing_size=8, dilation_size=8, drive_area=None):
    """
    Fill holes in the occupancy map using morphological operations.

    Args:
        occupancy_map (np.ndarray): The input occupancy map.

    Returns:
        np.ndarray: Processed occupancy map with holes filled.
    """
    if mode == "morphological":
        # Initialize the filled map with the same shape as the input map
        filled_map = np.full_like(occupancy_map, -1)  # Unknown regions initialized to -1

        unique_classes = np.unique(occupancy_map)
        for cls in unique_classes:
            if cls == -1:
                # Skip unknown regions
                continue

            # Create a binary mask for the current class
            class_mask = (occupancy_map == cls)

            if cls == drive_area:  # Free cells
                # Apply hole filling to the free region
                filled_class = binary_fill_holes(class_mask).astype(int)
            else:  # Occupied or other semantic classes
                # Apply binary closing to the regions to fill small holes
                filled_class = binary_closing(class_mask, structure=np.ones((gap_closing_size, gap_closing_size))).astype(int)

            # Update the filled map
            filled_map[filled_class == 1] = cls
        return filled_map

    elif mode == "interpolation":
        # Step 1: Initialize the filled map with the same shape
        filled_map = np.full_like(occupancy_map, -1)  # Unknown regions initialized to -1

        # Get all unique classes in the map, excluding unknown (-1)
        unique_classes = np.unique(occupancy_map)
        unique_classes = unique_classes[unique_classes != -1]  # Exclude unknown

        for cls in unique_classes:
            # Step 2: Create a binary mask for the current class
            class_mask = (occupancy_map == cls)

            # Step 3: Close small gaps in the current class
            closed_class = binary_closing(
                class_mask, structure=np.ones((gap_closing_size, gap_closing_size))
            ).astype(int)

            # Step 4: Fill holes in the free-space areas (if cls == 0)
            if cls == 0:  # Free cells
                closed_class = binary_fill_holes(closed_class).astype(int)

            # Step 5: Dilate the class region to smooth boundaries
            dilated_class = binary_dilation(
                closed_class, structure=np.ones((dilation_size, dilation_size))
            ).astype(int)

            # Update the filled map with the processed class
            filled_map[dilated_class == 1] = cls

        # Step 6: Interpolate unknown regions (-1) using the nearest-neighbor method
        road_boundary_mask = (filled_map != -1)  # Non-unknown areas
        known_coords = np.argwhere(road_boundary_mask)  # Coordinates of known cells
        known_values = filled_map[road_boundary_mask]   # Values of known cells

        # Create a grid for the map
        grid_z, grid_x = np.meshgrid(
            np.arange(filled_map.shape[0]),
            np.arange(filled_map.shape[1]),
            indexing='ij'
        )
        grid_coords = np.column_stack((grid_z.ravel(), grid_x.ravel()))

        # Interpolate missing values (-1)
        interpolated_map = griddata(
            known_coords, known_values, grid_coords, method='nearest', fill_value=-1
        )
        interpolated_map = interpolated_map.reshape(filled_map.shape)

        # Ensure interpolation doesn't overwrite unknown regions outside the boundary
        interpolated_map[~road_boundary_mask] = -1

        return interpolated_map

def create_SoM_keypts(img, sematic_info, drivable_indices, font_path='/home/trailbot/RAG/OpenGraph/config/arial.ttf', interval=20):
    '''
    Create a grid of dots and labels on the road surface, constrained by a mask.
    '''
    # Apply the mask to filter the point cloud and corresponding pixels
    image = img.copy()
    # Prepare the image for drawing
    width, height = image.shape[1], image.shape[0]
    def clamp(value, min_value, max_value):
            """Clamp a value to ensure it stays within min and max bounds."""
            return max(min_value, min(value, max_value))

    def adjust_label_position(label_pos_x, label_pos_y, label_width, label_height):
        """Adjust label position to keep it inside the image."""
        label_pos_x = clamp(label_pos_x, 0, width - label_width)
        label_pos_y = clamp(label_pos_y, label_height + 4, height)
        return int(label_pos_x), int(label_pos_y-5)

    opposite_color =(0, 0, 255)
    circle_radius = width // 140  # Adjust dot size if needed
    labels = list(string.ascii_lowercase)  # ['a', 'b', 'c', ..., 'z']
    label_positions = []
    for idx, info in enumerate(sematic_info):
        sematic_info[idx]["SoM_label"] = {}
        road_pix = info["pixels"]
        indicies = info["Occ_idx"]

        if road_pix.ndim == 1 and len(road_pix) == 2:
            road_pix = road_pix.reshape(1, 2)
        if len(info["pixels"]) > 0:
            for i in range(len(road_pix)):
                if road_pix.ndim == 1:
                    x = road_pix[1]
                    y = road_pix[0]
                else:
                    x = road_pix[i][1]
                    y = road_pix[i][0]
                # Draw the ellipse (dot) at the grid point

                # Draw the ellipse (dot) at the grid point

                if indicies is not None:
                    indices_x = indicies[i][0]
                    indices_y = indicies[i][1]
                    label_x = labels[indices_x //interval]
                    label_y = labels[indices_y //interval]
                    label_str = f"{str(label_x)+str(label_y)}"
                else:
                    label_str = f"{i}"

                sematic_info[idx]["SoM_label"][label_str] = np.round(info["SoM"][i], 1)
                # Adjust label position to ensure it remains inside the image
                font_size = 0.45
                label_size, _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
                label_width, label_height = label_size

                label_pos_x, label_pos_y = adjust_label_position(x, y, label_width, label_height)
                center = (int(label_pos_x), int(label_pos_y))
                radius = 10
                # Draw a filled rectangle for label background
                if sematic_info[idx]["img_bbox"] in drivable_indices:
                    cv2.rectangle(
                        image,
                        (label_pos_x, label_pos_y - label_height-2),
                        (label_pos_x + label_width, label_pos_y+3),
                        (0, 0, 0),  # black rectangle
                        -1  # Filled rectangle
                    )

                    # Put the label text
                    cv2.putText(
                        image,
                        label_str,
                        (label_pos_x, label_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_size,
                        (255, 255, 255),  # Black text
                        1
                    )

    return image

def get_random_points_from_semantic_map(semantic_map, valid_classes, num_points=2):
    """
    Randomly selects a specified number of valid points from the semantic map.

    Args:
        semantic_map (np.ndarray): 2D semantic map where each value represents a class.
        valid_classes (list): List of classes to consider as valid.
        num_points (int): Number of points to randomly select.

    Returns:
        list: List of (row, col) tuples representing the selected points.
    """
    # Find indices of valid points
    valid_indices = np.argwhere(np.isin(semantic_map, valid_classes))

    # Check if there are enough valid points
    if len(valid_indices) < num_points:
        raise ValueError("Not enough valid points in the semantic map to select from.")

    # Randomly sample points
    selected_indices = valid_indices[np.random.choice(len(valid_indices), num_points, replace=False)]

    # Convert to a list of tuples
    selected_points = [tuple(idx) for idx in selected_indices]

    return selected_points

def shift_objects_to_positive_plane(detect_object, resolution=0.25, map_origin=(0, 0, 0), map_size=(400, 800)):
    """
    Shifts all point clouds in detected_object so they lie in the positive coordinate plane.

    Args:
        detected_object (list): List of detected 3D objects. Each object has a 'pcd' attribute
                                (Open3D PointCloud) representing the point cloud.

    Returns:
        updated_detected_object: List of shifted objects with updated point clouds.
        translation_vector: Translation vector used to shift the objects.
    """
    # Step 1: Find global minimum x, y, z
    global_min_x = float('inf')
    global_min_y = float('inf')
    global_min_z = float('inf')

    for i, obj in enumerate(detect_object, start=0):
        # if objects[i]['image_idx'][-1] == idx:
        pcl = detect_object[i]["pcd"]
        points = np.asarray(pcl.points)  # Convert to numpy array

        # Update global minimum values
        global_min_x = min(global_min_x, points[:, 0].min())
        global_min_y = min(global_min_y, points[:, 1].min())
        global_min_z = min(global_min_z, points[:, 2].min())

    print(f"Global minimum values - X: {global_min_x}, Y: {global_min_y}, Z: {global_min_z}")

    # Step 2: Compute translation vector to shift the minimums to zero
    translation_vector = np.array([-global_min_x, -global_min_y, -global_min_z])

    # Step 3: Apply translation to all point clouds
    for i, obj in enumerate(detect_object, start=0):
        pcl = detect_object[i]["pcd"]
        points = np.asarray(pcl.points)

        # Shift points
        shifted_points = points + translation_vector
        # create a new point cloud
        shifted_pcl = o3d.geometry.PointCloud()
        shifted_pcl.points = o3d.utility.Vector3dVector(shifted_points)
        detect_object[i]["pcd"] = shifted_pcl
    labels =[]
    semantic_map = np.zeros(map_size, dtype=int)

    for i, obj in enumerate(detect_object, start=0):
        # if objects[i]['image_idx'][-1] == idx:
        pcl = detect_object[i]["pcd"]
        bbox = detect_object[i]['bbox']  # OrientedBoundingBox object
        img_bbx = detect_object[i]['img_bbox']
        get_center = getattr(bbox, "get_center", None)
        get_extent = getattr(bbox, "get_extent", None)
        center = get_center() if callable(get_center) else bbox.center
        extent = get_extent() if callable(get_extent) else bbox.extent
        rounded_center = np.round(center, 3)
        rounded_extent = np.round(extent, 3)
        caption_obj = detect_object[i]["caption"]
        if ", " in caption_obj:
            last_caption = caption_obj[caption_obj.rfind(', ')+2:]
        else:
            last_caption = caption_obj
        labels.append(last_caption)
        map_indices = world_to_map(pcl, resolution, map_origin)
        for x_map, y_map, z_map in map_indices:
            #if x_map < map_size[0] and z_map < map_size[1]:
            semantic_map[x_map, z_map] = i
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    plot_case = 2
    if plot_case == 1:
        plt.imshow(semantic_map.T, origin="lower", cmap="tab20")  # Use a categorical colormap
        plt.colorbar(label="Object Index")
        plt.title("2D Semantic Occupancy Map")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
    elif plot_case == 2:
        # Plot with colorbar
        fig, ax = plt.subplots()

        # Create a colormap and norm
        cmap = plt.cm.tab20  # Categorical colormap
        norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels) + 1) - 0.5, ncolors=len(labels))

        # Plot the semantic map
        im = ax.imshow(semantic_map.T, origin="lower", cmap=cmap, norm=norm)

        # Add a colorbar with labels
        cbar = plt.colorbar(im, ax=ax, ticks=np.arange(len(labels)))
        cbar.ax.set_yticklabels(labels)  # Set the custom labels
        cbar.set_label("Object Index")

        # Add labels and title
        plt.title("2D Semantic Occupancy Map")
        plt.xlabel("X-axis")
        plt.ylabel("Z-axis")

        plt.show()

def build_kdtree_for_semantic_map(semantic_map, valid_classes):
    """
    Build a KDTree for the valid points in the semantic map.

    Args:
        semantic_map (np.ndarray): 2D semantic map where each value represents a class.
        valid_classes (list): List of classes to include in the KDTree.

    Returns:
        KDTree: KDTree built from the valid points.
        np.ndarray: Array of valid point indices (row, col) used to build the KDTree.
    """
    # Find indices of valid points
    valid_indices = np.argwhere(np.isin(semantic_map, valid_classes))
    # Build KDTree using the valid indices
    kdtree = KDTree(valid_indices)
    return kdtree, valid_indices

def find_closest_point(kdtree, valid_indices, current_location):
    """
    Find the closest valid point to the robot's current location using the KDTree.

    Args:
        kdtree (KDTree): KDTree built from the valid points.
        valid_indices (np.ndarray): Array of valid point indices used to build the KDTree.
        current_location (tuple): Current location of the robot (row, col).

    Returns:
        tuple: Closest point indices (row, col) in the semantic map.
    """
    # Query the KDTree for the closest point
    distance, index = kdtree.query(current_location)
    # Get the closest point's indices
    closest_point = tuple(valid_indices[index])
    return closest_point

def downsample_semantic_map(semantic_map, original_res=0.1, new_res=0.5):
    factor = int(new_res / original_res)  # Downsampling factor (5 in this case)
    new_shape = (semantic_map.shape[0] // factor, semantic_map.shape[1] // factor)

    downsampled_map = np.zeros(new_shape, dtype=int)  # Initialize new map

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            block = semantic_map[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
            downsampled_map[i, j] = mode(block, axis=None).mode[0]  # Assign most frequent label

    return downsampled_map

def downsample_preserving_small_objects(semantic_map, original_res=0.1, new_res=0.5):
    factor = int(new_res / original_res)  # Downsampling factor
    new_shape = (semantic_map.shape[0] // factor, semantic_map.shape[1] // factor)

    downsampled_map = np.full(new_shape, -1, dtype=int)  # Initialize with -1

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            block = semantic_map[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]

            # count all classes except -1
            unique, counts = np.unique(block[block != -1], return_counts=True)

            if len(unique) > 0:
                # select most frequent class (original mode)
                most_common = unique[np.argmax(counts)]

                # if small/rare classes exist, give them higher weight
                if len(unique) > 1 and np.min(counts) <= (factor * factor * 0.2):  # 20% threshold
                    rare_class = unique[np.argmin(counts)]
                    downsampled_map[i, j] = rare_class  # prefer keeping small-object class
                else:
                    downsampled_map[i, j] = most_common  # use original mode

    return downsampled_map

def get_road_branches(bg_list):

    for obj in bg_list:
        pcd = obj['pcd']
        pcd_road, road_points = preprocess_point_cloud(pcd)

        # Filter ground points (X-Z plane)
        pcd_road_filtered, road_points_filtered = filter_ground_points(pcd_road)
        if len(road_points_filtered) == 0:
            print("No road points found after filtering. Exiting.")
            return None, None

        # Extract 2D skeleton (X-Z plane)
        skeleton_2d = extract_2d_skeleton(road_points_filtered)

        # Fit 3D centreline
        centreline_3d = fit_3d_centreline(skeleton_2d, road_points_filtered)

        # Visualize results
        if len(centreline_3d) > 0:
            pcd_centreline = o3d.geometry.PointCloud()
            pcd_centreline.points = o3d.utility.Vector3dVector(centreline_3d)
            pcd_centreline.paint_uniform_color([1, 0, 0])  # Red for centreline

            o3d.visualization.draw_geometries([pcd_road_filtered, pcd_centreline])
        else:
            print("No centreline extracted. Check input data or parameters.")

        return pcd_road_filtered, centreline_3d


# 1. Preprocess the Point Cloud
def preprocess_point_cloud(pcd, voxel_size=0.1):
    # Load point cloud
    points = np.asarray(pcd.points)
    print("Original points shape:", points.shape)

    # Statistical outlier removal (α=2 as per paper)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    points = np.asarray(pcd.points)
    print("Points shape after outlier removal:", points.shape)

    # Downsample for efficiency
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(pcd.points)

    # Remove isolated areas using DBSCAN (ε=2, minPts=10 as per paper)
    labels = np.array(pcd.cluster_dbscan(eps=2.0, min_points=10))
    unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
    road_points = points[labels >= 0]  # Keep non-noise points
    pcd_road = o3d.geometry.PointCloud()
    pcd_road.points = o3d.utility.Vector3dVector(road_points)

    return pcd_road, road_points

# 2. Ground Point Filtering (Grid-based RANSAC, X-Z plane)
def filter_ground_points(pcd, grid_size=10.0):
    points = np.asarray(pcd.points)
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])  # X
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])  # Z (instead of Y)

    road_points = []
    for x in np.arange(x_min, x_max, grid_size):
        for z in np.arange(z_min, z_max, grid_size):
            # Select points in current grid block (X-Z plane)
            mask = (points[:, 0] >= x) & (points[:, 0] < x + grid_size) & \
                   (points[:, 2] >= z) & (points[:, 2] < z + grid_size)
            block_points = points[mask]

            if len(block_points) > 0:
                print(f"Block (X: {x}-{x+grid_size}, Z: {z}-{z+grid_size}): {len(block_points)} points")
                # Fit plane using RANSAC (threshold=0.3, adjustable)
                try:
                    plane_model, inliers = pcd.select_by_index(np.where(mask)[0]).segment_plane(
                        distance_threshold=0.5,  # Increased from 0.3 for robustness
                        ransac_n=3,
                        num_iterations=1000)
                    a, b, c, d = plane_model  # Plane: ax + by + cz + d = 0
                    normal = np.array([a, b, c])
                    # Use Z-normal (c) for tilt angle in X-Z plane
                    tilt_angle = np.arccos(np.abs(normal[2]) / np.linalg.norm(normal)) * 180 / np.pi
                    print(f"Tilt angle: {tilt_angle}°")

                    # Keep planes with tilt < 70° (relaxed from 60° for robustness)
                    if tilt_angle < 85:
                        # Use moving threshold (Z-percentile, adjusted to 85th for sparse data)
                        z_values = block_points[:, 2]  # Z instead of Y
                        threshold = np.percentile(z_values, 85)  # Increased percentile
                        road_block = block_points[z_values <= threshold]
                        if len(road_block) > 0:
                            print(f"Road points in block: {len(road_block)}")
                            road_points.append(road_block)
                        else:
                            print("No road points after Z-thresholding.")
                    else:
                        print("Block discarded: Tilt angle too steep.")
                except RuntimeError as e:
                    print(f"RANSAC failed for block: {e}")
                    continue

    road_points = np.vstack(road_points) if road_points else np.array([])
    if len(road_points) == 0:
        print("Warning: No road points found. Check grid_size, RANSAC thresholds, or tilt angle.")
    pcd_road = o3d.geometry.PointCloud()
    pcd_road.points = o3d.utility.Vector3dVector(road_points) if len(road_points) > 0 else o3d.utility.Vector3dVector([])
    return pcd_road, road_points

# 3. Project to X-Z Plane and Skeletonize
def extract_2d_skeleton(road_points, resolution=0.1):
    if len(road_points) == 0:
        return np.array([])

    # Project to X-Z plane (ignore Y for 2D)
    xz_points = road_points[:, [0, 2]]  # X, Z coordinates

    # Create 2D grid (binary image)
    x_min, x_max = np.min(xz_points[:, 0]), np.max(xz_points[:, 0])
    z_min, z_max = np.min(xz_points[:, 1]), np.max(xz_points[:, 1])
    x_range = int((x_max - x_min) / resolution) + 1
    z_range = int((z_max - z_min) / resolution) + 1

    # Create binary image
    binary = np.zeros((z_range, x_range), dtype=np.uint8)
    for x, z in xz_points:
        i = int((x - x_min) / resolution)
        j = int((z - z_min) / resolution)
        if 0 <= i < x_range and 0 <= j < z_range:
            binary[j, i] = 255

    # Apply Gaussian blur for denoising (σ=5 as per paper)
    binary = cv2.GaussianBlur(binary, (5, 5), sigmaX=5)

    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    # Skeletonization using OpenCV thinning
    skeleton = cv2.ximgproc.thinning(binary)

    # Extract skeleton points
    skeleton_points = []
    for j in range(skeleton.shape[0]):
        for i in range(skeleton.shape[1]):
            if skeleton[j, i] == 255:
                x = x_min + i * resolution
                z = z_min + j * resolution
                skeleton_points.append([x, z])
    skeleton_points = np.array(skeleton_points)

    return skeleton_points

# 4. Smooth Skeleton and Back-Project to 3D
def fit_3d_centreline(skeleton_points, road_points):
    if len(skeleton_points) < 3:
        return np.array([])

    # Sort points by X for consistent ordering (adjust as needed)
    sorted_idx = np.argsort(skeleton_points[:, 0])
    skeleton_sorted = skeleton_points[sorted_idx]

    # Smooth 2D skeleton using Savitzky-Golay filter (window=5, polyorder=3)
    smoothed_x = savgol_filter(skeleton_sorted[:, 0], window_length=5, polyorder=3)
    smoothed_z = savgol_filter(skeleton_sorted[:, 1], window_length=5, polyorder=3)
    smoothed_2d = np.column_stack((smoothed_x, smoothed_z))

    # Back-project to 3D (match Y-values from road points, using X-Z)
    kdtree = KDTree(road_points[:, [0, 2]])  # Use X-Z for matching
    smoothed_3d = []
    for x, z in smoothed_2d:
        _, idx = kdtree.query([x, z], k=1)  # Find nearest point
        y = road_points[idx, 1]  # Use Y from nearest road point
        smoothed_3d.append([x, y, z])
    smoothed_3d = np.array(smoothed_3d)

    # Use region growing with normals (simplified)
    if len(smoothed_3d) > 0:
        pcd_skeleton = o3d.geometry.PointCloud()
        pcd_skeleton.points = o3d.utility.Vector3dVector(smoothed_3d)
        pcd_skeleton.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        # Region growing: Find nearby road points using K-D Tree
        kdtree_road = KDTree(road_points)
        final_centreline = []
        for point in smoothed_3d:
            _, idx = kdtree_road.query(point, k=1)  # Find nearest road point
            final_centreline.append(road_points[idx])
        final_centreline = np.array(final_centreline)

        return final_centreline
    return np.array([])

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.05, min_points=10) -> o3d.geometry.PointCloud:
    from collections import Counter

    ### Remove noise via clustering
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]

        # Create mask for points in the largest cluster
        largest_mask = pcd_clusters == most_common_label

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd
    return pcd

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def overlay_current_time(
    image: np.ndarray,
    current_time: float,
    *,
    tz: str = "America/Toronto",
    fmt: str = "%Y-%m-%d %H:%M:%S",
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float | None = None,
    thickness: int = 2,
    margin: int = 8,
    pad: int = 6,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    alpha: float = 1.0,  # <1.0 makes background semi-transparent
):
    """
    Draw current_time (epoch seconds) as 'YYYY-MM-DD HH:MM:SS' at top-left on a black background.
    Returns: (image_with_text, timestamp_string)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array (BGR)")

    h_img, w_img = image.shape[:2]
    if font_scale is None:
        font_scale = max(0.5, min(2.0, w_img / 800.0))

    dt = datetime.fromtimestamp(current_time, ZoneInfo(tz) if tz else timezone.utc)
    ts_str = dt.strftime(fmt)

    (tw, th), baseline = cv2.getTextSize(ts_str, font, font_scale, thickness)
    x1, y1 = margin, margin
    x2, y2 = x1 + tw + 2*pad, y1 + th + baseline + 2*pad

    out = image.copy()
    if alpha >= 1.0:
        cv2.rectangle(out, (x1, y1), (x2, y2), bg_color, thickness=-1)
    else:
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, thickness=-1)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    cv2.putText(out, ts_str, (x1 + pad, y1 + pad + th), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return out, ts_str

def run_scenegraph_generation(
    cfg : DictConfig,
    data_source,
    prepare_event,
    loop_event,
    last_graph_dir=None,
    iterator=iter_by_event_depth
):
    # Prepare the environment
    set_seed(123)
    objects, timestamps, poses, idx = init_scenegraph(last_graph_dir)

    # init the map
    mask_generator = load_models(cfg)
    # calib = load_calib(cfg)
    depth_to_rgb_matrix = load_depth_to_rgb_matrix(cfg)
    rgb_camera_matrix = load_3x3_matrix_from_cfg(cfg, "rgb_camera_matrix", DEFAULT_RGB_CAMERA_MATRIX)
    depth_camera_matrix = load_3x3_matrix_from_cfg(cfg, "depth_camera_matrix", DEFAULT_DEPTH_CAMERA_MATRIX)
    if prepare_event is not None:
        prepare_event.set()

    # Process background separately
    bg_objects, bg_fts = load_background_objects(cfg, BG_CAPTIONS_Pro_Sim, BG_CAPTIONS)
    point_clouds = []
    # Example parameters
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)
    enable_timing_logs = bool(getattr(cfg, "enable_timing_logs", False))
    timing_log_every_n = max(1, int(getattr(cfg, "timing_log_every_n", 1)))
    print_scenegraph_captions = bool(getattr(cfg, "print_scenegraph_captions", True))
    verbose = bool(getattr(cfg, "verbose", False))

    try:
        # for data_in_window in iterator(data_source, loop_event):
        data_iter = iterator(data_source, loop_event)
        first_sensor_timestamp = None
        while True:
            try:
                data_in_window = next(data_iter) # one frame
            except StopIteration:
                break
            
            t1 = time.time()
            file_names = idx
            save_path_vis = Path(os.path.join(cfg.save_vis_path, f"vis_{file_names}"))
            _, image, depth, pose, timestamp = data_in_window
            if first_sensor_timestamp is None:
                first_sensor_timestamp = timestamp
            display_timestamp = cfg.start_timestamp + (timestamp - first_sensor_timestamp)

            if cfg.projection_depth:
                save_path = Path(os.path.join(cfg.save_vis_proj_path, f"vis_{file_names}")).with_suffix(".png")
                out = visualize_depth_on_rgb(
                    depth,
                    depth_camera_matrix,
                    image,
                    rgb_camera_matrix,
                    depth_to_rgb_matrix,
                    (0.2, 30.0)
                )

                if not save_path.parent.exists():
                    os.makedirs(save_path.parent)
                cv2.imwrite(str(save_path), out)
                if verbose:
                    print(f"Save depth projection to {save_path}")

            data_loading_dt = time.time() - t1
            if enable_timing_logs and (idx % timing_log_every_n == 0):
                print(f"Data loading time: {data_loading_dt:.2f}s")
            t2 = time.time()
            masks_result, anotated_image_data = mask_generator.generate_DAM(image, cfg, save_path=save_path_vis, save_vis=cfg.save_vis)
            mask_generation_dt = time.time() - t2
            if enable_timing_logs and (idx % timing_log_every_n == 0):
                print(f"Mask generation time: {mask_generation_dt:.2f}s")
            # masks_result, anotated_image = mask_generator.generate_OpenGraph(image, save_path=save_path_vis, save_vis=cfg.save_vis)

            # Filter out distant objects
            if cfg.filter_dis:
                depth[depth > cfg.max_depth] = 0

            save_path = Path(os.path.join(cfg.save_vis_path, f"vis_{file_names}")).with_suffix(".png")
            if not save_path.parent.exists():
                os.makedirs(save_path.parent)

            t3 = time.time()

            file_names = idx
            gobs = masks_result
            detection_list, bg_list, _, _, valid_mask_indices = gobs_to_detection_list_depth(
                cfg = cfg,
                image = image,
                depth_array=depth,
                cam_K_depth=depth_camera_matrix,
                cam_K_rgb=rgb_camera_matrix,
                cam_R=depth_to_rgb_matrix,
                idx = idx,
                gobs = gobs,
                trans_pose = pose,
                bg_fts = bg_fts,
                BG_CAPTIONS_Pro = BG_CAPTIONS_Pro,
            )
            detection_list_dt = time.time() - t3
            if enable_timing_logs and (idx % timing_log_every_n == 0):
                print(f"Detection list generation time: {detection_list_dt:.2f}s")

            if verbose:
                print("detection_list:", len(detection_list))
            t4 = time.time()

            if len(bg_list) > 0:
                for detected_object in bg_list:
                    class_name = detected_object['bg_class']
                    if bg_objects[class_name] is None:
                        bg_objects[class_name] = detected_object
                    else:
                        matched_obj = bg_objects[class_name]
                        matched_det = detected_object
                        bg_objects[class_name] = merge_obj2_into_obj1(cfg, matched_obj, matched_det, bg=True, class_name = class_name)

            bg_merge_dt = time.time() - t4
            if enable_timing_logs and (idx % timing_log_every_n == 0):
                print(f"Background merging time: {bg_merge_dt:.2f}s")
            # occupancy_map, map_origin, sematic_map, sematic_info, _, _, object_indices_dict, robot_state, drivable_indices, terrain_map = create_2d_map_point(detection_list, bg_list, pose, prev_occupancy_map, prev_map_origin, pre_detection_list, pre_bg_list, resolution)

            # If there is no detection, skip this frame
            if len(detection_list) == 0:
                timestamps.append(timestamp)
                poses.append(pose)
                idx += 1
                if verbose:
                    print("No detections, skipping frame.")
                continue

            # Add all if it is the first frame
            if len(objects) == 0:
                for i in range(len(detection_list)):
                    objects.append(detection_list[i])
                    if print_scenegraph_captions:
                        print(f"[scenegraph-caption] frame={idx} det={i}: \033[96m{detection_list[i]['caption']}\033[0m")
                timestamps.append(timestamp)
                poses.append(pose)
                idx += 1

                continue

            if enable_timing_logs and (idx % timing_log_every_n == 0):
                print(f"denoise_objects time: {time.time() - t4:.2f}s")
            if cfg.vis_all:
                point_clouds.extend([detection_list[i]["pcd"] for i in range(len(detection_list))])
                point_clouds.extend([bg_objects[_]["pcd"] for _ in bg_objects if bg_objects[_] is not None])

            # Calculate similarities between detections and existing objects
            spatial_sim = compute_spatial_similarities(detection_list, objects)
            caption_sim = compute_caption_similarities(detection_list, objects)
            ft_sim = compute_ft_similarities(detection_list, objects)
            # volume_sim reuse spatial_sim since it's AABB IoU
            agg_sim = aggregate_similarities(cfg, spatial_sim, ft_sim, caption_sim)
            agg_sim[agg_sim < cfg.sim_threshold] = float('-inf')

            # Merge detections to existing objects
            objects, object_id = merge_detections_to_objects_2(
                cfg,
                detection_list,
                objects,
                agg_sim,
                valid_mask_indices,
                frame_idx=idx,
            )
            anotated_image = prepare_labeled_contours_merge(
                image=image,
                masks=anotated_image_data['masks'],
                detections=anotated_image_data['detections'],
                image_width=image.shape[1],
                image_height=image.shape[0],
                instance_id=object_id,
                valid_mask_indices =valid_mask_indices
            )

            anotated_image, ts = overlay_current_time(anotated_image, display_timestamp)
            if cfg.save_SoM:
                if not os.path.exists(cfg.annotated_rgb_path):
                    os.makedirs(cfg.annotated_rgb_path)
                annotated_bgr = cv2.cvtColor(anotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(cfg.annotated_rgb_path, f"annotated_rgb_{idx + cfg.start}.png"),
                    annotated_bgr,
                )

            if verbose:
                print(f"\033[91m@@@ Time elapsed: {idx//60:02d}:{idx%60:02d}\033[0m, wall time {time.time() - t1}")

            if idx % cfg.vis_interval == 0:
                visualize_objects_org(objects)
            timestamps.append(timestamp)
            poses.append(pose)
            idx += 1
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        print("Finalizing scene graph generation...")
        save_scene_graph(cfg, objects, timestamps, poses)

        # After building the map, downsample the map to reduce resolution and denoise it
        if bg_objects is not None:
            bg_objects = MapObjectList([_ for _ in bg_objects.values() if _ is not None])
            bg_objects = denoise_objects(cfg, bg_objects, bg = True)

if __name__ == "__main__":
    run_scenegraph_generation()
