from typing import List

import numpy as np
import open3d as o3d

from star.some_class.map_class import DetectionList


class ObjectVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='3D Scene Graph Viewer', width=960, height=720)
        self.vis.get_render_option().point_size = 2.0
        self.is_geometry_added = False

    def update(self, objects):
        self.vis.clear_geometries()
        for i, obj in enumerate(objects):
            self.vis.add_geometry(obj["pcd"])
            self.vis.add_geometry(obj["bbox"])
        self.vis.poll_events()
        self.vis.update_renderer()

    def update_pointclouds(self, point_clouds):
        self.vis.clear_geometries()
        for pc in point_clouds:
            self.vis.add_geometry(pc)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


def visualize_objects_org(objects: DetectionList) -> None:
    all_point_clouds: List[o3d.geometry.PointCloud] = []
    for obj in objects:
        all_point_clouds.append(obj['pcd'])
        all_point_clouds.append(obj['bbox'])
    o3d.visualization.draw_geometries(all_point_clouds)


def visualize_objects_org(objects: DetectionList) -> None:
    all_point_clouds: List[o3d.geometry.PointCloud] = []
    for obj in objects:
        all_point_clouds.append(obj['pcd'])
    o3d.visualization.draw_geometries(all_point_clouds)


def visualize_objects(objects: List[dict]) -> None:
    import copy
    
    all_geometries: List[o3d.geometry.Geometry] = []

    colors = [
        [1.0, 0.0, 0.0],    # 0 Red
        [0.0, 1.0, 0.0],    # 1 Green
        [0.0, 0.0, 1.0],    # 2 Blue
        [1.0, 0.0, 1.0],    # 3 Magenta
        [0.0, 1.0, 1.0],    # 4 Cyan
        [0.5, 0.0, 1.0],    # 5 Purple
        [0.0, 0.5, 0.5],    # 6 Teal
    ]


    for i, obj in enumerate(objects):
        color = colors[i % len(colors)]
        pcd_copy = copy.deepcopy(obj['pcd'])
        pcd_copy.paint_uniform_color(color)

        bbox_copy = copy.deepcopy(obj['bbox'])
        bbox_copy.color = color

        all_geometries.append(pcd_copy)
        all_geometries.append(bbox_copy)

    o3d.visualization.draw_geometries(all_geometries)

#     colors = [
# [1, 0, 0], # Red
# [0, 1, 0], # Green
# [0, 0, 1], # Blue
# [1, 1, 0], # Yellow
# [1, 0, 1], # Magenta
#     ]

#     for i, obj in enumerate(objects):
#         color = colors[i % len(colors)]

#         # Set color for point cloud
#         pcd = obj['pcd'].paint_uniform_color(color)

#         # Set color for OrientedBoundingBox
#         bbox = obj['bbox']

#         all_geometries.append(pcd)
#         all_geometries.append(bbox)

#     o3d.visualization.draw_geometries(all_geometries)



def rotate_R_roll(R: np.ndarray, roll: float) -> np.ndarray:
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])
    return rot @ R

def rotate_point_roll(points: np.ndarray, roll: float) -> np.ndarray:
    rot: np.ndarray = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])
    return points @ rot.T

def rotate_R_pitch(R: np.ndarray, pitch: float) -> np.ndarray:
    rot: np.ndarray = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    return rot @ R

def rotate_point_pitch(points: np.ndarray, pitch: float) -> np.ndarray:
    rot: np.ndarray = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    return points @ rot.T

def rotate_R_yaw(R: np.ndarray, yaw: float) -> np.ndarray:
    rot: np.ndarray = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return rot @ R

def rotate_point_yaw(points: np.ndarray, yaw: float) -> np.ndarray:
    rot: np.ndarray = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return points @ rot.T


def convert_pcd_to_global(
    pcd: o3d.geometry.PointCloud,
    roll: float,
    pitch: float,
    offset: np.ndarray
) -> o3d.geometry.PointCloud:
    global_points: np.ndarray = np.asarray(pcd.points)
    global_points = rotate_point_roll(global_points, roll)
    global_points = rotate_point_pitch(global_points, -pitch)
    pcd.points = o3d.utility.Vector3dVector(global_points + offset)
    return pcd

def convert_obj_pcd_to_global(obj_list: DetectionList, roll: float, pitch: float, offset: np.ndarray) -> list:
    frame_point_clouds: list = []
    for obj in obj_list:
        # Transform pointcloud data
        obj['pcd'] = convert_pcd_to_global(obj['pcd'], roll, pitch, offset)
        frame_point_clouds.append(obj['pcd'])

        # Transform bounding box (OrientedBoundingBox)
        bbox: o3d.geometry.OrientedBoundingBox = obj['bbox']
        bbox_center = np.asarray(bbox.center)
        bbox_center = rotate_point_roll(bbox_center.reshape(1, -1), roll).flatten()
        bbox_center = rotate_point_pitch(bbox_center.reshape(1, -1), -pitch).flatten()
        bbox_center += offset

        # Rotate bbox R matrix
        R = bbox.R.copy()
        R = rotate_R_roll(R, roll)
        R = rotate_R_pitch(R, -pitch)

        # Update bbox
        bbox.center = bbox_center
        bbox.R = R

    return frame_point_clouds
