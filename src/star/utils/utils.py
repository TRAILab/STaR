"""Utility functions for model loading, point-cloud processing, and object-map construction."""

import sys
import os
import time
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# #sys.path.append("/home/trailbot/trail_ws/src/TRAILBot/object_nav/third_parties/tokenize-anything")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
sys.path.append("original_documents/third_parties/GroundingDINO")
sys.path.append("third_parties/Tag2Text")
sys.path.append("original_documents/third_parties")
sys.path.append("original_documents/third_parties/tokenize-anything")
from tokenize_anything import model_registry

import open3d as o3d
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import spacy

import torch.nn.functional as F
import json
import faiss
import re
from openai import OpenAI
from termcolor import colored
from tqdm import trange

try:
    from ram.models import tag2text
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your PATH. ")
    raise e

try:
    from groundingdino.util.inference import Model
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

from star.some_class.map_class import MapObjectList
from some_class.amg_class import MyAutomaticMaskGenerator
from star.some_class.map_class import DetectionList


# Nouns that should be excluded when spaCy selects the first noun as the keyword.
CONFUSED_NOUNS = ["metal", "back", "part", "row", "triangular","patch"]
INTEREST_NOUNS = ["van", "house"]
INTEREST_ADJS = ["grassy","white"]
EGO_TO_CAM: np.ndarray = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])
CAM_TO_EGO: np.ndarray = np.linalg.inv(EGO_TO_CAM)

def print_to_cot_log(message, log_file="cot_log.txt", color=None):
    if color:
        print(colored(message, color))
    else:
        print(message)

    if log_file is None:
        return

    with open(log_file, "a") as f:
        f.write(message + "\n")

def load_models(cfg):
    '''
    Load the three large models and SBERT.
    '''
    print("\n[models] Loading scenegraph foundation models")
    print("[models] Loading Tag2Text")
    print(f"  weights                 {cfg.tag2text_path}")
    TAG2TEXT_CHECKPOINT_PATH = cfg.tag2text_path
    delete_tag_index = []
    for i in range(3012, 3429):
        delete_tag_index.append(i)
    tagging_model = tag2text(
        pretrained=TAG2TEXT_CHECKPOINT_PATH,
        image_size=384,
        vit='swin_b',
        delete_tag_index=delete_tag_index,
    ).to("cuda")
    tagging_model = tagging_model.eval().to("cuda")
    print("[models] Tag2Text ready")

    print("[models] Loading GroundingDINO")
    print(f"  config                  {cfg.gd_path}")
    print(f"  weights                 {cfg.gd_weights}")
    grounding_dino_model = Model(
        model_config_path = cfg.gd_path,
        model_checkpoint_path = cfg.gd_weights,
        device="cuda"
    )
    print("[models] GroundingDINO ready")

    print("[models] Loading TAP")
    model_type = "tap_vit_l"
    checkpoint = cfg.tap_path
    print(f"  checkpoint              {checkpoint}")
    tap_model = model_registry[model_type](checkpoint=checkpoint).to("cuda")

    concept_weights = cfg.tap_merge_path
    print(f"  concept weights         {concept_weights}")
    tap_model.concept_projector.reset_weights(concept_weights)
    tap_model.text_decoder.reset_cache(max_batch_size=1000)
    print("[models] TAP ready")

    sbert_name = 'sentence-transformers/all-MiniLM-L6-v2'
    print("[models] Loading SBERT")
    print(f"  model                   {sbert_name}")
    sbert_model = SentenceTransformer(sbert_name).to("cuda")
    print("[models] SBERT ready")

    mask_generator = MyAutomaticMaskGenerator(tagging_model=tagging_model, grounding_dino_model=grounding_dino_model, tap_model=tap_model, sbert_model=sbert_model)
    print("[models] Scenegraph foundation models ready\n")
    return mask_generator

def project(points, image, calib, cfg):
    '''
    Project a point cloud onto an image and return the corresponding image coordinates.
    '''
    points_homo = np.insert(points, 3, 1, axis=1).T

    front_axis = str(getattr(cfg, "front_axis", "y")).lower()
    if front_axis == 'x':
        idx = 0
    elif front_axis == 'y':
        idx = 1
    else:
        raise ValueError(f"front_axis must be 'x' or 'y', got {front_axis!r}.")

    pointCloud = np.delete(points, np.where(points_homo[idx, :] < 0), axis=0)
    points_homo = np.delete(points_homo, np.where(points_homo[idx, :] < 0), axis=1)
    # Camera-frame 3D points = camera projection * lidar-to-camera transform * lidar 3D points.

    camera_projection_matrix = calib["camera_projection_matrix"]
    lidar_to_camera_matrix = calib["lidar_to_camera_matrix"]
    proj_lidar = camera_projection_matrix.dot(lidar_to_camera_matrix).dot(points_homo)


    # Remove columns for projected points with depth z < 0, which are behind the image plane. # 3xN
    cam = np.delete(proj_lidar, np.where(proj_lidar[2, :] < 0), axis=1)
    pointCloud = np.delete(pointCloud, np.where(proj_lidar[2, :] < 0), axis=0)
    # Divide the first two rows by the third row to normalize onto the camera-frame z=1 plane.
    cam[:2, :] /= cam[2, :]
    # Project onto the image.
    IMG_H, IMG_W, _ = image.shape
    # Filter out points outside the camera image.
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    points = np.delete(pointCloud, np.where(outlier), axis=0)
    u,v,z  = cam
    pixels = np.dstack((v,u)).squeeze()
    # return points, pixels
    return points, pixels, z

def prepare_labeled_contours_merge(image, masks, detections, image_width, image_height, instance_id, valid_mask_indices):
    """
    Display object contours with merged map object IDs as labels.

    Args:
        image (np.ndarray): The original RGB image as a NumPy array.
        masks (list of np.ndarray): Binary masks for each object.
        detections (list or np.ndarray): Bounding boxes with coordinates [x_min, y_min, x_max, y_max].
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        instance_id (dict): Mapping from detection index (1-based) to map object index.

    Returns:
        np.ndarray: Annotated RGB image with contours and labels drawn.
    """

    # Define distinct colors for drawing contours
    distinct_colors = [
        '#A52A2A', '#5F9EA0', '#D2691E', '#9ACD32', '#DA70D6', '#7FFFD4',
        '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80',
        '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00',
        '#FF4500', '#2E8B57'
    ]
    distinct_colors_rgb = [tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for color in distinct_colors]

    # Copy the original image for annotation
    annotated_image = image.copy()

    # Track label positions
    label_positions = []
    offset_step = 5  # Offset for small contours

    def clamp(value, min_value, max_value):
        """Clamp a value within a specified range."""
        return max(min_value, min(value, max_value))

    def find_safe_label_position(contour, bbox, is_small):
        """Find a safe label position inside or near the object contour."""
        x_min, y_min, x_max, y_max = bbox
        bbox_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))

        if not is_small:
            return bbox_center
        return (
            clamp(x_max, 0, image_width),
            clamp(y_min - offset_step, 0, image_height)
        )

    label_dict = {}

    # Iterate over detections and masks

    for i, (mask, box) in enumerate(zip(masks, detections.xyxy)):
        if i not in valid_mask_indices:
            continue

        color = distinct_colors_rgb[instance_id.get(i) % len(distinct_colors_rgb)]
        # Validate mask
        if mask is None or mask.size == 0:
            print(f"Warning: Empty mask detected at index {i}")
            continue

        mask_uint8 = (mask.astype(np.uint8) * 255)
        if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
            mask_uint8 = mask_uint8.squeeze(0)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contours found for mask at index {i}")
            continue

        x_min, y_min, x_max, y_max = map(int, box)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        is_small = bbox_width < 30 or bbox_height < 30

        # Draw contour
        cv2.drawContours(annotated_image, contours, -1, color, 2)

        # Find safe label position
        label_pos_x, label_pos_y = find_safe_label_position(contours[0], (x_min, y_min, x_max, y_max), is_small)

        # ⚡ Use instance_id to get the merged object id
        detection_idx = i #+ 1  # because instance_id is 1-based
        map_object_idx = instance_id.get(detection_idx, -1)
        label_text = str(map_object_idx) if map_object_idx >= 0 else "?"

        font_size = 0.4
        label_width, label_height = 36, 14

        label_positions.append((label_pos_x, label_pos_y))
        label_dict[label_text] = (
            label_pos_x,
            label_pos_y - label_height,
            label_pos_x + label_width,
            label_pos_y,
            color,
            label_pos_x,
            label_pos_y - 3
        )

    # Draw all labels
    for key in label_dict.keys():
        cv2.rectangle(
            annotated_image,
            (label_dict[key][0], label_dict[key][1]),
            (label_dict[key][2], label_dict[key][3]),
            label_dict[key][4],
            -1  # Filled rectangle
        )
        cv2.putText(
            annotated_image,
            key,
            (label_dict[key][5], label_dict[key][6]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (0, 0, 0),  # Black text
            1
        )

    return annotated_image

def create_object_pcd(image, pc, pixels, mask, obj_color=None) -> o3d.geometry.PointCloud:
    '''
    Create an RGB point cloud.
    '''
    mask_for_pc = mask[pixels[:, 0].astype(int), pixels[:, 1].astype(int)]
    points = pc[mask_for_pc]
    pixels = pixels[mask_for_pc]
    colors = image[pixels[:,0].astype(int), pixels[:,1].astype(int)]/255.0
    # Slightly perturb points to avoid collinearity and hide lidar scan lines.
    #points += np.random.normal(0, 4e-3, points.shape)
    # Create an Open3D PointCloud object.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# def create_object_pcd(image, pc, pixels, mask, obj_color=None) -> o3d.geometry.PointCloud:
#     '''
#     Create an RGB point cloud.
#     '''
#     # Get the height and width of the mask (image)
#     height, width = mask.shape[:2]

#     # Ensure pixel values are within valid range and log invalid entries
#     pixels[:, 0] = np.clip(pixels[:, 0], 0, height - 1)
#     pixels[:, 1] = np.clip(pixels[:, 1], 0, width - 1)

#     # Debugging logs to see any strange pixel values
#     if np.any(pixels < 0):
#         print(f"Invalid pixel values (less than 0) found: {pixels[pixels < 0]}")

#     if np.any(pixels >= height) or np.any(pixels >= width):
#         print(f"Invalid pixel values (greater than mask size) found. Max values in pixels: {pixels.max(axis=0)}")

#     # Make sure the pixel indices are valid
#     try:
#         mask_for_pc = mask[pixels[:, 0].astype(int), pixels[:, 1].astype(int)]
#     except IndexError as e:
#         print(f"IndexError encountered! Pixels: {pixels}, Mask shape: {mask.shape}")
#         raise e  # Re-raise after logging

#     # Filter the point cloud using the valid mask points
#     points = pc[mask_for_pc]
#     pixels = pixels[mask_for_pc]

#     # Get colors for the valid points
#     colors = image[pixels[:, 0].astype(int), pixels[:, 1].astype(int)] / 255.0

#     # Add small noise to avoid collinearity
#     points += np.random.normal(0, 4e-3, points.shape)

#     # Create Open3D PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd

# def create_object_pcd(image, pc, pixels, mask, obj_color=None) -> o3d.geometry.PointCloud:
#     '''
#     Create an RGB point cloud.
#     '''
#     # Get the height and width of the mask (image)
#     height, width = mask.shape[:2]

#     # Ensure pixel values are within valid range
#     pixels[:, 0] = np.clip(pixels[:, 0], 0, height - 1)
#     pixels[:, 1] = np.clip(pixels[:, 1], 0, width - 1)

#     # Remove any NaN values in the pixel array
#     valid_mask = ~np.isnan(pixels).any(axis=1)  # Find rows without NaNs
#     if np.any(~valid_mask):
#         print(f"Found {np.sum(~valid_mask)} invalid (NaN) pixel entries, removing them.")

#     # Keep only valid pixels and corresponding point cloud points
#     pixels = pixels[valid_mask]
#     pc = pc[valid_mask]

#     # Make sure the pixel indices are valid
#     try:
#         mask_for_pc = mask[pixels[:, 0].astype(int), pixels[:, 1].astype(int)]
#     except IndexError as e:
#         print(f"IndexError encountered! Pixels: {pixels}, Mask shape: {mask.shape}")
#         raise e  # Re-raise after logging

#     # Filter the point cloud using the valid mask points
#     points = pc[mask_for_pc]
#     pixels = pixels[mask_for_pc]

#     # Get colors for the valid points
#     colors = image[pixels[:, 0].astype(int), pixels[:, 1].astype(int)] / 255.0

#     # Add small noise to avoid collinearity
#     points += np.random.normal(0, 4e-3, points.shape)

#     # Create Open3D PointCloud object
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd



def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.01, min_points=10) -> o3d.geometry.PointCloud:
    '''
    Remove noise through clustering.
    '''
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )
    # Convert to NumPy arrays.
    obj_points = np.asarray(pcd.points)
    obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)
    # Count all cluster labels.
    counter = Counter(pcd_clusters)
    # Remove the noise label.
    if counter and (-1 in counter):
        del counter[-1]
    if counter:
        # Find the label of the largest cluster.
        most_common_label, _ = counter.most_common(1)[0]
        # Create a mask for points in the largest cluster.
        largest_mask = pcd_clusters == most_common_label
        # Apply the mask.
        largest_cluster_points = obj_points[largest_mask]
        largest_cluster_colors = obj_colors[largest_mask]
        # Return the original point cloud if the largest cluster is too small.
        if len(largest_cluster_points) < 5:
            return pcd
        # Create a new PointCloud object.
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        pcd = largest_cluster_pcd
    return pcd

def process_pcd(cfg, pcd, use_db = True):
    '''
    Denoise the point cloud and remove outliers.
    '''
    # Downsample using the configured voxel size.
    pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
    # Debug by comparing the point cloud before and after denoising.
    # o3d.visualization.draw_geometries([pcd])
    # Denoise the point cloud, although this method may not work well for all point clouds.
    if cfg.dbscan_remove_noise and use_db:
        # cl, index = pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=1.0)
        pcd = pcd_denoise_dbscan(
            pcd,
            eps=cfg.dbscan_eps,
            min_points=cfg.dbscan_min_points
        )
    return pcd


def get_bounding_box(pcd, mode="obb"):
    '''
    Get the point-cloud bounding box using OBB or AABB geometry.
    '''
    mode = str(mode).lower()
    if mode not in {"obb", "aabb"}:
        raise ValueError(f"Unsupported bbox_mode '{mode}'. Use 'obb' or 'aabb'.")

    if mode == "aabb":
        return pcd.get_axis_aligned_bounding_box()

    # OBB mode falls back to AABB for sparse or degenerate point clouds.
    if len(pcd.points) >= 4:
        try:
            return pcd.get_oriented_bounding_box(robust=True)
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()


def gobs_to_detection_list_2(
    cfg,
    image,
    pc,
    pixels,
    idx,
    gobs,
    trans_pose = None,
    bg_fts = None,
    BG_CAPTIONS_Pro = None,
):
    '''
    Return a DetectionList from gobs containing all objects in the current frame.
    '''
    detection_lists = DetectionList()
    bg_list = DetectionList()
    # Return empty lists when there is no data.
    if len(gobs) == 0:
        return detection_lists, bg_list
    n_masks = len(gobs)
    # Process each mask.
    SoM = None
    mask_dic = {}
    valid_mask_indices = []
    bg_indices = []
    detect_indices = []

    for mask_idx in range(n_masks):
        mask = gobs[mask_idx]['mask'].squeeze()

        caption = gobs[mask_idx]['caption']
        caption_ft = gobs[mask_idx]['caption_ft']
        img_bbox = gobs[mask_idx]['img_bbox']
        # Create the point cloud.
        camera_object_pcd = create_object_pcd(
            image,
            pc,
            pixels,
            mask,
            obj_color = None
        )
        # Assign a random color to the instance.
        color = np.random.random(3)
        # Discard objects with fewer than five points.
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):#
            continue
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        valid_mask_indices.append(mask_idx)
        # Keep the largest cluster to filter out noise.
        global_object_pcd = process_pcd(cfg, global_object_pcd)

        pcd_bbox = get_bounding_box(
            global_object_pcd, mode=getattr(cfg, "bbox_mode", "obb")
        )
        pcd_bbox.color = [0,1,0]
        # Also discard objects that are too small.
        # if pcd_bbox.volume() < 1e-6:
        #     continue
        bg_class = None

        mask_dic[mask_idx+1] = mask_idx+1
        # When background classes are enabled, compare similarity against the background threshold.
        if cfg.use_bg:
            caption_ft_cuda = caption_ft.to("cuda")
            for i in range(len(bg_fts)):
                similarity = F.cosine_similarity(bg_fts[i], caption_ft_cuda, dim=-1)
                if similarity > cfg.bg_rate:
                    bg_class = BG_CAPTIONS_Pro[i]
                    SoM = None
                    #SoM = create_SoM_road(image, pc, pixels, mask, dots_size_w=10, dots_size_h=10, font_path='/home/mfyuan/local_folder/OpenGraph/config/arial.ttf')
                    # SoM = create_SoM_keypts(image, pc, mask, pixels, font_path='/home/mfyuan/local_folder/OpenGraph/config/arial.ttf')
                    # Stop once a sufficiently high similarity is found.
                    break
        # Store this object.
        detected_object = {
            'image_idx' : [idx],                             # which image is it from, be careful with the stride
            'num_detections' : 1,                            # how many times this object is detected
            'n_points': len(global_object_pcd.points),       # Number of points in this object.
            "inst_color": color,                             # Random instance color, later assigned according to SemanticKITTI.
            "bg_class": bg_class,                            # Background class for this instance, or None for foreground objects.
            # The following fields describe a global-map object, since this may be a new object.
            'class_sk':None,                                 # Instance class used for later visualization.
            'caption':caption,                               # Instance caption to be fused later.
            'captions_ft':None,                              # Encoded features of the fused caption.
            'ft':caption_ft,                                 # Encoded caption features.
            'pcd': global_object_pcd,                        # Point cloud.
            'bbox': pcd_bbox,                                # Instance bounding box.
            'img_bbox': mask_idx+1,                          # Instance image bounding box.
            }
        #print(f'image_idx {idx}, caption {caption}, bbox {pcd_bbox}')
        # Categorize the detection.
        if cfg.use_bg and bg_class is not None:
            bg_list.append(detected_object)
            bg_indices.append(mask_idx)
        else:
            detection_lists.append(detected_object)
            detect_indices.append(mask_idx)
    #print(f"the valid indices are {valid_mask_indices}, bg indices are {bg_indices}, detect indices are {detect_indices}")
                # detection_lists.append(detected_object)
    #print(f'---------------------above is for image {idx}---------------------')
    return detection_lists, bg_list, SoM, mask_dic, detect_indices

def gobs_to_detection_list(
    cfg,
    image,
    pc,
    pixels,
    idx,
    gobs,
    trans_pose = None,
    bg_fts = None,
    BG_CAPTIONS_Pro = None,
):
    '''
    Return a DetectionList from gobs containing all objects in the current frame.
    '''
    detection_lists = DetectionList()
    bg_list = DetectionList()
    # Return empty lists when there is no data.
    if len(gobs) == 0:
        return detection_lists, bg_list
    n_masks = len(gobs)
    if bool(getattr(cfg, "verbose", False)):
        print(f"Processing {n_masks} masks for image index {idx}...")
    # Process each mask.
    for mask_idx in range(n_masks):
        mask = gobs[mask_idx]['mask'].squeeze()

        caption = gobs[mask_idx]['caption']
        caption_ft = gobs[mask_idx]['caption_ft']
        img_bbox = gobs[mask_idx]['img_bbox']
        # Create the point cloud.
        camera_object_pcd = create_object_pcd(
            image,
            pc,
            pixels,
            mask,
            obj_color = None
        )
        # Assign a random color to the instance.
        color = np.random.random(3)
        # Discard objects with fewer than five points.
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        # Keep the largest cluster to filter out noise.
        global_object_pcd = process_pcd(cfg, global_object_pcd)
        pcd_bbox = get_bounding_box(
            global_object_pcd, mode=getattr(cfg, "bbox_mode", "obb")
        )
        pcd_bbox.color = [0,1,0]
        # Also discard objects that are too small.
        if pcd_bbox.volume() < 1e-6:
            continue
        bg_class = None
        # When background classes are enabled, compare similarity against the background threshold.
        if cfg.use_bg:
            caption_ft_cuda = caption_ft.to("cuda")
            for i in range(len(bg_fts)):
                similarity = F.cosine_similarity(bg_fts[i], caption_ft_cuda, dim=-1)
                if similarity > cfg.bg_rate:
                    bg_class = BG_CAPTIONS_Pro[i]
                    # Stop once a sufficiently high similarity is found.
                    break
        # Store this object.
        detected_object = {
            'image_idx' : [idx],                             # Image where this object was observed; account for stride.
            'num_detections' : 1,                            # Number of detections; currently one for a new object.
            'n_points': len(global_object_pcd.points),       # Number of points in this object.
            "inst_color": color,                             # Random instance color, later assigned according to SemanticKITTI.
            "bg_class": bg_class,                            # Background class for this instance, or None for foreground objects.
            # The following fields describe a global-map object, since this may be a new object.
            'class_sk':None,                                 # Instance class used for later visualization.
            'caption':caption,                               # Instance caption to be fused later.
            'captions_ft':None,                              # Encoded features of the fused caption.
            'ft':caption_ft,                                 # Encoded caption features.
            'pcd': global_object_pcd,                        # Point cloud.
            'bbox': pcd_bbox,                                # Instance bounding box.
            #'img_bbox': img_bbox,                           # Instance image bounding box.
            'img_bbox': mask_idx+1,
            }
        #print(f'image_idx {idx}, caption {caption}, bbox {pcd_bbox}')
        # Categorize the detection.
        if cfg.use_bg and bg_class is not None:
            bg_list.append(detected_object)
        else:
            detection_lists.append(detected_object)
    return detection_lists, bg_list


def gobs_to_detection_list_depth(
    cfg,
    image,
    depth_array,
    cam_K_depth,
    cam_K_rgb,
    cam_R,
    idx,
    gobs,
    trans_pose = None,
    bg_fts = None,
    BG_CAPTIONS_Pro = None,
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame.
    '''

    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()

    SoM = None
    mask_dic = {}
    detected_indices = []
    valid_mask_indices = []
    bg_indices = []
    detected_indices = []

    if len(gobs) == 0:
        return fg_detection_list, bg_detection_list
    n_masks = len(gobs)
    for mask_idx in range(n_masks):
        mask = gobs[mask_idx]['mask'].squeeze()

        caption = gobs[mask_idx]['caption']
        caption_ft = gobs[mask_idx]['caption_ft']

        # make the pcd and color it
        camera_object_pcd = create_object_pcd_with_extrinsics(
            depth_array,
            mask,
            cam_K_depth,
            image,
            cam_K_rgb,
            cam_R,
        )

        color = np.random.random(3)
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5):
            continue

        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd

        global_points = np.asarray(camera_object_pcd.points)
        mask = global_points[:, 2] <= 4.5
        global_object_pcd.points = o3d.utility.Vector3dVector(global_points[mask])
        colors = np.asarray(global_object_pcd.colors)
        global_object_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        valid_mask_indices.append(mask_idx)
        # get largest cluster, filter out noise
        global_object_pcd = process_pcd(cfg, global_object_pcd)
        pcd_bbox = get_bounding_box(
            global_object_pcd, mode=getattr(cfg, "bbox_mode", "obb")
        )
        pcd_bbox.color = [0,1,0]

        if pcd_bbox.volume() < 1e-6:
            continue
        bg_class = None
        mask_dic[mask_idx+1] = mask_idx+1
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection

        if cfg.use_bg:
            caption_ft_cuda = caption_ft.to("cuda")
            for i in range(len(bg_fts)):
                similarity = F.cosine_similarity(bg_fts[i], caption_ft_cuda, dim=-1)
                if similarity > cfg.bg_rate:
                    bg_class = BG_CAPTIONS_Pro[i]
                    break

        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'num_detections': 1,                            # number of detections in this object
            'n_points': [len(global_object_pcd.points)],
            "inst_color": color,
            "bg_class": bg_class,
            'class_sk':None,
            'caption':caption,
            'captions_ft':None,
            'ft':caption_ft,
            'pcd': global_object_pcd,
            'bbox': pcd_bbox,
            'img_bbox': mask_idx+1,
        }

        if cfg.use_bg and bg_class is not None:
            bg_detection_list.append(detected_object)
            bg_indices.append(mask_idx)
        else:
            fg_detection_list.append(detected_object)
            detected_indices.append(mask_idx)
    if bool(getattr(cfg, "verbose", False)):
        print(f"the valid indices are {valid_mask_indices}, bg indices are {bg_indices}, detect indices are {detected_indices}")

    return fg_detection_list, bg_detection_list, SoM, mask_dic, detected_indices


def denoise_objects(cfg, objects: MapObjectList, bg=False):
    '''
    Denoise the entire map.
    '''
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        if bg:
            objects[i]['pcd'] = process_pcd(cfg, objects[i]['pcd'], use_db=False)
        else:
            objects[i]['pcd'] = process_pcd(cfg, objects[i]['pcd'], use_db=True)
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(
            objects[i]['pcd'], mode=getattr(cfg, "bbox_mode", "obb")
        )
        objects[i]['bbox'].color = [0,1,0]
    return objects



def filter_objects(cfg, objects: MapObjectList):
    '''
    Post-process objects by removing those with too few points or observations.
    '''
    #print("Before final map filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    return objects

def compute_3d_iou(bbox1, bbox2, padding=0, use_iou=True):
    '''
    Compute 3D IoU; objects with insufficient overlap will not be considered for final fusion.
    '''
    # Get the coordinates of the first bounding box.
    bbox1_min = np.asarray(bbox1.get_min_bound()) - padding
    bbox1_max = np.asarray(bbox1.get_max_bound()) + padding
    # Get the coordinates of the second bounding box.
    bbox2_min = np.asarray(bbox2.get_min_bound()) - padding
    bbox2_max = np.asarray(bbox2.get_max_bound()) + padding
    # Compute the overlap between the two bounding boxes.
    overlap_min = np.maximum(bbox1_min, bbox2_min)
    overlap_max = np.minimum(bbox1_max, bbox2_max)
    overlap_size = np.maximum(overlap_max - overlap_min, 0.0)
    overlap_volume = np.prod(overlap_size)
    bbox1_volume = np.prod(bbox1_max - bbox1_min)
    bbox2_volume = np.prod(bbox2_max - bbox2_min)
    obj_1_overlap = overlap_volume / bbox1_volume
    obj_2_overlap = overlap_volume / bbox2_volume
    max_overlap = max(obj_1_overlap, obj_2_overlap)
    iou = overlap_volume / (bbox1_volume + bbox2_volume - overlap_volume)
    if use_iou:
        return iou
    else:
        return max_overlap


def compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    Compute pairwise object overlap using nearest-neighbor points. Assume a list of n point clouds,
    where each point cloud is an o3d.geometry.PointCloud object. Build an n-by-n matrix whose
    (i, j) entry is the ratio of points in point cloud i that are within the distance threshold
    of any point in point cloud j.
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    # Convert point clouds to NumPy arrays and then FAISS indices for efficient searching.
    point_arrays = [np.asarray(obj['pcd'].points, dtype=np.float32) for obj in objects]
    indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in point_arrays]
    # Add points from each NumPy array to its corresponding FAISS index.
    for index, arr in zip(indices, point_arrays):
        index.add(arr)
    # Compute pairwise overlap.
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal entries.
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                # Skip completely non-overlapping boxes to save computation.
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    continue
                # Use range_search to find points within the threshold.
                # _, I = indices[j].range_search(point_arrays[i], threshold ** 2)
                D, I = indices[j].search(point_arrays[i], 1)
                # Increase the overlap count for points found within the threshold.
                # overlap += sum([len(i) for i in I])
                overlap = (D < cfg.voxel_size ** 2).sum() # D contains squared distances.
                # Compute the ratio of points within the threshold.
                overlap_matrix[i, j] = overlap / len(point_arrays[i])
    return overlap_matrix


def to_numpy(tensor):
    '''
    Convert to a NumPy array.
    '''
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    '''
    Convert to a tensor.
    '''
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    '''
    Perform final post-processing by merging overlapping objects.
    '''
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]
    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]
    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        ft_sim = F.cosine_similarity(
            to_tensor(objects[i]['ft']),
            to_tensor(objects[j]['ft']),
            dim=0
        )
        if ratio > cfg.merge_overlap_thresh and ft_sim > cfg.merge_ft_thresh:
                if kept_objects[j]:
                    # Merge object i into object j.
                    from utils.merge import merge_obj2_into_obj1
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i])
                    kept_objects[i] = False
        else:
            break
    # Remove merged objects.
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    return objects

def merge_objects(cfg, objects: MapObjectList):
    '''
    Post-process by performing a final merge of highly overlapping objects.
    '''
    if cfg.merge_final:
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before final map fusion:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After Final Map Fusion:", len(objects))
    return objects


def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    '''
    Transform a tensor point cloud into global coordinates.
    '''
    transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = past_point_clouds.shape[0]
    xyz1 = torch.hstack([past_point_clouds, torch.ones(NP, 1)]).T
    past_point_clouds = (transformation @ xyz1).T[:, :3]
    return past_point_clouds


def timestamp_tensor(tensor, time):
    '''
    Append time as an additional column for determining whether points are dynamic.
    '''
    n_points = tensor.shape[0]
    time = time * torch.ones((n_points, 1))
    timestamped_tensor = torch.hstack([tensor, time])
    return timestamped_tensor


def accumulate_pc(cfg, mos_model, pc, pose, his_pcs, his_poses):
    '''
    Accumulate the current point cloud and pose with historical point clouds and poses.
    '''
    # Discard intensity values.
    pc = pc[:,:3]
    his_pcs = [arr[:,:3] for arr in his_pcs]
    # all_pcs and all_poses use reverse chronological order [9, 8, 7, ..., 0], where 9 is the current frame.
    all_pcs = []
    all_poses = []
    # Insert the current point cloud and pose at the beginning of the lists.
    all_pcs.insert(0, pc)
    all_pcs.extend(his_pcs)
    all_poses.insert(0, pose)
    all_poses.extend(his_poses)
    if bool(cfg.get("filter_dynamic", False)):
        # his_pcs and his_poses use chronological order [0, 1, 2, ..., 9], where 9 is the current frame.
        his_pcs_copy = all_pcs[:]
        his_poses_copy = all_poses[:]
        his_pcs_copy.reverse()
        his_poses_copy.reverse()
        his_pcs_copy = [torch.tensor(arr) for arr in his_pcs_copy]
        list_his_pcs = his_pcs_copy
        # Align the poses.
        inv_frame0 = np.linalg.inv(his_poses_copy[0])
        new_poses = []
        for pose in his_poses_copy:
            new_poses.append(inv_frame0.dot(pose))
        poses = np.array(new_poses)
        # Compute dynamic objects across the ten most recent point-cloud frames.
        for i, pcd in enumerate(list_his_pcs):
            from_pose = poses[i]
            to_pose = poses[-1]
            pcd = transform_point_cloud(pcd, from_pose, to_pose)
            time_index = i - cfg.stride + 1
            timestamp = round(time_index * 0.1, 3)
            list_his_pcs[i] = timestamp_tensor(pcd, timestamp)
        past_point_clouds = torch.cat(list_his_pcs, dim=0)
        past_point_clouds = past_point_clouds.to('cuda')
        past_point_clouds_list = []
        past_point_clouds_list.append(past_point_clouds)
        out = mos_model.forward(past_point_clouds_list)
        for step in range(cfg.stride):
            coords = out.coordinates_at(0)
            logits = out.features_at(0)
            t = round(-step * 0.1, 3)
            mask = coords[:, -1].isclose(torch.tensor(t))
            masked_logits = logits[mask]
            masked_logits[:, [0]] = -float("inf")
            pred_softmax = F.softmax(masked_logits, dim=1)
            pred_softmax = pred_softmax.detach().cpu().numpy()
            assert pred_softmax.shape[1] == 3
            assert pred_softmax.shape[0] >= 0
            sum = np.sum(pred_softmax[:, 1:3], axis=1)
            assert np.isclose(sum, np.ones_like(sum)).all()
            moving_confidence = pred_softmax[:, 2]
            # colors = np.zeros((all_pcs[step].shape[0], 3))
            # moving_mask = moving_confidence > cfg.moving_thre
            # colors[moving_mask] = [1, 0, 0]  # Set moving points to red
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(all_pcs[step])
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd])
            # Determine dynamic objects according to the threshold.
            moving_mask = moving_confidence < cfg.moving_thre
            all_pcs[step] = all_pcs[step][moving_mask]
    pose_inv = np.linalg.inv(all_poses[0])
    for i in range(len(all_poses)):
        if i == 0:
            accumulate_pcs = all_pcs[0]
        else:
            # Compute the relative pose.
            pose_rel = np.dot(pose_inv, all_poses[i])
            # Append a column to convert point-cloud coordinates to homogeneous coordinates.
            homogeneous_points = np.column_stack((all_pcs[i], np.ones(all_pcs[i].shape[0])))
            transformed_points = np.dot(homogeneous_points, pose_rel.T)
            # Remove the final column to obtain the transformed point-cloud coordinates.
            transformed_points = transformed_points[:, :3]
            accumulate_pcs = np.vstack((accumulate_pcs, transformed_points))
    return accumulate_pcs


def distance_filter(max_depth, pc):
    '''
    Filter out point-cloud points with excessive depth.
    '''
    # Compute the distance of each point.
    distances = np.linalg.norm(pc, axis=1)
    # Keep points whose distance is within max_depth.
    filtered_points = pc[distances <= max_depth]
    return filtered_points


def caption_extract(idx, spacy_nlp, caption_ori):
    # Process the sentence with spaCy.
    doc = spacy_nlp(str(caption_ori))
    tokens = [token.text for token in doc]
    main_noun = "none"
    main_adj = []
    extra_captions = []

    # Extract nouns.
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    # Extract adjectives.
    adjectives = [token.text for token in doc if token.pos_ == "ADJ"]

    # Use the first noun as the keyword, excluding ambiguous concepts in CONFUSED_NOUNS.
    for i in range(len(nouns)):
        if nouns[i] not in CONFUSED_NOUNS:
            main_noun = nouns[i]
            break
    for token in tokens:
        if token in INTEREST_NOUNS:
            main_noun = token
            break

    # Record the token index of the extracted subject.
    main_noun_idx = tokens.index(main_noun)

    # Keep only adjectives that appear before the subject.
    for adj in adjectives:
        if (tokens.index(adj) < main_noun_idx) and (len(main_adj)<2):
            main_adj += [adj]

    # If no valid adjective was extracted, check for missed adjectives of interest.
    if not main_adj:
        for token in tokens:
            if token in INTEREST_ADJS and (len(main_adj)<2) and (tokens.index(token) < main_noun_idx):
                main_adj += [token]


    extra_captions = main_adj + [main_noun]
    extra_captions = " ".join(extra_captions)
    return extra_captions

def class_objects(cfg, sbert_model, objects: MapObjectList, bg_objects: MapObjectList, generator):
    '''
    Classify objects into SemanticKITTI categories using captions and features, then set inst_color.
    '''
    # Load the semantic-class and color file.
    file_path = cfg.class_colors_json
    with open(file_path, 'r') as json_file:
        class_colors_sk_disk = json.load(json_file)
        class_names_sk = list(class_colors_sk_disk.keys())
        class_colors_sk = list(class_colors_sk_disk.values())
        class_colors_sk = [list(map(lambda x: x / 255.0 if isinstance(x, (int, float)) else x, color)) for color in class_colors_sk]
    if cfg.class_methods == "sbert" or cfg.class_methods == "llama" or cfg.class_methods == "chatgpt":
        # Compute features for all classes for SBERT and as a fallback for invalid Llama or GPT output.
        class_name_fts = None
        for class_name in class_names_sk:
            class_name_ft = sbert_model.encode(class_name, convert_to_tensor=True)
            class_name_ft = class_name_ft / class_name_ft.norm(dim=-1, keepdim=True)
            class_name_ft = class_name_ft.squeeze()
            if class_name_fts is None:
                class_name_fts = class_name_ft
            else:
                class_name_fts = torch.vstack((class_name_fts,class_name_ft))
    if cfg.class_methods == "sbert":
        # Optionally parse captions with spaCy before evaluating similarity.
        if cfg.spacy:
            # Load the English model.
            spacy_nlp = spacy.load("en_core_web_sm")
            print("Spacy English loaded successfully! Ready for caption extraction!")
        # Find the most similar semantic class for each object and record its color.
        for i in range(len(objects)): #(trange)
            # First compute features from the final caption.
            caption = objects[i]['caption']
            if cfg.spacy:
                caption = caption_extract(i, spacy_nlp, caption)
            caption_only_ft = sbert_model.encode(caption, convert_to_tensor=True)
            caption_only_ft = caption_only_ft / caption_only_ft.norm(dim=-1, keepdim=True)
            caption_only_ft = caption_only_ft.squeeze()
            # Then use the fused caption features.
            objects_sbert_fts = objects[i]["ft"]
            objects_sbert_fts = objects_sbert_fts.to("cuda")
            # Combine the two using weighted fusion.
            final_ft = caption_only_ft*cfg.vis_caption_weight+objects_sbert_fts*cfg.vis_ft_weight
            # Compute similarity with each class.
            similarities = F.cosine_similarity(class_name_fts, final_ft.unsqueeze(0), dim=-1)
            if cfg.spacy and cfg.caption_only:
                similarities = F.cosine_similarity(class_name_fts, caption_only_ft.unsqueeze(0), dim=-1)
            max_indices = torch.argmax(similarities)
            # Set the class and color.
            objects[i]['class_sk'] = class_names_sk[max_indices]
            objects[i]['inst_color'] = class_colors_sk[max_indices]
        if bg_objects is not None:
            for i in range(len(bg_objects)):#trange
                # First compute features from the final caption.
                caption = bg_objects[i]['caption']
                if cfg.spacy:
                    caption = caption_extract(i, spacy_nlp, caption)
                caption_only_ft = sbert_model.encode(caption, convert_to_tensor=True)
                caption_only_ft = caption_only_ft / caption_only_ft.norm(dim=-1, keepdim=True)
                caption_only_ft = caption_only_ft.squeeze()
                # Then use the fused caption features.
                objects_sbert_fts = bg_objects[i]["ft"]
                objects_sbert_fts = objects_sbert_fts.to("cuda")
                # Combine the two using weighted fusion.
                final_ft = caption_only_ft*0.5+objects_sbert_fts*0.5
                # Compute similarity with each class.
                similarities = F.cosine_similarity(class_name_fts, final_ft.unsqueeze(0), dim=-1)
                if cfg.spacy and cfg.caption_only:
                    similarities = F.cosine_similarity(class_name_fts, caption_only_ft.unsqueeze(0), dim=-1)
                max_indices = torch.argmax(similarities)
                # Set the class and color.
                bg_objects[i]['class_sk'] = class_names_sk[max_indices]
                bg_objects[i]['inst_color'] = class_colors_sk[max_indices]
    elif cfg.class_methods == "llama":
        # Prompt examples used for demonstration.
        caption_example1 = "a car parked on the street"
        caption_example2 = "a red and white sign"
        caption_example3 = "grass on the side of the road"
        caption_example4 = "a sign on a pole"
        DEFAULT_PROMPT = """
        You are a classifier that can categorize a caption phrase into one of the following categories based on a caption phrase.
        List of categories: [car, bicycle, motorcycle, truck, person, bicyclist, motorcyclist, road,
        parking, sidewalk, building, fence, vegetation, trunk, terrain, pole, traffic-sign].
        You only need to generate one category name which must be included in this list.
        The output format is 'Category name: [[your summarized category name itself]]'
        Emphasizing again: Do not provide words beyond the given list!!! Please test it yourself and regenerate it if it exceeds the list.
        """
        for i in trange(len(objects)):
            caption_obj = objects[i]["caption"]
            # Generate the Llama dialog.
            dialogs: List[Dialog] = [
                [{"role": "system",
                "content": DEFAULT_PROMPT}
                ,{"role": "user", "content": caption_example1}
                ,{"role": "assistant", "content": "Category name: [car]"}
                ,{"role": "user", "content": caption_example2}
                ,{"role": "assistant", "content": "Category name: [traffic-sign]"}
                ,{"role": "user", "content": caption_example3}
                ,{"role": "assistant", "content": "Category name: [terrain]"}
                ,{"role": "user", "content": caption_example4}
                ,{"role": "assistant", "content": "Category name: [traffic-sign]"}
                ,{"role": "user", "content": caption_obj}],
            ]
            # Generate the Llama response.
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len= None,
                temperature=0.6,
                top_p=0.9,
            )
            # Read generation content from the Llama response as the fused caption result.
            for dialog, result in zip(dialogs, results):
                input_text = result["generation"]["content"]
                pattern = r'\[([^]]+)\]'  # Match content inside square brackets.
                match = re.search(pattern, input_text)
                extracted_content = []
                if match:
                    extracted_content = match.group(1)
            # If the Llama-generated class is not in the given list, match it using SBERT features.
            if extracted_content not in class_colors_sk_disk:
                extracted_content_ft = sbert_model.encode(extracted_content, convert_to_tensor=True)
                extracted_content_ft = extracted_content_ft / extracted_content_ft.norm(dim=-1, keepdim=True)
                extracted_content_ft = extracted_content_ft.squeeze()
                # Compute similarity with each class.
                similarities = F.cosine_similarity(class_name_fts, extracted_content_ft.unsqueeze(0), dim=-1)
                max_indices = torch.argmax(similarities)
                # Set the class and color.
                objects[i]['class_sk'] = class_names_sk[max_indices]
                objects[i]['inst_color'] = class_colors_sk[max_indices]
            else:
                objects[i]["class_sk"] = extracted_content
                objects[i]['inst_color'] = np.array(class_colors_sk_disk[extracted_content])/255.0
        if bg_objects is not None:
            for i in trange(len(bg_objects)):
                caption_obj = bg_objects[i]["caption"]
                # Generate the Llama dialog.
                dialogs: List[Dialog] = [
                    [{"role": "system",
                    "content": DEFAULT_PROMPT}
                    ,{"role": "user", "content": caption_example1}
                    ,{"role": "assistant", "content": "Category name: [car]"}
                    ,{"role": "user", "content": caption_example2}
                    ,{"role": "assistant", "content": "Category name: [traffic-sign]"}
                    ,{"role": "user", "content": caption_example3}
                    ,{"role": "assistant", "content": "Category name: [terrain]"}
                    ,{"role": "user", "content": caption_example4}
                    ,{"role": "assistant", "content": "Category name: [traffic-sign]"}
                    ,{"role": "user", "content": caption_obj}],
                ]
                # Generate the Llama response.
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len= None,
                    temperature=0.6,
                    top_p=0.9,
                )
                # Read generation content from the Llama response as the fused caption result.
                for dialog, result in zip(dialogs, results):
                    input_text = result["generation"]["content"]
                    pattern = r'\[([^]]+)\]'  # Match content inside square brackets.
                    match = re.search(pattern, input_text)
                    extracted_content = []
                    if match:
                        extracted_content = match.group(1)
                # If the Llama-generated class is not in the given list, match it using SBERT features.
                if extracted_content not in class_colors_sk_disk:
                    extracted_content_ft = sbert_model.encode(extracted_content, convert_to_tensor=True)
                    extracted_content_ft = extracted_content_ft / extracted_content_ft.norm(dim=-1, keepdim=True)
                    extracted_content_ft = extracted_content_ft.squeeze()
                    # Compute similarity with each class.
                    similarities = F.cosine_similarity(class_name_fts, extracted_content_ft.unsqueeze(0), dim=-1)
                    max_indices = torch.argmax(similarities)
                    # Set the class and color.
                    bg_objects[i]['class_sk'] = class_names_sk[max_indices]
                    bg_objects[i]['inst_color'] = class_colors_sk[max_indices]
                else:
                    bg_objects[i]["class_sk"] = extracted_content
                    bg_objects[i]['inst_color'] = np.array(class_colors_sk_disk[extracted_content])/255.0
    elif cfg.class_methods == "chatgpt":
        print("Asking gpt for class")
        client = OpenAI()
        TIMEOUT = 25  # timeout in seconds
        DEFAULT_PROMPT = """
        You are a classifier that can categorize a caption phrase into one of the following categories based on a caption phrase.
        List of categories: [car, bicycle, motorcycle, truck, person, bicyclist, motorcyclist, road,
        parking, sidewalk, building, fence, vegetation, trunk, terrain, pole, traffic-sign]
        . You only need to generate one category name which must be included in this list.
        The output format is 'Category name: [[your summarized category name itself]]'
        Note that I may enter all the captions at the same time, please output them in order, the number of your generated category name MUST be same as the number of captions!!!
        Here's an example for you.
        Input:
        'a car parked on the street'
        'a red and white sign'
        'grass on the side of the road'.
        You should output like this:
        'Category name: [car]
        Category name: [traffic-sign]
        Category name: [terrain]
        '
        Make sure that the number of category names you output matches the number of captions; otherwise, regenerate them.
        """
        caption_objects = objects.get_stacked_str_torch("caption")
        batch_size = cfg.gpt_max_num
        num_batches = len(caption_objects) // batch_size + (len(caption_objects) % batch_size > 0)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            current_caption_batch = caption_objects[start_idx:end_idx]
            caption_obj_batch = '\n'.join(current_caption_batch)

            chat_completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + caption_obj_batch}],
                    timeout=TIMEOUT,  # Timeout in seconds
                    )

            input_text_batch = chat_completion.choices[0].message.content #chat_completion["choices"][0]["message"]["content"]
            input_text_batch = input_text_batch.split('\n')
            extracted_contents_batch = [re.search(r'\[([^]]+)\]', result).group(1) for result in input_text_batch if re.search(r'\[([^]]+)\]', result)]

            regenerated_time = 0
            while len(extracted_contents_batch) != len(current_caption_batch):
                print(f"Missing captions, regenerate {regenerated_time} times")
                PROMPT = """Your generated category names do not match the number of captions. Please regenerate them again until their numbers are the same."""
                chat_completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + caption_obj_batch + "\n\n" + PROMPT}],
                    timeout=TIMEOUT,  # Timeout in seconds
                    )
                input_text_batch = chat_completion.choices[0].message.content
                input_text_batch = input_text_batch.split('\n')
                extracted_contents_batch = [re.search(r'\[([^]]+)\]', result).group(1) for result in input_text_batch if re.search(r'\[([^]]+)\]', result)]
                regenerated_time += 1

            for i in range(len(current_caption_batch)):
                # If the GPT-generated class is not in the given list, match it using SBERT features.
                extracted_content = extracted_contents_batch[i]
                if extracted_content not in class_colors_sk_disk:
                    extracted_content_ft = sbert_model.encode(extracted_content, convert_to_tensor=True)
                    extracted_content_ft = extracted_content_ft / extracted_content_ft.norm(dim=-1, keepdim=True)
                    extracted_content_ft = extracted_content_ft.squeeze()
                    # Compute similarity with each class.
                    similarities = F.cosine_similarity(class_name_fts, extracted_content_ft.unsqueeze(0), dim=-1)
                    max_indices = torch.argmax(similarities)
                    # Set the class and color.
                    objects[start_idx+i]['class_sk'] = class_names_sk[max_indices]
                    objects[start_idx+i]['inst_color'] = class_colors_sk[max_indices]
                else:
                    objects[start_idx+i]["class_sk"] = extracted_content
                    objects[start_idx+i]['inst_color'] = np.array(class_colors_sk_disk[extracted_content])/255.0
        if bg_objects is not None:
            caption_obj = bg_objects.get_stacked_str_torch("caption")
            caption_obj = '\n'.join(caption_obj)
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": DEFAULT_PROMPT + "\n\n" + caption_obj}],
                timeout=TIMEOUT,  # Timeout in seconds
            )
            input_text = chat_completion.choices[0].message.content #chat_completion["choices"][0]["message"]["content"]
            input_text = input_text.split('\n')
            extracted_contents = [re.search(r'\[([^]]+)\]', result).group(1) for result in input_text if re.search(r'\[([^]]+)\]', result)]
            for i in range(len(bg_objects)):
                bg_objects[i]["class_sk"] = extracted_contents[i]
                bg_objects[i]['inst_color'] = np.array(class_colors_sk_disk[extracted_contents[i]])/255.0
    else:
        raise NotImplementedError
    return objects, bg_objects




def show_captions(objects: MapObjectList, bg_objects: MapObjectList):
    '''
    Display captions associated with objects for debugging.
    '''
    for i in range(len(objects)):
        caption_obj = objects[i]["caption"]
        class_obj = objects[i]["class_sk"]
        print(f"object id {i} capitons: {caption_obj} ******** class_name: {class_obj}")
    if bg_objects is not None:
        for i in range(len(bg_objects)):
            caption_obj = bg_objects[i]["caption"]
            class_obj = bg_objects[i]["class_sk"]
            print(f"bgobject id {i} capitons: {caption_obj} ******** class_name: {class_obj}")

def get_observation_by_window(observation):
    color = observation[-1][1]
    pointCloud = observation[-1][2]
    pose = observation[-1][3]
    timestamp = observation[-1][4]

    his_pointCloud, his_pose = [], []
    for i in range(len(observation)-1):
        his_index = len(observation)-i-1
        his_pointCloud.append(observation[his_index][2])
        his_pose.append(observation[his_index][3])

    return (
        color,
        pointCloud,
        pose,
        his_pointCloud,
        his_pose,
        timestamp
    )

def get_observation(stride, observation):
    '''
    retrieve the data from the dataset, including the image, point cloud, pose, and history
    '''
    color = observation[-1][1]
    pointCloud = observation[-1][2]
    pose = observation[-1][3]

    all_pc = observation
    all_poses = observation
    # Overlapping projection of historical frames

    his_pointCloud = []
    his_pose = []
    # if False:
    for i in range(stride-1):
        his_index = stride-i-1
        his_pointCloud.append(all_pc[his_index][2])
        his_pose.append(all_poses[his_index][3])

    return (
        color,
        pointCloud,
        pose,
        his_pointCloud,
        his_pose
    )

def load_calib(calib_source):
    '''
    Load LiDAR-to-camera projection calibration.

    Preferred config keys:
      - camera_projection_matrix: 3x4 camera projection matrix
      - lidar_to_camera_matrix: 4x4 transform from LiDAR frame to RGB camera frame

    A calibration file path is still supported for older configs.
    '''
    calib = {}
    if isinstance(calib_source, (dict,)):
        cfg = calib_source
    elif hasattr(calib_source, "keys"):
        cfg = calib_source
    else:
        cfg = None

    if cfg is not None:
        projection_cfg = cfg.get("camera_projection_matrix", cfg.get("projection_matrix"))
        lidar_to_camera_cfg = cfg.get("lidar_to_camera_matrix", cfg.get("cam2velo_matrix"))
        if projection_cfg is not None and lidar_to_camera_cfg is not None:
            camera_projection_matrix = np.asarray(projection_cfg, dtype=np.float64)
            lidar_to_camera_matrix = np.asarray(lidar_to_camera_cfg, dtype=np.float64)
            if camera_projection_matrix.shape != (3, 4):
                raise ValueError(
                    "camera_projection_matrix must be 3x4, "
                    f"got {camera_projection_matrix.shape}."
                )
            if lidar_to_camera_matrix.shape != (4, 4):
                raise ValueError(
                    "lidar_to_camera_matrix must be 4x4, "
                    f"got {lidar_to_camera_matrix.shape}."
                )
            calib["camera_projection_matrix"] = camera_projection_matrix
            calib["lidar_to_camera_matrix"] = lidar_to_camera_matrix
            # Backward-compatible aliases for older helper code.
            calib["P_rect_20"] = camera_projection_matrix
            calib["T_cam2_velo"] = lidar_to_camera_matrix
            print("[calibration] Loaded camera_projection_matrix and lidar_to_camera_matrix from config.")
            return calib

        calib_path = cfg.get("calib_path")
        if calib_path is None:
            raise ValueError(
                "Missing calibration. Provide camera_projection_matrix + "
                "lidar_to_camera_matrix, or provide calib_path."
            )
    else:
        calib_path = calib_source

    print(f"Loading calibration from {calib_path}...")
    with open(calib_path, "r") as calib_file:
        calib_lines = calib_file.readlines()
        # Load the camera intrinsic matrix.
        P_rect_line = calib_lines[2]
        P_rect_02 = np.array(list(map(float, P_rect_line.strip().split()[1:]))).reshape(3, 4)
        calib["camera_projection_matrix"] = P_rect_02
        calib["P_rect_20"] = P_rect_02
        # Load the camera extrinsic matrix.
        Tr_line = calib_lines[4]
        Tr = np.array(list(map(float, Tr_line.strip().split()[1:]))).reshape(3, 4)
        Tr = np.vstack([Tr, [0, 0, 0, 1]])
        calib["lidar_to_camera_matrix"] = Tr
        calib["T_cam2_velo"] = Tr
    return calib

def from_intrinsics_matrix(K: np.ndarray) -> tuple[float, float, float, float]:
    '''
    Get fx, fy, cx, cy from the intrinsics matrix

    return 4 scalars
    '''
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    return fx, fy, cx, cy

import numpy as np
import open3d as o3d

def create_object_pcd_with_extrinsics(
    depth_array,                  # H_d x W_d, depth in meters, converted before input if necessary.
    mask,                         # Same resolution as depth_array.
    K_d,                          # Depth-camera intrinsics [fx fy cx cy] or 3x3.
    image_rgb,                    # H_c x W_c x 3, uint8
    K_c,                          # Color-camera intrinsics [fx fy cx cy] or 3x3.
    T_c_from_d,                   # 4x4 homogeneous transform from depth-camera to color-camera coordinates.
) -> o3d.geometry.PointCloud:
    fx_d, fy_d, cx_d, cy_d = K_d[0,0], K_d[1,1], K_d[0,2], K_d[1,2]
    fx_c, fy_c, cx_c, cy_c = K_c[0,0], K_c[1,1], K_c[0,2], K_c[1,2]

    H_d, W_d = depth_array.shape
    H_c, W_c, _ = image_rgb.shape

    # Valid-depth mask.
    valid = np.isfinite(depth_array) & (depth_array > 0)
    if mask is not None:
        valid = np.logical_and(valid, mask)

    if valid.sum() == 0:
        return o3d.geometry.PointCloud()

    # Pixel grid.
    u = np.arange(W_d, dtype=np.float32) # + 0.5
    v = np.arange(H_d, dtype=np.float32) # + 0.5
    uu, vv = np.meshgrid(u, v)   # H_d x W_d

    z = depth_array[valid]                         # (N,)
    u_valid = uu[valid]
    v_valid = vv[valid]

    # 1) Back-project into depth-camera coordinates using the depth-camera intrinsics.
    Xd = (u_valid - cx_d) * z / fx_d
    Yd = (v_valid - cy_d) * z / fy_d
    Zd = z
    Pd = np.stack([Xd, Yd, Zd, np.ones_like(Zd)], axis=0)  # 4 x N

    # 2) Transform points into color-camera coordinates using the extrinsics.
    Pc = T_c_from_d @ Pd                                   # 4 x N
    Xc, Yc, Zc = Pc[0, :], Pc[1, :], Pc[2, :]

    # 3) Perspective-project onto the color-image plane in pixel coordinates.
    #    Note: This is valid only for points with Zc > 0.
    positive = Zc > 0
    Xc, Yc, Zc = Xc[positive], Yc[positive], Zc[positive]
    du, dv = -3.0, 0.0 # -3.0 0
    u_c = (fx_c * Xc / Zc) + cx_c + du
    v_c = (fy_c * Yc / Zc) + cy_c + dv

    # 4) Discard points projected outside the color image.
    u_round = np.rint(u_c).astype(np.int32)
    v_round = np.rint(v_c).astype(np.int32)
    in_img = (u_round >= 0) & (u_round < W_c) & (v_round >= 0) & (v_round < H_c)

    u_round = u_round[in_img]
    v_round = v_round[in_img]

    # Corresponding 3D points, represented in either depth- or color-camera coordinates.
    # Return the point cloud in color-camera coordinates. To return depth-camera coordinates,
    # apply the same filtering to the earlier Xd, Yd, and Zd values.
    Xc, Yc, Zc = Xc[in_img], Yc[in_img], Zc[in_img]
    points = np.stack([Xc, Yc, Zc], axis=1)  # (M, 3)
    colors = image_rgb[v_round, u_round, :].astype(np.float32) / 255.0  # (M,3)

    # height_mask = points[:, 2] <= 4.0
    # points = points[height_mask]
    # colors = colors[height_mask]

    # Build the point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def visualize_depth_on_rgb_new(
    depth_m,           # H_d x W_d, depth in meters; 0 or NaN is invalid.
    K_d,               # Depth-camera intrinsic matrix, 3x3.
    image_rgb,         # H_c x W_c x 3, uint8, processed as RGB.
    K_c,               # Color-camera intrinsic matrix, 3x3.
    T_c_from_d,        # 4x4 transform from depth-camera to color-camera coordinates.
    depth_clip=(0.2, 5.0),
    alpha=0.6,
    delta_uv=(0.0, 0.0),         # Constant pixel-offset compensation (delta_u, delta_v).
    scale_crop=None,             # Optional (sx, sy, ox, oy) values for resizing or cropping.
    undistort_maps=None          # Optional (map1, map2) values for applying undistortion first.
):
    import numpy as np, cv2

    # ---------- 0) Optional undistortion; preferably undistort RGB and depth separately using their own K. ----------
    if undistort_maps is not None:
        map1, map2 = undistort_maps
        image_rgb = cv2.remap(image_rgb, map1, map2, interpolation=cv2.INTER_LINEAR)

    # ---------- 1) Parse and correct intrinsics. ----------
    # If resizing or cropping occurred, correct K_c using (sx, sy, ox, oy).
    Kc = K_c.copy()
    if scale_crop is not None:
        sx, sy, ox, oy = scale_crop  # sx, sy: scale factors; ox, oy: crop offsets in pixels.
        Kc[0,0] *= sx
        Kc[1,1] *= sy
        Kc[0,2] = sx * Kc[0,2] + ox
        Kc[1,2] = sy * Kc[1,2] + oy

    fx_d, fy_d, cx_d, cy_d = float(K_d[0,0]), float(K_d[1,1]), float(K_d[0,2]), float(K_d[1,2])
    fx_c, fy_c, cx_c, cy_c = float(Kc[0,0]), float(Kc[1,1]), float(Kc[0,2]), float(Kc[1,2])

    H_d, W_d = depth_m.shape
    H_c, W_c = image_rgb.shape[:2]

    # ---------- 2) Valid depth. ----------
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return image_rgb.copy()

    # Using pixel centers (+0.5) avoids a systematic half-pixel offset.
    u = np.arange(W_d, dtype=np.float32) # + 0.5
    v = np.arange(H_d, dtype=np.float32) # + 0.5
    uu, vv = np.meshgrid(u, v)  # H_d x W_d

    z = depth_m[valid].astype(np.float32)
    u_valid = uu[valid]
    v_valid = vv[valid]

    # ---------- 3) Back-project into depth-camera coordinates. ----------
    Xd = (u_valid - cx_d) * z / fx_d
    Yd = (v_valid - cy_d) * z / fy_d
    Zd = z
    ones = np.ones_like(Zd)
    Pd = np.stack([Xd, Yd, Zd, ones], axis=0)  # 4 x N

    # ---------- 4) Transform into color-camera coordinates using the extrinsics. ----------
    Pc = T_c_from_d @ Pd
    Xc, Yc, Zc = Pc[0], Pc[1], Pc[2]

    front = Zc > 0
    if not np.any(front):
        return image_rgb.copy()
    Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]

    # ---------- 5) Project onto the color image and apply constant pixel compensation. ----------
    du, dv = float(delta_uv[0]), float(delta_uv[1])  # Constant offset in pixels.
    u_c = fx_c * (Xc / Zc) + cx_c + du
    v_c = fy_c * (Yc / Zc) + cy_c + dv

    # Do not align to pixel centers here; these pixel coordinates are later rounded to integer indices.
    u_i = np.round(u_c).astype(np.int32)
    v_i = np.round(v_c).astype(np.int32)

    in_img = (u_i >= 0) & (u_i < W_c) & (v_i >= 0) & (v_i < H_c)
    if not np.any(in_img):
        return image_rgb.copy()
    u_i, v_i, Zc = u_i[in_img], v_i[in_img], Zc[in_img]

    # ---------- 6) Color mapping. ----------
    z_vis = np.clip(Zc, depth_clip[0], depth_clip[1])
    z_norm = (1.0 - (z_vis - depth_clip[0]) / (depth_clip[1] - depth_clip[0]))
    z_norm = (np.clip(z_norm, 0, 1) * 255).astype(np.uint8)
    cmap_colors = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)       # BGR
    cmap_colors = cv2.cvtColor(cmap_colors, cv2.COLOR_BGR2RGB)      # RGB
    cmap_colors = cmap_colors.reshape(-1, 3)

    # ---------- 7) Z-buffer: prioritize the nearest point. ----------
    zbuffer = np.full((H_c, W_c), np.inf, dtype=np.float32)
    overlay = np.zeros((H_c, W_c, 3), dtype=np.uint8)

    lin = v_i * W_c + u_i
    np.minimum.at(zbuffer.ravel(), lin, Zc)
    keep = Zc == zbuffer.ravel()[lin]
    if not np.any(keep):
        return image_rgb.copy()

    overlay[v_i[keep], u_i[keep]] = cmap_colors[keep]

    # ---------- 8) Overlay. ----------
    out = (alpha * overlay.astype(np.float32) + (1 - alpha) * image_rgb.astype(np.float32)).astype(np.uint8)
    return out


def visualize_depth_on_rgb(
    depth_m,           # H_d x W_d, depth in meters; 0 or NaN is invalid.
    K_d,               # Depth-camera intrinsics [fx, fy, cx, cy] or 3x3.
    image_rgb,         # H_c x W_c x 3, uint8, either BGR or RGB; processed below as RGB.
    K_c,               # Color-camera intrinsics [fx, fy, cx, cy] or 3x3.
    T_c_from_d,        # 4x4 transform from depth-camera to color-camera coordinates.
    depth_clip=(0.2, 5.0),   # Depth range in meters used for visualization color mapping.
    alpha=0.6               # Overlay opacity.
):
    # --- 1) Parse intrinsics. ---
    fx_d, fy_d, cx_d, cy_d = K_d[0,0], K_d[1,1], K_d[0,2], K_d[1,2]
    fx_c, fy_c, cx_c, cy_c = K_c[0,0], K_c[1,1], K_c[0,2], K_c[1,2]

    H_d, W_d = depth_m.shape
    H_c, W_c = image_rgb.shape[:2]

    # --- 2) Valid depth. ---
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return image_rgb.copy()  # Return the original image when there is no valid depth.

    u = np.arange(W_d, dtype=np.float32) # + 0.5
    v = np.arange(H_d, dtype=np.float32) # + 0.5
    uu, vv = np.meshgrid(u, v)            # H_d x W_d

    z = depth_m[valid].astype(np.float32)  # (N,)
    u_valid = uu[valid]
    v_valid = vv[valid]

    # --- 3) Back-project into depth-camera coordinates. ---
    Xd = (u_valid - cx_d) * z / fx_d
    Yd = (v_valid - cy_d) * z / fy_d
    Zd = z
    ones = np.ones_like(Zd)
    Pd = np.stack([Xd, Yd, Zd, ones], axis=0)  # 4 x N

    # --- 4) Transform into color-camera coordinates using the extrinsics. ---
    Pc = T_c_from_d @ Pd
    Xc, Yc, Zc = Pc[0], Pc[1], Pc[2]

    # Keep only points in front of the camera.
    front = Zc > 0
    if not np.any(front):
        return image_rgb.copy()
    Xc, Yc, Zc = Xc[front], Yc[front], Zc[front]

    # --- 5) Project onto the color image. ---
    du, dv = -3.0, 0 #float(delta_uv[0]), float(delta_uv[1])  # Constant offset in pixels.
    # u_c = fx_c * (Xc / Zc) + cx_c + du
    # v_c = fy_c * (Yc / Zc) + cy_c + dv

    u_c = fx_c * (Xc / Zc) + cx_c + du
    v_c = fy_c * (Yc / Zc) + cy_c + dv
    u_i = np.rint(u_c).astype(np.int32)
    v_i = np.rint(v_c).astype(np.int32)

    in_img = (u_i >= 0) & (u_i < W_c) & (v_i >= 0) & (v_i < H_c)
    if not np.any(in_img):
        return image_rgb.copy()

    u_i, v_i, Zc = u_i[in_img], v_i[in_img], Zc[in_img]

    # --- 6) Normalize depth and map colors for visualization. ---
    z_vis = np.clip(Zc, depth_clip[0], depth_clip[1])
    z_norm = (z_vis - depth_clip[0]) / (depth_clip[1] - depth_clip[0])  # 0..1
    z_norm = (z_norm * 255).astype(np.uint8)

    # Use an OpenCV colormap such as JET or Rainbow.
    cmap_colors = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)  # BGR
    cmap_colors = cv2.cvtColor(cmap_colors, cv2.COLOR_BGR2RGB) # Convert to RGB.
    cmap_colors = cmap_colors.reshape(-1, 3)

    # --- 7) Z-buffer: keep the nearest point at each pixel. ---
    # Initialize a minimum-depth image with all values set to +inf.
    zbuffer = np.full((H_c, W_c), np.inf, dtype=np.float32)
    overlay = np.zeros((H_c, W_c, 3), dtype=np.uint8)

    # Use linear indices to simplify atomic updates.
    lin = v_i * W_c + u_i

    # Select the minimum depth at each position.
    np.minimum.at(zbuffer.ravel(), lin, Zc)

    # Find pixels corresponding to these minimum depths, which are the nearest points.
    keep = Zc == zbuffer.ravel()[lin]
    if not np.any(keep):
        return image_rgb.copy()

    u_k, v_k = u_i[keep], v_i[keep]
    colors_k = cmap_colors[keep]  # (M, 3)

    overlay[v_k, u_k] = colors_k

    # --- 8) Overlay onto the original image. ---
    base = image_rgb.astype(np.float32)
    over = overlay.astype(np.float32)
    out = (alpha * over + (1 - alpha) * base).astype(np.uint8)
    return out


import numpy as np
import cv2

def undistort_fisheye_equidistant6(
    image,
    K_src,                 # Original fisheye-camera intrinsics, 3x3; converts theta_d to pixel radius.
    k6,                    # [k0..k5]: six coefficients mapping theta to theta_d using an odd polynomial.
    output_size=None,      # (w_out, h_out)
    K_out=None,            # Output camera intrinsics; generated from fov_out when omitted.
    fov_out_deg=100.0,     # Target field of view, using a diagonal approximation; used only when K_out is None.
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_CONSTANT
):
    """
    Rectify a six-parameter equidistant fisheye image into a rectilinear pinhole image.
    """
    h_src, w_src = image.shape[:2]
    if output_size is None:
        w_out, h_out = w_src, h_src
    else:
        w_out, h_out = output_size

    K_src = np.asarray(K_src, dtype=np.float64).reshape(3, 3)
    fx_src, fy_src = K_src[0, 0], K_src[1, 1]
    cx_src, cy_src = K_src[0, 2], K_src[1, 2]

    k6 = np.asarray(k6, dtype=np.float64).ravel()
    if k6.size < 6:
        k6 = np.pad(k6, (0, 6 - k6.size))
    k0, k1, k2, k3, k4, k5 = k6[:6]

    # Generate output camera intrinsics for the perspective model.
    if K_out is None:
        # Given a diagonal field of view, estimate focal length from the smaller dimension.
        diag = np.sqrt(w_out**2 + h_out**2)
        f = (diag / 2.0) / np.tan(np.deg2rad(fov_out_deg) / 2.0)
        fx_out = fy_out = f
        cx_out, cy_out = (w_out - 1) / 2.0, (h_out - 1) / 2.0
        K_out = np.array([[fx_out, 0, cx_out],
                          [0, fy_out, cy_out],
                          [0,     0,     1]], dtype=np.float64)
    else:
        K_out = np.asarray(K_out, dtype=np.float64).reshape(3, 3)

    fx_out, fy_out = K_out[0, 0], K_out[1, 1]
    cx_out, cy_out = K_out[0, 2], K_out[1, 2]

    # --- Build the output pixel grid for inverse mapping. ---
    u = np.arange(w_out, dtype=np.float64)
    v = np.arange(h_out, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # Shape: (h_out, w_out).

    # Convert normalized perspective coordinates to viewing directions.
    x = (uu - cx_out) / fx_out
    y = (vv - cy_out) / fy_out

    # Obtain the incidence angle from the perspective model: theta = arctan(r), r = sqrt(x^2 + y^2).
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)  # Angle relative to the optical axis.

    # Azimuth angle.
    phi = np.arctan2(y, x)  # [-pi, pi]

    # --- Six-parameter equidistant distortion: theta -> theta_d using an odd polynomial. ---
    t2 = theta * theta
    theta_d = (k0 * theta +
               k1 * theta * t2 +
               k2 * theta * t2 * t2 +
               k3 * theta * t2 * t2 * t2 +
               k4 * theta * t2 * t2 * t2 * t2 +
               k5 * theta * t2 * t2 * t2 * t2 * t2)

    # Equidistant projection: r_d = f * theta_d.
    rdx = fx_src * theta_d
    rdy = fy_src * theta_d

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Source-image coordinates from polar coordinates to pixels, using anisotropic scaling for fx and fy.
    map_u = (cx_src + rdx * cos_phi).astype(np.float32)
    map_v = (cy_src + rdy * sin_phi).astype(np.float32)

    # --- Remap sampling. ---
    undistorted = cv2.remap(image, map_u, map_v,
                            interpolation=interpolation,
                            borderMode=border_mode)
    return undistorted, K_out
