from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, List, Optional, Tuple, Dict
from langchain_core.documents import Document
import numpy as np

from .memory import Memory, MemoryItem
from .ib_geometry import FloorFilterConfig, filter_floor_primitives

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
import torch.nn.functional as F
from termcolor import colored
import distinctipy

from clio_batch.ib_cluster import ClusterIB, ClusterIBConfig
from clio_batch.aib_helper import visualize_highlighted_clusters_open3d, visualize_graph_highlight, visualize_graph_in_3d_scene

from utils.world_map import visualize_objects_org

FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970

import numpy as np
from pathlib import Path
import os

import networkx as nx
import torch
import open3d as o3d
from collections import Counter

# ---- Helper: get absolute seconds for a caption doc ----
def _caption_time_seconds(doc, ref_time):
    """Return absolute seconds for a caption doc (handles float or [t, ...])."""
    mt = doc.metadata.get('time', 0.0)
    t = mt[0] if isinstance(mt, (list, tuple)) else mt
    return float(t + (ref_time or 0.0))

def _summarize_omitted_times(docs, ref_time, max_list=6):
    """Make a compact string listing omitted caption times and span."""
    if not docs:
        return ""
    ts = sorted(_caption_time_seconds(d, ref_time) for d in docs)
    # Pretty list (limit)
    shown = ts[:max_list]
    more = len(ts) - len(shown)
    # Span
    span = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    # Render
    from time import localtime, strftime
    def fmt(sec): return strftime('%Y-%m-%d %H:%M:%S', localtime(sec))
    shown_str = ", ".join(fmt(s) for s in shown)
    # if more > 0:
    #     shown_str += f", +{more} more"
    return f"The scene area was also observed at the following times, feel free to check. {shown_str}."


def _to_unit(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.float()
    if x.ndim == 1: x = x[None, :]
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

def cluster_task_scores_cosine(objects, clusters, task_features, pool="max"):
    """
    objects: list[dict] each with 'ft' tensor/ndarray
    clusters: list[list[int]] (indices into `objects`)
    task_features: (T,D) numpy/torch (already SBERT)
    pool: 'max' (best member), 'mean', or 'topk-mean:k'
    returns: scores (C,), winners (C,) best task per cluster
    """
    T = _to_unit(task_features)  # (T,D)
    scores, winners = [], []

    for members in clusters:
        if not members:
            scores.append(0.0); winners.append(-1); continue
        F = _to_unit(torch.stack([_to_unit(objects[i]['ft']).squeeze(0) for i in members]))  # (M,D)
        S = (T @ F.T)  # (T, M)

        if pool == "mean":
            v = S.mean(dim=1)                  # (T,)
        elif pool.startswith("topk-mean:"):
            k = int(pool.split(":")[1])
            vals, _ = torch.topk(S, k=min(k, S.shape[1]), dim=1)
            v = vals.mean(dim=1)               # (T,)
        else:  # 'max'
            v, _ = S.max(dim=1)                # (T,)

        best_val, best_task = torch.max(v, dim=0)
        scores.append(float(best_val.item()))
        winners.append(int(best_task.item()))
    return np.array(scores), np.array(winners)


def aabb_overlaps(aabb_i: o3d.geometry.AxisAlignedBoundingBox,
                  aabb_j: o3d.geometry.AxisAlignedBoundingBox,
                  eps: float = 0.0) -> bool:
    """Return True if two AABBs overlap (with optional dilation eps)."""
    mi = np.asarray(aabb_i.get_min_bound())
    Mi = np.asarray(aabb_i.get_max_bound())
    mj = np.asarray(aabb_j.get_min_bound())
    Mj = np.asarray(aabb_j.get_max_bound())
    # overlap if intervals intersect on all 3 axes
    return np.all(mi <= Mj + eps) and np.all(mj <= Mi + eps)


def _bbox_center(bbox):
    """Return a center vector for either an Open3D AABB or OBB."""
    get_center = getattr(bbox, "get_center", None)
    return np.asarray(get_center() if callable(get_center) else bbox.center)


def _bbox_extent(bbox):
    """Return an extent vector for either an Open3D AABB or OBB."""
    get_extent = getattr(bbox, "get_extent", None)
    return np.asarray(get_extent() if callable(get_extent) else bbox.extent)


def _bbox_as_aabb(bbox):
    """Return an AABB without assuming the input box type."""
    if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
        return bbox
    return bbox.get_axis_aligned_bounding_box()

def build_object_graph(objects, region_features, eps=0.0):
    G = nx.Graph()

    for i, obj in enumerate(objects):
        G.add_node(
            i,
            position=_bbox_center(obj['bbox']),
            semantic_feature=region_features[i].reshape(-1, 1),
            bounding_box=obj['bbox'],
        )

    # O(N^2) pass; fine for moderate N
    for i in range(len(objects)):
        aabb_i = _bbox_as_aabb(objects[i]['bbox'])
        for j in range(i + 1, len(objects)):
            aabb_j = _bbox_as_aabb(objects[j]['bbox'])
            if aabb_overlaps(aabb_i, aabb_j, eps=eps):
                G.add_edge(i, j)

    return G

def _axis_index(axis):
    if isinstance(axis, int):
        return axis
    return {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

def _aabb_corners_from_obj(obj, dilate_eps=0.0):
    """Return (8,3) AABB corners; optionally dilate by eps (meters)."""
    aabb = _bbox_as_aabb(obj['bbox'])
    if dilate_eps > 0.0:
        mn = np.asarray(aabb.get_min_bound(), dtype=np.float32)
        mx = np.asarray(aabb.get_max_bound(), dtype=np.float32)
        ctr = 0.5 * (mn + mx)
        ext = 0.5 * (mx - mn) + dilate_eps
        aabb = o3d.geometry.AxisAlignedBoundingBox(ctr - ext, ctr + ext)
    return np.asarray(aabb.get_box_points(), dtype=np.float32)

def _vertical_overlap_mask(mn, mx, v_ax, use_overlap=True, slack=0.25):
    """(N,N) upper-tri mask for vertical overlap (or center diff <= slack)."""
    if use_overlap:
        vmin = mn[:, v_ax][:, None]; vmax = mx[:, v_ax][:, None]
        vmin2 = mn[:, v_ax][None, :]; vmax2 = mx[:, v_ax][None, :]
        inter_v = torch.minimum(vmax, vmax2) - torch.maximum(vmin, vmin2)
        M = (inter_v > 0)
    else:
        vc  = ((mn[:, v_ax] + mx[:, v_ax]) * 0.5)[:, None]
        vc2 = ((mn[:, v_ax] + mx[:, v_ax]) * 0.5)[None, :]
        M = (torch.abs(vc - vc2) <= slack)
    return torch.triu(M, diagonal=1)

def _ground_like_mask(mn, mx, v_ax, down_positive, height_thresh=0.05, floor_thresh=0.05):
    """
    Returns (N,) bool ground-ish: vertically thin and touching the scene floor.
    For down_positive=True (+Y is down): floor is at LARGE positive coordinate.
    For down_positive=False (usual +Z up): floor is at SMALL coordinate.
    """
    extent_v = (mx[:, v_ax] - mn[:, v_ax])              # thickness along vertical axis
    base_v = mx[:, v_ax] if down_positive else mn[:, v_ax]
    floor_v = torch.max(base_v) if down_positive else torch.min(base_v)
    is_thin  = extent_v < height_thresh
    is_at_floor = torch.abs(base_v - floor_v) <= floor_thresh
    return is_thin & is_at_floor


def build_object_graph_ground_safe(
    objects,
    region_features,
    *,
    vertical_axis='y',            # <-- your scene uses Y as vertical
    down_positive=True,           # <-- +Y is down toward the floor
    dist_radius=2.0,              # meters
    z_overlap=False,               # vertical interval overlap vs. slack
    z_slack=0.25,                 # meters, only used when z_overlap=False
    iou_thresh=0.02,              # small >0 to avoid floor-touch bridges
    covis_min=0,                  # require shared frames; 0 disables
    knn=6,                        # sparsify; None to add all gated edges
    ground_height_thresh=0.05,    # “thin” threshold on vertical extent
    ground_floor_thresh=0.05,     # floor contact threshold along vertical axis
    dilate_eps=0.02               # inflate all AABBs by a few cm (stability)
):
    """
    Build a sparse, well-gated object adjacency graph robust to Y-down frames.
    """
    G = nx.Graph()
    N = len(objects)
    v_ax = _axis_index(vertical_axis)

    # Add nodes
    for i, obj in enumerate(objects):
        G.add_node(
            i,
            position=_bbox_center(obj['bbox']),
            semantic_feature=region_features[i].reshape(-1, 1),
            bounding_box=obj['bbox'],
        )
    if N <= 1:
        return G

    # Gather (N,8,3) corners (optionally dilated) and image sets
    corners = np.stack([_aabb_corners_from_obj(o, dilate_eps=dilate_eps) for o in objects], axis=0)
    corners = torch.from_numpy(corners)  # (N,8,3) torch.float32
    img_sets = [set(o.get('image_idx', [])) for o in objects]

    # Min/max/centers
    mn = corners.min(dim=1).values                              # (N,3)
    mx = corners.max(dim=1).values                              # (N,3)
    centers = 0.5 * (mn + mx)                                   # (N,3)

    # Pairwise distance gating
    D = torch.cdist(centers, centers, p=2)                      # (N,N)
    dist_mask = torch.triu((D <= dist_radius), diagonal=1)

    # Vertical gating (using vertical_axis)
    v_mask = _vertical_overlap_mask(mn, mx, v_ax, use_overlap=z_overlap, slack=z_slack)

    # AABB IoU (vectorized)
    b1_min = mn[:, None, :]; b1_max = mx[:, None, :]
    b2_min = mn[None, :, :]; b2_max = mx[None, :, :]
    inter_min = torch.maximum(b1_min, b2_min)
    inter_max = torch.minimum(b1_max, b2_max)
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)
    vol1 = torch.prod(b1_max - b1_min, dim=2)
    vol2 = torch.prod(b2_max - b2_min, dim=2)
    union = vol1 + vol2 - inter_vol + 1e-10
    iou = inter_vol / union
    iou_mask = torch.triu((iou > iou_thresh), diagonal=1)

    # Co-visibility gating
    if covis_min > 0:
        covis = torch.zeros((N, N), dtype=torch.bool)
        for i in range(N):
            Si = img_sets[i]
            for j in range(i + 1, N):
                if len(Si & img_sets[j]) >= covis_min:
                    covis[i, j] = True
        covis_mask = covis
    else:
        covis_mask = torch.triu(torch.ones((N, N), dtype=torch.bool), diagonal=1)

    # Ground hygiene (block any pair where either is ground-like along vertical_axis)
    is_ground = _ground_like_mask(mn, mx, v_ax, down_positive,
                                  height_thresh=ground_height_thresh,
                                  floor_thresh=ground_floor_thresh)
    not_ground = ~is_ground
    ground_mask = torch.triu((not_ground[:, None] & not_ground[None, :]), diagonal=1)

    # Combine all gates
    gate = dist_mask & v_mask & iou_mask & covis_mask & ground_mask

    # Sparsify with kNN among gated neighbors (per row)
    edges = []
    if knn is None:
        ii, jj = torch.where(gate)
        edges = [(int(i), int(j)) for i, j in zip(ii, jj)]
    else:
        for i in range(N):
            js = torch.nonzero(gate[i], as_tuple=False).flatten()
            if js.numel() == 0:
                continue
            dij = D[i, js]
            k = min(knn, js.numel())
            chosen = js[torch.topk(-dij, k).indices]  # k smallest distances
            edges.extend([(i, int(j)) for j in chosen])

    G.add_edges_from(edges)
    return G

def write_default_cluster_config(cfg_path, overrides=None, overwrite=False):
    """Write the IB config, optionally replacing stale values for a debug run."""
    if os.path.exists(cfg_path) and not overwrite:
        return
    config = {
        "debug": False,
        "debug_folder": "./ib_debug",
        "sims_thres": 0.20,
        "delta": 0.10,
        "top_k_tasks": 1,
        "cumulative": False,
        "use_lerf_loss": False,
        "lerf_loss_cannonical_phrases": [],
    }
    if overrides:
        config.update(overrides)

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        f.write(
            f"debug: {str(bool(config['debug'])).lower()}\n"
            f"debug_folder: {config['debug_folder']}\n"
            f"sims_thres: {float(config['sims_thres'])}\n"
            f"delta: {float(config['delta'])}\n"
            f"top_k_tasks: {int(config['top_k_tasks'])}\n"
            f"cumulative: {str(bool(config['cumulative'])).lower()}\n"
            f"use_lerf_loss: {str(bool(config['use_lerf_loss'])).lower()}\n"
            "lerf_loss_cannonical_phrases: []\n"
        )


def run_ib_clustering(region_features, task_features, G_nx, cfg_path):
    """
    Run graph-constrained Agglomerative Information Bottleneck (AIB).

    Symbol mapping:
    - X: input objects/regions
    - Y: task labels (null-task + query/task prompts)
    - C: compressed cluster labels

    Inputs:
    - region_features: (N, D), one embedding per object x
    - task_features: (T, D), one embedding per task/query
    - G_nx: graph over objects; only graph-connected pairs may merge

    Internal probabilities built by the solver:
    - p(x): prior over objects, uniform here
    - p(y|x): task distribution for each object from cosine similarity
    - p(c|x): cluster assignment, initialized as one-hot identity
    - p(y|c): task distribution of each merged cluster

    Output:
    - list[list[int]] where each inner list is one final cluster of object ids
    """
    ib_cfg = ClusterIBConfig(cfg_path)
    ib = ClusterIB(ib_cfg)
    ib.setup_py_x(region_features, task_features)                 # p(x), p(y|x), p(y)
    ib.update_delta_as_part(region_features, task_features)       # scale for factorized case
    ib.initialize_nx_graph(G_nx)                                  # nodes/edges
    clusters = ib.find_clusters()                                  # list of lists (indices into objects)
    return clusters


def merge_cluster(objects, idxs):
    """Merge members into a single object dict (pcd, ft, caption, bbox, etc.)."""
    merged_pcd = o3d.geometry.PointCloud()
    fts, caps = [], []
    for k in idxs:
        merged_pcd += objects[k]['pcd']
        ft_k = objects[k]['ft']
        if torch.is_tensor(ft_k):
            ft_k = ft_k.detach().cpu().numpy()
        fts.append(np.asarray(ft_k, dtype=np.float32))
        caps.append(objects[k]['caption'])

    ft_mean = np.mean(fts, axis=0)
    norm = np.linalg.norm(ft_mean) + 1e-12
    ft_mean = (ft_mean / norm).astype(np.float32)

    caption = Counter(caps).most_common(1)[0][0]
    merged_bbox = merged_pcd.get_oriented_bounding_box()

    return {
        'image_idx': sum((objects[k]['image_idx'] for k in idxs), []),
        'num_detections': sum(objects[k]['num_detections'] for k in idxs),
        'n_points': np.asarray(merged_pcd.points).shape[0],
        'inst_color': objects[idxs[0]].get('inst_color', None),
        'bg_class': None,
        'class_sk': None,
        'caption': caption,
        'captions_ft': None,
        'ft': ft_mean,                  # merged SBERT embedding
        'pcd': merged_pcd,
        'bbox': merged_bbox,
        'img_bbox': None,
        'cluster_members': idxs,
    }


def merge_all_clusters(objects, clusters):
    return [merge_cluster(objects, c) for c in clusters]


def cosine_max_to_tasks(task_features, ft_vec):
    """Max cosine(task, vector)."""
    sims = task_features @ ft_vec.reshape(-1, 1)  # (M, 1)
    return float(sims.max())


# import numpy as np

def filter_clusters_by_task_topk(clustered_objects, task_features, k=6):
    """Keep top-k clusters whose embedding is most similar to the tasks."""
    # Compute similarities for each object
    sims = []
    for obj in clustered_objects:
        s = cosine_max_to_tasks(task_features, obj['ft'])
        sims.append((s, obj))
    
    # Sort by similarity (descending) and pick top k
    sims.sort(key=lambda x: x[0], reverse=True)
    kept = [obj for _, obj in sims[:k]]
    
    return kept


def filter_clusters_by_task(clustered_objects, task_features, thresh=0.23):
    """Keep clusters whose embedding is sufficiently similar to any task."""
    kept = []
    for obj in clustered_objects:
        s = cosine_max_to_tasks(task_features, obj['ft'])
        
        if s >= thresh:
            kept.append(obj)
    
    return kept

def obb_to_lineset(obb, color=(0, 0, 0)):
    """Convert an Open3D AABB/OBB using Open3D's canonical edge topology."""
    if isinstance(obb, o3d.geometry.OrientedBoundingBox):
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    elif isinstance(obb, o3d.geometry.AxisAlignedBoundingBox):
        line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(obb)
    elif hasattr(obb, "get_axis_aligned_bounding_box"):
        line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            obb.get_axis_aligned_bounding_box()
        )
    else:
        raise TypeError(f"Unsupported bounding-box type: {type(obb)}")

    line_set.paint_uniform_color(np.asarray(color, dtype=float))
    return line_set


def bbox_to_cylinder_meshes(bbox, color=(0, 0, 0), radius=0.03):
    """Render bbox edges as cylinders, matching scripts/vis_3D.py."""
    line_set = obb_to_lineset(bbox, color=color)
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    meshes = []

    for start_idx, end_idx in lines:
        start = points[start_idx]
        end = points[end_idx]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=float(radius),
            height=length,
        )
        cylinder.compute_vertex_normals()

        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        direction_unit = direction / length
        cross = np.cross(z_axis, direction_unit)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0))

        if cross_norm > 1e-8:
            axis = cross / cross_norm
            angle = float(np.arccos(dot))
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                axis * angle
            )
            cylinder.rotate(rotation, center=np.zeros(3))
        elif dot < 0.0:
            rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0]) * np.pi
            )
            cylinder.rotate(rotation, center=np.zeros(3))

        cylinder.translate((start + end) / 2.0)
        cylinder.paint_uniform_color(np.asarray(color, dtype=float))
        meshes.append(cylinder)

    return meshes

def visualize_clusters_open3d(
    objects,
    clusters,
    save_dir=None,
    show=True,
    highlight_idxs=None,
    cluster_scores=None,
    bbox_mode="obb",
    unrelated_color=(0.35, 0.35, 0.35),
    unrelated_point_scale=0.45,
    cluster_bbox_dark_factor=0.55,
    primitive_bbox_light_mix=0.55,
    bbox_line_radius=0.03,
    show_unrelated_bboxes=False,
    primitive_scores=None,
    primitive_task_threshold=0.40,
    hide_unrelated_primitives=False,
):
    """
    Color each cluster, draw its objects' PCDs + bboxes.
    objects[i] should have 'pcd' (o3d.geometry.PointCloud) and 'bbox' (Open3D OBB/AABB).
    """
    bbox_mode = str(bbox_mode).lower()
    if bbox_mode not in {"aabb", "obb"}:
        raise ValueError(f"Unsupported bbox_mode '{bbox_mode}'. Use 'aabb' or 'obb'.")
    if not 0.0 <= cluster_bbox_dark_factor <= 1.0:
        raise ValueError("cluster_bbox_dark_factor must be within [0, 1].")
    if not 0.0 <= primitive_bbox_light_mix <= 1.0:
        raise ValueError("primitive_bbox_light_mix must be within [0, 1].")
    if bbox_line_radius < 0.0:
        raise ValueError("bbox_line_radius must be non-negative.")
    if not -1.0 <= primitive_task_threshold <= 1.0:
        raise ValueError("primitive_task_threshold must be within [-1, 1].")

    geoms = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    highlight_set = (
        None if highlight_idxs is None else {int(idx) for idx in highlight_idxs}
    )
    if highlight_set is None:
        relevant_cluster_ids = list(range(len(clusters)))
    else:
        relevant_cluster_ids = sorted(highlight_set)

    palette = distinctipy.get_colors(max(1, len(relevant_cluster_ids)))
    relevant_colors = {
        cluster_id: np.asarray(palette[color_idx], dtype=float)
        for color_idx, cluster_id in enumerate(relevant_cluster_ids)
    }

    if highlight_set is not None:
        print(colored(
            f"IB visualization: {len(highlight_set)} relevant cluster(s), "
            f"{len(clusters) - len(highlight_set)} unrelated cluster(s).",
            "cyan",
        ))

    for ci, idxs in enumerate(clusters):
        is_relevant = highlight_set is None or ci in highlight_set
        color = (
            relevant_colors[ci]
            if is_relevant
            else np.asarray(unrelated_color, dtype=float)
        )
        draw_bboxes = is_relevant or show_unrelated_bboxes
        pcolor = np.clip(
            color * (0.8 if is_relevant else unrelated_point_scale),
            0.0,
            1.0,
        )
        cluster_bbox_color = np.clip(
            color * cluster_bbox_dark_factor,
            0.0,
            1.0,
        )
        primitive_bbox_color = np.clip(
            color + (1.0 - color) * primitive_bbox_light_mix,
            0.0,
            1.0,
        )
        score = (
            float(cluster_scores[ci])
            if cluster_scores is not None and ci < len(cluster_scores)
            else None
        )
        if is_relevant:
            score_text = f", score={score:.4f}" if score is not None else ""
            print(colored(
                f"Relevant cluster {ci}: {len(idxs)} object(s){score_text}",
                "green",
            ))

        # Merge cluster pcd for a big bbox (optional)
        cluster_pcd = o3d.geometry.PointCloud()
        task_cluster_pcd = o3d.geometry.PointCloud()
        task_primitive_count = 0
        for k in idxs:
            if not 0 <= int(k) < len(objects):
                print(colored(
                    f"Skipping cluster {ci} object index {k}: out of range.",
                    "yellow",
                ))
                continue

            obj = objects[int(k)]
            pcd = obj.get('pcd')
            if not isinstance(pcd, o3d.geometry.PointCloud):
                print(colored(
                    f"Skipping cluster {ci} object {k}: missing point cloud.",
                    "yellow",
                ))
                continue

            primitive_score = (
                float(primitive_scores[int(k)])
                if primitive_scores is not None
                and int(k) < len(primitive_scores)
                else None
            )
            primitive_is_task_related = (
                primitive_score is None
                or primitive_score >= primitive_task_threshold
            )
            if primitive_is_task_related:
                task_primitive_count += 1
                task_cluster_pcd += pcd

            # color the points
            num_pts = np.asarray(pcd.points).shape[0]
            if num_pts > 0:
                cluster_pcd += pcd
                if primitive_is_task_related or not hide_unrelated_primitives:
                    display_point_color = (
                        pcolor
                        if primitive_is_task_related
                        else np.asarray(unrelated_color) * unrelated_point_scale
                    )
                    pcd_colored = o3d.geometry.PointCloud(pcd)  # copy
                    pcd_colored.colors = o3d.utility.Vector3dVector(
                        np.tile(display_point_color, (num_pts, 1))
                    )
                    geoms.append(pcd_colored)

            # add each object bbox in cluster color
            bbox = obj.get('bbox')
            draw_primitive_bbox = draw_bboxes and (
                primitive_is_task_related or show_unrelated_bboxes
            )
            if draw_primitive_bbox and bbox is not None:
                try:
                    if bbox_mode == "aabb":
                        display_bbox = _bbox_as_aabb(bbox)
                    elif isinstance(bbox, o3d.geometry.OrientedBoundingBox):
                        display_bbox = bbox
                    else:
                        display_bbox = pcd.get_oriented_bounding_box(robust=True)
                    if bbox_line_radius > 0.0:
                        geoms.extend(bbox_to_cylinder_meshes(
                            display_bbox,
                            color=(
                                primitive_bbox_color
                                if primitive_is_task_related
                                else unrelated_color
                            ),
                            radius=bbox_line_radius,
                        ))
                    else:
                        geoms.append(obb_to_lineset(
                            display_bbox,
                            color=(
                                primitive_bbox_color
                                if primitive_is_task_related
                                else unrelated_color
                            ),
                        ))
                except Exception as bbox_error:
                    print(colored(
                        f"Skipping bbox for cluster {ci} object {k}: {bbox_error}",
                        "yellow",
                    ))

        if is_relevant and primitive_scores is not None:
            print(colored(
                f"Cluster {ci}: {task_primitive_count}/{len(idxs)} primitive(s) "
                f"meet task threshold {primitive_task_threshold:.2f}.",
                "cyan",
            ))

        # Save useful cluster geometry even if aggregate bbox generation fails.
        if save_dir and len(cluster_pcd.points) > 0:
            relation = "relevant" if is_relevant else "unrelated"
            score_suffix = f"_score_{score:.4f}" if score is not None else ""
            cluster_path = os.path.join(
                save_dir,
                f"cluster_{ci:03d}_{relation}{score_suffix}.ply",
            )
            if o3d.io.write_point_cloud(cluster_path, cluster_pcd):
                print(f"Saved IB cluster {ci} to {cluster_path}")
            else:
                print(colored(f"Failed to save IB cluster {ci} to {cluster_path}", "yellow"))

        # Bound task-related members only, while saving the complete cluster PCD.
        bbox_source_pcd = (
            task_cluster_pcd if primitive_scores is not None else cluster_pcd
        )
        if draw_bboxes and len(bbox_source_pcd.points) > 0:
            if bbox_mode == "aabb":
                cluster_bbox = bbox_source_pcd.get_axis_aligned_bounding_box()
            else:
                try:
                    cluster_bbox = bbox_source_pcd.get_oriented_bounding_box(
                        robust=True
                    )
                except Exception as obb_error:
                    print(colored(
                        f"Cluster {ci} OBB failed; using axis-aligned bbox: {obb_error}",
                        "yellow",
                    ))
                    cluster_bbox = bbox_source_pcd.get_axis_aligned_bounding_box()

            try:
                if bbox_line_radius > 0.0:
                    geoms.extend(bbox_to_cylinder_meshes(
                        cluster_bbox,
                        color=cluster_bbox_color,
                        radius=bbox_line_radius,
                    ))
                else:
                    geoms.append(obb_to_lineset(
                        cluster_bbox,
                        color=cluster_bbox_color,
                    ))
            except Exception as cluster_bbox_error:
                print(colored(
                    f"Skipping aggregate bbox for cluster {ci}: {cluster_bbox_error}",
                    "yellow",
                ))

    if show:
        if geoms:
            o3d.visualization.draw_geometries(geoms)
        else:
            print(colored("No valid IB cluster geometry to visualize.", "yellow"))


def print_task_related_captions(objects, clusters, task_texts, task_features, sim_threshold=0.3):
    """
    For each cluster, find the most related task and print captions of objects
    in that cluster that are related to that task.
    """
    for ci, idxs in enumerate(clusters):
        # 1. Average the cluster's object embeddings
        cluster_vecs = [objects[k]['ft'] for k in idxs if objects[k].get('ft') is not None]
        if len(cluster_vecs) == 0:
            continue
        cluster_mean = np.mean(np.stack(cluster_vecs), axis=0)  # (D,)
        cluster_mean = cluster_mean / np.linalg.norm(cluster_mean)

        # 2. Cosine sim with all task features
        sims = np.dot(task_features, cluster_mean)  # (num_tasks,)
        best_task_idx = int(np.argmax(sims))
        best_task_name = task_texts[best_task_idx]
        best_task_score = sims[best_task_idx]

        print(f"\n=== Cluster {ci} → Task: '{best_task_name}' (score={best_task_score:.3f}) ===")

        # 3. Print captions that match the best task above threshold
        for k in idxs:
            obj_ft = objects[k]['ft']
            if obj_ft is None:
                continue
            score = np.dot(task_features[best_task_idx], obj_ft)
            if score >= sim_threshold:
                print(f"  • {objects[k]['caption']} (sim={score:.3f})")


def _encode_query_sbert(sbert_model, query: str):
    q = sbert_model.encode(query, convert_to_tensor=True)
    q = q / q.norm(dim=-1, keepdim=True)
    return q.squeeze()

def _parse_object_ids(meta_val):
    """
    meta_val can be:
      - a comma-separated string: '2225,2240,...'
      - a list[int] or list[str]
    Returns: List[int]
    """
    if meta_val is None:
        return []
    if isinstance(meta_val, str):
        return [int(x) for x in meta_val.split(',') if x.strip() != ""]
    if isinstance(meta_val, (list, tuple)):
        out = []
        for v in meta_val:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    return []

def _global_ids_to_local_indices(global_ids, objid_to_idx=None):
    """
    Map global object ids to indices into self.objects.
    If objid_to_idx is None we assume objects are stored at index == global_id.
    """
    if objid_to_idx is None:
        return [gid for gid in global_ids if 0 <= gid]
    idxs = []
    for gid in global_ids:
        if gid in objid_to_idx:
            idxs.append(objid_to_idx[gid])
    return idxs


def _build_subgraph_from_indices(objects, object_cap_features_np, idxs, graph_builder_kwargs):
    sub_objects = []
    new_idxs = []   
    for i in idxs:
        # print(colored(f"Processing index {i} for subgraph building {_bbox_extent(objects[i]['bbox'])}", "cyan"))
        if 0 <= i < len(objects): 
            L, W, H = _bbox_extent(objects[i]['bbox'])
            if L * W < 10.0 and L < 2.50 and H <2.50:
                sub_objects.append(objects[i])
                new_idxs.append(i)
            # else:
            #     print(colored(f"Filtered objects {objects[i]['caption']} {_bbox_extent(objects[i]['bbox'])}", "blue"))
        else:
            print(colored(f"Warning: index {i} is out of bounds [0, {len(objects)-1}]", "red"))
    # sub_objects = [objects[i] for i in idxs]
    # sub_feats   = object_cap_features_np[idxs]
    sub_feats   = object_cap_features_np[np.array(new_idxs, dtype=int)]

    G_sub = build_object_graph_ground_safe(sub_objects, sub_feats, **graph_builder_kwargs)
    return G_sub, sub_objects, sub_feats

def _build_subgraph_from_indices_v0(objects, object_cap_features_np, idxs, graph_builder_kwargs):
    sub_objects = []
    for i in idxs:
        if 0 <= i < len(objects):
            sub_objects.append(objects[i])
        else:
            print(colored(f"Warning: index {i} is out of bounds [0, {len(objects)-1}]", "red"))
    # sub_objects = [objects[i] for i in idxs]
    sub_feats   = object_cap_features_np[idxs]
    G_sub = build_object_graph_ground_safe(sub_objects, sub_feats, **graph_builder_kwargs)
    return G_sub, sub_objects, sub_feats

def _time_to_frame_idx(t, time_offset, fps, stride=1):
    """
    Map absolute caption time `t` to a scene-graph frame index space.
    - time_offset: beginning-of-memory time (same origin for caps and SG)
    - fps: frames per second of the SG (or 1/period for keyframes)
    - stride: if you only kept every `stride`-th frame in SG objects['image_idx']
    """
    print(colored(f"Mapping caption time {t} with offset {time_offset}, fps {fps}, stride {stride}", "red"))
    rel = max(0.0, float(time_offset +t))
    frame = int(round(rel * fps))
    if stride > 1:
        frame //= int(stride)
    return frame

def _collect_obj_ids_in_time_window(objects, frame_idx, window_frames=3):
    """
    Return set of object indices observed in [f - w, f + w].
    Assumes objects[i]['image_idx'] is a list of frame indices.
    """
    lo, hi = frame_idx - window_frames, frame_idx + window_frames
    keep = []
    for oid, obj in enumerate(objects):
        idxs = obj.get('image_idx', [])
        if any((lo <= f <= hi) for f in idxs):
            keep.append(oid)
    return keep

def _subselect_features(features_np, idxs):
    # features_np: (N, D) numpy (SBERT/CLIP features for objects)
    return features_np[idxs]

def _build_subgraph(objects, object_cap_features, idxs, graph_builder_kwargs):
    # Reuse your Y-down safe builder, but pass only subset `idxs`.
    sub_objects = [objects[i] for i in idxs]
    sub_feats   = _subselect_features(object_cap_features, np.array(idxs))
    G_sub = build_object_graph_ground_safe(sub_objects, sub_feats, **graph_builder_kwargs)
    return G_sub, sub_objects, sub_feats

def _run_ib_subset(object_cap_features, task_features, G_nx, cfg_yaml_path):
    # thin wrapper to your existing IB entrypoint
    return run_ib_clustering(object_cap_features, task_features, G_nx, cfg_yaml_path)

def _group_caps_by_cluster(caps, cap_scores, cap_to_obj_ids, clusters, subset_obj_global_ids):
    """
    caps: list of retrieved docs (LangChain Document)
    cap_scores: parallel list of float scores (lower distance → better)
    cap_to_obj_ids: list[ set[int] ] objects touched by each caption (GLOBAL ids)
    clusters: list[list[int]] cluster members (subset index space)
    subset_obj_global_ids: list[int] mapping subset index → global object id

    Returns: dict cluster_id -> list of (doc_idx, score)
    """
    # Map global object id → cluster id
    obj_global_to_cluster = {}
    for cid, member_subset_idxs in enumerate(clusters):
        for sub_idx in member_subset_idxs:
            g = subset_obj_global_ids[sub_idx]
            obj_global_to_cluster[g] = cid

    groups = {}
    for i, obj_ids in enumerate(cap_to_obj_ids):
        # a caption may touch multiple objs → assign to the most frequent cluster hit
        cluster_hits = [obj_global_to_cluster.get(gid, None) for gid in obj_ids]
        cluster_hits = [h for h in cluster_hits if h is not None]
        if not cluster_hits:
            continue
        # pick the dominant cluster (or just the first)
        cid = max(set(cluster_hits), key=cluster_hits.count)
        groups.setdefault(cid, []).append((i, cap_scores[i]))
    return groups

def _pick_best_per_group(groups, caps, cap_scores, k):
    """
    Select at most one caption per group (lowest distance score is best), up to k groups.
    If there are more groups than k, choose the k groups with the best representatives.
    """
    representatives = []
    for cid, items in groups.items():
        # items: list[(doc_idx, score)] where score is distance/sim metric from vectorstore
        # lower score often means more similar (depending on store). If yours is "similarity",
        # flip the sign accordingly.
        best_idx, best_score = min(items, key=lambda t: t[1])
        representatives.append((cid, best_idx, best_score))

    # sort groups by their best item
    representatives.sort(key=lambda t: t[2])
    representatives = representatives[:k]
    return representatives


class MilvusWrapper:

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection)


    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000),
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
            FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),
            FieldSchema(name='object_id', dtype=DataType.VARCHAR, description='object id', max_length=3000),

        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            # expr="id > 3",
            output_fields=["id", "text_embedding"]
        )

        return res




class MilvusMemory(Memory):


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT, embedder=None, args=None):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset
        self.args = args

        self.embedder = embedder[0] or HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.sbert_model = embedder[1]
        self.working_memory = []
        
        self.reset(drop_collection=False)
        self.scene_graph = None
        self.search_position = {
            "method_name": "position_search",
            "data": []
        }
        self.search_time = {
            "method_name": "time_search",
            "data": []
        }
        self.search_text = {}
        # self.search_text = {
        #     "method_name": "text_search",
        #     "data": []
        # }
        # self.search_SG = {
        #     "method_name": "scenegraph_search",
        #     "data": []
        # }
        self.search_SG = {}

    def truncate_utf8(self,text: str, max_bytes: int = 3000) -> str:
        if text is None:
            return ""
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text
        return data[:max_bytes].decode("utf-8", errors="ignore")

    def insert(self, item: MemoryItem, text_embedding=None):
        # Convert the dataclass item to a dictionary
        memory_dict = asdict(item)
        # Assign a unique ID based on current timestamp
        memory_dict['id'] = str(time.time())
        
        memory_dict['caption'] = self.truncate_utf8(memory_dict.get('caption', ''), 3000)
        # print("memory dict: ", memory_dict)
        # If no embedding is provided, compute one using the embedder
        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])
            print("text embedding is none, therefore generate it online")
        # Adjust the timestamp by a time offset (e.g. the start time of the current memory collection)
        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5})


        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
        )

        self.time_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='time',
            text_field='caption',
        )
    
    def search_by_position(self, query: tuple) -> str:
        t1 = time.time()
        docs_with_scores = similarity_search_with_score_by_vector(
            self.position_vector_db, np.array(query).astype(float)
        )
        self.working_memory += [doc for doc, _ in docs_with_scores]

        # extract the time and score from the documents
        data = []
        for doc, score in docs_with_scores:
            t = doc.metadata['time']
            data.append({
                "time": t + self.time_offset,
                "score": score,
            })
        self.search_position = {
            "method_name": "position_search",
            "data": data
        }

        docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores])
        print(colored(f"Position search took {time.time() - t1:.3f} seconds", "green"))
        #print("docs for Position search: ", docs_str)
        return docs_str


    def search_by_time(self, hms_time: str) -> str:
        t1 = time.time()
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res:
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time, template).timetuple()) - self.time_offset

        docs_with_scores = similarity_search_with_score_by_vector(
            self.time_vector_db, np.array([query, 0])
        )

        self.working_memory += [doc for doc, _ in docs_with_scores]

        # Extract data for plotting
        data = []
        for doc, score in docs_with_scores:
            t = doc.metadata['time']
            data.append({
                "time": t + self.time_offset,
                "score": score,
            })
        self.search_time = {
            "method_name": "time_search",
            "data": data
        }

        docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores])
        print(colored(f"Time search took {time.time() - t1:.3f} seconds", "green"))
        print("docs for time search: ", docs_str)
        return docs_str


    def search_by_text_org(self, query: str, k=12) -> str:
        # Retrieve more results (e.g., top-50) to inspect the overall distribution,
        # but only add the top-k results to the working memory for downstream tasks.
        docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=100)
        k = self.args.topk

        # ✅ Add only the top-k documents to the working memory.
        self.working_memory += [doc for doc, _ in docs_with_scores[:k]]

        # ✅ Extract data for plotting.
        data = []
        scores = []
        for doc, score in docs_with_scores:
            t = doc.metadata['time']
            data.append({
                "time": t + self.time_offset,  # Shift time by the offset for real-time alignment.
                "score": score,
            })
            scores.append(score)

        # ✅ Normalize scores between 0 and 1.
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score == 0:
            normalized_scores = [0.0 for _ in scores]
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        # ✅ Add normalized scores to the plot data.
        for i in range(len(data)):
            data[i]['score_normalized'] = normalized_scores[i]

        # ✅ Store the plot data for later visualization.
        # self.search_text = {
        #     "method_name": f"{query}",
        #     "data": data,
        #     "score_min": min_score,
        #     "score_max": max_score,
        # }
        if not hasattr(self, 'search_text'):
            self.search_text = {}

    
        self.search_text[query] = {
        "method_name": f"cue: {query}",
        "data": data,
        "score_min": min_score,
        "score_max": max_score,
        }
        # ✅ Return string representation of the top-k results for downstream use.
        docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores[:k]])
        return docs_str

    def search_by_text_IB_diverse(self, query: str, k: int = 6) -> str:
        """
        Retrieve a large caption pool for a query, identify related objects through caption-object links,
        run IB clustering over the task-conditioned object subgraph, and return diverse representative
        captions from task-relevant clusters.

        Main stages:
        1. Retrieve top-M captions from vector store
        2. Normalize retrieval scores and filter candidate captions
        3. Extract object subset covered by retained captions
        4. Build object subgraph and run IB clustering
        5. Score clusters against the query embedding
        6. Assign captions to dominant clusters
        7. Select temporally diverse representative captions per cluster
        8. Save debug metadata and return formatted memory
        """
        start_time = time.time()

        # ---------------------------------------------------------------------
        # Debug / visualization controls
        # ---------------------------------------------------------------------
        debug_enabled = getattr(self.args, "verbose", True)
        save_debug_artifacts = True
        show_visualization = True

        def log_info(msg: str):
            if debug_enabled:
                print(colored(msg, "cyan"))

        def log_warn(msg: str):
            print(colored(msg, "yellow"))

        def log_error(msg: str):
            print(colored(msg, "red"))

        def log_success(msg: str):
            print(colored(msg, "green"))

        def stage_header(title: str):
            if debug_enabled:
                print(colored(f"\n{'=' * 20} {title} {'=' * 20}", "blue"))

        # ---------------------------------------------------------------------
        # Tunable parameters
        # ---------------------------------------------------------------------
        RETRIEVAL_POOL_SIZE = 100
        CAPTION_KEEP_THRESHOLD = 0.35

        CLUSTER_RELEVANCE_THRESHOLD = 0.40
        MIN_RETAINED_CAPTIONS = 10
        FALLBACK_MIN_CAPTIONS = 12

        
        MIN_TIME_GAP_BETWEEN_CAPTIONS = 90.0  # seconds (1.5 minutes) - to encourage temporal diversity in selected captions
        MAX_CAPTIONS_PER_CLUSTER = 3
        MIN_CAPTIONS_PER_CLUSTER = 1

        GRAPH_BUILD_CONFIG = dict(
            vertical_axis='z',
            down_positive=True,
            dist_radius=3.0,
            z_overlap=False,
            z_slack=0.40,
            iou_thresh=0.01,
            covis_min=0,
            knn=6,
            ground_height_thresh=0.1,
            ground_floor_thresh=0.05,
            dilate_eps=0.02
        )

        # ---------------------------------------------------------------------
        # Small helpers
        # ---------------------------------------------------------------------
        def normalize_retrieval_scores(raw_scores):
            """
            Convert raw vector-store scores into normalized values in [0, 1].
            Assumes smaller score = better match if returned values behave like distances.
            """
            if not raw_scores:
                return []

            score_min = min(raw_scores)
            score_max = max(raw_scores)

            if abs(score_max - score_min) < 1e-9:
                return [1.0 for _ in raw_scores]

            return [1.0 - (s - score_min) / (score_max - score_min) for s in raw_scores]

        def select_caption_candidates(normalized_scores):
            """
            Keep captions above threshold. If too few survive, fallback to top-N by normalized score.
            """
            retained_indices = [
                idx for idx, score in enumerate(normalized_scores)
                if score >= CAPTION_KEEP_THRESHOLD
            ]

            if len(retained_indices) < FALLBACK_MIN_CAPTIONS:
                sorted_indices = sorted(
                    range(len(normalized_scores)),
                    key=lambda idx: normalized_scores[idx],
                    reverse=True
                )
                retained_indices = sorted_indices[:min(MIN_RETAINED_CAPTIONS, len(sorted_indices))]

            if not retained_indices:
                retained_indices = list(range(min(RETRIEVAL_POOL_SIZE, len(normalized_scores))))

            return retained_indices

        def extract_caption_object_links(caption_docs):
            """
            Parse object ids from caption metadata and map them into local object indices.
            Returns:
                caption_to_local_object_ids: list[list[int]]
                unique_local_object_ids: sorted list[int]
            """
            caption_to_local_object_ids = []
            unique_object_id_set = set()

            for doc in caption_docs:
                global_object_ids = _parse_object_ids(doc.metadata.get("object_id"))
                local_object_ids = _global_ids_to_local_indices(
                    global_object_ids,
                    objid_to_idx=getattr(self, "objid_to_idx", None)
                )
                local_object_ids = list(set(local_object_ids))
                caption_to_local_object_ids.append(local_object_ids)
                unique_object_id_set.update(local_object_ids)

            return caption_to_local_object_ids, sorted(unique_object_id_set)

        def ensure_axis_aligned_bbox(bbox):
            """
            Convert supported bbox formats into Open3D AxisAlignedBoundingBox if possible.
            """
            if bbox is None:
                return None

            if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
                return bbox

            if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
                return bbox.get_axis_aligned_bounding_box()

            try:
                bbox_np = np.asarray(bbox)
                if bbox_np.ndim == 2 and bbox_np.shape[1] == 3:
                    return o3d.geometry.AxisAlignedBoundingBox(
                        bbox_np.min(axis=0),
                        bbox_np.max(axis=0)
                    )
            except Exception:
                pass

            return None

        def summarize_cluster_objects(cluster_id, clustered_objects):
            """
            Build a compact text summary for objects inside one cluster.
            Useful for debugging cluster semantics and geometry.
            """
            object_descriptions = []
            cluster_member_indices = clusters_in_subset[cluster_id]

            for member_idx in cluster_member_indices:
                obj = clustered_objects[member_idx]
                object_id = obj.get("global_id", obj.get("object_id", obj.get("id", f"idx_{member_idx}")))
                object_caption = obj.get("caption", obj.get("desc", ""))

                bbox = ensure_axis_aligned_bbox(obj.get("bbox"))
                if bbox is None:
                    object_descriptions.append(f"Object {object_id}: bbox=None")
                    continue

                center = np.round(np.asarray(bbox.get_center(), dtype=float), 3)
                extent = np.round(np.asarray(bbox.get_extent(), dtype=float), 3)

                if object_caption:
                    object_descriptions.append(
                        f"Object {object_id}: center={center.tolist()}, extent={extent.tolist()}, caption={object_caption}"
                    )
                else:
                    object_descriptions.append(
                        f"Object {object_id}: center={center.tolist()}, extent={extent.tolist()}"
                    )

            if not object_descriptions:
                return "No valid objects in this cluster."

            return "\n".join(object_descriptions)

        def debug_print_retrieved_captions(caption_docs, raw_scores, normalized_scores, title="Retrieved captions"):
            if not debug_enabled:
                return
            stage_header(title)
            for idx, (doc, raw_score, norm_score) in enumerate(zip(caption_docs, raw_scores, normalized_scores)):
                caption_time = doc.metadata.get("time", 0.0)
                if isinstance(caption_time, (list, tuple)):
                    caption_time = caption_time[0]
                preview = doc.page_content[:120].replace("\n", " ")
                print(
                    f"[{idx:03d}] raw={raw_score:.4f} norm={norm_score:.4f} "
                    f"time={caption_time} object_id={doc.metadata.get('object_id', None)} "
                    f"text='{preview}...'"
                )

        def debug_print_cluster_scores(cluster_scores, cluster_indices_to_keep):
            if not debug_enabled:
                return
            stage_header("Cluster relevance scores")
            for cluster_id, score in enumerate(cluster_scores):
                marker = "*" if cluster_id in set(cluster_indices_to_keep.tolist() if isinstance(cluster_indices_to_keep, np.ndarray) else cluster_indices_to_keep) else " "
                print(f"{marker} cluster={cluster_id:02d} score={score:.4f}")
        
        def maybe_visualize_subgraph(
            subgraph,
            subgraph_objects,
            clusters_in_subset,
            kept_cluster_indices,
            cluster_best_task_idx,
            debug_output_dir
        ):
            if not save_debug_artifacts:
                return

            os.makedirs(debug_output_dir, exist_ok=True)

            log_info("[Visualization] Preparing debug visualizations...")
            log_info(f"[Visualization] debug_output_dir = {debug_output_dir}")
            log_info(f"[Visualization] num_subgraph_objects = {len(subgraph_objects) if subgraph_objects is not None else 'None'}")
            log_info(f"[Visualization] num_clusters = {len(clusters_in_subset) if clusters_in_subset is not None else 'None'}")
            log_info(f"[Visualization] kept_cluster_indices = {kept_cluster_indices}")
            log_info(f"[Visualization] cluster_best_task_idx type = {type(cluster_best_task_idx)}")

            # ------------------------------------------------------------
            # 1) 2D graph view
            # ------------------------------------------------------------
            try:
                visualize_graph_highlight(
                    subgraph,
                    clusters_in_subset,
                    kept_cluster_indices,
                    out_png=os.path.join(debug_output_dir, "graph_hybrid.png"),
                    show=False,
                    objects=subgraph_objects,
                    layout_mode="hybrid",
                    title=f"Graph Hybrid View: {query}"
                )
                log_success("[Visualization] Saved 2D graph visualization.")
            except Exception as exc:
                log_warn(f"[Visualization] Graph visualization failed: {exc}")

            # ------------------------------------------------------------
            # 2) Existing Open3D cluster visualization
            # ------------------------------------------------------------
            try:
                if subgraph_objects is None or len(subgraph_objects) == 0:
                    log_warn("[Visualization] Skip Open3D cluster visualization: subgraph_objects is empty.")
                elif clusters_in_subset is None or len(clusters_in_subset) == 0:
                    log_warn("[Visualization] Skip Open3D cluster visualization: clusters_in_subset is empty.")
                elif kept_cluster_indices is None or len(kept_cluster_indices) == 0:
                    log_warn("[Visualization] Skip Open3D cluster visualization: kept_cluster_indices is empty.")
                elif cluster_best_task_idx is None:
                    log_warn("[Visualization] Skip Open3D cluster visualization: cluster_best_task_idx is None.")
                else:
                    visualize_highlighted_clusters_open3d(
                        subgraph_objects,
                        clusters_in_subset,
                        kept_cluster_indices,
                        cluster_best_task_idx,
                        [query],
                        dim_alpha=0.10,
                        save_dir=os.path.join(debug_output_dir, "clusters_vis"),
                        show=False
                    )
                    log_success("[Visualization] Saved existing Open3D cluster visualization.")
            except Exception as exc:
                log_warn(f"[Visualization] 3D cluster visualization failed: {exc}")

            # ------------------------------------------------------------
            # 3) New lifted 3D scene graph visualization
            # ------------------------------------------------------------
            try:
                if subgraph_objects is None or len(subgraph_objects) == 0:
                    log_warn("[Visualization] Skip lifted 3D scene graph visualization: subgraph_objects is empty.")
                elif subgraph is None:
                    log_warn("[Visualization] Skip lifted 3D scene graph visualization: subgraph is None.")
                else:
                    visualize_graph_in_3d_scene(
                        objects=subgraph_objects,
                        graph=subgraph,
                        clusters=clusters_in_subset,
                        kept_cluster_ids=kept_cluster_indices,
                        save_path=os.path.join(debug_output_dir, "graph_3d_scene.png"),
                        show=True,
                        lift_height=2,
                        node_radius=0.05,
                        edge_radius=0.012,
                        vertical_edge_radius=0.006,
                        use_bbox=True,
                        use_mesh=True,
                        show_vertical_connectors=False,
                    )
                    log_success("[Visualization] Saved lifted 3D scene graph visualization.")
            except Exception as exc:
                log_warn(f"[Visualization] Lifted 3D scene graph visualization failed: {exc}")
        
        try:
            # -----------------------------------------------------------------
            # Stage 1: Retrieve caption candidates
            # -----------------------------------------------------------------
            stage_header("Stage 1 - Retrieve captions")
            retrieved_docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(
                query,
                k=RETRIEVAL_POOL_SIZE
            )

            if not retrieved_docs_with_scores:
                log_error("No captions found for the query.")
                return ""

            retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
            retrieved_raw_scores = [score for _, score in retrieved_docs_with_scores]
            retrieved_norm_scores = normalize_retrieval_scores(retrieved_raw_scores)

            debug_print_retrieved_captions(
                retrieved_docs,
                retrieved_raw_scores,
                retrieved_norm_scores,
                title="All retrieved captions"
            )

            # -----------------------------------------------------------------
            # Stage 2: Retain promising captions
            # -----------------------------------------------------------------
            stage_header("Stage 2 - Filter candidate captions")
            retained_caption_indices = select_caption_candidates(retrieved_norm_scores)

            retained_docs = [retrieved_docs[idx] for idx in retained_caption_indices]
            retained_raw_scores = [retrieved_raw_scores[idx] for idx in retained_caption_indices]
            retained_norm_scores = [retrieved_norm_scores[idx] for idx in retained_caption_indices]

            log_info(
                f"Retained {len(retained_caption_indices)} / {len(retrieved_docs)} captions "
                f"(threshold={CAPTION_KEEP_THRESHOLD:.2f})"
            )

            debug_print_retrieved_captions(
                retained_docs,
                retained_raw_scores,
                retained_norm_scores,
                title="Retained captions after filtering"
            )

            # -----------------------------------------------------------------
            # Stage 3: Extract related objects from retained captions
            # -----------------------------------------------------------------
            stage_header("Stage 3 - Extract related objects")
            caption_to_local_object_ids, subset_local_object_ids = extract_caption_object_links(retained_docs)

            log_info(f"Found {len(subset_local_object_ids)} unique objects covered by retained captions.")

            if debug_enabled:
                for caption_idx, local_ids in enumerate(caption_to_local_object_ids):
                    print(f"caption={caption_idx:03d} -> local_object_ids={local_ids}")

            if len(subset_local_object_ids) == 0:
                log_warn("No objects linked to retained captions. Falling back to plain top-k retrieval.")
                fallback_docs = [doc for doc, _ in retrieved_docs_with_scores[:k]]
                self.working_memory += fallback_docs
                return self.memory_to_string(fallback_docs)

            # -----------------------------------------------------------------
            # Stage 4: Build object subgraph and compute query embedding
            # -----------------------------------------------------------------
            stage_header("Stage 4 - Build subgraph and encode query")
            object_feature_matrix = np.array([obj["ft"].flatten() for obj in self.scene_graph])

            subgraph, subgraph_objects, subgraph_features = _build_subgraph_from_indices(
                self.scene_graph,
                object_feature_matrix,
                subset_local_object_ids,
                GRAPH_BUILD_CONFIG
            )

            query_feature = _encode_query_sbert(self.sbert_model, query).detach().cpu().numpy()[None, :]

            log_info(
                f"Subgraph built with {len(subgraph_objects)} objects "
                f"and feature shape {subgraph_features.shape}"
            )

            # Optional object visualization
            if debug_enabled:
                try:
                    visualize_objects_org(subgraph_objects)
                except Exception as exc:
                    log_warn(f"Subgraph object visualization failed: {exc}")

            # -----------------------------------------------------------------
            # Stage 5: Run IB clustering
            # -----------------------------------------------------------------
            stage_header("Stage 5 - Run IB clustering")
            debug_output_dir = getattr(self, "out_dir", "./_tmp")
            Path(debug_output_dir).mkdir(parents=True, exist_ok=True)

            cluster_config_path = os.path.join(debug_output_dir, "cluster_config.yaml")
            write_default_cluster_config(cluster_config_path)

            clusters_in_subset = run_ib_clustering(
                subgraph_features,
                query_feature,
                subgraph,
                cluster_config_path
            )

            log_info(
                f"IB clustering formed {len(clusters_in_subset)} clusters "
                f"from {len(subset_local_object_ids)} subset objects."
            )

            if debug_enabled:
                for cluster_id, members in enumerate(clusters_in_subset):
                    print(f"cluster={cluster_id:02d} members={members}")
                    print(summarize_cluster_objects(cluster_id, subgraph_objects))

            # -----------------------------------------------------------------
            # Stage 6: Score clusters against the query
            # -----------------------------------------------------------------
            stage_header("Stage 6 - Score cluster relevance")
            cluster_scores, cluster_best_task_idx = cluster_task_scores_cosine(
                subgraph_objects,
                clusters_in_subset,
                query_feature,
                pool="max"
            )

            kept_cluster_indices = np.where(cluster_scores >= CLUSTER_RELEVANCE_THRESHOLD)[0]
            if kept_cluster_indices.size < k:
                kept_cluster_indices = np.argsort(-cluster_scores)[:min(k, len(clusters_in_subset))]
                log_warn(
                    f"No sufficient clusters above threshold={CLUSTER_RELEVANCE_THRESHOLD:.2f}. "
                    f"Fallback to top-{len(kept_cluster_indices)} clusters by score."
                )
            else:
                kept_cluster_indices = np.argsort(-cluster_scores)[:min(k, len(clusters_in_subset))]
                log_info(
                    f"Keeping top {len(kept_cluster_indices)} clusters ranked by query relevance."
                )

            debug_print_cluster_scores(cluster_scores, kept_cluster_indices)


            maybe_visualize_subgraph(
                subgraph=subgraph,
                subgraph_objects=subgraph_objects,
                clusters_in_subset=clusters_in_subset,
                kept_cluster_indices=kept_cluster_indices,
                cluster_best_task_idx=cluster_best_task_idx,
                debug_output_dir=debug_output_dir
            )        
            # main pipeline

            # -----------------------------------------------------------------
            # Stage 7: Map captions to dominant clusters
            # -----------------------------------------------------------------
            stage_header("Stage 7 - Assign captions to dominant clusters")
            subset_local_id_to_subgraph_pos = {
                local_id: pos for pos, local_id in enumerate(subset_local_object_ids)
            }

            object_position_to_cluster_id = {}
            for cluster_id in kept_cluster_indices:
                for object_pos in clusters_in_subset[cluster_id]:
                    object_position_to_cluster_id[object_pos] = int(cluster_id)

            cluster_to_caption_candidates = {int(cluster_id): [] for cluster_id in kept_cluster_indices}

            for caption_idx, local_object_ids in enumerate(caption_to_local_object_ids):
                matched_cluster_ids = []

                for local_id in local_object_ids:
                    subgraph_pos = subset_local_id_to_subgraph_pos.get(local_id)
                    if subgraph_pos is None:
                        continue

                    cluster_id = object_position_to_cluster_id.get(subgraph_pos)
                    if cluster_id is not None:
                        matched_cluster_ids.append(cluster_id)

                if not matched_cluster_ids:
                    continue

                dominant_cluster_id = max(set(matched_cluster_ids), key=matched_cluster_ids.count)
                cluster_to_caption_candidates[dominant_cluster_id].append(
                    (caption_idx, retained_raw_scores[caption_idx])
                )

                log_info(
                    f"Caption {caption_idx} assigned to dominant cluster {dominant_cluster_id} "
                    f"via local_object_ids={local_object_ids}"
                )

            if debug_enabled:
                for cluster_id, items in cluster_to_caption_candidates.items():
                    print(f"cluster={cluster_id:02d} -> caption_candidates={items}")

            # -----------------------------------------------------------------
            # Stage 8: Select temporally diverse representatives per cluster
            # -----------------------------------------------------------------
            stage_header("Stage 8 - Select diverse representatives")
            selected_representatives = []
            time_reference_offset = getattr(self, "time_offset", 0.0)

            for cluster_id in kept_cluster_indices:
                candidate_items = cluster_to_caption_candidates.get(int(cluster_id), [])
                if not candidate_items:
                    continue

                # NOTE:
                # If your retriever score is a distance, lower is better.
                # Therefore ascending sort is correct here.
                candidate_items_sorted = sorted(candidate_items, key=lambda item: item[1])

                chosen_caption_records = []
                chosen_times = []
                chosen_docs = []

                for caption_idx, raw_score in candidate_items_sorted:
                    doc = retained_docs[caption_idx]
                    caption_time_abs = _caption_time_seconds(doc, time_reference_offset)

                    if all(abs(caption_time_abs - prev_time) >= MIN_TIME_GAP_BETWEEN_CAPTIONS for prev_time in chosen_times):
                        chosen_caption_records.append((int(cluster_id), caption_idx, raw_score, caption_time_abs))
                        chosen_times.append(caption_time_abs)
                        chosen_docs.append(doc)

                    if len(chosen_caption_records) >= MAX_CAPTIONS_PER_CLUSTER:
                        break

                if len(chosen_caption_records) < MIN_CAPTIONS_PER_CLUSTER and candidate_items_sorted:
                    first_caption_idx, first_raw_score = candidate_items_sorted[0]
                    first_doc = retained_docs[first_caption_idx]
                    first_time_abs = _caption_time_seconds(first_doc, time_reference_offset)

                    chosen_caption_records = [(int(cluster_id), first_caption_idx, first_raw_score, first_time_abs)]
                    chosen_times = [first_time_abs]
                    chosen_docs = [first_doc]

                selected_caption_idx_set = {cap_idx for _, cap_idx, _, _ in chosen_caption_records}
                omitted_items = [
                    (cap_idx, score) for cap_idx, score in candidate_items_sorted
                    if cap_idx not in selected_caption_idx_set
                ]
                omitted_docs = [retained_docs[cap_idx] for cap_idx, _ in omitted_items]
                omitted_summary = _summarize_omitted_times(omitted_docs, time_reference_offset)

                for doc in chosen_docs:
                    doc.page_content += f" {omitted_summary}"

                if debug_enabled:
                    print(colored(f"\nCluster {cluster_id} selected representatives:", "yellow"))
                    for _, caption_idx, raw_score, caption_time_abs in chosen_caption_records:
                        preview = retained_docs[caption_idx].page_content[:120].replace("\n", " ")
                        print(
                            f"  [SELECTED] cluster={cluster_id} caption={caption_idx} "
                            f"score={raw_score:.4f} time={caption_time_abs:.2f}s text='{preview}...'"
                        )

                selected_representatives.extend(
                    [(cluster_id, caption_idx, raw_score) for cluster_id, caption_idx, raw_score, _ in chosen_caption_records]
                )

            # -----------------------------------------------------------------
            # Stage 9: Apply global cap
            # -----------------------------------------------------------------
            stage_header("Stage 9 - Apply global top-k cap")
            selected_representatives.sort(key=lambda item: item[2])  # lower score = better if distance
            selected_representatives = selected_representatives[:k]

            final_docs = [retained_docs[caption_idx] for _, caption_idx, _ in selected_representatives]
            selected_caption_indices = [caption_idx for _, caption_idx, _ in selected_representatives]
            selected_caption_index_set = set(selected_caption_indices)

            if debug_enabled:
                print(f"Final selected caption indices: {selected_caption_indices}")

            # -----------------------------------------------------------------
            # Stage 10: Save debug metadata for later analysis
            # -----------------------------------------------------------------
            stage_header("Stage 10 - Save debug metadata")
            time_offset = getattr(self, "time_offset", 0.0)
            all_retrieved_debug_data = []

            for retrieved_idx, (doc, raw_score) in enumerate(retrieved_docs_with_scores):
                metadata_time = doc.metadata.get("time", 0.0)
                if isinstance(metadata_time, (list, tuple)):
                    metadata_time = metadata_time[0]

                all_retrieved_debug_data.append({
                    "idx": retrieved_idx,
                    "time": float(metadata_time + time_offset),
                    "score": float(raw_score),
                    "selected": (retrieved_idx in selected_caption_index_set)
                })

            if not hasattr(self, "search_text"):
                self.search_text = {}

            self.search_text[query] = {
                "method_name": f"cue: {query}",
                "data": all_retrieved_debug_data,
                "representatives": selected_representatives,
                "selected_indices": selected_caption_index_set,
                "retained_caption_indices": retained_caption_indices,
                "subset_local_object_ids": subset_local_object_ids,
                "cluster_scores": cluster_scores.tolist() if isinstance(cluster_scores, np.ndarray) else cluster_scores,
                "kept_cluster_indices": kept_cluster_indices.tolist() if isinstance(kept_cluster_indices, np.ndarray) else kept_cluster_indices,
            }

            log_success(f"[search_by_text_ib_diverse] Total time: {time.time() - start_time:.2f}s")
            return self.memory_to_string(final_docs)

        except Exception as exc:
            import traceback
            log_error(f"[search_by_text_ib_diverse] ERROR: {exc}")
            traceback.print_exc()

            # Graceful fallback
            fallback_docs = [
                doc for doc, _ in self.text_retriever.vectorstore.similarity_search_with_score(query, k=k)
            ]
            self.working_memory += fallback_docs
            return self.memory_to_string(fallback_docs)
    
    def search_by_text_IB_isaacsim(self, query: str, k=6) -> str:
        """
        Retrieve many captions, restrict to those covering objects (object_id) related to the query,
        cluster the union subset via IB (task = query), and return one best caption per cluster (diverse).
        """
        t1 = time.time()

        def record_fallback_plot_data(retrieved_docs_with_scores):
            """Retain plain-retrieval results when the IB path cannot be used."""
            time_offset = getattr(self, 'time_offset', 0.0)
            selected_count = min(k, len(retrieved_docs_with_scores))
            data = []

            for idx, (doc, score) in enumerate(retrieved_docs_with_scores):
                metadata_time = doc.metadata.get('time', 0.0)
                if isinstance(metadata_time, (list, tuple)):
                    metadata_time = metadata_time[0]
                data.append({
                    "idx": idx,
                    "time": float(metadata_time + time_offset),
                    "score": float(score),
                    "selected": idx < selected_count,
                })

            if not hasattr(self, 'search_text'):
                self.search_text = {}
            self.search_text[query] = {
                "method_name": f"cue: {query} (plain fallback)",
                "data": data,
                "selected_indices": set(range(selected_count)),
            }

        try:
            # ---------- Tunables ----------
            TOP_M        = 100     # retrieve a big pool
            KEEP_THRESH  = 0.35    # normalized similarity threshold to keep captions
            # TASK_FILTER_THRESH = 0.4  # SBERT task relevance after merging
            GRAPH_KW = dict(
                vertical_axis=getattr(self.args, 'vertical_axis', 'y'),
                down_positive=bool(getattr(
                    self.args,
                    'vertical_axis_down_positive',
                    getattr(self.args, 'vertical_axis', 'y') == 'y',
                )),
                dist_radius=float(getattr(self.args, 'graph_dist_radius', 3.0)),
                z_overlap=False,
                z_slack=float(getattr(self.args, 'graph_z_slack', 0.40)),
                iou_thresh=float(getattr(self.args, 'graph_iou_thresh', 0.01)),
                covis_min=0,
                knn=int(getattr(self.args, 'graph_knn', 6)),
                ground_height_thresh=0.1, ground_floor_thresh=0.05,
                dilate_eps=float(getattr(self.args, 'graph_dilate_eps', 0.02)),
            )

            # ---------- 1) Retrieve ----------
            docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=TOP_M)
            if not docs_with_scores:
                print(colored("Warning: No captions found for the query.", "red"))
                return ""

            # Normalize scores to [0,1]
            raw_scores = [s for _, s in docs_with_scores]
            smin, smax = min(raw_scores), max(raw_scores)
            if smax - smin < 1e-9:
                norm = [1.0 for _ in raw_scores]
            else:
                # If vectorstore returns DISTANCE (smaller=better), convert to similarity
                norm = [1.0 - (s - smin) / (smax - smin) for s in raw_scores]

            # TODO: can be removed 
            keep_idx = [i for i, v in enumerate(norm) if v >= KEEP_THRESH]
            # fallback: force top-10 (or fewer if not enough items)
            if len(keep_idx) < 12:
                idx_sorted = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
                keep_idx = idx_sorted[:min(10, len(idx_sorted))]
                # print top-10 norm scores
                #print(colored(f"Top-10 norm scores: {[norm[i] for i in idx_sorted[:10]]}", "yellow"))
            if self.args.verbose:
                print(colored(f"Keeping {len(keep_idx)} / {len(norm)} captions (>= {KEEP_THRESH:.2f})", "yellow"))
            if not keep_idx:
                keep_idx = list(range(min(TOP_M, len(docs_with_scores))))
            #keep_idx = list(range(len(docs_with_scores)))
            kept_docs   = [docs_with_scores[i][0] for i in keep_idx]
            kept_scores = [docs_with_scores[i][1] for i in keep_idx]
            
            kept_times = [docs_with_scores[i][0].metadata.get('time', 0.0) for i in keep_idx]
            # for doc, score, t in zip(kept_docs, kept_scores, kept_times):
            #     t = t[0] if isinstance(t, (list, tuple)) else t
            #     t += self.time_offset
            #     t = localtime(t)
            #     t = strftime('%Y-%m-%d %H:%M:%S', t)
            #     print(f"  [KEPT-CAND] id={doc.metadata.get('id', 'unknown')} score={score:.3f} text='At {t}{doc.page_content}...'")
            kept_norm   = [norm[i] for i in keep_idx]

            # ---------- 2) Build subset of objects from object_id ----------
            cap_to_local_obj_ids = []
            subset_local_ids_set = set()
            for doc in kept_docs:
                gids = _parse_object_ids(doc.metadata.get('object_id'))
                lids = _global_ids_to_local_indices(gids, objid_to_idx=getattr(self, 'objid_to_idx', None))
                lids = list(set(lids))
                cap_to_local_obj_ids.append(lids)
                subset_local_ids_set.update(lids)

            subset_local_ids = sorted(list(subset_local_ids_set))
            floor_filter_config = FloorFilterConfig(
                enabled=bool(getattr(self.args, 'exclude_floor', False)),
                vertical_axis=getattr(self.args, 'vertical_axis', 'z'),
                max_thickness=float(
                    getattr(self.args, 'floor_max_thickness', 0.35)
                ),
                min_footprint_area=float(
                    getattr(self.args, 'floor_min_area', 4.0)
                ),
                semantic_threshold=float(
                    getattr(self.args, 'floor_semantic_threshold', 0.5)
                ),
            )
            subset_local_ids, removed_floor_objects = filter_floor_primitives(
                self.scene_graph,
                subset_local_ids,
                floor_filter_config,
            )
            for object_id, decision in removed_floor_objects:
                caption = str(self.scene_graph[object_id].get('caption', ''))[:100]
                print(colored(
                    f"Removed floor primitive {object_id}: reason={decision.reason}, "
                    f"semantic={decision.semantic_score:.2f}, "
                    f"robust_thickness={decision.vertical_thickness:.3f}, "
                    f"bbox_thickness={decision.bbox_vertical_thickness:.3f}, "
                    f"area={decision.footprint_area:.3f}, caption={caption!r}",
                    "yellow",
                ))
            if removed_floor_objects:
                print(colored(
                    f"Floor filter removed {len(removed_floor_objects)} primitive(s); "
                    f"{len(subset_local_ids)} candidate primitive(s) remain.",
                    "cyan",
                ))
            if len(subset_local_ids) == 0:
                print(colored("No objects covered by captions; falling back to top-k plain retrieval.", "red"))
                record_fallback_plot_data(docs_with_scores)
                topk_docs = [doc for doc, _ in docs_with_scores[:k]]
                self.working_memory += topk_docs
                return self.memory_to_string(topk_docs)

            # ---------- 3) Subgraph + IB on subset ----------
            # NOTE: if your objects live in self.objects, replace self.scene_graph with self.objects below
            object_cap_features_np = np.array([obj['ft'].flatten() for obj in self.scene_graph])  # (N,D)
            # visualize_objects_org(self.scene_graph)
            
            G_sub, sub_objects, sub_feats = _build_subgraph_from_indices_v0(
                self.scene_graph, object_cap_features_np, subset_local_ids, GRAPH_KW
            )
            # visualize_objects_org(sub_objects)
            # Task feature from query (SBERT) -> (1, D)
            task_ft = _encode_query_sbert(self.sbert_model, query).detach().cpu().numpy()[None, :]
            normalized_sub_feats = sub_feats / np.maximum(
                np.linalg.norm(sub_feats, axis=1, keepdims=True),
                1e-12,
            )
            normalized_task_ft = task_ft / np.maximum(
                np.linalg.norm(task_ft, axis=1, keepdims=True),
                1e-12,
            )
            primitive_scores = np.max(
                normalized_sub_feats @ normalized_task_ft.T,
                axis=1,
            )

            # IB config
            out_path = getattr(self, 'out_dir', './_tmp')
            Path(out_path).mkdir(parents=True, exist_ok=True)
            cfg_yaml_path = os.path.join(out_path, "cluster_config.yaml")
            override_ib_config = bool(getattr(
                self.args, 'ib_override_config', False
            ))
            ib_config_overrides = None
            if override_ib_config:
                ib_config_overrides = {
                    'sims_thres': float(getattr(
                        self.args, 'ib_sims_thres', 0.20
                    )),
                    'delta': float(getattr(self.args, 'ib_delta', 0.10)),
                    'top_k_tasks': int(getattr(
                        self.args, 'ib_top_k_tasks', 1
                    )),
                    'cumulative': bool(getattr(
                        self.args, 'ib_cumulative', False
                    )),
                }
            write_default_cluster_config(
                cfg_yaml_path,
                overrides=ib_config_overrides,
                overwrite=override_ib_config,
            )
            if override_ib_config:
                print(colored(
                    f"Runtime IB config: {ib_config_overrides}; "
                    f"graph={GRAPH_KW}; file={cfg_yaml_path}",
                    "cyan",
                ))

            # IB clusters in subset index space
            clusters_subset = run_ib_clustering(sub_feats, task_ft, G_sub, cfg_yaml_path)


            # ----- 5) Score clusters by task & keep only task-relevant ones -----
            # print("\n=== Step 5: Scoring clusters by task relevance ===")
            scores, winners = cluster_task_scores_cosine(sub_objects, clusters_subset, task_ft, pool="max")
            #print(f"Cluster scores vs task: {scores}")
            #print(f"Best matching task index per cluster: {winners}")

            highlight_threshold = float(
                getattr(self.args, 'highlight_threshold', 0.4)
            )
            max_highlight_clusters = max(
                1, int(getattr(self.args, 'max_highlight_clusters', k))
            )

            above_threshold = np.where(scores >= highlight_threshold)[0]
            if above_threshold.size == 0:
                fallback_count = min(k, max_highlight_clusters, len(clusters_subset))
                keep_cluster_idxs = np.argsort(-scores)[:fallback_count]
                print(
                    f"No clusters above {highlight_threshold:.2f}; keeping top "
                    f"{len(keep_cluster_idxs)} by score.\n{keep_cluster_idxs}"
                )
            else:
                ranked_above_threshold = above_threshold[
                    np.argsort(-scores[above_threshold])
                ]
                keep_cluster_idxs = ranked_above_threshold[:max_highlight_clusters]
                print(
                    f"Keeping {len(keep_cluster_idxs)} clusters above "
                    f"{highlight_threshold:.2f} (cap={max_highlight_clusters}).\n"
                    f"{keep_cluster_idxs}"
                )

            # ----- 6) Map captions → dominant kept cluster via object ids -----
            if self.args.verbose:
                for cid in keep_cluster_idxs:
                    print(f"cluster {cid}: score={scores[cid]:.4f}")
                print(colored(f"Building subgraph from {len(subset_local_ids)} objects", "blue"))
                print(colored(f"[AIB] Formed {len(clusters_subset)} clusters from {len(subset_local_ids)} primitives.", "cyan"))

                print("\n=== Step 6: Mapping captions to dominant clusters ===")
            
            subset_local_to_pos = {lid: pos for pos, lid in enumerate(subset_local_ids)}
            objpos_to_keptcid = {}
            for cid in keep_cluster_idxs:
                for pos in clusters_subset[cid]:
                    objpos_to_keptcid[pos] = cid
            if self.args.verbose:
                print(f"subset_local_to_pos: {subset_local_to_pos}")
                print(f"objpos_to_keptcid: {objpos_to_keptcid}")

            groups = {int(cid): [] for cid in keep_cluster_idxs}
            for cap_i, lids in enumerate(cap_to_local_obj_ids):
                hits = []
                for lid in lids:
                    pos = subset_local_to_pos.get(lid, None)
                    if pos is None:
                        continue
                    cid = objpos_to_keptcid.get(pos, None)
                    if cid is not None:
                        hits.append(int(cid))
                if not hits:
                    continue
                dom = max(set(hits), key=hits.count)
                groups[dom].append((cap_i, kept_scores[cap_i]))
                if self.args.verbose:
                    print(f"Under query: {query}; Caption {cap_i} with lids {lids} -> dominant cluster {dom}")
            if self.args.verbose:
                print(f"Grouped captions by cluster (before selection):\n{groups}")

                # ----- 7) Pick top-N per cluster (by score) for diversity -----
                print("\n=== Step 7: Selecting top captions per cluster ===")

            # ---- Parameters (tune) ----
            TIME_SEP = 90.0            # min temporal separation (seconds) within a cluster’s chosen reps
            AUTO_MAX_PER_CLUSTER = 3   # hard cap (we’ll choose as many as pass TIME_SEP, up to this)
            MIN_PER_CLUSTER = 1        # always keep at least one if the cluster has any candidates

            representatives = []
            ref_time = getattr(self, 'time_offset', 0.0)
            def _ensure_o3d_aabb(bbox):
                            """
                            Convert to Open3D AxisAlignedBoundingBox if possible.
                            - Keep AABB as-is
                            - Convert OOBB -> AABB
                            - Convert Nx3 array -> AABB
                            """
                            if bbox is None:
                                return None
                            if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
                                return bbox
                            if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
                                return bbox.get_axis_aligned_bounding_box()
                            try:
                                arr = np.asarray(bbox)
                                if arr.ndim == 2 and arr.shape[1] == 3:
                                    return o3d.geometry.AxisAlignedBoundingBox(arr.min(axis=0), arr.max(axis=0))
                            except Exception:
                                pass
                            return None
            # Build a compact per-cluster object summary string
            def _cluster_object_summary(cid, subset_local_ids):
                members = clusters_subset[cid]
                print(colored(f"Cluster {cid} has {members} members.", "cyan"))
                
                parts = []
                for m in members:
                    o = sub_objects[m]
                    # Prefer global id if present; fallback to any available identifier
                    oid = o.get('global_id', o.get('object_id', o.get('id', f"{subset_local_ids[m]}")))
                    cap = o.get('caption', o.get('desc', ''))

                    bbox  = _ensure_o3d_aabb(o.get("bbox", None))
                    if bbox is None:
                        print(colored(f"Warning: No bbox for object {oid} in cluster {cid}.", "yellow"))
                        continue

                    center = np.round(np.asarray(bbox.get_center(), dtype=float), 3)  # (cx, cy, cz)
                    extent = np.round(np.asarray(bbox.get_extent(), dtype=float), 3)  # (dx, dy, dz)

                    if center is None: center = "center=None"
                    if extent is None: extent = "extent=None"
                    if cap:
                        # parts.append(f"Object {oid}: caption= {cap}; center: {center}: extent:{extent}")
                        parts.append(f"Object {oid}: center: {center}")
                    else:
                        parts.append(f"[obj:{oid}] {center} {extent}")
                if not parts:
                    return "Objects: (none)"
                return "Task relavante objects: " + " | ".join(parts)

            for cid in keep_cluster_idxs:
                items = groups.get(int(cid), [])
                if not items:
                    # print(f"\n=== Cluster {cid} ===")
                    # print("No captions.")
                    continue

                items_sorted = sorted(items, key=lambda t: t[1])  # adjust if similarity instead of distance

                chosen = []
                chosen_times = []
                chosen_docs = []
                for cap_i, sc in items_sorted:
                    doc = kept_docs[cap_i]
                    t_abs = _caption_time_seconds(doc, ref_time)
                    if all(abs(t_abs - prev) >= TIME_SEP for prev in chosen_times):
                        chosen.append((cid, cap_i, sc, t_abs))
                        chosen_times.append(t_abs)
                        chosen_docs.append(doc)
                    if len(chosen) >= AUTO_MAX_PER_CLUSTER:
                        break

                if len(chosen) < MIN_PER_CLUSTER:
                    cap_i, sc = items_sorted[0]
                    doc = kept_docs[cap_i]
                    t_abs = _caption_time_seconds(doc, ref_time)
                    chosen = [(cid, cap_i, sc, t_abs)]
                    chosen_times = [t_abs]
                    chosen_docs = [doc]

                chosen_set = set((cap_i for (_, cap_i, _, _) in chosen))
                omitted_items = [(cap_i, sc) for (cap_i, _) in items_sorted if cap_i not in chosen_set]
                omitted_docs = [kept_docs[cap_i] for (cap_i, _) in omitted_items]

                omitted_summary = _summarize_omitted_times(omitted_docs, ref_time)
                # NEW: Build object summary for this cluster (object ids, captions, bbox centers/extents)
                # obj_summary = _cluster_object_summary(int(cid), subset_local_ids)

                # Append both summaries to each selected doc's text
                addon = f" {omitted_summary}" # {obj_summary}"
                for doc in chosen_docs:
                    doc.page_content += addon
                # Append omitted summary to each selected doc's text
                # for doc in chosen_docs:
                #     doc.page_content += f" {omitted_summary}"
                if self.args.verbose:
                    print(f"\n=== Cluster {cid} ===")
                    for _, cap_i, sc, t_abs in chosen:
                        print(f"  [SELECTED] cap={cap_i} score={sc:.3f} t={t_abs:.2f}s")

                # print(colored(f"\nSelected captions for Cluster {cid}:", "yellow"))
                # _ = self.memory_to_string(chosen_docs)  # will now include omitted summary at end

                representatives.extend([(cid, cap_i, sc) for (_, cap_i, sc, _) in chosen])
            
            # Global cap across clusters
            representatives.sort(key=lambda t: t[2])   # lower distance = better
            representatives = representatives[:k]
    
            final_docs = [kept_docs[cap_i] for (_, cap_i, _) in representatives]
            final_docs_with_scores = [(kept_docs[cap_i], kept_scores[cap_i]) for (_, cap_i, _) in representatives]
            # final_docs = [doc for doc, _ in final_docs_with_scores]
            
            # # Selected indices (caption indices within the TOP_M pool)
            selected_indices = [cap_i for (_, cap_i, _) in representatives]
            selected_set = set(selected_indices)
            # print(colored(f"Selected caption indices: {selected_indices}"), "green")

            # # Build a full list for the top-M retrieved pool (times + original scores)
            time_offset = getattr(self, 'time_offset', 0.0)
            all_retrieved = []  # every retrieved item (top-M), marked if selected
            for cap_i, (doc, score) in enumerate(docs_with_scores):
                mt = doc.metadata.get('time', 0.0)
                t_val = mt[0] if isinstance(mt, (list, tuple)) else mt
                all_retrieved.append({
                    "idx": cap_i,                                # caption index within the top-M pool
                    "time": float(t_val + time_offset),          # absolute time
                    "score": float(score),                       # original score from retriever
                    "selected": (cap_i in selected_set)          # whether it was picked as representative
                })

            # Optional: also keep the final docs in working memory (comment out if you don't want this)
            # self.working_memory += final_docs

            # Persist to self.search_text
            if not hasattr(self, 'search_text'):
                self.search_text = {}

            self.search_text[query] = {
                "method_name": f"cue: {query}",
                # full top-M pool: times, scores, selection flags
                "data": all_retrieved,
                # the tuples you already produce: (cluster_id, caption_idx, score)
                "representatives": representatives,
                # the plain list of selected caption indices (easy to consume later)
                "selected_indices": selected_set,
            }
            print(colored(f"[search_by_text_IB] Total time: {time.time() - t1:.2f}s", "green"))
            '''visualize_highlighted_clusters_open3d(
                sub_objects, clusters_subset, keep_cluster_idxs, winners, [query],
                dim_alpha=0.10, save_dir=os.path.join(out_path, "clusters_vis"), show=True
            )'''
            # visualize_graph_highlight(
            #     G_sub, clusters_subset, keep_cluster_idxs,
            #     out_png=os.path.join(out_path, "graph_highlight.png"), show=True
            # )
            return self.memory_to_string(final_docs)
        except Exception as e:
                import traceback
                print(colored(f"[search_by_text_IB] ERROR: {e}", "red"))
                traceback.print_exc()
                # graceful fallback to simple top-k
                fallback_results = self.text_retriever.vectorstore.similarity_search_with_score(query, k=k)
                record_fallback_plot_data(fallback_results)
                docs = [doc for doc, _ in fallback_results]
                self.working_memory += docs
                return self.memory_to_string(docs)
        
    def search_by_text_IB(self, query: str, k=6) -> str:
        """
        Retrieve many captions, restrict to those covering objects (object_id) related to the query,
        cluster the union subset via IB (task = query), and return one best caption per cluster (diverse).
        """
        k = int(getattr(getattr(self, "args", None), "topk", k))
        t1 = time.time()
        try:
            # ---------- Tunables ----------
            TOP_M        = 100     # retrieve a big pool
            KEEP_THRESH  = 0.35    # normalized similarity threshold to keep captions
            # TASK_FILTER_THRESH = 0.4  # SBERT task relevance after merging
            GRAPH_KW = dict(
                vertical_axis='y', down_positive=True,
                dist_radius=3.0, z_overlap=False, z_slack=0.40,
                iou_thresh=0.01, covis_min=0, knn=6,
                ground_height_thresh=0.1, ground_floor_thresh=0.05,
                dilate_eps=0.02
            )

            # ---------- 1) Retrieve ----------
            docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=TOP_M)
            if not docs_with_scores:
                print(colored("Warning: No captions found for the query.", "red"))
                return ""

            # Normalize scores to [0,1]
            raw_scores = [s for _, s in docs_with_scores]
            smin, smax = min(raw_scores), max(raw_scores)
            if smax - smin < 1e-9:
                norm = [1.0 for _ in raw_scores]
            else:
                # If vectorstore returns DISTANCE (smaller=better), convert to similarity
                norm = [1.0 - (s - smin) / (smax - smin) for s in raw_scores]
            # print(colored(f"Normalized scores: {norm}", "yellow"))
            # TODO: can be removed 
            keep_idx = [i for i, v in enumerate(norm) if v >= KEEP_THRESH]
            # fallback: force top-10 (or fewer if not enough items)
            if len(keep_idx) < 12:
                idx_sorted = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
                keep_idx = idx_sorted[:min(10, len(idx_sorted))]
                # print top-10 norm scores
                #print(colored(f"Top-10 norm scores: {[norm[i] for i in idx_sorted[:10]]}", "yellow"))
            if self.args.verbose:
                print(colored(f"Keeping {len(keep_idx)} / {len(norm)} captions (>= {KEEP_THRESH:.2f})", "yellow"))
            if not keep_idx:
                keep_idx = list(range(min(TOP_M, len(docs_with_scores))))
            #keep_idx = list(range(len(docs_with_scores)))
            kept_docs   = [docs_with_scores[i][0] for i in keep_idx]
            kept_scores = [docs_with_scores[i][1] for i in keep_idx]
            
            kept_times = [docs_with_scores[i][0].metadata.get('time', 0.0) for i in keep_idx]
            # for doc, score, t in zip(kept_docs, kept_scores, kept_times):
            #     t = t[0] if isinstance(t, (list, tuple)) else t
            #     t += self.time_offset
            #     t = localtime(t)
            #     t = strftime('%Y-%m-%d %H:%M:%S', t)
            #     print(f"  [KEPT-CAND] id={doc.metadata.get('id', 'unknown')} score={score:.3f} text='At {t}{doc.page_content}...'")
            kept_norm   = [norm[i] for i in keep_idx]

            # ---------- 2) Build subset of objects from object_id ----------
            cap_to_local_obj_ids = []
            subset_local_ids_set = set()
            for doc in kept_docs:
                gids = _parse_object_ids(doc.metadata.get('object_id'))
                lids = _global_ids_to_local_indices(gids, objid_to_idx=getattr(self, 'objid_to_idx', None))
                lids = list(set(lids))
                cap_to_local_obj_ids.append(lids)
                subset_local_ids_set.update(lids)

            subset_local_ids = sorted(list(subset_local_ids_set))
            if len(subset_local_ids) == 0:
                print(colored("No objects covered by captions; falling back to top-k plain retrieval.", "red"))
                topk_docs = [doc for doc, _ in docs_with_scores[:k]]
                selected_set = set(range(min(k, len(docs_with_scores))))
                time_offset = getattr(self, 'time_offset', 0.0)
                all_retrieved = []
                for cap_i, (doc, score) in enumerate(docs_with_scores):
                    mt = doc.metadata.get('time', 0.0)
                    t_val = mt[0] if isinstance(mt, (list, tuple)) else mt
                    all_retrieved.append({
                        "idx": cap_i,
                        "time": float(t_val + time_offset),
                        "score": float(score),
                        "selected": (cap_i in selected_set),
                    })
                if not hasattr(self, 'search_text'):
                    self.search_text = {}
                self.search_text[query] = {
                    "method_name": f"cue: {query}",
                    "data": all_retrieved,
                    "selected_indices": selected_set,
                    "fallback": "topk_plain_retrieval",
                }
                self.working_memory += topk_docs
                return self.memory_to_string(topk_docs)

            # ---------- 3) Subgraph + IB on subset ----------
            # NOTE: if your objects live in self.objects, replace self.scene_graph with self.objects below
            object_cap_features_np = np.array([obj['ft'].flatten() for obj in self.scene_graph])  # (N,D)
            # visualize_objects_org(self.scene_graph)
            
            G_sub, sub_objects, sub_feats = _build_subgraph_from_indices_v0(
                self.scene_graph, object_cap_features_np, subset_local_ids, GRAPH_KW
            )
            # visualize_objects_org(sub_objects)
            # Task feature from query (SBERT) -> (1, D)
            task_ft = _encode_query_sbert(self.sbert_model, query).detach().cpu().numpy()[None, :]

            # IB config
            out_path = getattr(self, 'out_dir', './_tmp')
            Path(out_path).mkdir(parents=True, exist_ok=True)
            cfg_yaml_path = os.path.join(out_path, "cluster_config.yaml")
            write_default_cluster_config(cfg_yaml_path)

            # IB clusters in subset index space
            clusters_subset = run_ib_clustering(sub_feats, task_ft, G_sub, cfg_yaml_path)
            
            # ---------- 4) Merge + filter by task relevance ----------
            # clustered_objects = merge_all_clusters(sub_objects, clusters_subset)
            # print(colored(f"[AIB] After merge: {len(clustered_objects)} objects.", "cyan"))

            # clustered_objects_filtered = filter_clusters_by_task(clustered_objects, task_ft, TASK_FILTER_THRESH)
            # # clustered_objects_filtered = filter_clusters_by_task_topk(clustered_objects, task_ft, topk=6)
            # print(colored(f"[AIB] Filtered to {len(clustered_objects_filtered)} task-relevant objects.", "cyan"))


            # ----- 5) Score clusters by task & keep only task-relevant ones -----
            # print("\n=== Step 5: Scoring clusters by task relevance ===")
            scores, winners = cluster_task_scores_cosine(sub_objects, clusters_subset, task_ft, pool="max")
            #print(f"Cluster scores vs task: {scores}")
            #print(f"Best matching task index per cluster: {winners}")

            HIGHLIGHT_THR = 0.4  # align with TASK_FILTER_THRESH
            
            keep_cluster_idxs = np.where(scores >= HIGHLIGHT_THR)[0]
            if keep_cluster_idxs.size < k:
                keep_cluster_idxs = np.argsort(-scores)[:k] #min(k, len(clusters_subset))
                print(f"No clusters above {HIGHLIGHT_THR:.2f}, keeping top {len(keep_cluster_idxs)} by score. \n{keep_cluster_idxs}")
            else:
                keep_cluster_idxs = np.argsort(-scores)[:min(k, len(clusters_subset))]
                print(f"Keeping {len(keep_cluster_idxs)} clusters above {HIGHLIGHT_THR:.2f} \n{keep_cluster_idxs}")
            
            
            # ----- 6) Map captions → dominant kept cluster via object ids -----
            if self.args.verbose:
                for cid in keep_cluster_idxs:
                    print(f"cluster {cid}: score={scores[cid]:.4f}")
                print(colored(f"Building subgraph from {len(subset_local_ids)} objects", "blue"))
                print(colored(f"[AIB] Formed {len(clusters_subset)} clusters from {len(subset_local_ids)} primitives.", "cyan"))

                print("\n=== Step 6: Mapping captions to dominant clusters ===")
            
            subset_local_to_pos = {lid: pos for pos, lid in enumerate(subset_local_ids)}
            objpos_to_keptcid = {}
            for cid in keep_cluster_idxs:
                for pos in clusters_subset[cid]:
                    objpos_to_keptcid[pos] = cid
            if self.args.verbose:
                print(f"subset_local_to_pos: {subset_local_to_pos}")
                print(f"objpos_to_keptcid: {objpos_to_keptcid}")

            groups = {int(cid): [] for cid in keep_cluster_idxs}
            for cap_i, lids in enumerate(cap_to_local_obj_ids):
                hits = []
                for lid in lids:
                    pos = subset_local_to_pos.get(lid, None)
                    if pos is None:
                        continue
                    cid = objpos_to_keptcid.get(pos, None)
                    if cid is not None:
                        hits.append(int(cid))
                if not hits:
                    continue
                dom = max(set(hits), key=hits.count)
                groups[dom].append((cap_i, kept_scores[cap_i]))
                if self.args.verbose:
                    print(f"Under query: {query}; Caption {cap_i} with lids {lids} -> dominant cluster {dom}")
            if self.args.verbose:
                print(f"Grouped captions by cluster (before selection):\n{groups}")

                # ----- 7) Pick top-N per cluster (by score) for diversity -----
                print("\n=== Step 7: Selecting top captions per cluster ===")

            # ---- Parameters (tune) ----
            TIME_SEP = 90.0            # min temporal separation (seconds) within a cluster’s chosen reps
            AUTO_MAX_PER_CLUSTER = 3   # hard cap (we’ll choose as many as pass TIME_SEP, up to this)
            MIN_PER_CLUSTER = 1        # always keep at least one if the cluster has any candidates

            representatives = []
            ref_time = getattr(self, 'time_offset', 0.0)
            def _ensure_o3d_aabb(bbox):
                            """
                            Convert to Open3D AxisAlignedBoundingBox if possible.
                            - Keep AABB as-is
                            - Convert OOBB -> AABB
                            - Convert Nx3 array -> AABB
                            """
                            if bbox is None:
                                return None
                            if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
                                return bbox
                            if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
                                return bbox.get_axis_aligned_bounding_box()
                            try:
                                arr = np.asarray(bbox)
                                if arr.ndim == 2 and arr.shape[1] == 3:
                                    return o3d.geometry.AxisAlignedBoundingBox(arr.min(axis=0), arr.max(axis=0))
                            except Exception:
                                pass
                            return None
            # Build a compact per-cluster object summary string
            def _cluster_object_summary(cid, subset_local_ids):
                members = clusters_subset[cid]
                print(colored(f"Cluster {cid} has {members} members.", "cyan"))
                
                parts = []
                for m in members:
                    o = sub_objects[m]
                    # Prefer global id if present; fallback to any available identifier
                    oid = o.get('global_id', o.get('object_id', o.get('id', f"{subset_local_ids[m]}")))
                    cap = o.get('caption', o.get('desc', ''))

                    bbox  = _ensure_o3d_aabb(o.get("bbox", None))
                    if bbox is None:
                        print(colored(f"Warning: No bbox for object {oid} in cluster {cid}.", "yellow"))
                        continue

                    center = np.round(np.asarray(bbox.get_center(), dtype=float), 3)  # (cx, cy, cz)
                    extent = np.round(np.asarray(bbox.get_extent(), dtype=float), 3)  # (dx, dy, dz)

                    if center is None: center = "center=None"
                    if extent is None: extent = "extent=None"
                    if cap:
                        # parts.append(f"Object {oid}: caption= {cap}; center: {center}: extent:{extent}")
                        parts.append(f"Object {oid}: center: {center}")
                    else:
                        parts.append(f"[obj:{oid}] {center} {extent}")
                if not parts:
                    return "Objects: (none)"
                return "Task relavante objects: " + " | ".join(parts)

            for cid in keep_cluster_idxs:
                items = groups.get(int(cid), [])
                if not items:
                    # print(f"\n=== Cluster {cid} ===")
                    # print("No captions.")
                    continue

                items_sorted = sorted(items, key=lambda t: t[1])  # adjust if similarity instead of distance

                chosen = []
                chosen_times = []
                chosen_docs = []
                for cap_i, sc in items_sorted:
                    doc = kept_docs[cap_i]
                    t_abs = _caption_time_seconds(doc, ref_time)
                    if all(abs(t_abs - prev) >= TIME_SEP for prev in chosen_times):
                        chosen.append((cid, cap_i, sc, t_abs))
                        chosen_times.append(t_abs)
                        chosen_docs.append(doc)
                    if len(chosen) >= AUTO_MAX_PER_CLUSTER:
                        break

                if len(chosen) < MIN_PER_CLUSTER:
                    cap_i, sc = items_sorted[0]
                    doc = kept_docs[cap_i]
                    t_abs = _caption_time_seconds(doc, ref_time)
                    chosen = [(cid, cap_i, sc, t_abs)]
                    chosen_times = [t_abs]
                    chosen_docs = [doc]

                chosen_set = set((cap_i for (_, cap_i, _, _) in chosen))
                omitted_items = [(cap_i, sc) for (cap_i, _) in items_sorted if cap_i not in chosen_set]
                omitted_docs = [kept_docs[cap_i] for (cap_i, _) in omitted_items]

                omitted_summary = _summarize_omitted_times(omitted_docs, ref_time)
                # NEW: Build object summary for this cluster (object ids, captions, bbox centers/extents)
                # obj_summary = _cluster_object_summary(int(cid), subset_local_ids)

                # Append both summaries to each selected doc's text
                addon = f" {omitted_summary}" # {obj_summary}"
                for doc in chosen_docs:
                    doc.page_content += addon
                # Append omitted summary to each selected doc's text
                # for doc in chosen_docs:
                #     doc.page_content += f" {omitted_summary}"
                if self.args.verbose:
                    print(f"\n=== Cluster {cid} ===")
                    for _, cap_i, sc, t_abs in chosen:
                        print(f"  [SELECTED] cap={cap_i} score={sc:.3f} t={t_abs:.2f}s")

                # print(colored(f"\nSelected captions for Cluster {cid}:", "yellow"))
                # _ = self.memory_to_string(chosen_docs)  # will now include omitted summary at end

                representatives.extend([(cid, cap_i, sc) for (_, cap_i, sc, _) in chosen])
            
            # Global cap across clusters
            representatives.sort(key=lambda t: t[2])   # lower distance = better
            representatives = representatives[:k]
    
            final_docs = [kept_docs[cap_i] for (_, cap_i, _) in representatives]
            final_docs_with_scores = [(kept_docs[cap_i], kept_scores[cap_i]) for (_, cap_i, _) in representatives]
            # final_docs = [doc for doc, _ in final_docs_with_scores]
            
            # # Selected indices (caption indices within the TOP_M pool)
            selected_indices = [keep_idx[cap_i] for (_, cap_i, _) in representatives]
            if len(final_docs) < k:
                used_doc_ids = {id(doc) for doc in final_docs}
                for retrieved_idx, (doc, _) in enumerate(docs_with_scores):
                    if id(doc) in used_doc_ids:
                        continue
                    final_docs.append(doc)
                    used_doc_ids.add(id(doc))
                    selected_indices.append(retrieved_idx)
                    if len(final_docs) >= k:
                        break
                if self.args.verbose:
                    print(colored(
                        f"Filled selected captions to {len(final_docs)} / {k} using plain retrieval.",
                        "yellow",
                    ))
            selected_set = set(selected_indices)
            # print(colored(f"Selected caption indices: {selected_indices}"), "green")

            # # Build a full list for the top-M retrieved pool (times + original scores)
            time_offset = getattr(self, 'time_offset', 0.0)
            all_retrieved = []  # every retrieved item (top-M), marked if selected
            for cap_i, (doc, score) in enumerate(docs_with_scores):
                mt = doc.metadata.get('time', 0.0)
                t_val = mt[0] if isinstance(mt, (list, tuple)) else mt
                all_retrieved.append({
                    "idx": cap_i,                                # caption index within the top-M pool
                    "time": float(t_val + time_offset),          # absolute time
                    "score": float(score),                       # original score from retriever
                    "selected": (cap_i in selected_set)          # whether it was picked as representative
                })

            # Optional: also keep the final docs in working memory (comment out if you don't want this)
            # self.working_memory += final_docs

            # Persist to self.search_text
            if not hasattr(self, 'search_text'):
                self.search_text = {}

            self.search_text[query] = {
                "method_name": f"cue: {query}",
                # full top-M pool: times, scores, selection flags
                "data": all_retrieved,
                # the tuples you already produce: (cluster_id, caption_idx, score)
                "representatives": representatives,
                # the plain list of selected caption indices (easy to consume later)
                "selected_indices": selected_set,
            }
            print(colored(f"[search_by_text_IB] Total time: {time.time() - t1:.2f}s", "green"))
            '''visualize_highlighted_clusters_open3d(
                sub_objects, clusters_subset, keep_cluster_idxs, winners, [query],
                dim_alpha=0.10, save_dir=os.path.join(out_path, "clusters_vis"), show=True
            )'''
            # visualize_graph_highlight(
            #     G_sub, clusters_subset, keep_cluster_idxs,
            #     out_png=os.path.join(out_path, "graph_highlight.png"), show=True
            # )
            return self.memory_to_string(final_docs)
        except Exception as e:
            import traceback
            print(colored(f"[search_by_text_IB] ERROR: {e}", "red"))
            traceback.print_exc()
            # graceful fallback to simple top-k
            docs = [doc for doc, _ in self.text_retriever.vectorstore.similarity_search_with_score(query, k=k)]
            self.working_memory += docs
            return self.memory_to_string(docs)
        
    def search_by_text_IB_working(self, query: str, k=6) -> str:
        """
        Retrieve many captions, restrict to those covering objects (object_id) related to the query,
        cluster the union subset via IB (task = query), and return one best caption per cluster (diverse).
        """
        try:
            verbose = bool(getattr(getattr(self, "args", None), "verbose", False))
            # ---------- Tunables ----------
            TOP_M        = 100     # retrieve a big pool
            KEEP_THRESH  = 0.45    # normalized similarity threshold to keep captions
            TASK_FILTER_THRESH = 0.55  # SBERT task relevance after merging
            GRAPH_KW = dict(
                vertical_axis='y', down_positive=True,
                dist_radius=3.0, z_overlap=False, z_slack=0.40,
                iou_thresh=0.01, covis_min=0, knn=6,
                ground_height_thresh=0.1, ground_floor_thresh=0.05,
                dilate_eps=0.02
            )

            # ---------- 1) Retrieve ----------
            docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=TOP_M)
            if not docs_with_scores:
                print(colored("Warning: No captions found for the query.", "red"))
                return ""

            # Normalize scores to [0,1]
            raw_scores = [s for _, s in docs_with_scores]
            smin, smax = min(raw_scores), max(raw_scores)
            if smax - smin < 1e-9:
                norm = [1.0 for _ in raw_scores]
            else:
                # If vectorstore returns DISTANCE (smaller=better), convert to similarity
                norm = [1.0 - (s - smin) / (smax - smin) for s in raw_scores]

            keep_idx = [i for i, v in enumerate(norm) if v >= KEEP_THRESH]
            if verbose:
                print(colored(f"Keeping {len(keep_idx)} / {len(norm)} captions (>= {KEEP_THRESH:.2f})", "yellow"))
            if not keep_idx:
                keep_idx = list(range(min(TOP_M, len(docs_with_scores))))

            kept_docs   = [docs_with_scores[i][0] for i in keep_idx]
            kept_scores = [docs_with_scores[i][1] for i in keep_idx]
            kept_norm   = [norm[i] for i in keep_idx]

            # ---------- 2) Build subset of objects from object_id ----------
            cap_to_local_obj_ids = []
            subset_local_ids_set = set()
            for doc in kept_docs:
                gids = _parse_object_ids(doc.metadata.get('object_id'))
                lids = _global_ids_to_local_indices(gids, objid_to_idx=getattr(self, 'objid_to_idx', None))
                lids = list(set(lids))
                cap_to_local_obj_ids.append(lids)
                subset_local_ids_set.update(lids)

            subset_local_ids = sorted(list(subset_local_ids_set))
            if len(subset_local_ids) == 0:
                print(colored("No objects covered by captions; falling back to top-k plain retrieval.", "red"))
                topk_docs = [doc for doc, _ in docs_with_scores[:k]]
                self.working_memory += topk_docs
                return self.memory_to_string(topk_docs)

            # ---------- 3) Subgraph + IB on subset ----------
            # NOTE: if your objects live in self.objects, replace self.scene_graph with self.objects below
            object_cap_features_np = np.array([obj['ft'].flatten() for obj in self.scene_graph])  # (N,D)
            if verbose:
                print(colored(f"Building subgraph from {len(subset_local_ids)} objects", "blue"))
            G_sub, sub_objects, sub_feats = _build_subgraph_from_indices(
                self.scene_graph, object_cap_features_np, subset_local_ids, GRAPH_KW
            )

            # Task feature from query (SBERT) -> (1, D)
            task_ft = _encode_query_sbert(self.sbert_model, query).detach().cpu().numpy()[None, :]

            # IB config
            out_path = getattr(self, 'out_dir', './_tmp')
            Path(out_path).mkdir(parents=True, exist_ok=True)
            cfg_yaml_path = os.path.join(out_path, "cluster_config.yaml")
            write_default_cluster_config(cfg_yaml_path)

            # IB clusters in subset index space
            clusters_subset = run_ib_clustering(sub_feats, task_ft, G_sub, cfg_yaml_path)
            if verbose:
                print(colored(f"[AIB] Formed {len(clusters_subset)} clusters from {len(subset_local_ids)} primitives.", "cyan"))

            # ---------- 4) Merge + filter by task relevance ----------
            clustered_objects = merge_all_clusters(sub_objects, clusters_subset)
            if verbose:
                print(colored(f"[AIB] After merge: {len(clustered_objects)} objects.", "cyan"))

            clustered_objects_filtered = filter_clusters_by_task(clustered_objects, task_ft, TASK_FILTER_THRESH)
            if verbose:
                print(colored(f"[AIB] Filtered to {len(clustered_objects_filtered)} task-relevant objects.", "cyan"))

            # ---------- 5) Group captions by IB cluster and pick one per cluster (diversity) ----------
            # map: subset local id -> subset position (0..Ns-1)
            subset_local_to_pos = {lid: pos for pos, lid in enumerate(subset_local_ids)}
            # map: obj subset pos -> cluster id
            objpos_to_cid = {}
            for cid, members in enumerate(clusters_subset):
                for pos in members:
                    objpos_to_cid[pos] = cid

            # group captions by the cluster they “hit” most through their object ids
            groups = {}
            for cap_i, lids in enumerate(cap_to_local_obj_ids):
                hits = []
                for lid in lids:
                    pos = subset_local_to_pos.get(lid, None)
                    if pos is None:
                        continue
                    cid = objpos_to_cid.get(pos, None)
                    if cid is not None:
                        hits.append(cid)
                if not hits:
                    continue
                cid = max(set(hits), key=hits.count)
                groups.setdefault(cid, []).append((cap_i, kept_scores[cap_i]))

            # pick best doc per cluster by score (assuming score is distance → lower is better)
            reps = []
            for cid, items in groups.items():
                best_cap_i, best_score = min(items, key=lambda t: t[1])
                reps.append((cid, best_cap_i, best_score))
            reps.sort(key=lambda t: t[2])
            reps = reps[:k]

            final_docs = [kept_docs[best_i] for (_, best_i, _) in reps]

            # Fill if < k (optional): add other strong-but-unused captions
            if len(final_docs) < k:
                used = set(id(d) for d in final_docs)
                for d, _ in docs_with_scores:
                    if id(d) not in used:
                        final_docs.append(d)
                    if len(final_docs) >= k:
                        break

            # ---------- 6) Optional diagnostics / visuals ----------
            # Print per cluster
            task_texts = [query]
            if verbose:
                print_task_related_captions(sub_objects, clusters_subset, task_texts, task_ft, sim_threshold=0.45)

            # ---------- 7) Bookkeeping + plotting payload ----------
            self.working_memory += final_docs

            # Store plot data for all M retrieved caps
            data = []
            for (doc, score), n in zip(docs_with_scores, norm):
                mt = doc.metadata.get('time', 0.0)
                t_val = mt[0] if isinstance(mt, (list, tuple)) else mt
                data.append({
                    "time": t_val + getattr(self, 'time_offset', 0.0),
                    "score": score,
                    "score_normalized": n
                })
            if not hasattr(self, 'search_text'):
                self.search_text = {}
            self.search_text[query] = {
                "method_name": f"text_search_diverse_{query}",
                "data": data,
                "score_min": smin,
                "score_max": smax,
            }

            # ---------- 8) Return concise, diverse memory string ----------
            return self.memory_to_string(final_docs)

        except Exception as e:
            import traceback
            print(colored(f"[search_by_text_IB] ERROR: {e}", "red"))
            traceback.print_exc()
            # graceful fallback to simple top-k
            docs = [doc for doc, _ in self.text_retriever.vectorstore.similarity_search_with_score(query, k=k)]
            self.working_memory += docs
            return self.memory_to_string(docs)



    def search_by_text_hybrid(self, query: str, k =5) -> str:

        docs = self.text_retriever.invoke(query, k=k)
        # docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k)

        self.working_memory += docs
        memory_list = docs
        docs = self.memory_to_string(docs)

        """Look up things online."""        
        return docs, memory_list
    
    

    ### Doc formatting for the last LLM
    def memory_to_string(self, memory_list: list[MemoryItem], ref_time: float=None):
        if ref_time == None:
            ref_time = self.time_offset
        # print(f"memory_list: {memory_list}")
        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        # print(f"retrieved video captions: \n{out_string}")
        print(colored(f"Retrieved video captions: \n{out_string}", "yellow"))
        cot_log_file = getattr(self, "cot_log_file", None)
        if cot_log_file and out_string:
            try:
                os.makedirs(os.path.dirname(cot_log_file), exist_ok=True)
                with open(cot_log_file, "a") as f:
                    f.write("System: Retrieved video captions:\n")
                    f.write(out_string)
                    if not out_string.endswith("\n"):
                        f.write("\n")
            except OSError as exc:
                print(colored(f"Failed to write retrieved captions to CoT log: {exc}", "red"))
        return out_string
    
    def format_scene_objects_for_llm(self, objects: List[Dict]) -> str:
        """
        Format retrieved scene objects into a readable string for LLM consumption.
        Each object will be described with its caption, ID, and approximate location from bbox center.
        """
        if not objects:
            return "No scene objects matched the query."

        lines = ["Here are the top matched scene objects:"]
        for idx, obj in enumerate(objects, start=1):
            caption = obj.get('caption', 'unknown')
            obj_id = obj.get('obj_id', 'N/A')
            # time info
            times = obj.get('time', 'unknown')
            # time_dt = [datetime.fromtimestamp(t) for t in times]
            time_dt = [datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') for t in times]
            intervals = self.merge_time_intervals(time_dt)
            description = self.format_intervals_compact(intervals)
            try:
                center = _bbox_center(obj['bbox'])
                pos_str = f"at position [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
            except Exception:
                pos_str = "with unknown position"
            lines.append(f"Object ID {obj_id}: Located at {pos_str}; \"{caption}\". detected at times {description}.") #f"{idx}. Object ID {obj_id}: \"{caption}\" {pos_str}. detected at times {time_dt}.")
            print(colored(f"Object ID {obj_id}: Located at {pos_str}; \"{caption}\". detected at times {description}.", "yellow"))
    
        return "\n".join(lines)
    
    # from datetime import datetime, timedelta

    def merge_time_intervals(self, time_list_str, max_gap_sec=1):
        times = sorted([datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in time_list_str])
        if not times:
            return []

        intervals = []
        start = times[0]
        end = times[0]

        for curr in times[1:]:
            if (curr - end).total_seconds() <= max_gap_sec:
                end = curr
            else:
                intervals.append((start, end))
                start = curr
                end = curr
        intervals.append((start, end))

        return intervals

    def format_intervals_compact(self, intervals):
        if not intervals:
            return "No observation intervals found."

        first_day = intervals[0][0].strftime('%Y-%m-%d')
        lines = [f"The object was observed on {first_day} during:"]
        
        for s, e in intervals:
            if s.date() != e.date():
                # Display intervals separately when they span multiple days.
                lines.append(f" {s.strftime('%Y-%m-%d %H:%M:%S')} ~ {e.strftime('%Y-%m-%d %H:%M:%S')}")
            elif s == e:
                lines.append(f" {s.strftime('%H:%M:%S')}")
            else:
                lines.append(f" {s.strftime('%H:%M:%S')} ~ {e.strftime('%H:%M:%S')}")

        return ",".join(lines)

    
    def set_scene_graph(self, scene_graph):
        """Attach a SceneGraph for scene object retrieval."""
        self.scene_graph = scene_graph
    
    def search_scenegraph(self, query: str, top_k_scene=10) -> str:
        """
        Search scene objects using SBERT features for matching and store plotting data.
        """
        text_query_ft = self.sbert_model.encode([query], convert_to_tensor=True)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        top_k_scene = self.args.topk
        scored_objects = []
        for obj in self.scene_graph:
            if 'ft' not in obj or obj['ft'] is None:
                print(f"Object {obj.get('id', 'unknown')} does not have ft, skipping.")
                continue
            
            obj_ft = torch.tensor(obj['ft'], device=text_query_ft.device)
            obj_ft = obj_ft / obj_ft.norm(dim=-1, keepdim=True)

            score = F.cosine_similarity(text_query_ft, obj_ft.unsqueeze(0), dim=-1).item()
            scored_objects.append((obj, score))

        scored_objects.sort(key=lambda x: -x[1])

        # Store plot data
        plot_data = []
        for idx, (obj, score) in enumerate(scored_objects[:top_k_scene]):
            times = obj.get('time', [])
            if not isinstance(times, list):
                times = [times]  # Make sure it's a list

            plot_data.append({
                "object_id": obj.get('obj_id', f"obj_{idx}"),
                "times": times,
                "score": score
            })
        # Save for downstream plotting
        # self.search_SG = {
        #     "method_name": "scenegraph_search",
        #     "data": plot_data
        # }
        if not hasattr(self, 'search_text'):
            self.search_text = {}
    
        self.search_SG[query] = {
        "method_name": f"SG_{query}",
        "data": plot_data,
        # "score_min": min_score,
        # "score_max": max_score,
        }

        retrieved_objects = [obj for obj, _ in scored_objects[:top_k_scene]]
        doc = self.format_scene_objects_for_llm(retrieved_objects)
        print(colored(f"docs for scene graph search: \n, {doc}", "yellow"))
        cot_log_file = getattr(self, "cot_log_file", None)
        if cot_log_file and doc:
            try:
                os.makedirs(os.path.dirname(cot_log_file), exist_ok=True)
                with open(cot_log_file, "a") as f:
                    f.write("System: Retrieved scene graph objects:\n")
                    f.write(doc)
                    if not doc.endswith("\n"):
                        f.write("\n")
            except OSError as exc:
                print(colored(f"Failed to write scene graph retrieval to CoT log: {exc}", "red"))
        return doc



def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params

        # Determine result metadata fields with PK.
        output_fields = pos_db.fields[:]
        # output_fields.remove(pos_db._vector_field) # NOTE: Only thing removed
        timeout = pos_db.timeout or timeout
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = pos_db._parse_document(data)
            pair = (doc, result.score)
            ret.append(pair)

        return ret #[doc for doc, _ in ret]
