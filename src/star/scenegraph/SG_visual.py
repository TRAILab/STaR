
import re
import pandas as pd
import sys
import glob
import uuid
import time
import json
import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).resolve().parent.parent))

import gzip
import pickle
import os, sys
import argparse
import traceback 
import distinctipy
from datetime import datetime

import numpy as np
import pickle as pkl
from termcolor import colored
from dataclasses import asdict
from PIL import Image as PILImage
from time import strftime, localtime
from star.some_class.map_class import MapObjectList

from star.utils.world_map import ObjectVisualizer, visualize_objects, visualize_objects_org #visualize_objects_selected

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
sys.path.append(sys.path[0] + '/..')


from star.utils.util import get_caption, assign_frame_indices, assign_object_ids, extract_gt_times, plot_multi_method_scores, plot_scenegraph_scores # plot_retrieval_scores_with_gt


import torch
import open3d as o3d
import networkx as nx
from pathlib import Path
from termcolor import colored
from collections import Counter
import clio_batch.helpers as helpers
from sentence_transformers import SentenceTransformer
from clio_batch.ib_cluster import ClusterIB, ClusterIBConfig
from clio_batch.aib_helper import cluster_task_scores, select_relevant_clusters, visualize_highlighted_clusters_open3d, visualize_graph_highlight, build_object_graph_smart

from copy import deepcopy

# Vivid, high-contrast base palette (RGB in [0,1]); adapted from Tableau/Glasbey-like sets
_BASE_VIVID = [
    (0.121, 0.466, 0.705),  # blue
    (1.000, 0.498, 0.054),  # orange
    (0.172, 0.627, 0.172),  # green
    (0.839, 0.152, 0.156),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.811),  # cyan
    (0.000, 0.000, 0.000),  # black
    (1.000, 0.000, 0.000),  # bright red
]


# Fixed palette of readable colors (name -> RGB)
_PALETTE = [
    ("red",        (1.0, 0.0, 0.0)),
    ("blue",       (0.0, 0.0, 1.0)),
    ("green",      (0.0, 1.0, 0.0)),
    ("orange",     (1.0, 0.5, 0.0)),
    ("purple",     (0.5, 0.0, 0.5)),
    ("cyan",       (0.0, 1.0, 1.0)),
    ("magenta",    (1.0, 0.0, 1.0)),
    ("yellow",     (1.0, 1.0, 0.0)),
    ("teal",       (0.0, 0.5, 0.5)),
    ("pink",       (1.0, 0.4, 0.7)),
    ("lime",       (0.7, 1.0, 0.0)),
    ("indigo",     (0.3, 0.0, 0.5)),
]

def _get_distinct_palette(n: int, seed: int = 7):
    """
    Return a palette of n distinct colors in [0,1], guaranteeing at least 10
    highly distinct colors by starting from a vivid base palette and extending
    with distinctipy if necessary.
    """
    m = max(n, 10)  # ensure at least 10 colors
    if m <= len(_BASE_VIVID):
        return _BASE_VIVID[:m]
    # Extend palette with additional distinct colors while excluding the base ones
    extra = distinctipy.get_colors(m - len(_BASE_VIVID), exclude_colors=_BASE_VIVID, rng=seed)
    return _BASE_VIVID + extra


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def evaluate_output(qa_instance, predicted):

    out_error = {}

    q_type = qa_instance['type']
    if 'position' in q_type:

        answer = np.array(qa_instance['answers']['position'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        pred_pos = np.array(predicted['position'])

        dist = np.linalg.norm(answer - pred_pos)

        out_error['position_error'] = dist

    elif 'binary' in q_type:

        answer = qa_instance['answers']['text'][1] # we made this assumption in other examples that binary answer is the second one

        if 'binary' in predicted and (predicted['binary'].lower() == "yes" or predicted['binary'].lower() == "no"):
            # get correct/incorrect label
            if predicted['binary'].lower() == answer.lower():
                correct = 1
            else:
                correct = 0

            out_error['binary_iscorrect'] = correct

    elif 'time' in q_type:

        answer = np.array(qa_instance['answers']['time'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['time']) == str:
            predicted['time'] = eval(predicted['time'])
        pred_time = np.array(predicted['time'])

        dist = abs(answer - pred_time)

        out_error['time_error'] = dist

    elif 'duration' in q_type:

        answer = np.array(qa_instance['answers']['duration'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['duration']) == str:
            predicted['duration'] = eval(predicted['duration'])
        pred_time = np.array(predicted['duration'])

        dist = abs(answer - pred_time)

        out_error['duration_error'] = dist

    elif 'text' in q_type:
        answer = qa_instance['answers']['text']
        out_error = {'answer': answer}

    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def answer_squad_question(model, question, qa_instance):

    print(colored(f'Question: {question}', "green", attrs=["bold"]))

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question) # the key part of the code

            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)

            out_error = evaluate_output(qa_instance, parsed)

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed

        return return_dict



def _rand_color_from_index(i: int) -> list:
    """Deterministic bright-ish RGB color from an index."""
    rng = np.random.default_rng(i + 12345)
    # avoid too-dark colors
    return (rng.random(3) * 0.6 + 0.3).tolist()

def _ensure_o3d_aabb(bbox) -> o3d.geometry.AxisAlignedBoundingBox | None:
    """
    Convert various bbox formats to an Open3D AxisAlignedBoundingBox if needed.
    - If already an Open3D bbox (AABB/OOBB), return it.
    - If array-like of shape (N, 3), build an AABB from min/max.
    - Otherwise, return None.
    """
    if bbox is None:
        return None

    # Already an Open3D bbox?
    if hasattr(bbox, "get_center") and (
        isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox) or
        isinstance(bbox, o3d.geometry.OrientedBoundingBox)
    ):
        # Convert OOBB to AABB just for consistent drawing (optional)
        if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
            return bbox
        else:
            return bbox.get_axis_aligned_bounding_box()

    # Try array-like
    try:
        arr = np.asarray(bbox)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return o3d.geometry.AxisAlignedBoundingBox(arr.min(axis=0), arr.max(axis=0))
    except Exception:
        pass

    return None

def _pcd_copy(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Safe copy of an Open3D PointCloud."""
    if not isinstance(pcd, o3d.geometry.PointCloud):
        return None
    p = o3d.geometry.PointCloud()
    p.points = deepcopy(pcd.points)
    if pcd.has_colors():
        p.colors = deepcopy(pcd.colors)
    if pcd.has_normals():
        p.normals = deepcopy(pcd.normals)
    return p
'''
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


def _color_from_index(i, total):
    """Generate distinct colors based on index (HSV → RGB)."""
    hue = i / max(1, total)
    return list(o3d.utility.Vector3dVector(
        np.array([[(hue + 0.3) % 1, 1.0, 1.0]])
    ).to_numpy()[0])  # quick hack, but works as stable color


def visualize_all_with_highlight(objects_all, selected_idxs):
    """
    Show all pcds unchanged, highlight selected with different-colored bboxes.
    Prints index ↔ color mapping.

    Args:
        objects_all: list of dicts, each with keys incl. 'pcd' and 'bbox'
        selected_idxs: list[int], indices to highlight
    """
    geoms = []

    # 1) Add all PCDs as-is
    for obj in objects_all:
        pcd = obj.get("pcd", None)
        if isinstance(pcd, o3d.geometry.PointCloud) and len(pcd.points) > 0:
            geoms.append(pcd)

    # 2) Selected → bboxes with distinct colors
    nsel = len(selected_idxs)
    if nsel > 0:
        print("\n=== Highlighted primitives ===")
    for rank, i in enumerate(selected_idxs):
        if i < 0 or i >= len(objects_all):
            continue
        bbox = _ensure_o3d_aabb(objects_all[i].get("bbox", None))
        if bbox is not None:
            color = _color_from_index(rank, nsel) if nsel > 1 else [1.0, 0.0, 0.0]
            bbox.color = color
            geoms.append(bbox)
            print(f"Primitive {i} → color {np.round(color, 2).tolist()}")
    if nsel > 0:
        print("==============================\n")

    if not geoms:
        print("[visualize_all_with_highlight] Nothing to render.")
        return

    o3d.visualization.draw_geometries(geoms)

# =========================
# Interactive loop example
# =========================

def interactive_check(objects_all: list[dict], caption_key: str = "caption"):
    """
    Type comma-separated indices each round to:
      1) print captions and centers
      2) visualize pcd + bbox + center markers in an Open3D window
    """
    while True:
        raw_input_str = input("Enter primitive indices (e.g., '0, 3, 7'), or 'q' to quit: ").strip()
        if raw_input_str.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        # Parse indices
        try:
            idxs = [int(x.strip()) for x in raw_input_str.split(",") if x.strip()]
            if not idxs:
                print("No indices provided.")
                continue
        except ValueError:
            print("Invalid input: please provide integers separated by commas.")
            continue
        
        info = visualize_all_with_highlight(objects_all, idxs)
        # Access returned results
        for idx, vals in info.items():
            print(f"Primitive {idx}: center={vals['center']}, extent={vals['extent']}, color={vals['color']}")


        # Collect info (uses the function we wrote earlier)
        # bboxes, captions, centers = collect_bbox_caption_centers(objects_all, idxs, caption_key=caption_key)

        # # Print summary

        # # Visualize

'''
def load_scenegraph(result, start_time=None, end_time=None, dataset_start_timestamp=None, fps=10):
    '''
    Load the full scene graph, and optionally filter objects by time window.

    Args:
        result_path (str): path to scene graph pkl
        start_time (float): task start time (absolute, e.g., seconds since epoch)
        end_time (float): task end time
        dataset_start_timestamp (float): the timestamp corresponding to image_idx=0
        fps (int): frames per second (default 10)
    '''
    objects, objects_all = result[0], result[1]
        
        
    #     if results['bg_objects'] is None:
    #         bg_objects = None
    #     else:
    #         bg_objects = MapObjectList()
    #         bg_objects.load_serializable(results["bg_objects"])
        
    # elif isinstance(results, list):
    #     objects = MapObjectList()
    #     objects.load_serializable(results)
    #     bg_objects = None
    # else:
    #     raise ValueError("Unknown results type: ", type(results))
    

    if start_time is not None and end_time is not None and dataset_start_timestamp is not None:
        filtered_objects = []
        for obj in objects:
            if 'image_idx' not in obj:
                continue
            # Remove duplicates and sort indices
            idx_list = obj['image_idx']
            matched = False

            # Check if any index falls into the time window
            for idx in idx_list:
                obj_time = dataset_start_timestamp + (idx*10 / fps)
                if start_time <= obj_time <= end_time:
                    matched = True
                    break
            if matched:
                filtered_objects.append(obj)

        print(colored(f"[load_result] Total objects in scene graph: {len(objects)}", "white", attrs=["dark"]))
        print(colored(f"[load_result] Objects kept after time filtering: {len(filtered_objects)}", "white", attrs=["dark"]))
        objects = MapObjectList(filtered_objects)  # only keep filtered objects

    # # Regenerate instance colors.

    return objects, objects_all#, bg_objects, instance_colors

def load_scenegraph_working(result_path, start_time=None, end_time=None, dataset_start_timestamp=None, fps=10):
    '''
    Load the full scene graph, and optionally filter objects by time window.

    Args:
        result_path (str): path to scene graph pkl
        start_time (float): task start time (absolute, e.g., seconds since epoch)
        end_time (float): task end time
        dataset_start_timestamp (float): the timestamp corresponding to image_idx=0
        fps (int): frames per second (default 10)
    '''
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
        
    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])
        
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
    else:
        raise ValueError("Unknown results type: ", type(results))
    
    for i, obj in enumerate(objects):
        objects[i]['caption'] = get_caption(obj['caption'], method='lastone')
        objects[i]['image_idx'] = sorted(set(obj['image_idx']))
        objects[i]['obj_id'] = i
        objects[i]['time'] = [dataset_start_timestamp + (idx * 10 / fps) for idx in obj['image_idx']]
    objects_all = objects.copy()

    if start_time is not None and end_time is not None and dataset_start_timestamp is not None:
        filtered_objects = []
        for obj in objects:
            if 'image_idx' not in obj:
                continue
            # Remove duplicates and sort indices
            idx_list = obj['image_idx']
            matched = False

            # Check if any index falls into the time window
            for idx in idx_list:
                obj_time = dataset_start_timestamp + (idx*10 / fps)
                if start_time <= obj_time <= end_time:
                    matched = True
                    break
            if matched:
                filtered_objects.append(obj)

        print(colored(f"[load_result] Total objects in scene graph: {len(objects)}", "white", attrs=["dark"]))
        print(colored(f"[load_result] Objects kept after time filtering: {len(filtered_objects)}", "white", attrs=["dark"]))
        objects = MapObjectList(filtered_objects)  # only keep filtered objects

    # # Regenerate instance colors.

    return objects, objects_all#, bg_objects, instance_colors

def load_memory(args, qa_instance, use_milvus=True, use_optimal_context=False, ip_address='127.0.0.1', embedder=None, objects=None, objects_all=None):
    # Here we load everything needed to load a MilvusDB instance neatly
    captions_path = os.path.join(args.data_dir, 'captions', str(args.sequence_id), 'captions', f'{args.caption_file}_{str(args.sequence_id)}.json')
    with open(captions_path, 'r') as f:
        out = json.load(f)

    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])

    if args.all_mem:
        # if we want to use the full memory, we need to set the end time to the last frame
        start_time = all_start_times[0]
        end_time = all_end_times[-1]
    else:
        start_time = np.float64(qa_instance['start_time'])
        end_time = qa_instance['end_time']


    if use_milvus:
        memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time, embedder=embedder)
    elif 'vlm' in args.model:
        memory = VideoMemory()
    else:
        memory = TextMemory()

    memory.reset()

    outputs = []

    # Compute start idx
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))
    # Compute end idx
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))


    # updata caption to include frame index
    out = assign_frame_indices(out, all_start_times, all_end_times, duration=3.0)


    pkl_files = glob.glob(os.path.join(args.coda_dir, str(args.sequence_id), '*.pkl'))
    pkl_files.sort(key=lambda x: float(x.split('/')[-1][:-4]))
    
    
    objects, objects_all = load_scenegraph(
            result=(objects, objects_all),
            start_time= start_time, #qa_instance['start_time'],
            end_time= end_time, #qa_instance['end_time'],
            dataset_start_timestamp=all_start_times[0],
            fps=10  # or your real fps
        )
    
    out = assign_object_ids(out, objects)
    for i in range(start_idx, end_idx+1):

        item = out[i]
        obj_id = item.get('object_id', None)
        if isinstance(obj_id, list):
            object_id = ','.join(map(str, obj_id))
        elif isinstance(obj_id, int):
            object_id = str(obj_id)
        elif obj_id is None:
            object_id = ''

        entity = {
            'position': item['position'],
            'theta': item['theta'], # ignoring rotation
            'time': item['time'], 
            'caption': item['caption'],
            'object_id': object_id,
        }

        outputs.append(entity)

        if type(memory) == VideoMemory:

            qa_start_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i]['file_start'])
            qa_end_path = os.path.join(args.coda_dir, str(args.sequence_id), out[i+1]['file_start'])

            qa_start_idx = pkl_files.index(qa_start_path)
            qa_end_idx = pkl_files.index(qa_end_path)
            idxs = np.linspace(qa_start_idx, qa_end_idx, 6, dtype=int)

            for pkl_idx in idxs:
                pkl_path = pkl_files[pkl_idx]
                with open(pkl_path, 'rb') as f:
                    pkl_data = pkl.load(f)
                entity['image'] = PILImage.fromarray(pkl_data['cam0'].astype('uint8'), 'RGB')

            entity = ImageMemoryItem.from_dict(entity)
        else:
            entity = MemoryItem.from_dict(entity)

        if use_milvus:
            
            memory.insert(entity, text_embedding=item['text_embedding'])
            if args.all_mem:
                memory.set_scene_graph(objects_all)
                print(colored(f"[load_memory] Using full scene graph with {len(objects_all)} objects", "white", attrs=["dark"]))
            else:
                print(colored(f"[load_memory] Using scene graph with {len(objects)} objects", "white", attrs=["dark"]))
                memory.set_scene_graph(objects)
        else:
            memory.insert(entity)

    if use_optimal_context:
        # then replace the full memory with the optimal context
        memory = TextMemory()
        memory.insert(qa_instance['context'])


    return memory, outputs, all_start_times[0], objects

def load_SG_data(fps=10):
    '''
    Load the full scene graph, and optionally filter objects by time window.

    Args:
        result_path (str): path to scene graph pkl
        start_time (float): task start time (absolute, e.g., seconds since epoch)
        end_time (float): task end time
        dataset_start_timestamp (float): the timestamp corresponding to image_idx=0
        fps (int): frames per second (default 10)
    '''
    scene_graph_path = "/workspace/local_folder/RADAR/sceneVLN/evaluation/output/results/pcd/full_pcd.pkl.gz"
    print("scene_graph_path\n", scene_graph_path)
    captions_path = os.path.join(args.data_dir, 'captions', str(args.sequence_id), 'captions', f'{args.caption_file}_{str(args.sequence_id)}.json')
    with open(captions_path, 'r') as f:
        out = json.load(f)
    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    dataset_start_timestamp = all_start_times[0]

    with gzip.open(scene_graph_path, "rb") as f:
        results = pickle.load(f)
        
    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        
        if results['bg_objects'] is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])
        
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
    else:
        raise ValueError("Unknown results type: ", type(results))
    
    for i, obj in enumerate(objects):
        objects[i]['caption'] = get_caption(obj['caption'], method='majority')
        objects[i]['image_idx'] = sorted(set(obj['image_idx']))
        objects[i]['obj_id'] = i
        objects[i]['time'] = [dataset_start_timestamp + (idx * 10 / fps) for idx in obj['image_idx']]
    objects_all = objects.copy()
    return objects, objects_all

@torch.no_grad()
def sbert_encode_texts(model, texts):
    """Encode text prompts with SBERT and return L2-normalized numpy (M, D)."""
    if isinstance(texts, str):
        texts = [texts]
    embs = model.encode(texts, convert_to_tensor=True)         # torch [M, D]
    embs = torch.nn.functional.normalize(embs, dim=-1)
    return embs.detach().cpu().numpy().astype(np.float32)      # np [M, D]


def get_caption_features_from_objects(objects):
    """Collect per-object SBERT embeddings into (N, D) numpy array."""
    feats = []
    for obj in objects:
        ft = obj['ft']
        if torch.is_tensor(ft):
            ft = ft.detach().cpu().numpy()
        feats.append(np.asarray(ft, dtype=np.float32))
    return np.stack(feats, axis=0)  # (N, D)


def patch_helpers_for_sbert():
    """Patch Clio's helper to use cosine(task, region) on SBERT features."""
    def _sbert_cosine_sim(task_feats, region_feats, *_unused, **__unused):
        sims = task_feats @ region_feats.T   # (M, N), both L2-normalized
        sims = np.clip(sims, 0.0, None)      # match Clio behavior (no negatives)
        return sims
    helpers.compute_sim_to_tasks = _sbert_cosine_sim

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

def build_object_graph(objects, region_features, eps=0.0):
    G = nx.Graph()

    for i, obj in enumerate(objects):
        G.add_node(
            i,
            position=np.asarray(obj['bbox'].center),
            semantic_feature=region_features[i].reshape(-1, 1),
            bounding_box=obj['bbox'],
        )

    # O(N^2) pass; fine for moderate N
    for i in range(len(objects)):
        aabb_i = objects[i]['bbox'].get_axis_aligned_bounding_box()
        for j in range(i + 1, len(objects)):
            aabb_j = objects[j]['bbox'].get_axis_aligned_bounding_box()
            if aabb_overlaps(aabb_i, aabb_j, eps=eps):
                G.add_edge(i, j)

    return G

def _axis_index(axis):
    if isinstance(axis, int):
        return axis
    return {'x': 0, 'y': 1, 'z': 2}[axis.lower()]

def _aabb_corners_from_obj(obj, dilate_eps=0.0):
    """Return (8,3) AABB corners; optionally dilate by eps (meters)."""
    aabb = obj['bbox'].get_axis_aligned_bounding_box()
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
    Returns (N,) bool ground-ish: thin on vertical OR touching floor side.
    For down_positive=True (+Y is down): floor is at LARGE positive coordinate.
    For down_positive=False (usual +Z up): floor is at SMALL coordinate.
    """
    extent_v = (mx[:, v_ax] - mn[:, v_ax])              # thickness along vertical axis
    base_v   = (mx[:, v_ax] if down_positive else mn[:, v_ax])
    is_thin  = extent_v < height_thresh
    is_floor = (base_v > floor_thresh) if down_positive else (base_v < floor_thresh)
    return (is_thin | is_floor)

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
            position=np.asarray(obj['bbox'].center),
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
    gate = iou_mask & v_mask #dist_mask & v_mask & iou_mask & covis_mask & ground_mask

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

def write_default_cluster_config(cfg_path):
    """Write a reasonable default IB config if not present."""
    if os.path.exists(cfg_path):
        return
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        f.write(
            "debug: false\n"
            "debug_folder: ./ib_debug\n"
            "sims_thres: 0.20\n"          # null-task floor; try 0.15–0.30 for SBERT
            "delta: 0.10\n"               # higher => more merging
            "top_k_tasks: 1\n"            # crisp relevance
            "cumulative: false\n"
            "use_lerf_loss: false\n"
            "lerf_loss_cannonical_phrases: []\n"
        )


def run_ib_clustering(region_features, task_features, G_nx, cfg_path):
    """Run Agglomerative IB and return list[list[int]] cluster indices."""
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


def filter_clusters_by_task(clustered_objects, task_features, thresh=0.23):
    """Keep clusters whose embedding is sufficiently similar to any task."""
    kept = []
    for obj in clustered_objects:
        s = cosine_max_to_tasks(task_features, obj['ft'])
        if s >= thresh:
            kept.append(obj)
    return kept

def obb_to_lineset(obb, color=(0, 0, 0)):
    """Convert an Open3D OrientedBoundingBox/AABB to a LineSet for visualization."""
    if hasattr(obb, "get_box_points"):
        pts = np.asarray(obb.get_box_points())
    else:
        # OrientedBoundingBox: sample its corner points by converting to AABB or using its methods
        pts = np.asarray(obb.get_axis_aligned_bounding_box().get_box_points())
    lines = [
        [0,1],[0,2],[0,3],
        [4,5],[4,6],[4,7],
        [1,4],[2,6],[3,7],
        [1,5],[2,7],[3,6],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return ls

def visualize_clusters_open3d(objects, clusters, save_dir=None, show=True):
    """
    Color each cluster, draw its objects' PCDs + bboxes.
    objects[i] should have 'pcd' (o3d.geometry.PointCloud) and 'bbox' (Open3D OBB/AABB).
    """
    geoms = []

    # Build a palette that is guaranteed to have >= 10 very distinct colors
    full_palette = _get_distinct_palette(len(clusters))
    # Only take as many colors as clusters
    palette = full_palette[:len(clusters)]

    for ci, idxs in enumerate(clusters):
        color = np.array(palette[ci], dtype=float)
        # Slightly dimmer point color for better bbox contrast
        pcolor = np.clip(color * 0.8, 0.0, 1.0)

        # Merge cluster pcd for a big bbox (optional)
        cluster_pcd = o3d.geometry.PointCloud()
        for k in idxs:
            pcd = objects[k]['pcd']

            # color the points
            num_pts = np.asarray(pcd.points).shape[0]
            if num_pts > 0:
                pcd_colored = o3d.geometry.PointCloud(pcd)  # copy
                pcd_colored.colors = o3d.utility.Vector3dVector(
                    np.tile(pcolor, (num_pts, 1))
                )
                geoms.append(pcd_colored)
                cluster_pcd += pcd

            # add each object bbox in cluster color
            ls = obb_to_lineset(objects[k]['bbox'], color=color)
            geoms.append(ls)

        # add one big bbox for the cluster (overview)
        if len(cluster_pcd.points) > 0:
            big_obb = cluster_pcd.get_oriented_bounding_box()
            # Some Open3D versions draw OBB color via .color on OBB or via LineSet.
            # Here we also add a LineSet to ensure visible color across versions.
            try:
                big_obb.color = color
                geoms.append(big_obb)
            except Exception:
                # Fallback: draw as colored LineSet
                geoms.append(obb_to_lineset(big_obb, color=color))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # save each cluster merged PLY
        for ci, idxs in enumerate(clusters):
            merged = o3d.geometry.PointCloud()
            for k in idxs:
                merged += objects[k]['pcd']
            o3d.io.write_point_cloud(os.path.join(save_dir, f"cluster_{ci:03d}.ply"), merged)

    if show:
        o3d.visualization.draw_geometries(geoms)


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

# -----------------------------
# Main: step-by-step
# -----------------------------




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

def _get_color_name_and_rgb(k: int):
    """Pick a color name + rgb from the fixed palette, cycling as needed."""
    name, rgb = _PALETTE[k % len(_PALETTE)]
    return name, rgb

def _get_caption(obj: dict, caption_key: str = "caption") -> str:
    """Best-effort caption retrieval + cleanup."""
    cap = obj.get(caption_key)
    if not cap:
        cap = obj.get("captions_ft", "")
    if isinstance(cap, (list, tuple)):
        # dedupe, strip, drop empties; choose the longest as default
        clean = {c.strip() for c in cap if isinstance(c, str) and c.strip()}
        return max(clean, key=len, default="") if clean else ""
    if isinstance(cap, str):
        return cap.strip()
    return ""

def visualize_all_with_highlight(objects_all, selected_idxs, caption_key="caption", show=True):
    """
    Show all point clouds unchanged; highlight only selected indices with bboxes.
    Prints: primitive index, color NAME, caption, center, extent.
    Returns: dict[idx] = {"center": np.ndarray(3), "extent": np.ndarray(3), "color_name": str, "caption": str}
    """
    geoms = []
    results = {}

    # 1) Add all PCDs as-is
    for obj in objects_all:
        pcd = obj.get("pcd", None)
        if isinstance(pcd, o3d.geometry.PointCloud) and len(pcd.points) > 0:
            geoms.append(pcd)

    # 2) Selected → bboxes colored from palette, print info
    nsel = len(selected_idxs)
    if nsel > 0:
        print("\n=== Highlighted primitives ===")
    for rank, idx in enumerate(selected_idxs):
        if idx < 0 or idx >= len(objects_all):
            continue

        obj   = objects_all[idx]
        bbox  = _ensure_o3d_aabb(obj.get("bbox", None))
        if bbox is None:
            continue

        color_name, rgb = _get_color_name_and_rgb(rank)
        bbox.color = rgb
        geoms.append(bbox)

        center = np.asarray(bbox.get_center(), dtype=float)
        extent = np.asarray(bbox.get_extent(), dtype=float)  # (dx, dy, dz)
        caption = _get_caption(obj, caption_key=caption_key)

        results[idx] = {
            "center": np.round(center, 3).tolist(),
            "extent": np.round(extent, 3).tolist(),
            "color_name": color_name,
            "caption": caption,
        }

        # Print with color NAME, not RGB
        print(f"Primitive {idx} → {color_name}; "
              f"caption='{caption if caption else '(empty)'}'; "
              f"center={np.round(center, 3).tolist()}; "
              f"extent={np.round(extent, 3).tolist()}")
    if nsel > 0:
        print("==============================\n")

    if not geoms:
        print("[visualize_all_with_highlight] Nothing to render.")
        return results
    if show:
        # --- Add robot location as red dot ---
            
        o3d.visualization.draw_geometries(geoms)
    return results

# Optional interactive loop
def interactive_highlight(objects_all, caption_key="caption", show=True):
    while True:
        raw = input("Enter indices to HIGHLIGHT (comma separated), or 'q' to quit: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break
        try:
            idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("Invalid input: please enter integers separated by commas.")
            continue

        visualize_all_with_highlight(objects_all, idxs, caption_key=caption_key, show=show)

def get_object_geometry(objects_all, caption_key="caption", user_input = '', show=True):
    # while True:
    raw = user_input#("Enter indices to HIGHLIGHT (comma separated), or 'q' to quit: ").strip()
    if raw.lower() in {"q", "quit", "exit"}:
        print("Exiting.")
     
    try:
        idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        print("Invalid input: please enter integers separated by commas.")

    result = visualize_all_with_highlight(objects_all, idxs, caption_key=caption_key, show=show)
    return result
'''
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


def _color_from_index(i, total):
    """Generate distinct colors based on index (HSV → RGB)."""
    hue = i / max(1, total)
    # Simple HSV->RGB conversion
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return [r, g, b]


def visualize_all_with_highlight(objects_all, selected_idxs):
    """
    Show all pcds unchanged, highlight selected with different-colored bboxes.
    Prints index ↔ color mapping.
    Returns: dict of {idx: {"center": np.ndarray, "extent": np.ndarray, "color": [r,g,b]}}
    """
    geoms = []
    results = {}

    # 1) Add all PCDs as-is
    for obj in objects_all:
        pcd = obj.get("pcd", None)
        if isinstance(pcd, o3d.geometry.PointCloud) and len(pcd.points) > 0:
            geoms.append(pcd)

    # 2) Selected → bboxes with distinct colors
    nsel = len(selected_idxs)
    if nsel > 0:
        print("\n=== Highlighted primitives ===")
    for rank, i in enumerate(selected_idxs):
        if i < 0 or i >= len(objects_all):
            continue
        bbox = _ensure_o3d_aabb(objects_all[i].get("bbox", None))
        if bbox is not None:
            color = _color_from_index(rank, nsel) if nsel > 1 else [1.0, 0.0, 0.0]
            bbox.color = color
            geoms.append(bbox)

            center = bbox.get_center()
            extent = bbox.get_extent()  # (dx, dy, dz)

            results[i] = {
                "center": np.array(center),
                "extent": np.array(extent),
                "color": color
            }

            print(f"Primitive {i} → color {np.round(color, 2).tolist()}, "
                  f"center={np.round(center, 3).tolist()}, "
                  f"extent={np.round(extent, 3).tolist()}")
    if nsel > 0:
        print("==============================\n")

    if not geoms:
        print("[visualize_all_with_highlight] Nothing to render.")
        return results

    o3d.visualization.draw_geometries(geoms)
    return results
'''

def find_col(df, want: str) -> str:
    """Return the actual column name matching `want` (case/space-insensitive)."""
    want_norm = re.sub(r"\s+", " ", want.strip().lower())
    for c in df.columns:
        if re.sub(r"\s+", " ", c.strip().lower()) == want_norm:
            return c
    raise KeyError(f"Column '{want}' not found. Got columns: {list(df.columns)}")

def main(args):

    '''use_milvus = False
    use_optimal_context = False
    if 'ragbot' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = RagAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True
    if 'remembr' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = ReMEmbRAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True
    elif 'optimal' in args.model:
        base_llm = args.model.split('+')[-1]
        agent = NonAgent(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_optimal_context = True
    elif 'vlm' in args.model:
        agent = VLMNonAgent(llm_type='gpt-4o')
    else:
        agent = NonAgent(llm_type=args.model, num_ctx=args.num_ctx*4, temperature=args.temperature)

    # Load question data
    data_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), args.qa_file+'.json')

    data = json.load(open(data_path, 'r'))
    data = data['data']

    # below is the variable to keep track of the evaluation
    running_successes = 0
    num_binary = 0

    running_pos_error = 0
    num_position = 0

    running_time_error = 0
    num_time = 0

    running_duration_error = 0
    num_duration = 0
    
    responses = []
    
    embedder = HuggingFaceEmbeddings(
        model_name='mixedbread-ai/mxbai-embed-large-v1',
        model_kwargs={
            "device": "cuda:0",  #  TITAN Xp
            #"torch_dtype": "auto",
        }
    )
    print("Initializing SBERT model...")
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sbert_model = sbert_model.to("cuda")
    print(colored("Done initializing SBERT model.", "white", attrs=["dark"]))
    # save the outputs
    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.log_file)
    os.makedirs(out_path, exist_ok=True)
    # for i in tqdm.tqdm(range(0, len(data)), total=len(data)):
    i = 0'''
    objects, objects_all = load_SG_data(fps=10)




    Mode = "Manual" # "Auto" or "Manual"
    if Mode == "Manual":
        interactive_highlight(objects_all, caption_key="caption", show = True)
    else:
        csv_path = "/home/mfyuan/local_folder/OpenNav_v2/opennav_mem/opennav_mem/data/navqa/Tasklist_updated_0_16_21_22_3_4.csv"
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        print(f"Loaded {len(df)} rows from {csv_path}")
        df.columns = (
            df.columns
            .str.replace("\ufeff", "", regex=False)         # remove BOM if present
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)           # collapse multiple spaces
            )
            # Resolve the real column names once
        TYPE_COL      = find_col(df, "Type (binary, position, time, text)")
        QUESTION_COL  = find_col(df, "Question")
        SEQID_COL     = find_col(df, "Seq ID")
        TS_COL        = find_col(df, "Timestamp with answer")   # or "Timestamp with answer" depending on file
        CENTER_COL    = find_col(df, "Center") if "Center" in [c.title() for c in df.columns] else "Center"
        EXTENT_COL    = find_col(df, "Extent") if "Extent" in [c.title() for c in df.columns] else "Extent"
        OBJECT_ID     = find_col(df, "Object ID") if "Object_ID" in [c.title() for c in df.columns] else "Object_ID"

        # Ensure 'Center' and 'Extent' columns exist (create if missing)
        if CENTER_COL not in df.columns: df[CENTER_COL] = None
        if EXTENT_COL not in df.columns: df[EXTENT_COL] = None
        if OBJECT_ID not in df.columns: df[OBJECT_ID] = None
    # 1) Define/keep collect_bbox_caption_centers from earlier.
    # 2) Then run:
        # Load question data
        # === Main loop ===
        # for idx, row in df.iterrows():
        # for idx, row in df.head(5).iterrows():
        for idx, row in df.iloc[180:210].iterrows():
            if str(row[TYPE_COL]).strip().lower() == "position":
                question  = row[QUESTION_COL]
                print(colored(f"---\nProcessing row {idx}: Question='{question}'", "yellow"))
                seq_id    = row[SEQID_COL]
                timestamp = row[TS_COL]
                object_id = row[OBJECT_ID]
                
                captions_path = os.path.join(args.data_dir, 'captions', str(seq_id), 'captions', f'{args.caption_file}_{str(seq_id)}.json')
                print(colored(f"Processing row {idx}: Seq ID={seq_id}, Timestamp={timestamp}, Question='{question}'", "yellow"))
                with open(captions_path, 'r') as f:
                    out = json.load(f)

                all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
                all_end_times = np.array([float(x['file_end'][:-4]) for x in out])
                
                timestamp_str = str(timestamp).strip()             # e.g., "10:57:53"
                ref = all_start_times[0]                           # e.g., "2025-08-21 10:00:00" or a datetime
                print(f"Row {idx}: Seq ID={seq_id}, Timestamp={timestamp_str}, Ref={ref}")
                ref_dt = datetime.fromtimestamp(ref)
                print(f"Reference datetime: {ref_dt}")

                # Combine
                combined_str = f"{ref_dt.strftime('%Y-%m-%d')} {timestamp_str}"
                combined_dt  = datetime.strptime(combined_str, "%Y-%m-%d %H:%M:%S")
                print(f"Processing row {idx}: Seq ID={seq_id}, Timestamp={combined_str}")
                _, img_id =  build_image_paths_from_full_timestamps([combined_dt], all_start_times[0])
                output_path = f"/home/mfyuan/local_folder/OpenNav_v2/remembr/remembr/data/scene_graphs/{str(seq_id)}/annotated_rgb/annotated_rgb_{str(round(img_id[0] / 10) * 10)}.png"
                # Step 1: Get target ID from ChatGPT
                target_id = get_chatgpt_output(question, output_path)
                print(f"Target ID from ChatGPT: {target_id}")
                # Step 2: Lookup geometry
                geom = get_object_geometry(objects_all, caption_key="caption", user_input = target_id, show=False)
                print(f"Geometry found: {geom}")
                if not geom:
                    print(f"❌ No geometry found for target ID {target_id}. Skipping row {idx}.")
                    continue
                df.at[idx, CENTER_COL] = geom[int(target_id)]["center"]
                df.at[idx, EXTENT_COL] = geom[int(target_id)]["extent"]
                df.at[idx, OBJECT_ID] = target_id
                # Step 3: Update CSV row
                # df.at[idx, "Center"] = geom["center"]
                # df.at[idx, "Extent"] = geom["extent"]
                #df.at[idx, "Color"] = geom["color_name"]    
        
        df.to_csv("/home/mfyuan/local_folder/opennav_mem/opennav_mem/data/navqa/Tasklist_updated_0_16_21_22_3_4_6.csv", index=False)
        print("✅ Tasklist_updated.csv saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    # data-specific args
    parser.add_argument("--sequence_id", type=int, default=4)
    parser.add_argument("--model", type=str, default="remembr+gpt-4.1") #remembr+llama3 gpt-4.1
    # ChatGPT models: gpt-4o, gpt-4o-mini, gpt-4.1-mini, gpt-4o-mini; issue with gpt-4.1-nano 
    parser.add_argument("--qa_file", type=str, default="human_qa")
    parser.add_argument("--log_file", type=str, default="test_log") #test_log
    parser.add_argument("--caption_file", type=str, default="captions_vila") #captions_nvila
    parser.add_argument("--VDB", type=str, default="search_DB") #captions_nvila
    parser.add_argument("--SG", type=str, default="search_SG") #captions_nvila
    parser.add_argument("--coda_dir", type=str, default="./coda_data/")
    parser.add_argument("--data_dir", type=str, default="/workspace/Local_data/data/")

    parser.add_argument("--scenegraph_file", type=str, default="test_1") #seq: full_pcd_dam_{seq}

    parser.add_argument("--out_dir", type=str, default="./out/")

    parser.add_argument("--postfix", type=str, default='_IB') #_long  _medium _short
    parser.add_argument("--all_mem", type=bool, default=True, help="Whether to use the full memory. Default is True.")
    parser.add_argument("--evaluation_mode", type=bool, default=False, help="Whether to run in evaluation mode. Default is False.")




    # all model args

    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_ctx", type=int, default=8192*4)

    # remembr specific args
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--db_name", type=str, default='test')
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')


    args = parser.parse_args()
    main(args)
