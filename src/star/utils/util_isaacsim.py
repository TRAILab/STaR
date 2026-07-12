
from PIL import Image
import numpy as np
import torch
import cv2
from collections import Counter
from termcolor import colored

import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import re

from typing import Iterable, List, Tuple, Dict, Any

# ------------------------------
# Generic, no prior object lists
# ------------------------------

_SPLIT_RE = re.compile(r"\s*(?:\.\,|,|\.)\s*")  # split on '.,' or ',' or '.'
_TOKEN_RE = re.compile(r"[a-z0-9]+")

PREP_START = re.compile(r"^(on|in|at|by|near|inside|outside|under|over|behind|around)\b")

def _split_captions(x) -> List[str]:
    """Accept string or iterable; split strings on '.,' or ',' or '.'; trim empties."""
    if isinstance(x, str):
        parts = [p.strip() for p in _SPLIT_RE.split(x) if p and p.strip()]
    elif isinstance(x, Iterable):
        parts = [str(p).strip() for p in x if p and str(p).strip()]
    else:
        parts = []
    return parts

def _normalize_text(s: str) -> str:
    """Lowercase, collapse spaces, strip surrounding punctuation/spaces."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.\s]+$", "", s)
    return s

def _tokenize(s: str) -> List[str]:
    """Tokenize into alphanumerics; very lightweight, library-free."""
    return _TOKEN_RE.findall(s.lower())

def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity over token sets."""
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

def _is_background_like(s: str) -> bool:
    """
    Heuristic: treat phrases that look like leading prepositional/location snippets
    as background-like (e.g., 'on the bus', 'in the building').
    This is generic and does not rely on a fixed object vocabulary.
    """
    return bool(PREP_START.match(s))

def _cluster_by_similarity(
    phrases: List[str],
    jaccard_threshold: float = 0.8
) -> Tuple[Dict[int, int], List[str]]:
    """
    Greedy clustering of phrases by token Jaccard similarity.
    Returns:
        assign: mapping from phrase index -> cluster id
        reps:   representative string per cluster (initial seed phrase)
    """
    clusters_tokens: List[List[str]] = []
    reps: List[str] = []
    assign: Dict[int, int] = {}

    for i, p in enumerate(phrases):
        toks = _tokenize(p)
        best_cid, best_sim = -1, 0.0
        for cid, ctoks in enumerate(clusters_tokens):
            sim = _jaccard(toks, ctoks)
            if sim > best_sim:
                best_sim, best_cid = sim, cid
        if best_sim >= jaccard_threshold:
            assign[i] = best_cid
            # optional: keep cluster tokens as the intersection for stability
            clusters_tokens[best_cid] = list(set(clusters_tokens[best_cid]) | set(toks))
        else:
            cid = len(clusters_tokens)
            clusters_tokens.append(toks)
            reps.append(p)          # use first phrase as the representative
            assign[i] = cid

    return assign, reps

def get_caption(
    caption,
    method: str = "majority",
    jaccard_threshold: float = 0.8,
    return_debug: bool = False
) -> str | Tuple[str, Dict[str, Any]]:
    """
    Stable caption selection without prior knowledge of possible captions.

    Parameters:
        caption            : str or iterable of str
        method             : 'majority' (default), 'longest', 'last'
        jaccard_threshold  : threshold to merge near-duplicate captions by token Jaccard
        return_debug       : return (final, debug) if True

    Majority voting steps (no prior lists):
      1) split -> normalize -> remove empties
      2) cluster by token Jaccard to merge near-duplicates
      3) vote by cluster frequency
      4) tie-break (generic):
         a) prefer non-background-like (not starting with a preposition)
         b) prefer longer representative (more informative)
         c) prefer earliest occurrence in input
    """
    raw_list = _split_captions(caption)
    if not raw_list:
        return ("", {"reason": "empty_input"}) if return_debug else ""

    cleaned = [_normalize_text(r) for r in raw_list if r and _normalize_text(r)]
    if not cleaned:
        return ("", {"reason": "empty_after_normalize"}) if return_debug else ""

    if method == "last":
        final = cleaned[-1]
        return (final, {"method": "last"}) if return_debug else final

    if method == "longest":
        final = max(cleaned, key=len)
        return (final, {"method": "longest"}) if return_debug else final

    # Majority voting with clustering
    assign, reps = _cluster_by_similarity(cleaned, jaccard_threshold=jaccard_threshold)

    # Count cluster frequencies
    freq = Counter(assign.values())
    maxf = max(freq.values())
    top_clusters = [cid for cid, n in freq.items() if n == maxf]

    # Precompute earliest index per cluster
    earliest_idx: Dict[int, int] = {}
    for i, cid in assign.items():
        earliest_idx[cid] = min(earliest_idx.get(cid, i), i)

    def tie_key(cid: int) -> Tuple[int, int, int]:
        rep = reps[cid]
        return (
            0 if not _is_background_like(rep) else 1,  # prefer non-background-like
            -len(rep),                                  # prefer longer representative
            earliest_idx[cid],                          # prefer earlier occurrence
        )

    best_cid = min(top_clusters, key=tie_key)
    final = reps[best_cid]

    if return_debug:
        dbg = {
            "method": "majority",
            "cleaned": cleaned,
            "assign": assign,
            "reps": reps,
            "freq": dict(freq),
            "top_clusters": top_clusters,
            "best_cluster": best_cid,
            "final": final,
            "jaccard_threshold": jaccard_threshold
        }
        return final, dbg

    return final




def get_frames(file):
    import cv2

    vidcap = cv2.VideoCapture(file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = frame_count

    if fps == None or frame_count == None:
        # if one of fps or frame_count is None, still recompute
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Video file not found. return empty images.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames
    
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print("frame_interval is equal to 0. return empty image.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, frame_count-1 , num_frames, dtype=int)

    while success:
        if frame_count >= num_frames:
            success, frame = vidcap.read()
            if count in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                if len(images) >= num_frames:
                    return images
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                count += 1
            elif count >= 1:
                width, height = images[-1].size
                images = [Image.new("RGB", (width, height))] * (num_frames - len(images)) + images
                print("padding frames:", (num_frames - len(images)))
                return images
            else: 
                break
    raise ValueError("Did not find enough frames in the video. return empty image.")
        

def file_to_string(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read().strip()
    


def collect_object_captions(objects: list, frame_index: int) -> str:
    """
    Collects the latest caption of each object that appears in the given frame.

    Args:
        objects (list): List of dictionaries, each describing a detected object.
        frame_index (int): The target image frame index.

    Returns:
        str: Concatenated instance-level captions for the given frame.
    """
    captions = []

    for obj in objects:
        if obj["image_idx"][-1] in [frame_index - 2, frame_index - 1, frame_index]:
            # Use the last non-empty caption from the list
            caption_list = [c.strip() for c in obj["caption"].split(",") if c.strip()]
            if caption_list:
                captions.append(caption_list[-1])

    return "; ".join(captions) if captions else "No objects detected in this frame."

def parse_scene_levels(model_response: str):
    import re
    """
    Extract Level 1 and Level 2 area names from the model's text output.

    Args:
        model_response (str): Full text output from the scene model (e.g., VILA).

    Returns:
        tuple: (level1_name, level2_name), or (None, None) if not found.
    """
    level1_match = re.search(r"Level 1:\s*(.+)", model_response)
    level2_match = re.search(r"Level 2:\s*(.+)", model_response)

    level1 = level1_match.group(1).strip() if level1_match else None
    level2 = level2_match.group(1).strip() if level2_match else None

    return level1, level2


def assign_frame_indices(out, all_start_times, all_end_times, duration=3.0, fps=10):
    """
    Assign frame indices to each caption based on start and end times.

    Args:
        out (list of dict): List of caption dictionaries containing 'file_start' and 'file_end' keys.
        frame_interval (float): Time interval between frames, default is 1.0 second.

    Returns:
        list of dict: Updated list where each dict has an additional 'frame_idx' key.
    """

    # Global start and end times
    global_start_time = all_start_times[0]
    global_end_time = all_end_times[-1]
    # Assign frame indices to each caption
    for i in range(0, len(out)):
        start_time = all_start_times[i]-global_start_time#all_start_times[i+1]-global_start_time
        frame_indices = list(range(np.ceil(start_time).astype(int), np.ceil(start_time+duration).astype(int)))
        out[i]["frame_idx"] = frame_indices
    return out


def assign_object_ids(out, objects):
    """
    Assign object IDs to each caption in 'out' based on the frames it covers.

    Args:
        out (list of dict): List of caption dictionaries containing 'frame_idx' key.
        objects (list of dict): List of object dictionaries, each with 'obj_id' and 'image_idx' key.

    Returns:
        list of dict: Updated 'out' with an additional 'object_id' key for each caption.
    """
    from collections import defaultdict
    # Step 1: Build a frame-to-object mapping
    frame_to_objects = defaultdict(set)

    for obj in objects:
        obj_id = obj['obj_id']
        frames = obj.get('image_idx', [])  # Use 'image_idx' field for frames
        for frame in frames:
            frame_to_objects[frame].add(obj_id)
    # Step 2: Assign object IDs to each caption based on covered frames
    for i, x in enumerate(out):
        covered_obj_ids = set()
        # for frame in x['frame_idx']:
        #HERE
        for frame in x['frame_indices']:
            covered_obj_ids.update(frame_to_objects.get(frame, []))
        # Save the collected object IDs as a list
        x['object_id'] = sorted(list(covered_obj_ids))  # Optional: sorted for consistent order
    return out

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, eps=0.02, min_points=10) -> o3d.geometry.PointCloud:
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


def _plot_multi_method_scores_compressed(plot_data_list, k=6, gt_info=None, title="Retrieved Video Captions", save_path=None):
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import matplotlib.dates as mdates
    import numpy as np
    from datetime import datetime, timedelta
    """
    Plot multiple retrieval method scores over time, with top-k highlighted.

    Args:
        plot_data_list (list of dict): 
            Each dict: {
                'method_name': str,
                'data': [{'time': float or array, 'score': float}, ...]
            }
        k (int): Number of top results to highlight (default 5)
        gt_info (dict, optional): {
            'gt_type': 'single' or 'range',
            'timestamps': [t1] or [t1, t2]
        }
        title (str): Plot title.
    """
    shape_styles = ['o', 's', '^', 'D', 'P', '*', 'X']  # different marker shapes

    # Plot
    fig, ax = plt.subplots(figsize=(14, 4))
        # === NEW: build a compressor that removes big blank time ranges ===
    def _collect_all_time_nums(plot_data_list, gt_info):
        """Gather ALL timestamps (methods + GT) as matplotlib date numbers."""
        all_nums = []

        # methods
        for pd in plot_data_list:
            for item in pd.get('data', []):
                t = item['time']
                if isinstance(t, (list, np.ndarray)):
                    t = t[0]
                all_nums.append(mdates.date2num(datetime.fromtimestamp(t)))

        # ground truth
        if gt_info and 'timestamps' in gt_info:
            for ts in gt_info['timestamps']:
                all_nums.append(mdates.date2num(datetime.fromtimestamp(ts)))

        return np.array(all_nums, dtype=float)

    def _make_time_compressor(all_time_nums, gap_seconds=1800, visual_gap_seconds=60):
        """
        Returns (forward, inverse, segments) where:
          - forward(date_num) -> compressed_x (float)
          - inverse(compressed_x) -> original date_num (float) or np.nan
          - segments: list of (a, b, base) where [a,b] is original date range, base is compressed start
        """
        if all_time_nums.size == 0:
            # Identity mapping if no data
            return lambda x: x, lambda y: y, []

        # Sort & unique
        tn = np.unique(np.sort(all_time_nums))
        gap_days = gap_seconds / 86400.0
        vis_gap_days = visual_gap_seconds / 86400.0

        # Split into contiguous segments where gaps <= gap_seconds
        segments_bounds = []
        start = tn[0]
        prev = tn[0]
        for t in tn[1:]:
            if (t - prev) > gap_days:
                segments_bounds.append((start, prev))
                start = t
            prev = t
        segments_bounds.append((start, prev))

        # Build piecewise-linear mapping
        segments = []
        cur_base = 0.0
        for (a, b) in segments_bounds:
            segments.append((a, b, cur_base))
            cur_base += (b - a) + vis_gap_days  # insert small visual gap

        def forward(x):
            # Map a date number x into compressed coordinate
            for (a, b, base) in segments:
                if a <= x <= b:
                    return base + (x - a)
            return np.nan  # outside any segment → skipped

        def inverse(y):
            # Map compressed coordinate y back into original date number (for tick labels)
            for (a, b, base) in segments:
                span = (b - a)
                if base <= y <= base + span:
                    return a + (y - base)
            return np.nan

        return forward, inverse, segments

        # === NEW: compute compressor based on all timestamps ===
    
    _ALL_TIME_NUMS = _collect_all_time_nums(plot_data_list, gt_info)
    # Configure thresholds here:
    GAP_SECONDS = 1800       # treat gaps > 30 minutes as "blank" to skip
    VISUAL_GAP_SECONDS = 60  # small spacer between kept segments (purely visual)

    fwd_map, inv_map, _segments = _make_time_compressor(
        _ALL_TIME_NUMS, gap_seconds=GAP_SECONDS, visual_gap_seconds=VISUAL_GAP_SECONDS
    )


    for idx, plot_data in enumerate(plot_data_list):
        method = plot_data['method_name']
        data = plot_data['data']
        print(colored(f"selected_indices: {plot_data.get('selected_indices', None)}", "red"))
        selected_indices = plot_data.get('selected_indices', None)
        
        if len(data) == 0:
            print(colored(f"Warning: No data for method '{method}', skipping.", "grey"))
            continue

        # Extract times and scores
        times = []
        scores = []
        for item in data:
            t = item['time']
            if isinstance(t, (list, np.ndarray)):
                t = t[0]  # take the first value if array
            times.append(t)
            scores.append(item['score'])
            print(item)
        times = np.array(times)
        scores = np.array(scores)

        # Normalize: higher = better
        min_score = 0.0 #np.min(scores)
        max_score = 400 # np.max(scores)
        max_score = max(400, np.max(scores))  # Avoid division by zero
        if np.max(scores) > 400:
            print(colored(f"Warning: max score {np.max(scores)} is greater than 400, setting to 400!!!!!", "red"))
        norm_scores = 1 - (scores - min_score) / (max_score - min_score + 1e-8)

        # Get top-k indices (most relevant)

        if selected_indices is not None:
            top_k_indices = list(selected_indices) #np.argsort(scores)[:k]
            top_k_indices = sorted(top_k_indices)  # Ensure indices are sorted
        else:
            top_k_indices = np.argsort(norm_scores)[-k:]

        # Convert to datetime objects
        times_dt = [datetime.fromtimestamp(ts) for ts in times]

        # === NEW: compress times ===
        times_num = mdates.date2num(times_dt)
        times_comp = np.array([fwd_map(x) for x in times_num], dtype=float)
        valid_mask = ~np.isnan(times_comp)  # drop points that fall into removed gaps


        # Plot non-top-k (blue)
        mask = np.ones_like(scores, dtype=bool)
        mask[top_k_indices] = False
                # Plot non-top-k (blue) -- only valid points
        other_mask = mask & valid_mask

        plt.scatter(times_comp[other_mask], norm_scores[other_mask],
                    marker=shape_styles[idx % len(shape_styles)],
                    color='blue', alpha=0.6, label=f"{method} (other)")

        # Plot top-k (red) -- only valid points
        top_mask = np.zeros_like(scores, dtype=bool)
        top_mask[top_k_indices] = True
        top_mask &= valid_mask
        plt.scatter(times_comp[top_mask], norm_scores[top_mask],
                    marker=shape_styles[idx % len(shape_styles)],
                    color='red', alpha=0.9, label=f"{method} (top {k})")

        # # Plot top-k (red)

    # Plot ground truth
        # Plot ground truth (compressed)
    if gt_info:
        def _plot_vline_if_in_segment(ts, color, label):
            x = mdates.date2num(datetime.fromtimestamp(ts))
            xc = fwd_map(x)
            if not np.isnan(xc):
                plt.axvline(xc, color=color, linestyle='--', label=label)

        if gt_info['gt_type'] == 'single':
            for ts in gt_info['timestamps']:
                _plot_vline_if_in_segment(ts, 'green', 'GT Time')
        elif gt_info['gt_type'] == 'range':
            start_ts, end_ts = gt_info['timestamps']
            _plot_vline_if_in_segment(start_ts, 'green', 'GT Start')
            _plot_vline_if_in_segment(end_ts, 'orange', 'GT End')

    '''if gt_info:
        if gt_info['gt_type'] == 'single':
            for ts in gt_info['timestamps']:
                plt.axvline(datetime.fromtimestamp(ts), color='green', linestyle='--', label='GT Time')
        elif gt_info['gt_type'] == 'range':
            start_ts, end_ts = gt_info['timestamps']
            plt.axvline(datetime.fromtimestamp(start_ts), color='green', linestyle='--', label='GT Start')
            plt.axvline(datetime.fromtimestamp(end_ts), color='orange', linestyle='--', label='GT End')
    '''
    # Format x-axis to HH:MM:SS
    # # ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

    # # Improve x-axis readability
    # # plt.show()
    ax = plt.gca()

    # --- Smart x-axis (absolute HH:MM[:SS], no confusing offsets) ---
    '''def apply_time_axis(ax):
        # span in seconds
        x0_num, x1_num = ax.get_xlim()
        x0, x1 = mdates.num2date(x0_num), mdates.num2date(x1_num)
        span_sec = max(1.0, (x1 - x0).total_seconds())

        # choose locators + formatter
        if span_sec <= 120:          # ≤ 2 min
            major = mdates.SecondLocator(interval=10)   # 10s
            minor = mdates.SecondLocator(interval=5)
            fmt   = '%H:%M:%S'
        elif span_sec <= 300:        # ≤ 5 min
            major = mdates.SecondLocator(interval=30)   # 30s
            minor = mdates.SecondLocator(interval=10)
            fmt   = '%H:%M:%S'
        elif span_sec <= 900:        # ≤ 15 min
            major = mdates.MinuteLocator(interval=1)    # 1 min
            minor = mdates.SecondLocator(interval=15)
            fmt   = '%H:%M'
        elif span_sec <= 1800:       # ≤ 30 min
            major = mdates.MinuteLocator(interval=2)    # 2 min
            minor = mdates.MinuteLocator(interval=1)
            fmt   = '%H:%M'
        else:                        # > 30 min
            major = mdates.MinuteLocator(interval=5)    # 5 min
            minor = mdates.MinuteLocator(interval=1)
            fmt   = '%H:%M'

        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))  # full time, no offset text
    '''
        # === NEW: smart formatter that shows real wall-clock time on compressed axis ===
    def _compressed_time_formatter(inv_fn, fmt='%Y-%m-%d %H:%M:%S'):
        def _fmt(y, pos):
            x_orig = inv_fn(y)
            if np.isnan(x_orig):
                return ''
            return mdates.num2date(x_orig).strftime(fmt)
        return FuncFormatter(_fmt)

    # Choose a compact format depending on spread of original times
    if _ALL_TIME_NUMS.size > 0:
        span_days = (_ALL_TIME_NUMS.max() - _ALL_TIME_NUMS.min())
        if span_days > 1.0:
            fmt = '%Y-%m-%d %H:%M'
        elif span_days > (10 / 1440.0):  # >10 minutes
            fmt = '%H:%M:%S'
        else:
            fmt = '%H:%M:%S'

        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, prune=None))
        ax.xaxis.set_minor_locator(MaxNLocator(nbins=16, prune=None))
        ax.xaxis.set_major_formatter(_compressed_time_formatter(inv_map, fmt=fmt))

    # --- Y axis fixed to [0,1] ---
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))  # optional: 0,.2,.4,.6,.8,1

    # labels / title / grid
    plt.xlabel("Time")
    plt.ylabel("Relevance Score")  # higher = better
    plt.title(title)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)

    # dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # layout
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_multi_method_scores(plot_data_list, k=6, gt_info=None,
                             title="Retrieved Video Captions", save_path=None):
    """Plot valid retrieval scores against their real wall-clock timestamps."""
    import matplotlib.dates as mdates
    import numpy as np
    from datetime import datetime

    shape_styles = ['o', 's', '^', 'D', 'P', '*', 'X']
    fig, ax = plt.subplots(figsize=(14, 4))
    plotted_points = 0

    for method_idx, plot_data in enumerate(plot_data_list):
        method = plot_data.get('method_name', f'method_{method_idx}')
        data = plot_data.get('data', [])
        raw_selected_indices = plot_data.get('selected_indices')
        selected_indices = (
            set(raw_selected_indices) if raw_selected_indices is not None else set()
        )

        valid_rows = []
        for original_idx, item in enumerate(data):
            try:
                timestamp = item['time']
                if isinstance(timestamp, (list, tuple, np.ndarray)):
                    if len(timestamp) == 0:
                        continue
                    timestamp = timestamp[0]

                timestamp = float(timestamp)
                score = float(item['score'])
                if not np.isfinite(timestamp) or not np.isfinite(score):
                    continue
                valid_rows.append((original_idx, timestamp, score))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue

        print(
            f"Retrieval plot method '{method}': "
            f"{len(valid_rows)} valid point(s) out of {len(data)}"
        )
        if not valid_rows:
            continue

        original_indices = np.array([row[0] for row in valid_rows], dtype=int)
        times = np.array([row[1] for row in valid_rows], dtype=float)
        scores = np.array([row[2] for row in valid_rows], dtype=float)

        max_score = max(400.0, float(np.max(scores)))
        norm_scores = 1.0 - (scores / (max_score + 1e-8))

        if selected_indices:
            top_mask = np.array(
                [idx in selected_indices for idx in original_indices], dtype=bool
            )
        else:
            top_mask = np.zeros(len(scores), dtype=bool)
            top_mask[np.argsort(scores)[:min(k, len(scores))]] = True

        times_dt = np.array([datetime.fromtimestamp(ts) for ts in times])
        other_mask = ~top_mask
        marker = shape_styles[method_idx % len(shape_styles)]

        if np.any(other_mask):
            ax.scatter(
                times_dt[other_mask], norm_scores[other_mask], marker=marker,
                color='blue', alpha=0.6, label=f"{method} (other)"
            )
        if np.any(top_mask):
            ax.scatter(
                times_dt[top_mask], norm_scores[top_mask], marker=marker,
                color='red', alpha=0.9, label=f"{method} (top {k})"
            )
        plotted_points += len(valid_rows)

    if plotted_points == 0:
        print("No valid retrieval data to plot; retrieval figure was not saved.")
        plt.close(fig)
        return False

    if gt_info:
        gt_type = gt_info.get('gt_type')
        gt_timestamps = gt_info.get('timestamps', [])
        if gt_type == 'single':
            for timestamp in gt_timestamps:
                ax.axvline(
                    datetime.fromtimestamp(timestamp), color='green',
                    linestyle='--', label='GT Time'
                )
        elif gt_type == 'range' and len(gt_timestamps) >= 2:
            ax.axvline(
                datetime.fromtimestamp(gt_timestamps[0]), color='green',
                linestyle='--', label='GT Start'
            )
            ax.axvline(
                datetime.fromtimestamp(gt_timestamps[1]), color='orange',
                linestyle='--', label='GT End'
            )

    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Time (YYYY-MM-DD HH:MM:SS)')
    ax.set_ylabel('Relevance Score')
    ax.set_title(title)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return True


def plot_scenegraph_scores(sg_data, gt_info=None, title="Scene Graph Search: object Scores vs Time", save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from datetime import datetime
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if not sg_data:
        print("No sg_data provided. Skipping plot.")
        return

    # ===== Helpers for time-gap compression (borrowed/adapted from plot_multi_method_scores) =====
    def _collect_all_time_nums(sg_data, gt_info):
        """Gather ALL timestamps (methods + GT) as matplotlib date numbers."""
        all_nums = []
        # scene graph data
        for entry in sg_data:
            for obj_data in entry.get("data", []):
                for t in obj_data.get('times', []):
                    all_nums.append(mdates.date2num(datetime.fromtimestamp(t)))
        # ground truth
        if gt_info and 'timestamps' in gt_info:
            for ts in gt_info['timestamps']:
                all_nums.append(mdates.date2num(datetime.fromtimestamp(ts)))
        return np.array(all_nums, dtype=float)

    def _make_time_compressor(all_time_nums, gap_seconds=1800, visual_gap_seconds=60):
        """
        Returns (forward, inverse, segments) where:
          - forward(date_num) -> compressed_x (float)
          - inverse(compressed_x) -> original date_num (float) or np.nan
          - segments: list of (a, b, base) where [a,b] is original date range, base is compressed start
        """
        if all_time_nums.size == 0:
            return lambda x: x, lambda y: y, []

        tn = np.unique(np.sort(all_time_nums))
        gap_days = gap_seconds / 86400.0
        vis_gap_days = visual_gap_seconds / 86400.0

        # split contiguous segments
        segments_bounds = []
        start = tn[0]
        prev = tn[0]
        for t in tn[1:]:
            if (t - prev) > gap_days:
                segments_bounds.append((start, prev))
                start = t
            prev = t
        segments_bounds.append((start, prev))

        segments = []
        cur_base = 0.0
        for (a, b) in segments_bounds:
            segments.append((a, b, cur_base))
            cur_base += (b - a) + vis_gap_days  # insert small visual gap

        def forward(x):
            for (a, b, base) in segments:
                if a <= x <= b:
                    return base + (x - a)
            return np.nan  # outside any segment

        def inverse(y):
            for (a, b, base) in segments:
                span = (b - a)
                if base <= y <= base + span:
                    return a + (y - base)
            return np.nan

        return forward, inverse, segments

    _ALL_TIME_NUMS = _collect_all_time_nums(sg_data, gt_info)
    GAP_SECONDS = 1800       # treat gaps > 30 minutes as "blank" to skip
    VISUAL_GAP_SECONDS = 60  # small spacer between kept segments (purely visual)

    fwd_map, inv_map, _segments = _make_time_compressor(
        _ALL_TIME_NUMS, gap_seconds=GAP_SECONDS, visual_gap_seconds=VISUAL_GAP_SECONDS
    )

    # ===== Original plotting logic (x to compressed coord) =====
    fig, ax = plt.subplots(figsize=(14, 5))

    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for sg_idx, entry in enumerate(sg_data):
        method_name = entry.get("method_name", f"SG_{sg_idx}")
        data = entry.get("data", [])

        if not data:
            continue

        for obj_idx, obj_data in enumerate(data):
            obj_id = obj_data['object_id']
            times = obj_data['times']
            score = obj_data['score']

            if not times:
                continue

            # --- map to compressed x ---
            time_nums = mdates.date2num([datetime.fromtimestamp(t) for t in times])
            times_comp = np.array([fwd_map(x) for x in time_nums], dtype=float)
            valid_mask = ~np.isnan(times_comp)

            if not np.any(valid_mask):
                continue

            scores = np.full(np.count_nonzero(valid_mask), fill_value=score, dtype=float)

            label = f"{method_name}: {obj_id}"
            ax.scatter(
                times_comp[valid_mask],
                scores,
                label=label,
                marker=markers[obj_idx % len(markers)],
                color=color_cycle[sg_idx % len(color_cycle)],
                s=80
            )

    # ✅ Plot GT info if available (using compressed axis)
    if gt_info:
        timestamps = gt_info.get('timestamps', [])
        gt_type = gt_info.get('gt_type', '')

        def _plot_vline_if_in_segment(ts, color, label):
            x = mdates.date2num(datetime.fromtimestamp(ts))
            xc = fwd_map(x)
            if not np.isnan(xc):
                ax.axvline(xc, color=color, linestyle='--', label=label)

        if gt_type == 'single':
            for ts in timestamps:
                _plot_vline_if_in_segment(ts, 'green', 'GT Time')
        elif gt_type == 'range' and len(timestamps) == 2:
            _plot_vline_if_in_segment(timestamps[0], 'green', 'GT Start')
            _plot_vline_if_in_segment(timestamps[1], 'orange', 'GT End')
        else:
            print("GT info provided but format is not recognized.")

    if not ax.collections:
        print("No valid points were plotted. Skipping figure.")
        plt.close(fig)
        return

    # --- Compressed x-axis: show real wall-clock formatting via inverse map ---
    def _compressed_time_formatter(inv_fn, fmt='%H:%M:%S'):
        def _fmt(y, pos):
            x_orig = inv_fn(y)
            if np.isnan(x_orig):
                return ''
            return mdates.num2date(x_orig).strftime(fmt)
        return FuncFormatter(_compressed_time_formatter(inv_map))


    if _ALL_TIME_NUMS.size > 0:
        span_days = (_ALL_TIME_NUMS.max() - _ALL_TIME_NUMS.min())
        if span_days > 1.0:
            fmt = '%Y-%m-%d %H:%M'
        else:
            fmt = '%H:%M:%S'

        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.xaxis.set_minor_locator(MaxNLocator(nbins=16))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, pos: (
            '' if np.isnan(inv_map(y)) else mdates.num2date(inv_map(y)).strftime(fmt)
        )))

    ax.set_xlabel('Time')
    ax.set_ylabel('Similarity Score')
    ax.set_title(title)

    ax.legend(
        title='Query + Object ID',
        bbox_to_anchor=(0.5, -0.25),
        loc='upper center',
        ncol=6,
        fontsize=9
    )

    plt.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_scenegraph_scores_pre(sg_data, gt_info=None, title="Scene Graph Search: object Scores vs Time", save_path=None):
    if not sg_data:
        print("No sg_data provided. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+']
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for sg_idx, entry in enumerate(sg_data):
        method_name = entry.get("method_name", f"SG_{sg_idx}")
        data = entry.get("data", [])

        if not data:
            continue

        for obj_idx, obj_data in enumerate(data):
            obj_id = obj_data['object_id']
            times = obj_data['times']
            score = obj_data['score']

            if not times:
                continue

            time_dt = [datetime.fromtimestamp(t) for t in times]
            scores = [score] * len(times)

            label = f"{method_name}: {obj_id}"
            ax.scatter(
                time_dt,
                scores,
                label=label,
                marker=markers[obj_idx % len(markers)],
                color=color_cycle[sg_idx % len(color_cycle)],
                s=80
            )

    # ✅ Plot GT info if available
    if gt_info:
        timestamps = gt_info.get('timestamps', [])
        gt_type = gt_info.get('gt_type', '')

        if gt_type == 'single':
            for ts in timestamps:
                ax.axvline(datetime.fromtimestamp(ts), color='green', linestyle='--', label='GT Time')
        elif gt_type == 'range' and len(timestamps) == 2:
            ax.axvline(datetime.fromtimestamp(timestamps[0]), color='green', linestyle='--', label='GT Start')
            ax.axvline(datetime.fromtimestamp(timestamps[1]), color='orange', linestyle='--', label='GT End')
        else:
            print("GT info provided but format is not recognized.")

    if not ax.collections:
        print("No valid points were plotted. Skipping figure.")
        plt.close(fig)
        return

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    locator = mdates.SecondLocator(interval=60)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)

    ax.set_xlabel('Time')
    ax.set_ylabel('Similarity Score')
    ax.set_title(title)

    ax.legend(
        title='Query + Object ID',
        bbox_to_anchor=(0.5, -0.25),
        loc='upper center',
        ncol=6,
        fontsize=9
    )

    plt.grid(True)
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

#     # ✅ Check if there is data
#     if not data:
#         print("No data to plot. Skipping plot.")
#         return  # Exit early

#     fig, ax = plt.subplots(figsize=(14, 5))

#     markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '+']
    
#     for idx, obj_data in enumerate(data):
#         obj_id = obj_data['object_id']
#         times = obj_data['times']
#         score = obj_data['score']

#         if not times:
#             continue  # Skip if no time info

#         time_dt = [datetime.fromtimestamp(t) for t in times]
#         scores = [score] * len(times)

#         ax.scatter(time_dt, scores, label=obj_id, marker=markers[idx % len(markers)], s=80)

#     # ✅ Plot GT info if available
#     if gt_info:
#         timestamps = gt_info.get('timestamps', [])
#         gt_type = gt_info.get('gt_type', '')

#         if gt_type == 'single':
#             for ts in timestamps:
#                 ax.axvline(datetime.fromtimestamp(ts), color='green', linestyle='--', label='GT Time')
#         elif gt_type == 'range' and len(timestamps) == 2:
#             ax.axvline(datetime.fromtimestamp(timestamps[0]), color='green', linestyle='--', label='GT Start')
#             ax.axvline(datetime.fromtimestamp(timestamps[1]), color='orange', linestyle='--', label='GT End')
#         else:
#             print("GT info provided but format is not recognized.")

#     # Check again if there are any points actually plotted
#     if not ax.collections:
#         print("No valid points were plotted. Skipping figure.")
#         plt.close(fig)
#         return

#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
#     # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
#     locator = mdates.SecondLocator(interval=30)  # Try changing to 10, 20, 60 as needed
#     ax.xaxis.set_major_locator(locator)

#     ax.set_xlabel('Time')
#     ax.set_ylabel('Similarity Score')
#     ax.set_title(title)

#     # ax.legend(title='Object ID')
#     ax.legend(
#     title='Object ID',
# bbox_to_anchor=(0.5, -0.15), # Put to the bottom center of the figure
#     loc='upper center',
# ncol=12, # Show at most 5 colomns, can be configured
#     fontsize=9
# )
    
#     plt.grid(True)
#     # Improve x-axis readability
#     fig.autofmt_xdate()
#     # plt.subplots_adjust(bottom=0.5)
#     plt.tight_layout()
#     # plt.show()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to {save_path}")
#     else:
#         plt.show()





import re
from time import mktime, strptime

def extract_gt_times(qa_instance: dict) -> dict:
    """
    Extract ground truth time information from a QA instance.

    Args:
        qa_instance (dict): The QA item containing question, type, context, and answers.

    Returns:
        dict: A dictionary with:
            - 'gt_type': 'single', 'range', or 'none'
            - 'timestamps': list of timestamps (e.g., [t1] or [t1, t2]) in epoch format
    """
    q_type = qa_instance.get('type', '')
    context = qa_instance.get('context', '')

    # Define a regex pattern to extract time strings like '2023-01-16 10:56:33'
    time_pattern = r'At time=([\d\-: ]+),'

    # Find all time mentions in the context
    mt = []
    for t in context:
        mt += re.findall(time_pattern, t)    
    matched_times = mt
    print("timestamp with context:", matched_times)

    if q_type != 'duration': #'time' or q_type == 'position':
        if matched_times:
            epoch_time =[]
            # Take all matched times
            for time_str in matched_times:
                epoch_time.append(mktime(strptime(time_str, '%Y-%m-%d %H:%M:%S')))
            return {'gt_type': 'single', 'timestamps': epoch_time}
            # Convert to epoch timestamp
            # return {'gt_type': 'single', 'timestamps': [epoch_time]}
        else:
            return {'gt_type': 'none', 'timestamps': []}

    elif q_type == 'duration':
        if len(matched_times) >= 2:
            # Take first and last timestamp as the range
            start_str = matched_times[0]
            end_str = matched_times[-1]
            start_epoch = mktime(strptime(start_str, '%Y-%m-%d %H:%M:%S'))
            end_epoch = mktime(strptime(end_str, '%Y-%m-%d %H:%M:%S'))
            return {'gt_type': 'range', 'timestamps': [start_epoch, end_epoch]}
        else:
            return {'gt_type': 'none', 'timestamps': []}

    else:
        return {'gt_type': 'none', 'timestamps': []}
