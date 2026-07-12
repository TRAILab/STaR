
from PIL import Image
# from debian import timestamp
import logging
import numpy as np
import time
import cv2
from termcolor import colored
from collections import deque
from typing import List, Dict, Any, Optional 

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


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


def get_caption(caption, method='longest'):
    caption_string = caption.strip()
    captions = caption_string.split(".,")
    captions = [c.strip() for c in captions if c.strip()]
    if method == 'longest':
        caption = max(captions, key=lambda x: len(x))
    else:# take the last one
        caption = captions[-1]
    return caption

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
        for frame in x['frame_idx']:
            covered_obj_ids.update(frame_to_objects.get(frame, []))
        # Save the collected object IDs as a list
        x['object_id'] = sorted(list(covered_obj_ids))  # Optional: sorted for consistent order
    return out


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
            print(f"Warning: Mask index {i} is not valid.")
            continue
        color = distinct_colors_rgb[i % len(distinct_colors_rgb)]

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
        label_width, label_height = 28, 14

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


def plot_multi_method_scores(plot_data_list, k=6, gt_info=None, title="Retrieved Video Captions", save_path=None):
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

    for idx, plot_data in enumerate(plot_data_list):
        method = plot_data['method_name']
        data = plot_data['data']
        raw_selected_indices = plot_data.get('selected_indices')
        selected_indices = (
            set(raw_selected_indices) if raw_selected_indices is not None else None
        )

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
        times = np.array(times)
        scores = np.array(scores)

        # Normalize: higher = better
        min_score = 0.0 #np.min(scores)
        max_score = 400 # np.max(scores)
        max_score = max(400, np.max(scores))  # Avoid division by zero
        if np.max(scores) > 400:
            print(colored(f"Warning: max score {np.max(scores)} is greater than 400, setting to 400!!!!!", "red"))
        print(f"min_score: {min_score}, max_score: {max_score}")
        norm_scores = 1 - (scores - min_score) / (max_score - min_score + 1e-8)

        # Highlight the actual selected memories when the retriever provides them.
        if selected_indices is not None:
            top_k_indices = np.array(
                sorted(i for i in selected_indices if 0 <= i < len(scores)),
                dtype=int,
            )
            highlight_label = f"{method} (selected)"
        else:
            top_k_indices = np.argsort(scores)[:k]
            highlight_label = f"{method} (top {k})"

        # Convert to datetime objects
        times_dt = [datetime.fromtimestamp(ts) for ts in times]

        # Plot non-top-k (blue)
        mask = np.ones_like(scores, dtype=bool)
        mask[top_k_indices] = False
        plt.scatter(np.array(times_dt)[mask], norm_scores[mask],
                    marker=shape_styles[idx % len(shape_styles)],
                    color='blue', alpha=0.6, label=f"{method} (other)")

        # Plot top-k (red)
        if len(top_k_indices) > 0:
            plt.scatter(np.array(times_dt)[top_k_indices], norm_scores[top_k_indices],
                        marker=shape_styles[idx % len(shape_styles)],
                        color='red', alpha=0.9, label=highlight_label)

    # Plot ground truth
    if gt_info:
        if gt_info['gt_type'] == 'single':
            for ts in gt_info['timestamps']:
                plt.axvline(datetime.fromtimestamp(ts), color='green', linestyle='--', label='GT Time')
        elif gt_info['gt_type'] == 'range':
            start_ts, end_ts = gt_info['timestamps']
            plt.axvline(datetime.fromtimestamp(start_ts), color='green', linestyle='--', label='GT Start')
            plt.axvline(datetime.fromtimestamp(end_ts), color='orange', linestyle='--', label='GT End')

    # Format x-axis to HH:MM:SS
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    locator = mdates.SecondLocator(interval=30)  # Try changing to 10, 20, 60 as needed
    ax.xaxis.set_major_locator(locator)

    plt.xlabel("Time (HH:MM:SS)")
    plt.ylabel("Normalized Similarity (L2 Distance)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # Improve x-axis readability
    fig.autofmt_xdate()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_scenegraph_scores(sg_data, gt_info=None, title="Scene Graph Search: object Scores vs Time", save_path=None):
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

    # Plot GT info if available
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


class ListJitterBuffer: 
    def __init__(self, desired_timewindow: float) -> None: 
        self.__desired_timewindow: float = desired_timewindow 
        self.__session_gap_threshold: float = max(desired_timewindow * 5.0, desired_timewindow + 1.0)
        self.__tickets: float = desired_timewindow 
        self.__started: bool = True 
        self.__ready: bool = False 
        self.__buffers: List[deque] = [ (deque()) for _ in range(2) ] 
        self.reset() 

    def reset(self) -> None: 
        self.__last_timestamp: float = 0.0 
        # self.__start: float = 0.0 
        self.__tickets = self.__desired_timewindow
        self.__ready = False
        self.__started = True
        self.__write_index: int = 0 
        self.__read_index: int = 1 
        for buffer in self.__buffers:
            buffer.clear()

    def _restart_session(self, timestamp: float) -> None:
        logging.info(
            "[jitter][caption] restarting local buffer session at timestamp=%.3f",
            timestamp,
        )
        self.reset()
        self.__last_timestamp = timestamp
        self.__started = False

    def add(self, data_in_window: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]: 
        if not data_in_window: 
            return None 

        # Initialize if not started 
        if self.__started: 
            first_ts = data_in_window[0]['timestamps'] 
            # self.__start = first_ts 
            self.__last_timestamp = first_ts 
            self.__started = False 

        for data in data_in_window: 
            timestamp = data['timestamps'] 
            delta = timestamp - self.__last_timestamp

            if delta > self.__session_gap_threshold:
                self._restart_session(timestamp)
                delta = 0.0

            # 1️⃣ Cal delta t 
            # 2️⃣ Reduce tickets 
            self.__tickets -= delta 
            # 3️⃣ Update last_timestamp 
            self.__last_timestamp = timestamp 
            # 4️⃣ If tickets are used up, switch buffer once and fold any large
            # timestamp jump back into the current window without emitting empty frames.
            if self.__tickets < 0: 
                overflow = -self.__tickets
                self.__switch_buffer() 
                self.__ready = True 
                remainder = overflow % self.__desired_timewindow
                self.__tickets = (
                    self.__desired_timewindow - remainder
                    if remainder > 0
                    else self.__desired_timewindow
                )

            # 5️⃣ Write current buffer 
            self.__buffers[self.__write_index].append(data) 
    
        # 6️⃣ If a frame is ready, return 
        if self.__ready: 
            self.__ready = False 
            frame = self.__pop_read_buffer()
            return frame if frame else None
        return None 
    
    def close(self) -> Optional[List[Dict[str, Any]]]: 
        final_data: List[Dict[str, Any]] = self.__pop_write_buffer()
        return final_data

    def __switch_buffer(self) -> None: 
        self.__write_index, self.__read_index = ( self.__read_index, self.__write_index ) 

    def __pop_write_buffer(self) -> None: 
        buffer: deque = self.__buffers[self.__write_index] 
        frame: List[Dict[str, Any]] = [] 
        while buffer: 
            frame.append(buffer.popleft()) 
        return frame 

    def __pop_read_buffer(self) -> List[Dict[str, Any]]: 
        buffer: deque = self.__buffers[self.__read_index] 
        frame: List[Dict[str, Any]] = [] 
        while buffer: 
            frame.append(buffer.popleft()) 
        return frame 

    @property 
    def desired_timewindow(self) -> float: 
        return self.__desired_timewindow 

    @property 
    def tickets(self) -> float: 
        return self.__tickets


class JitterBuffer:
    def __init__(self, desired_timewindow: float) -> None: 
        self.__desired_timewindow: float = desired_timewindow 
        self.__session_gap_threshold: float = max(desired_timewindow * 5.0, desired_timewindow + 1.0)
        self.__tickets: float = desired_timewindow 
        self.__started: bool = True 
        self.__ready: bool = False 
        self.__buffers: List[deque] = [ (deque()) for _ in range(2) ] 
        self.reset() 

    def reset(self) -> None: 
        self.__last_timestamp: float = 0.0 
        self.__start: Optional[float] = None 
        self.__tickets = self.__desired_timewindow
        self.__ready = False
        self.__started = True
        self.__write_index: int = 0 
        self.__read_index: int = 1 
        for buffer in self.__buffers:
            buffer.clear()

    def _restart_session(self, timestamp: float) -> None:
        logging.info(
            "[jitter][producer] detected large timestamp gap; restarting local buffer session at timestamp=%.3f",
            timestamp,
        )
        self.reset()
        self.__start = timestamp
        self.__last_timestamp = timestamp
        self.__started = False

    def add(self, data: Any, timestamp: float) -> Optional[List[Dict[str, Any]]]: 
        if not data: 
            return None 

        # Initialize if not started 
        if self.__started: 
            first_ts = timestamp
            self.__start = first_ts 
            self.__last_timestamp = first_ts 
            self.__started = False 
            self.__buffers[self.__write_index].append(data) 
          
        if timestamp <= self.__last_timestamp:
            time.sleep(0.005)  # Sleep briefly to allow time to advance
            return None

        # 1️⃣ Cal delta t 
        delta = timestamp - self.__last_timestamp 

        if delta > self.__session_gap_threshold:
            self._restart_session(timestamp)
            self.__buffers[self.__write_index].append(data)
            return None

        # 2️⃣ Reduce tickets 
        self.__tickets -= delta 
        # 3️⃣ Update last_timestamp 
        self.__last_timestamp = timestamp 
        # 4️⃣ If tickets are used up, switch buffer 
        if self.__tickets < 0: 
            self.__switch_buffer() 
            self.__ready = True 
            # Add excess time to next buffer
            self.__tickets += self.__desired_timewindow 

        # 5️⃣ Write current buffer 
        self.__buffers[self.__write_index].append(data) 
    
        # 6️⃣ If a frame is ready, return 
        if self.__ready: 
            self.__ready = False 
            return self.__pop_read_buffer() 
        return None
    
    def close(self) -> Optional[List[Dict[str, Any]]]: 
        final_data: List[Dict[str, Any]] = self.__pop_write_buffer()
        return final_data
    
    def __switch_buffer(self) -> None: 
        self.__write_index, self.__read_index = ( self.__read_index, self.__write_index ) 

    def __pop_write_buffer(self) -> None: 
        buffer: deque = self.__buffers[self.__write_index] 
        frame: List[Dict[str, Any]] = [] 
        while buffer: 
            frame.append(buffer.popleft()) 
        return frame 
    
    def __pop_read_buffer(self) -> List[Dict[str, Any]]: 
        buffer: deque = self.__buffers[self.__read_index] 
        frame: List[Dict[str, Any]] = [] 
        while buffer: 
            frame.append(buffer.popleft()) 
        return frame 

    @property 
    def desired_timewindow(self) -> float: 
        return self.__desired_timewindow 

    @property 
    def tickets(self) -> float: 
        return self.__tickets
     
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
    matched_times = re.findall(time_pattern, context)

    if q_type != 'duration': #'time' or q_type == 'position':
        if matched_times:
            # Take the first timestamp found as the GT
            time_str = matched_times[0]
            # Convert to epoch timestamp
            epoch_time = mktime(strptime(time_str, '%Y-%m-%d %H:%M:%S'))
            return {'gt_type': 'single', 'timestamps': [epoch_time]}
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
