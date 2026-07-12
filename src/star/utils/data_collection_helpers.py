import time
from pathlib import Path
from typing import Dict

import numpy as np


def process_cfg(cfg):
    cfg.basedir = Path(cfg.basedir)
    cfg.save_vis_path = Path(cfg.save_vis_path)
    cfg.save_cap_path = Path(cfg.save_cap_path)
    cfg.start_timestamp = time.time()
    time_struct = time.localtime(cfg.start_timestamp)
    cfg.sequence = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)

    return cfg


def read_space_separated_matrix(string):
    """
    convert space separated matrix string to np matrix
    """
    lines = string.strip().split('\n')
    matrix = []
    for line in lines:
        values = line.split()  # Exclude the first element 'rotation_matrix'
        matrix.append([float(value) for value in values])
    numpy_matrix = np.array(matrix)
    return numpy_matrix


def preprocess_position_matrix(position_matrix: np.ndarray) -> tuple:
    # x: horizontal, y: vertical, z: depth
    position: np.ndarray = position_matrix[:3, 3]
    R: np.ndarray = position_matrix[:3, :3]
    theta: float = np.arcsin(R[2, 0]) # radian
    return position, theta


def to_seconds(t) -> float:
    return t.sec + t.nanosec * 1e-9


def load_4x4_matrix_from_cfg(cfg, keys: tuple[str, ...], display_name: str) -> np.ndarray:
    matrix = None
    found_key = None
    for key in keys:
        if key in cfg and cfg[key] is not None:
            matrix = cfg[key]
            found_key = key
            break

    if matrix is None:
        expected = ", ".join(keys)
        raise ValueError(f"Missing {display_name} in config. Expected one of: {expected}.")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{found_key} must be 4x4, got shape {matrix.shape}.")
    return matrix


def shutdown_process(proc, name: str, timeout: float = 5.0) -> None:
    """Gracefully join a child process, then terminate if it is still alive."""
    if proc is None:
        return

    try:
        if proc.is_alive():
            proc.join(timeout=timeout)
        if proc.is_alive():
            print(f"{name} did not exit within {timeout:.1f}s, terminating...")
            proc.terminate()
            proc.join(timeout=timeout)
        if proc.is_alive():
            print(f"{name} is still alive after terminate; manual cleanup may be required.")
    except Exception as exc:
        print(f"Failed to shut down {name}: {exc}")


def safe_queue_size(queue_obj) -> int | str:
    try:
        return queue_obj.qsize()
    except (NotImplementedError, AttributeError, OSError):
        return "NA"


def format_buffer_status(observation_buffers: Dict[str, object]) -> str:
    scenegraph_size = safe_queue_size(observation_buffers["scenegraph"])
    caption_size = safe_queue_size(observation_buffers["captioner"])
    return f"scenegraph_buffer={scenegraph_size} video_caption_buffer={caption_size}"


def buffers_are_empty(observation_buffers: Dict[str, object]) -> bool:
    scenegraph_size = safe_queue_size(observation_buffers["scenegraph"])
    caption_size = safe_queue_size(observation_buffers["captioner"])
    return scenegraph_size == 0 and caption_size == 0
