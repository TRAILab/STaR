from typing import Any

import numpy as np

from star.utils.data_collection_helpers import read_space_separated_matrix


MAP_FRAME = "map"
BASE_FRAME = "base_link"


def load_base_link_to_lidar_matrix(cfg) -> np.ndarray:
    """Load the transform from robot base_link frame to LiDAR frame."""
    matrix = None
    if "base_link_to_lidar_matrix" in cfg and cfg.base_link_to_lidar_matrix is not None:
        matrix = cfg.base_link_to_lidar_matrix
    elif "base_to_lidar_matrix" in cfg and cfg.base_to_lidar_matrix is not None:
        matrix = cfg.base_to_lidar_matrix
    elif "T_baselink_to_lidar" in cfg and cfg.T_baselink_to_lidar is not None:
        matrix = cfg.T_baselink_to_lidar

    if matrix is None:
        raise ValueError(
            "Missing LiDAR extrinsic in config. Expected "
            "'base_link_to_lidar_matrix' (preferred), 'base_to_lidar_matrix', "
            "or 'T_baselink_to_lidar'."
        )

    matrix = np.array(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"base_link_to_lidar_matrix must be 4x4, got shape {matrix.shape}.")
    return matrix


def load_base_to_lidar_matrix(cfg) -> np.ndarray:
    """Backward-compatible alias for older code paths."""
    return load_base_link_to_lidar_matrix(cfg)


def startup_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def startup_kv(label: str, value: Any) -> None:
    print(f"  {label:<24} {value}")


def format_matrix(matrix: Any, precision: int = 4) -> str:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return str(matrix)
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(f"{value:.{precision}f}" for value in row) + "]")
    return "\n".join(f"    {row}" for row in rows)


def get_camera_intrinsic_matrix(cfg) -> np.ndarray | None:
    if "camera_intrinsic_matrix" in cfg and cfg.camera_intrinsic_matrix is not None:
        return np.asarray(cfg.camera_intrinsic_matrix, dtype=float)
    if "camera_transformation_k" in cfg and cfg.camera_transformation_k is not None:
        return read_space_separated_matrix(str(cfg.camera_transformation_k))
    if "camera_projection_matrix" in cfg and cfg.camera_projection_matrix is not None:
        return np.asarray(cfg.camera_projection_matrix, dtype=float)[:3, :3]
    if "projection_matrix" in cfg and cfg.projection_matrix is not None:
        return np.asarray(cfg.projection_matrix, dtype=float)[:3, :3]
    return None


def get_camera_projection_matrix(cfg) -> np.ndarray | None:
    if "camera_projection_matrix" in cfg and cfg.camera_projection_matrix is not None:
        return np.asarray(cfg.camera_projection_matrix, dtype=float)
    if "projection_matrix" in cfg and cfg.projection_matrix is not None:
        return np.asarray(cfg.projection_matrix, dtype=float)
    return None


def get_lidar_to_camera_matrix(cfg) -> np.ndarray | None:
    if "lidar_to_camera_matrix" in cfg and cfg.lidar_to_camera_matrix is not None:
        return np.asarray(cfg.lidar_to_camera_matrix, dtype=float)
    if "cam2velo_matrix" in cfg and cfg.cam2velo_matrix is not None:
        return np.asarray(cfg.cam2velo_matrix, dtype=float)
    return None


def get_hydra_choice(group: str, fallback: str) -> str:
    try:
        from hydra.core.hydra_config import HydraConfig
        choices = HydraConfig.get().runtime.choices
        return str(choices.get(group, fallback))
    except Exception:
        return fallback


def print_startup_summary(cfg) -> None:
    print("\n" + "=" * 72)
    print("STAR LiDAR Data Collection")
    print("=" * 72)

    startup_section("Run")
    startup_kv("sequence", cfg.sequence)
    startup_kv("mode", "LiDAR + RGB camera")
    startup_kv("image encoding", "rgb8")
    startup_kv("image size", f"{cfg.image_width} x {cfg.image_height}")
    startup_kv("buffer size", cfg.observation_buffer_size)
    startup_kv("object bbox mode", str(getattr(cfg, "bbox_mode", "obb")).lower())

    startup_section("Parameter Files")
    scenegraph_cfg = get_hydra_choice("scenegraph", "collection_docker_coda")
    dataset_cfg = get_hydra_choice("dataset", "CODa_docker")
    inference_cfg = get_hydra_choice("inference", "docker")
    startup_kv("main config", "configs/config.yaml")
    startup_kv("scenegraph tuning", f"configs/scenegraph/{scenegraph_cfg}.yaml")
    startup_kv("dataset tuning", f"configs/dataset/{dataset_cfg}.yaml")
    startup_kv("inference tuning", f"configs/inference/{inference_cfg}.yaml")
    startup_kv("scenegraph params", "topics, calibration, model paths, buffers, logging")
    startup_kv("dataset params", "dataset root, sequence selection")

    startup_section("ROS")
    startup_kv("RGB camera topic", cfg.rgb_cam_topic)
    startup_kv("LiDAR topic", cfg.lidar_topic)
    startup_kv("TF", f"{MAP_FRAME} <- {BASE_FRAME}")
    startup_kv(
        "TF lookup",
        f"{getattr(cfg, 'tf_lookup_mode', 'latest')} "
        f"(timeout={float(getattr(cfg, 'tf_lookup_timeout_sec', 0.05)):.3f}s)",
    )

    startup_section("Camera Intrinsics")
    intrinsic = get_camera_intrinsic_matrix(cfg)
    if intrinsic is None:
        startup_kv("camera_intrinsic_matrix", "not configured")
    else:
        startup_kv("camera_intrinsic_matrix", "RGB camera K")
        print(format_matrix(intrinsic))
    projection = get_camera_projection_matrix(cfg)
    if projection is not None:
        startup_kv("camera_projection_matrix", "RGB camera projection P")
        print(format_matrix(projection))

    startup_section("Extrinsics")
    startup_kv("base_link_to_lidar", "base_link -> LiDAR")
    print(format_matrix(load_base_link_to_lidar_matrix(cfg)))
    lidar_to_camera = get_lidar_to_camera_matrix(cfg)
    if lidar_to_camera is not None:
        startup_kv("lidar_to_camera", "LiDAR -> RGB camera")
        print(format_matrix(lidar_to_camera))

    startup_section("Outputs")
    startup_kv("annotated RGB", cfg.annotated_rgb_path)
    startup_kv("captions", cfg.output_dir)
    startup_kv("point cloud map", f"{cfg.save_pcd_path}full_pcd.pkl.gz")
    startup_kv("completion idle", f"{float(getattr(cfg, 'finish_idle_sec', 30.0)):.1f}s")

    startup_section("Logging")
    startup_kv("timing logs", getattr(cfg, "enable_timing_logs", True))
    startup_kv("timing interval", f"every {max(1, int(getattr(cfg, 'timing_log_every_n', 1)))} step(s)")
    startup_kv("scenegraph captions", getattr(cfg, "print_scenegraph_captions", False))
    startup_kv("video captions", getattr(cfg, "print_video_captions", False))
    startup_kv("buffer status", getattr(cfg, "print_buffer_status_every_frame", True))
    startup_kv("verbose debug", getattr(cfg, "verbose", False))

    startup_section("Models")
    startup_kv("scenegraph", "Tag2Text + GroundingDINO + TAP + SBERT")
    startup_kv("captioner", cfg.model_name)
    startup_kv("caption model path", cfg.model_path)
    print("=" * 72 + "\n")
