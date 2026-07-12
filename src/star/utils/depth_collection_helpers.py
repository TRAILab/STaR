from typing import Any

import numpy as np


MAP_FRAME = "map"
BASE_FRAME = "base_link"


def startup_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def startup_kv(label: str, value: Any) -> None:
    print(f"  {label:<24} {value}")


def format_matrix(matrix: Any, precision: int = 4) -> str:
    try:
        if isinstance(matrix, str):
            rows = [[float(value) for value in line.split()] for line in matrix.strip().splitlines()]
            arr = np.asarray(rows, dtype=float)
        else:
            arr = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return str(matrix)
    if arr.ndim != 2:
        return str(matrix)
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(f"{value:.{precision}f}" for value in row) + "]")
    return "\n".join(f"    {row}" for row in rows)


def get_hydra_choice(group: str, fallback: str) -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        choices = HydraConfig.get().runtime.choices
        return str(choices.get(group, fallback))
    except Exception:
        return fallback


def get_matrix(cfg, keys: tuple[str, ...]) -> tuple[str | None, Any | None]:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return key, cfg[key]
    return None, None


def print_matrix_or_missing(label: str, matrix: Any | None, description: str) -> None:
    if matrix is None:
        startup_kv(label, "not configured")
        return
    startup_kv(label, description)
    print(format_matrix(matrix))


def print_startup_summary(cfg) -> None:
    print("\n" + "=" * 72)
    print("STAR Depth Data Collection")
    print("=" * 72)

    startup_section("Run")
    startup_kv("sequence", cfg.sequence)
    startup_kv("mode", "Depth + RGB camera")
    startup_kv("image encoding", "rgb8")
    startup_kv("depth encoding", "16UC1 or 32FC1")
    startup_kv("buffer size", cfg.observation_buffer_size)
    startup_kv("object bbox mode", str(getattr(cfg, "bbox_mode", "obb")).lower())

    startup_section("Parameter Files")
    scenegraph_cfg = get_hydra_choice("scenegraph", "collection_docker_sim")
    dataset_cfg = get_hydra_choice("dataset", "CODa_docker")
    inference_cfg = get_hydra_choice("inference", "docker")
    startup_kv("main config", "configs/config.yaml")
    startup_kv("scenegraph tuning", f"configs/scenegraph/{scenegraph_cfg}.yaml")
    startup_kv("dataset tuning", f"configs/dataset/{dataset_cfg}.yaml")
    startup_kv("inference tuning", f"configs/inference/{inference_cfg}.yaml")
    startup_kv("scenegraph params", "topics, calibration, model paths, buffers, logging")

    startup_section("ROS")
    startup_kv("RGB camera topic", cfg.rgb_cam_topic)
    startup_kv("depth topic", cfg.depth_cam_topic)
    startup_kv("TF", f"{MAP_FRAME} <- {BASE_FRAME}")
    startup_kv(
        "TF lookup",
        f"{getattr(cfg, 'tf_lookup_mode', 'latest')} "
        f"(timeout={float(getattr(cfg, 'tf_lookup_timeout_sec', 0.1)):.3f}s)",
    )

    startup_section("Camera Intrinsics")
    _, rgb_k = get_matrix(cfg, ("rgb_camera_matrix", "camera_intrinsic_matrix", "camera_transformation_k"))
    _, depth_k = get_matrix(cfg, ("depth_camera_matrix",))
    print_matrix_or_missing("rgb_camera_matrix", rgb_k, "RGB camera K")
    print_matrix_or_missing("depth_camera_matrix", depth_k, "depth camera K")

    startup_section("Extrinsics")
    _, depth_to_rgb = get_matrix(cfg, ("depth_to_rgb_matrix", "depth_to_left_matrix"))
    base_key, base_to_cam = get_matrix(
        cfg,
        ("base_to_cam_matrix", "base_link_to_depth_cam_matrix", "T_baselink_to_depth_cam"),
    )
    print_matrix_or_missing("depth_to_rgb", depth_to_rgb, "depth camera -> RGB camera")
    print_matrix_or_missing(base_key or "base_to_cam", base_to_cam, "base_link -> depth/RGB camera")

    startup_section("Outputs")
    startup_kv("annotated RGB", cfg.annotated_rgb_path)
    startup_kv("captions", cfg.output_dir)
    startup_kv("point cloud map", f"{cfg.save_pcd_path}full_pcd.pkl.gz")
    startup_kv("video frames", cfg.save_video_path)

    startup_section("Logging")
    startup_kv("timing logs", getattr(cfg, "enable_timing_logs", True))
    startup_kv("timing interval", f"every {max(1, int(getattr(cfg, 'timing_log_every_n', 1)))} step(s)")
    startup_kv("scenegraph captions", getattr(cfg, "print_scenegraph_captions", False))
    startup_kv("video captions", getattr(cfg, "print_video_captions", False))
    startup_kv("verbose debug", getattr(cfg, "verbose", False))

    startup_section("Models")
    startup_kv("scenegraph", "Tag2Text + GroundingDINO + TAP + SBERT")
    startup_kv("captioner", cfg.model_name)
    startup_kv("caption model path", cfg.model_path)
    print("=" * 72 + "\n")
