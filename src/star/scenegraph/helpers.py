import os
import sys
import gzip
import pickle
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import trange
from sentence_transformers import SentenceTransformer

from ..some_class.map_class import MapObjectList
from ..utils.utils import get_observation_by_window

def iter_by_event_depth(observation_buffer, loop_event):
    while not loop_event.is_set():
        data = observation_buffer.get()
        yield data

def safe_queue_size(queue_obj):
    try:
        return queue_obj.qsize()
    except (NotImplementedError, AttributeError, OSError):
        return "NA"

def iter_by_event_pcd(observation_buffer, loop_event, cfg=None):
    consumed_windows = 0
    enable_timing_logs = bool(getattr(cfg, "enable_timing_logs", False))
    timing_log_every_n = max(1, int(getattr(cfg, "timing_log_every_n", 1)))
    while not loop_event.is_set():
        get_start = time.perf_counter()
        data_in_window = observation_buffer.get()
        queue_get_dt = time.perf_counter() - get_start

        unpack_start = time.perf_counter()
        data = get_observation_by_window(data_in_window) # 10 frames
        unpack_dt = time.perf_counter() - unpack_start
        consumed_windows += 1

        # if enable_timing_logs and (consumed_windows % timing_log_every_n == 0):
        #     print(
        #         "[timing][scenegraph-fetch] "
        #         f"consumed_windows={consumed_windows} "
        #         f"q_after_get={safe_queue_size(observation_buffer)} "
        #         f"window_frames={len(data_in_window)} "
        #         f"queue_get={queue_get_dt:.3f}s "
        #         f"window_unpack={unpack_dt:.3f}s "
        #         f"total={queue_get_dt + unpack_dt:.3f}s",
        #         flush=True,
        #     )
        yield data

def iter_by_dataset(datasets, *args):
    for idx in trange(len(datasets)):
        data = datasets[idx]
        yield data

def read_scenegraph(scene_graph_path):
    with gzip.open(scene_graph_path, "rb") as f:
        data = pickle.load(f)

    objects = MapObjectList(device="cuda")
    objects.load_serializable(data["objects"])
    timestamps = data["timestamps"]
    poses = data["poses"]

    all_indices = set()
    for obj in objects:
        indices = obj['image_idx']
        all_indices.update(indices)
    idx = max(all_indices) + 1
    for obj in objects:
        obj['bbox'].color = (0, 1, 0)  # Reset bbox color to green

    return objects, timestamps, poses, idx

def init_scenegraph(last_graph_dir):
    if last_graph_dir is None:
        objects, timestamps, poses, idx = MapObjectList(device="cuda"), [], [], 0 # (idx, timestamp)
    else:
        objects, timestamps, poses, idx = read_scenegraph(last_graph_dir)
        print(f"Loaded last scene graph from {last_graph_dir}, containing {len(objects)} objects.")
    return objects, timestamps, poses, idx

def _import_mos4d_models():
    """Import 4DMOS models, adding common Docker checkout paths if needed."""
    try:
        import mos4d.models.models as models
        return models
    except ModuleNotFoundError:
        pass

    candidate_paths = [
        os.environ.get("STAR_4DMOS_PATH"),
        "/workspace/third_parties/4DMOS",
        "/workspace/third_parties/4DMOS/src",
        "/workspace/third_parties/4DMOS/src/mos4d",
    ]
    for candidate_path in candidate_paths:
        if candidate_path and os.path.isdir(candidate_path) and candidate_path not in sys.path:
            sys.path.append(candidate_path)
            try:
                import mos4d.models.models as models
                return models
            except ModuleNotFoundError:
                continue

    search_root = Path("/workspace/third_parties/4DMOS")
    if search_root.is_dir():
        for models_file in search_root.rglob("mos4d/models/models.py"):
            import_root = str(models_file.parent.parent.parent)
            if import_root not in sys.path:
                sys.path.append(import_root)
            try:
                import mos4d.models.models as models
                return models
            except ModuleNotFoundError:
                continue

    tried = [path for path in candidate_paths if path]
    raise ModuleNotFoundError(
        "Could not import 4DMOS package 'mos4d.models.models'. "
        "Install/checkout 4DMOS under /workspace/third_parties/4DMOS or set "
        "STAR_4DMOS_PATH to the directory that contains the mos4d package. "
        f"Tried: {tried}"
    )

def prepare_mos_model(cfg):
    mos_model = None
    if cfg.filter_dynamic:
        try:
            models = _import_mos4d_models()
            weights = cfg.mos_path
            mos_cfg = torch.load(weights)["hyper_parameters"]
            ckpt = torch.load(weights)
            mos_model = models.MOSNet(mos_cfg)
            mos_model.load_state_dict(ckpt["state_dict"])
            mos_model = mos_model.to("cuda")
            mos_model.eval()
            mos_model.freeze()
        except Exception as exc:
            cfg.filter_dynamic = False
            print(
                "[scenegraph][4DMOS] Dynamic filtering requested, but 4DMOS "
                f"could not be loaded ({exc}). Continuing with filter_dynamic=False.",
                flush=True,
            )
    return mos_model

def load_background_objects(cfg, BG_CAPTIONS_Pro_Sim, BG_CAPTIONS):
    if cfg.use_bg:
        bg_objects = {c: None for c in BG_CAPTIONS_Pro_Sim}
        # Load SBERT model for background caption encoding
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        sbert_model = sbert_model.to("cuda")
        # Encode background captions
        bg_fts = []
        for bg_cation in BG_CAPTIONS:
            bg_ft = sbert_model.encode(bg_cation, convert_to_tensor=True)
            bg_ft = bg_ft / bg_ft.norm(dim=-1, keepdim=True)
            bg_ft = bg_ft.squeeze()
            bg_fts.append(bg_ft)
    else:
        bg_objects = None
        bg_fts = []
    return bg_objects, bg_fts

def save_scene_graph(configs, objects, timestamps, poses):
    if configs.save_pcd:
        print("[memory][finalize] Saving scene graph memory...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'objects': objects.to_serializable(),
            # 'bg_objects': None if bg_objects is None else bg_objects.to_serializable(),
            'cfg': configs,
            'timestamps': timestamps,
            'poses': poses
        }
        pcd_save_path = configs.save_pcd_path + "full_pcd.pkl.gz"
        # If it does not exist, create the new directory
        os.makedirs(os.path.dirname(pcd_save_path), exist_ok=True)
        with gzip.open(pcd_save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"[memory][save] Saved point cloud map to {pcd_save_path}")
        print("[memory][complete] Scene graph memory construction finished successfully.")
    else:
        print("[memory][skip] Point cloud map not saved because save_pcd is disabled.")
