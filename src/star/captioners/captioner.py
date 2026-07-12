import os
import json
import time
import logging
from datetime import datetime
from typing import Protocol

import numpy as np
from PIL import Image
from omegaconf import DictConfig
from termcolor import colored
from langchain_huggingface import HuggingFaceEmbeddings

class Captioner(Protocol):
    def caption(self, images: list[Image.Image], *args, **kwargs) -> str:
        pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def format_size_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"

def estimate_size_bytes(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, list):
        return sum(estimate_size_bytes(item) for item in obj)
    if isinstance(obj, dict):
        return sum(estimate_size_bytes(v) for v in obj.values())
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return 8
    return 0

class CaptionManager:
    def __init__(self, args: DictConfig, captioner: Captioner) -> None:
        self.args: DictConfig = args
        self.init_json()
        self.embedder: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.save_every = int(getattr(args, "save_every", 10))
        self.last_save_time = time.time()
        self.last_saved_count = 0
        self.save_interval_sec = float(getattr(args, "save_interval_sec", 30.0))
        self.memory_log_interval = int(getattr(args, "memory_log_interval", 10))
        self.enable_timing_logs = bool(getattr(args, "enable_timing_logs", True))
        self.timing_log_every_n = max(1, int(getattr(args, "timing_log_every_n", 1)))
        self.print_caption_text = bool(getattr(args, "print_video_captions", False))
        self.verbose = bool(getattr(args, "verbose", False))
        # self.embedder: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        self.captioner: Captioner = captioner

    def init_json(self) -> None:
        self.outputs: list = []
        seq_timestamp: float = datetime.fromtimestamp(time.time())
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.output_path: str = os.path.join(self.args.output_dir, f"captions_{self.args.model_name}.json")
        with open(self.output_path, 'w') as f:
            json.dump(self.outputs, f, cls=NumpyEncoder)

    # TODO: Directly save to vector DB later
    def save_caption_data(self) -> None:
        logging.info(f"Saving caption data {len(self.outputs)}...")
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.outputs, f, cls=NumpyEncoder, indent=2)
        logging.info(f"Caption data saved to {self.output_path}")
        self.last_save_time = time.time()
        self.last_saved_count = len(self.outputs)

    def maybe_save_caption_data(self, force: bool = False) -> None:
        if len(self.outputs) == 0:
            if force and self.last_saved_count != 0:
                self.save_caption_data()
            return

        has_unsaved_outputs = len(self.outputs) > self.last_saved_count
        if force:
            if has_unsaved_outputs:
                self.save_caption_data()
            return

        due_to_count = has_unsaved_outputs and (len(self.outputs) % self.save_every) == 0
        due_to_time = has_unsaved_outputs and (time.time() - self.last_save_time) >= self.save_interval_sec
        if due_to_count or due_to_time:
            self.save_caption_data()

    def caption_video(
            self, 
            data_in_timewindow: dict, 
            query: str, 
            max_retries: int = 2,
            max_new_tokens: int = 48
        ) -> None:
        total_start = time.perf_counter()
        position: np.ndarray = np.array(data_in_timewindow['position'])
        rotation: np.ndarray = np.array(data_in_timewindow['rotation'])
        timestamps: np.ndarray = np.array(data_in_timewindow['timestamps'])
        frame_indices = data_in_timewindow.get('frame_indices', [])
        if frame_indices:
            frame_label = f"{frame_indices[0]}" if frame_indices[0] == frame_indices[-1] else f"{frame_indices[0]}-{frame_indices[-1]}"
        else:
            frame_label = "NA"

        preprocess_start = time.perf_counter()
        images: list = data_in_timewindow.pop('images') # Pop out the images
        images = images[::30 // self.args.num_video_frames]
        preprocess_dt = time.perf_counter() - preprocess_start

        inference_dt = 0.0
        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                out_text: str = self.captioner.caption(images, query=query, max_new_tokens=max_new_tokens)
                inference_dt = time.perf_counter() - start
                if self.verbose:
                    print(f"[memory][caption] generated in {inference_dt:.2f}s")
                if self.print_caption_text:
                    print(colored(f"[video-caption] frames={frame_label}: {out_text}", "yellow"), flush=True)
                break
            except json.JSONDecodeError as e:
                print(colored(f"[WARNING] JSON decoding failed (attempt {attempt+1}/{max_retries})", "red"))
                print(colored(f"Raw output:\n{repr(out_text)}", "yellow"))
                if attempt < max_retries - 1:
                    logging.info("[INFO] Retrying caption generation...")
                else:
                    logging.info(colored("[ERROR] All caption retries failed. Skipping this segment.", "red"))
                    scene: str = "Invalid caption"
                    location: str = "Unknown"
                    out_text = "Invalid caption"

        embed_start = time.perf_counter()
        text_embedding = self.embedder.embed_query(out_text)
        embed_dt = time.perf_counter() - embed_start

        append_start = time.perf_counter()
        entity: dict = {
            'id': timestamps[0],
            'position': position.mean(axis=0).tolist(),
            'theta': rotation.mean(axis=0).tolist(),
            'time': timestamps.mean(),
            'caption': out_text,
            'file_start': data_in_timewindow['file_start'],
            'file_end': data_in_timewindow['file_end'],
            'frame_indices': frame_indices,
            'text_embedding': text_embedding
        }

        self.outputs.append(entity)
        append_dt = time.perf_counter() - append_start
        if self.memory_log_interval > 0 and len(self.outputs) % self.memory_log_interval == 0:
            print(
                "[memory][caption] "
                f"segments={len(self.outputs)} "
                f"buffer_size={format_size_bytes(estimate_size_bytes(self.outputs))}",
                flush=True,
            )

        total_dt = time.perf_counter() - total_start
        if self.enable_timing_logs and (len(self.outputs) % self.timing_log_every_n == 0):
            print(
                colored(
                    "[timing][captioner] "
                    f"window={timestamps[0]:.3f}-{timestamps[-1]:.3f} "
                    f"frames={len(timestamps)} "
                    f"preprocess={preprocess_dt:.3f}s "
                    f"inference={inference_dt:.3f}s "
                    f"embed={embed_dt:.3f}s "
                    f"append={append_dt:.3f}s "
                    f"total={total_dt:.3f}s",
                    "white",
                    attrs=["dark"],
                ),
                flush=True,
            )
        del images  # Free up memory
