import argparse
import gzip
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import distinctipy
import numpy as np
import open3d as o3d
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(sys.path[0]).resolve().parent))

from star.some_class.map_class import MapObjectList


_BASE_VIVID = [
    (0.121, 0.466, 0.705),
    (1.000, 0.498, 0.054),
    (0.172, 0.627, 0.172),
    (0.839, 0.152, 0.156),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811),
    (0.000, 0.000, 0.000),
    (1.000, 0.000, 0.000),
]

_PALETTE = [
    ("red", (1.0, 0.0, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("green", (0.0, 1.0, 0.0)),
    ("orange", (1.0, 0.5, 0.0)),
    ("purple", (0.5, 0.0, 0.5)),
    ("cyan", (0.0, 1.0, 1.0)),
    ("magenta", (1.0, 0.0, 1.0)),
    ("yellow", (1.0, 1.0, 0.0)),
    ("teal", (0.0, 0.5, 0.5)),
    ("pink", (1.0, 0.4, 0.7)),
    ("lime", (0.7, 1.0, 0.0)),
    ("indigo", (0.3, 0.0, 0.5)),
]

_BBOX_LINE_RADIUS = 0.03


def _scene_graph_path(sequence_id: str) -> str:
    return f"/workspace/results/{sequence_id}/pcd/full_pcd.pkl.gz"


def _annotated_rgb_path(sequence_id: str) -> str:
    return f"/workspace/results/{sequence_id}/annotated_rgb/"


def _startup_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _startup_kv(label: str, value) -> None:
    print(f"  {label:<24} {value}")


def print_startup_summary(args) -> None:
    scene_graph_path = _scene_graph_path(args.sequence_id)
    annotated_rgb_path = _annotated_rgb_path(args.sequence_id)

    print("\n" + "=" * 72)
    print("STAR 3D Scene Graph Visualizer")
    print("=" * 72)

    _startup_section("Input")
    _startup_kv("sequence id", args.sequence_id)
    _startup_kv("scene graph file", scene_graph_path)
    _startup_kv("annotated images", annotated_rgb_path)
    _startup_kv("SBERT model", args.sbert_model_path)

    _startup_section("Visualization")
    _startup_kv("bbox mode", args.bbox_mode)
    _startup_kv("similarity threshold", args.similarity_threshold)
    _startup_kv("max search results", args.max_search_results)

    _startup_section("How To Use")
    print("  Object indices are the numbers shown on the annotated RGB images.")
    print("  1,5,8                    highlight objects by index")
    print("  search chair             search objects by caption/feature similarity")
    print("  q                        quit")
    print("=" * 72 + "\n")


def _get_distinct_palette(n: int, seed: int = 7):
    m = max(n, 10)
    if m <= len(_BASE_VIVID):
        return _BASE_VIVID[:m]
    extra = distinctipy.get_colors(m - len(_BASE_VIVID), exclude_colors=_BASE_VIVID, rng=seed)
    return _BASE_VIVID + extra


def _ensure_o3d_aabb(bbox):
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


def _select_bbox(bbox, bbox_mode: str = "aabb"):
    if bbox_mode == "obb":
        if isinstance(bbox, (o3d.geometry.OrientedBoundingBox, o3d.geometry.AxisAlignedBoundingBox)):
            return bbox
        try:
            arr = np.asarray(bbox)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return o3d.geometry.OrientedBoundingBox.create_from_points(
                    o3d.utility.Vector3dVector(arr)
                )
        except Exception:
            pass
    return _ensure_o3d_aabb(bbox)


def _bbox_extent_array(bbox):
    if bbox is None:
        return None
    if hasattr(bbox, "extent"):
        return np.asarray(bbox.extent, dtype=float)
    if hasattr(bbox, "get_extent"):
        return np.asarray(bbox.get_extent(), dtype=float)
    return None


def _pcd_copy(pcd: o3d.geometry.PointCloud):
    if not isinstance(pcd, o3d.geometry.PointCloud):
        return None
    out = o3d.geometry.PointCloud()
    out.points = deepcopy(pcd.points)
    if pcd.has_colors():
        out.colors = deepcopy(pcd.colors)
    if pcd.has_normals():
        out.normals = deepcopy(pcd.normals)
    return out


def _get_color_name_and_rgb(rank: int):
    return _PALETTE[rank % len(_PALETTE)]


def _get_caption(obj: dict, caption_key: str = "caption") -> str:
    caption = obj.get(caption_key)
    if not caption:
        caption = obj.get("captions_ft", "")
    if isinstance(caption, (list, tuple)):
        clean = {item.strip() for item in caption if isinstance(item, str) and item.strip()}
        return max(clean, key=len, default="") if clean else ""
    if isinstance(caption, str):
        return caption.strip()
    return ""


def _line_segment_to_cylinder(start, end, color=(0, 0, 0), radius=_BBOX_LINE_RADIUS):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length <= 1e-8:
        return None

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=float(length))
    cylinder.compute_vertex_normals()

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    direction_unit = direction / length
    cross = np.cross(z_axis, direction_unit)
    cross_norm = np.linalg.norm(cross)
    dot = float(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0))

    if cross_norm > 1e-8:
        axis = cross / cross_norm
        angle = np.arccos(dot)
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(rotation, center=np.zeros(3))
    elif dot < 0.0:
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        cylinder.rotate(rotation, center=np.zeros(3))

    cylinder.translate((start + end) / 2.0)
    cylinder.paint_uniform_color(color)
    return cylinder


def _create_bbox_lineset(bbox):
    if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
        return o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
        return o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

    if hasattr(bbox, "get_axis_aligned_bounding_box"):
        aabb = bbox.get_axis_aligned_bounding_box()
        return o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    return None


def obb_to_lineset(obb, color=(0, 0, 0), radius=_BBOX_LINE_RADIUS):
    line_set = _create_bbox_lineset(obb)
    if line_set is None:
        return []

    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    meshes = []
    for start_idx, end_idx in lines:
        cylinder = _line_segment_to_cylinder(points[start_idx], points[end_idx], color=color, radius=radius)
        if cylinder is not None:
            meshes.append(cylinder)
    return meshes


def visualize_clusters_open3d(objects, clusters, save_dir=None, show=True):
    geoms = []
    palette = _get_distinct_palette(len(clusters))[: len(clusters)]

    for cluster_index, object_indices in enumerate(clusters):
        color = np.array(palette[cluster_index], dtype=float)
        point_color = np.clip(color * 0.8, 0.0, 1.0)
        cluster_pcd = o3d.geometry.PointCloud()

        for object_index in object_indices:
            pcd = objects[object_index]["pcd"]
            pcd_copy = _pcd_copy(pcd)
            if pcd_copy is None:
                continue

            num_points = np.asarray(pcd_copy.points).shape[0]
            if num_points == 0:
                continue

            pcd_copy.colors = o3d.utility.Vector3dVector(np.tile(point_color, (num_points, 1)))
            geoms.append(pcd_copy)
            cluster_pcd += pcd
            geoms.extend(obb_to_lineset(objects[object_index]["bbox"], color=color))

        if len(cluster_pcd.points) > 0:
            big_obb = cluster_pcd.get_oriented_bounding_box()
            try:
                big_obb.color = color
                geoms.append(big_obb)
            except Exception:
                geoms.extend(obb_to_lineset(big_obb, color=color))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for cluster_index, object_indices in enumerate(clusters):
            merged = o3d.geometry.PointCloud()
            for object_index in object_indices:
                merged += objects[object_index]["pcd"]
            o3d.io.write_point_cloud(os.path.join(save_dir, f"cluster_{cluster_index:03d}.ply"), merged)

    if show and geoms:
        o3d.visualization.draw_geometries(geoms)


def load_sg_data(sequence_id: str):
    scene_graph_path = _scene_graph_path(sequence_id)
    print(f"[visualizer] Loading scene graph: {scene_graph_path}")

    with gzip.open(scene_graph_path, "rb") as handle:
        results = pickle.load(handle)

    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
    else:
        raise ValueError(f"Unknown results type: {type(results)}")

    return objects, objects.copy()


def visualize_all_with_highlight(objects_all, selected_idxs, caption_key="caption", show=True, bbox_mode="aabb"):
    geoms = []
    results = {}

    for obj in objects_all:
        pcd = obj.get("pcd")
        if isinstance(pcd, o3d.geometry.PointCloud) and len(pcd.points) > 0:
            geoms.append(pcd)

    if selected_idxs:
        print("\n=== Highlighted primitives ===")

    for rank, idx in enumerate(selected_idxs):
        if idx < 0 or idx >= len(objects_all):
            continue

        obj = objects_all[idx]
        bbox = _select_bbox(obj.get("bbox"), bbox_mode=bbox_mode)
        if bbox is None:
            continue

        color_name, rgb = _get_color_name_and_rgb(rank)
        bbox.color = rgb
        geoms.extend(obb_to_lineset(bbox, color=rgb))

        center = np.asarray(bbox.get_center(), dtype=float)
        extent = _bbox_extent_array(bbox)
        if extent is None:
            continue
        caption = _get_caption(obj, caption_key=caption_key)

        results[idx] = {
            "center": np.round(center, 3).tolist(),
            "extent": np.round(extent, 3).tolist(),
            "color_name": color_name,
            "caption": caption,
        }
        # Available info: image_idx, num_detections, n_points, inst_color, bg_class,  caption, color_name
        # class_sk, caption, caption_ft, bbox (center, extent), pcd, etc. 
        
        print(
            f"Primitive {idx} -> {color_name}; "
            f"caption='{caption if caption else '(empty)'}'; "
            f"center={np.round(center, 3).tolist()}; "
            f"extent={np.round(extent, 3).tolist()}"
        )

    if selected_idxs:
        print("==============================\n")

    if not geoms:
        print("[visualize_all_with_highlight] Nothing to render.")
        return results

    if show:
        o3d.visualization.draw_geometries(geoms)
    return results


def _as_feature_vector(feature):
    if feature is None:
        return None
    if hasattr(feature, "detach"):
        feature = feature.detach()
    if hasattr(feature, "cpu"):
        feature = feature.cpu()

    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if vector.size == 0 or not np.isfinite(vector).all() or norm <= 1e-8:
        return None
    return vector / norm


def search_objects_by_keyword(
    objects_all,
    keyword,
    sbert_model,
    caption_key="caption",
    similarity_threshold=0.0,
):
    """Return object indices ranked by cosine similarity to an SBERT text query."""
    keyword = keyword.strip()
    if not keyword:
        return []

    query_ft = sbert_model.encode(keyword, convert_to_numpy=True)
    query_ft = _as_feature_vector(query_ft)
    if query_ft is None:
        print("Could not create a valid embedding for the search keyword.")
        return []

    matches = []
    skipped = 0
    for idx, obj in enumerate(objects_all):
        object_ft = _as_feature_vector(obj.get("ft"))
        if object_ft is None or object_ft.shape != query_ft.shape:
            skipped += 1
            continue

        similarity = float(np.dot(query_ft, object_ft))
        if similarity >= similarity_threshold:
            matches.append((idx, similarity, _get_caption(obj, caption_key=caption_key)))

    matches.sort(key=lambda item: item[1], reverse=True)
    if skipped:
        print(f"Skipped {skipped} objects with missing or incompatible SBERT features.")
    return matches


def _print_search_matches(keyword, matches):
    print(f"\n=== Matches for '{keyword}' ===")
    if not matches:
        print("No matching objects found.")
    else:
        for rank, (idx, similarity, caption) in enumerate(matches, start=1):
            print(
                f"{rank:3d}. object {idx:4d} | similarity={similarity:.4f} | "
                f"caption='{caption if caption else '(empty)'}'"
            )
    print("==============================\n")


def _choose_match_count(num_matches):
    while True:
        raw = input(
            f"How many top matches should be visualized? [1-{num_matches}, 0 to cancel]: "
        ).strip()
        try:
            count = int(raw)
        except ValueError:
            print("Invalid input: please enter an integer.")
            continue

        if 0 <= count <= num_matches:
            return count
        print(f"Please enter a number from 0 to {num_matches}.")


def interactive_highlight(
    objects_all,
    sbert_model,
    caption_key="caption",
    show=True,
    bbox_mode="aabb",
    similarity_threshold=0.0,
    max_search_results=30,
):
    while True:
        raw = input(
            "Enter indices to highlight, 'search <keyword>', or 'q' to quit: "
        ).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            print("Exiting.")
            break

        if raw.lower().startswith("search "):
            keyword = raw[7:].strip()
            if not keyword:
                print("Please provide a keyword after 'search'.")
                continue

            matches = search_objects_by_keyword(
                objects_all,
                keyword,
                sbert_model,
                caption_key=caption_key,
                similarity_threshold=similarity_threshold,
            )
            displayed_matches = matches[:max_search_results]
            _print_search_matches(keyword, displayed_matches)
            if len(matches) > len(displayed_matches):
                print(
                    f"Showing the top {len(displayed_matches)} of {len(matches)} matches. "
                    "Use --max_search_results to change this limit.\n"
                )
            if not displayed_matches:
                continue

            count = _choose_match_count(len(displayed_matches))
            if count > 0:
                selected_idxs = [idx for idx, _, _ in displayed_matches[:count]]
                visualize_all_with_highlight(
                    objects_all,
                    selected_idxs,
                    caption_key=caption_key,
                    show=show,
                    bbox_mode=bbox_mode,
                )
            continue

        try:
            idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("Invalid input: enter comma-separated indices or 'search <keyword>'.")
            continue

        visualize_all_with_highlight(objects_all, idxs, caption_key=caption_key, show=show, bbox_mode=bbox_mode)


def get_object_geometry(objects_all, caption_key="caption", user_input="", show=True, bbox_mode="aabb"):
    raw = user_input.strip()
    if raw.lower() in {"q", "quit", "exit"}:
        print("Exiting.")
        return {}

    try:
        idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        print("Invalid input: please enter integers separated by commas.")
        return {}

    return visualize_all_with_highlight(objects_all, idxs, caption_key=caption_key, show=show, bbox_mode=bbox_mode)


def main(args):
    print_startup_summary(args)
    _, objects_all = load_sg_data(args.sequence_id)
    print(f"[visualizer] Loading SBERT model: {args.sbert_model_path}")
    sbert_model = SentenceTransformer(args.sbert_model_path)
    interactive_highlight(
        objects_all,
        sbert_model,
        caption_key="caption",
        show=True,
        bbox_mode=args.bbox_mode,
        similarity_threshold=args.similarity_threshold,
        max_search_results=args.max_search_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="vis_3D",
        description="Interactive 3D scene graph visualization",
    )
    parser.add_argument("--sequence_id", type=str, default="3")
    parser.add_argument("--bbox_mode", type=str, choices=["aabb", "obb"], default="aabb")
    parser.add_argument(
        "--sbert_model_path",
        type=str,
        default="/workspace/star/weights/all-MiniLM-L6-v2",
        help="SBERT model used to create the object 'ft' embeddings.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.5,
        help="Minimum cosine similarity included in keyword search results.",
    )
    parser.add_argument(
        "--max_search_results",
        type=int,
        default=60,
        help="Maximum number of ranked matches to print and make selectable.",
    )

    args = parser.parse_args()
    main(args)
