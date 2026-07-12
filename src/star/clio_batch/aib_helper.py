import os
import torch
import numpy as np
import distinctipy
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph_in_3d_scene(
    objects,
    graph,
    clusters=None,
    kept_cluster_ids=None,
    save_path=None,
    show=True,
    lift_height=0.8,
    node_radius=0.06,
    edge_radius=0.015,
    vertical_edge_radius=0.008,
    use_bbox=True,
    use_mesh=False,
    show_vertical_connectors=False,
    dim_unselected=True,
    background_bbox_color=(0.7, 0.7, 0.7),
    background_node_color=(0.6, 0.6, 0.6),
    background_edge_color=(0.5, 0.5, 0.5),
):
    """
    Visualize an object graph directly in 3D scene coordinates using Open3D.

    Parameters
    ----------
    objects : list[dict]
        Object list aligned with graph node indices.
        Each object may contain:
            - 'bbox': Open3D AABB/OBB or bbox-like object
            - 'pcd': Open3D point cloud (optional)
            - 'mesh': Open3D mesh (optional)
            - 'caption' / 'id' / 'object_id' / 'global_id' (optional)

    graph : networkx.Graph
        Graph whose node ids should align with object indices.

    clusters : list[list[int]] or None
        Cluster membership in graph node index space.

    kept_cluster_ids : list[int] or None
        Cluster ids to highlight.

    save_path : str or None
        Optional path to save camera screenshot, if your Open3D version/environment supports it.

    show : bool
        Whether to open the Open3D window.

    lift_height : float
        Height offset added to each object centroid for graph nodes.

    node_radius : float
        Radius of the lifted graph node spheres.

    edge_radius : float
        Radius of graph edge cylinders.

    vertical_edge_radius : float
        Radius of centroid-to-lifted-node connector cylinders.

    use_bbox : bool
        Whether to render bbox for each object.

    use_mesh : bool
        Whether to render mesh/pcd if available.

    dim_unselected : bool
        Whether to dim objects not in highlighted clusters.
    """
    import os
    import math
    import copy
    import numpy as np
    import open3d as o3d
    import networkx as nx

    def log(msg):
        print(f"[visualize_graph_in_3d_scene] {msg}")

    def safe_bbox(obj):
        bbox = obj.get("bbox", None)
        if bbox is None:
            return None

        if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
            return bbox
        if isinstance(bbox, o3d.geometry.OrientedBoundingBox):
            return bbox

        try:
            arr = np.asarray(bbox)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return o3d.geometry.AxisAlignedBoundingBox(arr.min(axis=0), arr.max(axis=0))
        except Exception:
            pass

        return None

    def bbox_center_and_extent(obj):
        bbox = safe_bbox(obj)
        if bbox is None:
            return None, None

        try:
            center = np.asarray(bbox.get_center(), dtype=float)
        except Exception:
            center = None

        try:
            if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
                extent = np.asarray(bbox.get_extent(), dtype=float)
            elif isinstance(bbox, o3d.geometry.OrientedBoundingBox):
                extent = np.asarray(bbox.extent, dtype=float)
            else:
                extent = None
        except Exception:
            extent = None

        return center, extent

    def make_sphere(center, radius, color):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        return sphere

    def make_box_marker(center, size, color):
        box = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        box.compute_vertex_normals()
        box.paint_uniform_color(color)
        box.translate(center - np.array([size / 2, size / 2, size / 2]))
        return box

    def make_cylinder_between_points(p0, p1, radius, color, resolution=20):
        """
        Create a cylinder mesh between two 3D points.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        vec = p1 - p0
        length = np.linalg.norm(vec)

        if length < 1e-8:
            return None

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=resolution
        )
        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)

        # Default cylinder axis in Open3D is +Z
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        direction = vec / length

        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)

        if np.linalg.norm(v) < 1e-8:
            if c > 0:
                R = np.eye(3)
            else:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        else:
            s = np.linalg.norm(v)
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ], dtype=float)
            R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

        cylinder.rotate(R, center=np.zeros(3))
        cylinder.translate((p0 + p1) / 2.0)
        return cylinder

    def make_line_set(points, lines, color):
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=int))
        colors = np.tile(np.asarray(color, dtype=float).reshape(1, 3), (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def object_display_name(obj, idx):
        return obj.get("global_id", obj.get("object_id", obj.get("id", idx)))

    # ------------------------------------------------------------
    # Prepare cluster highlighting
    # ------------------------------------------------------------
    if kept_cluster_ids is None:
        kept_cluster_ids = []

    kept_cluster_ids = [int(cid) for cid in kept_cluster_ids]
    highlighted_nodes = set()
    cluster_node_map = {}

    if clusters is not None:
        for cid in kept_cluster_ids:
            if 0 <= cid < len(clusters):
                valid_nodes = [n for n in clusters[cid] if n in graph.nodes]
                cluster_node_map[cid] = valid_nodes
                highlighted_nodes.update(valid_nodes)

    # Cluster colors
    cluster_palette = [
        (0.90, 0.20, 0.20),  # red
        (0.20, 0.60, 0.95),  # blue
        (0.20, 0.75, 0.35),  # green
        (0.95, 0.65, 0.15),  # orange
        (0.70, 0.35, 0.90),  # purple
        (0.10, 0.75, 0.75),  # cyan
        (0.95, 0.35, 0.65),  # pink
        (0.60, 0.50, 0.20),  # olive
    ]
    cluster_colors = {
        cid: cluster_palette[i % len(cluster_palette)]
        for i, cid in enumerate(kept_cluster_ids)
    }

    # ------------------------------------------------------------
    # Build geometries
    # ------------------------------------------------------------
    geometries = []
    lifted_positions = {}
    object_centers = {}

    log(f"Num objects: {len(objects)}")
    log(f"Num graph nodes: {graph.number_of_nodes()}")
    log(f"Num graph edges: {graph.number_of_edges()}")

    # 1) Draw base objects and lifted nodes
    for node_id in graph.nodes():
        if not (0 <= int(node_id) < len(objects)):
            log(f"Skip node {node_id}: out of object range.")
            continue

        obj = objects[int(node_id)]
        center, extent = bbox_center_and_extent(obj)
        if center is None:
            log(f"Skip node {node_id}: no valid bbox center.")
            continue

        object_centers[node_id] = center

        is_highlighted = (node_id in highlighted_nodes) if len(highlighted_nodes) > 0 else True

        # Determine color
        obj_color = background_bbox_color
        node_color = background_node_color

        if is_highlighted:
            assigned_cluster = None
            for cid, nodes in cluster_node_map.items():
                if node_id in nodes:
                    assigned_cluster = cid
                    break
            if assigned_cluster is not None:
                obj_color = cluster_colors[assigned_cluster]
                node_color = cluster_colors[assigned_cluster]
        elif dim_unselected:
            obj_color = tuple(np.array(background_bbox_color) * 0.8)
            node_color = tuple(np.array(background_node_color) * 0.8)

        # Render bbox
        if use_bbox:
            bbox = safe_bbox(obj)
            if bbox is not None:
                bbox_draw = copy.deepcopy(bbox)
                bbox_draw.color = obj_color
                geometries.append(bbox_draw)

        # Render pcd/mesh if available
        if use_mesh:
            if "mesh" in obj and obj["mesh"] is not None:
                try:
                    mesh = copy.deepcopy(obj["mesh"])
                    mesh.compute_vertex_normals()
                    if is_highlighted:
                        mesh.paint_uniform_color(obj_color)
                    else:
                        mesh.paint_uniform_color((0.75, 0.75, 0.75))
                    geometries.append(mesh)
                except Exception as e:
                    log(f"Failed to draw mesh for node {node_id}: {e}")

            elif "pcd" in obj and obj["pcd"] is not None:
                try:
                    pcd = copy.deepcopy(obj["pcd"])
                    if not pcd.has_colors():
                        if is_highlighted:
                            pcd.paint_uniform_color(obj_color)
                        else:
                            pcd.paint_uniform_color((0.75, 0.75, 0.75))
                    geometries.append(pcd)
                except Exception as e:
                    log(f"Failed to draw pcd for node {node_id}: {e}")

        # Lifted graph node position
        lifted_center = center.copy()
        lifted_center[2] += lift_height
        lifted_positions[node_id] = lifted_center

        # Lifted node marker
        marker = make_sphere(lifted_center, node_radius, node_color)
        geometries.append(marker)

        # Vertical connector from object center to lifted node
        if show_vertical_connectors:
            vertical_color = node_color if is_highlighted else background_edge_color
            vertical_connector = make_cylinder_between_points(
                center, lifted_center, vertical_edge_radius, vertical_color
            )
            if vertical_connector is not None:
                geometries.append(vertical_connector)

    # 2) Draw graph edges between lifted nodes
    for u, v in graph.edges():
        if u not in lifted_positions or v not in lifted_positions:
            continue

        u_highlighted = (u in highlighted_nodes) if len(highlighted_nodes) > 0 else True
        v_highlighted = (v in highlighted_nodes) if len(highlighted_nodes) > 0 else True

        edge_color = background_edge_color

        # If both endpoints belong to the same highlighted cluster, use cluster color
        same_cluster_color = None
        for cid, nodes in cluster_node_map.items():
            if u in nodes and v in nodes:
                same_cluster_color = cluster_colors[cid]
                break

        if same_cluster_color is not None:
            edge_color = same_cluster_color
        elif u_highlighted or v_highlighted:
            edge_color = (0.2, 0.2, 0.2)

        edge_mesh = make_cylinder_between_points(
            lifted_positions[u], lifted_positions[v], edge_radius, edge_color
        )
        if edge_mesh is not None:
            geometries.append(edge_mesh)

    # 3) Optional: world coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)

    log(f"Generated {len(geometries)} geometries.")

    # ------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------
    if show:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Object Graph View", width=1600, height=1000)
        for g in geometries:
            vis.add_geometry(g)

        render_option = vis.get_render_option()
        render_option.background_color = np.array([1.0, 1.0, 1.0])
        render_option.mesh_show_back_face = True
        render_option.line_width = 2.0

        vis.poll_events()
        vis.update_renderer()

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            vis.capture_screen_image(save_path)
            log(f"Saved screenshot to {save_path}")

        vis.run()
        vis.destroy_window()

    return geometries




def cluster_task_scores(objects, clusters, task_features):
    """
    objects: list of obj dicts with 'ft' (L2-normalized np/tensor)
    clusters: list[list[int]]
    task_features: (M, D) L2-normalized numpy array (SBERT)
    returns:
      scores: list[float] — max cosine(task, cluster_mean)
      winners: list[int] — argmax task index per cluster
      cluster_means: list[np.ndarray] — (D,) mean embedding per cluster
    """
    scores, winners, cluster_means = [], [], []
    for idxs in clusters:
        vecs = []
        for k in idxs:
            ft = objects[k]['ft']
            ft = ft.detach().cpu().numpy() if hasattr(ft, 'device') else np.asarray(ft)
            vecs.append(ft)
        if len(vecs) == 0:
            scores.append(0.0); winners.append(-1); cluster_means.append(None); continue
        mean = np.mean(np.stack(vecs), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        sims = task_features @ mean  # (M,)
        j = int(np.argmax(sims))
        scores.append(float(sims[j]))
        winners.append(j)
        cluster_means.append(mean)
    return scores, winners, cluster_means

def select_relevant_clusters(scores, thr=0.35):
    """
    scores: list[float]
    thr: clusters with score >= thr are considered relevant
    returns: set of indices to highlight
    """
    return {i for i, s in enumerate(scores) if s >= thr}

def obb_to_lineset(obb, color=(0,0,0)):
    pts = np.asarray(obb.get_box_points())
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

def visualize_highlighted_clusters_open3d(
    objects, clusters, highlight_idxs, winners, task_texts,
    dim_alpha=0.05, save_dir=None, show=True, task_tf=None
):
    """
    - highlight_idxs: set of cluster indices to highlight
    - winners: per-cluster task index (from cluster_task_scores)
    - task_texts: list[str], used for console labels
    - dim_alpha: how much to dim non-relevant points (0..1)
    """
    geoms = []
    colors = distinctipy.get_colors(len(clusters))

    for ci, idxs in enumerate(clusters):
        is_hot = ci in highlight_idxs
        base_color = np.array(colors[ci])
        pcolor = base_color if is_hot else base_color * dim_alpha
        lcolor = base_color if is_hot else base_color * dim_alpha
        # print highlighted cluster
        if is_hot:
            print(f"[HIGHLIGHT] Cluster {ci} with {len(idxs)} objects, task: {task_texts[winners[ci]] if (winners[ci]>=0) else 'N/A'}")
        merged = o3d.geometry.PointCloud()
        for k in idxs:
            pcd = objects[k]['pcd']
            # print caption of each object in cluster
            caption = objects[k].get('caption', 'N/A')
            if is_hot:
                print(f"Object {k} has caption: {caption}")
            # calculate the score of each object relative to the task
            '''
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

            scored_objects.sort(key=lambda x: -x[1])'''

            pts = np.asarray(pcd.points)
            if pts.size == 0:
                continue
            col = np.tile(pcolor, (pts.shape[0], 1))
            pcd_col = o3d.geometry.PointCloud(pcd)  # copy
            pcd_col.colors = o3d.utility.Vector3dVector(col)
            geoms.append(pcd_col)
            merged += pcd

            # object bbox

        # cluster bbox

    # --- Add robot location as red dot ---
    robot_pos = None #(368.89, 39.6, 11.03)
    if robot_pos is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)  # small radius
        sphere.paint_uniform_color([1, 0, 0])  # red
        sphere.translate(robot_pos)
        geoms.append(sphere)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ci, idxs in enumerate(clusters):
            merged = o3d.geometry.PointCloud()
            for k in idxs:
                merged += objects[k]['pcd']
            o3d.io.write_point_cloud(os.path.join(save_dir, f"cluster_{ci:03d}.ply"), merged)

    if show:
        o3d.visualization.draw_geometries(geoms)




def visualize_graph_highlight(G_nx, clusters, highlight_idxs, out_png=None, show=True):
    pos = nx.spring_layout(G_nx, seed=42, k=0.25)
    hot_nodes, dim_nodes = [], []
    for ci, idxs in enumerate(clusters):
        (hot_nodes if ci in highlight_idxs else dim_nodes).extend(idxs)

    plt.figure(figsize=(10,10))
    nx.draw_networkx_edges(G_nx, pos, alpha=0.25, width=0.5)
    nx.draw_networkx_nodes(G_nx, pos, nodelist=dim_nodes, node_size=18, node_color="#BBBBBB", alpha=0.6)
    nx.draw_networkx_nodes(G_nx, pos, nodelist=hot_nodes, node_size=28, node_color="#3A86FF", alpha=0.95)
    plt.axis('off')
    if out_png:
        plt.savefig(out_png, bbox_inches='tight', dpi=200)
        plt.close()
    elif show:
        plt.show()





def objects_to_aabb_corners(objects, device='cpu'):
    pts = []
    for obj in objects:
        aabb = obj['bbox'].get_axis_aligned_bounding_box()
        pts.append(np.asarray(aabb.get_box_points(), dtype=np.float32))
    return torch.from_numpy(np.stack(pts)).to(device)  # (N,8,3)

def aabb_minmax_from_corners(corners):
    # corners: (N,8,3)
    mn, _ = corners.min(dim=1)  # (N,3)
    mx, _ = corners.max(dim=1)  # (N,3)
    return mn, mx

def pairwise_center_distance(centers):
    return torch.cdist(centers, centers, p=2)  # (N,N)

def compute_iou_batch(b1, b2):
    # Your batch IoU (AABB) function: b1=(M,8,3), b2=(N,8,3) -> (M,N)
    b1_min, b1_max = b1.min(1).values, b1.max(1).values
    b2_min, b2_max = b2.min(1).values, b2.max(1).values
    b1_min = b1_min[:,None,:]; b1_max = b1_max[:,None,:]
    b2_min = b2_min[None,:,:]; b2_max = b2_max[None,:,:]
    inter_min = torch.maximum(b1_min, b2_min)
    inter_max = torch.minimum(b1_max, b2_max)
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)
    vol1 = torch.prod(b1_max - b1_min, dim=2)
    vol2 = torch.prod(b2_max - b2_min, dim=2)
    union = vol1 + vol2 - inter_vol + 1e-10
    return inter_vol / union  # (M,N)

def build_object_graph_smart(
    objects,
    region_features,
    *,
    device='cpu',
    dist_radius=3.0,           # meters (tight)
    z_overlap=False,            # require z-interval overlap
    z_slack=0.25,              # meters of slack on z if not using strict overlap
    iou_thresh=0.02,           # small but >0 to avoid “touching” via floor
    covis_min=2,               # co-visibility frames threshold; set 0 to disable
    knn=6,                     # sparsify to k nearest neighbors among gated pairs
    ignore_ground=True,        # don’t create edges from a ground node
    ground_height_thresh=0.05, # detect ground-ish by small thickness + low z
):
    """
    Builds a sparse, well-gated adjacency graph.

    objects[i] needs: 'bbox' (Open3D OBB), 'image_idx' (list of frame IDs), possibly 'caption'/'ft'
    region_features: (N,D) numpy used as node attr
    """
    N = len(objects)
    G = nx.Graph()
    for i, obj in enumerate(objects):
        G.add_node(
            i,
            position=np.asarray(obj['bbox'].center),
            semantic_feature=region_features[i].reshape(-1, 1),
            bounding_box=obj['bbox'],
        )
    if N <= 1: return G

    # Prepare geometry
    corners = objects_to_aabb_corners(objects, device=device)          # (N,8,3)
    mn, mx = aabb_minmax_from_corners(corners)                         # (N,3),(N,3)
    centers = (mn + mx) * 0.5
    extents = (mx - mn)                                                # (N,3)

    # Optional: label ground-ish nodes (thin in z and close to floor)
    is_ground = torch.zeros(N, dtype=torch.bool, device=device)
    if ignore_ground:
        z_thickness = extents[:, 2]
        z_base = mn[:, 2]
        is_ground = (z_thickness < ground_height_thresh) | (z_base < ground_height_thresh)

    # 1) distance gate
    D = pairwise_center_distance(centers)                              # (N,N)
    dist_mask = (D <= dist_radius)

    # 2) vertical criterion
    if z_overlap:
        # Overlap on z intervals
        zmin = mn[:, 2][:, None]; zmax = mx[:, 2][:, None]
        zmin2 = mn[:, 2][None, :]; zmax2 = mx[:, 2][None, :]
        inter_z = torch.minimum(zmax, zmax2) - torch.maximum(zmin, zmin2)
        z_mask = inter_z > 0
    else:
        # allow |z centers| within slack
        zc = centers[:, 2][:, None]
        zc2 = centers[:, 2][None, :]
        z_mask = (torch.abs(zc - zc2) <= z_slack)

    # 3) IoU gate (AABB)
    iou = compute_iou_batch(corners, corners)                          # (N,N)
    iou = torch.triu(iou, diagonal=1)                                  # keep upper triangle

    iou_mask = iou > iou_thresh

    # 4) co-visibility gate
    if covis_min > 0:
        # Build a small boolean matrix by set intersection counts
        covis = torch.zeros((N, N), dtype=torch.bool, device=device)
        img_sets = [set(objects[i].get('image_idx', [])) for i in range(N)]
        for i in range(N):
            Si = img_sets[i]
            # you can restrict j to neighborhood by dist_mask[i] to speed up:
            for j in range(i + 1, N):
                if len(Si.intersection(img_sets[j])) >= covis_min:
                    covis[i, j] = True
        covis_mask = covis
    else:
        covis_mask = torch.ones((N, N), dtype=torch.bool, device=device)

    # 5) ground hygiene: no edges from ground-ish nodes
    if ignore_ground:
        # build a mask that zeros rows and cols for ground nodes
        not_ground = ~is_ground
        ng_row = not_ground[:, None].expand(N, N)
        ng_col = not_ground[None, :].expand(N, N)
        ground_mask = ng_row & ng_col
    else:
        ground_mask = torch.ones((N, N), dtype=torch.bool, device=device)

    # Combine gates
    gate = dist_mask & z_mask & iou_mask & covis_mask & ground_mask    # (N,N), upper triangle only
    gate = torch.triu(gate, diagonal=1)

    # 6) sparsify by kNN on distance among the gated pairs
    if knn is not None and knn > 0:
        edges = []
        for i in range(N):
            # candidates j where gated and j>i
            mask_row = gate[i]
            js = torch.nonzero(mask_row, as_tuple=False).flatten()
            if js.numel() == 0:
                continue
            dij = D[i, js]
            # take k smallest distances
            k = min(knn, js.numel())
            topk = torch.topk(-dij, k).indices  # negative to get smallest
            chosen = js[topk]
            edges.extend([(i, int(j)) for j in chosen])
        G.add_edges_from(edges)
    else:
        ii, jj = torch.where(gate)
        G.add_edges_from([(int(i), int(j)) for i, j in zip(ii, jj)])

    return G
