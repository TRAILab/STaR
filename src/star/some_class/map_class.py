
'''Data structures for storing, manipulating, and serializing map objects.'''
from collections.abc import Iterable
import copy
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d


def to_numpy(tensor):
    """Convert torch tensors to numpy arrays without importing heavy utility modules."""
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def to_tensor(numpy_array, device=None):
    """Convert numpy arrays to torch tensors without importing heavy utility modules."""
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    tensor = torch.from_numpy(numpy_array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _build_bbox(bbox_np, pcd_np, bbox_type=None):
    """Rebuild a bbox robustly, falling back to AABB for degenerate serialized boxes."""
    bbox_arr = np.asarray(bbox_np, dtype=np.float64)
    pcd_arr = np.asarray(pcd_np, dtype=np.float64)

    if bbox_type == "aabb":
        source = pcd_arr if len(pcd_arr) > 0 else bbox_arr
        if source.ndim == 2 and source.shape[1] == 3 and len(source) > 0:
            return o3d.geometry.AxisAlignedBoundingBox(
                source.min(axis=0), source.max(axis=0)
            )

    if bbox_arr.ndim == 2 and bbox_arr.shape[1] == 3 and len(bbox_arr) > 0:
        unique = np.unique(bbox_arr, axis=0)
        if len(unique) >= 4:
            centered = unique - unique.mean(axis=0, keepdims=True)
            if np.linalg.matrix_rank(centered) >= 3:
                try:
                    return o3d.geometry.OrientedBoundingBox.create_from_points(
                        o3d.utility.Vector3dVector(bbox_arr)
                    )
                except RuntimeError:
                    pass

    if pcd_arr.ndim == 2 and pcd_arr.shape[1] == 3 and len(pcd_arr) > 0:
        mins = pcd_arr.min(axis=0)
        maxs = pcd_arr.max(axis=0)
        return o3d.geometry.AxisAlignedBoundingBox(mins, maxs)

    if bbox_arr.ndim == 2 and bbox_arr.shape[1] == 3 and len(bbox_arr) > 0:
        mins = bbox_arr.min(axis=0)
        maxs = bbox_arr.max(axis=0)
        return o3d.geometry.AxisAlignedBoundingBox(mins, maxs)

    zero = np.zeros(3, dtype=np.float64)
    return o3d.geometry.AxisAlignedBoundingBox(zero, zero)


class DetectionList(list):
    '''
    Temporarily store objects detected within a camera frame.
    '''
    def get_values(self, key, idx:int=None):
        '''
        Get values for a specified object attribute.
        '''
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]

    def get_stacked_values_torch(self, key, idx:int=None):
        '''
        Return stacked values as a torch tensor.
        '''
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            # Convert bounding boxes to their corner points before stacking.
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            # Convert all NumPy arrays to torch tensors.
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0)

    def get_stacked_values_numpy(self, key, idx:int=None):
        '''
        Return stacked values as a NumPy array.
        '''
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)

    def get_stacked_str_torch(self, key, idx:int=None):
        '''
        Return caption strings collected from the detections.
        '''
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            values.append(v)
        return values

    def __add__(self, other):
        '''
        Create a copied list and append the values from another list.
        '''
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list

    def __iadd__(self, other):
        '''
        Append values in place without copying.
        '''
        self.extend(other)
        return self

    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist selected by indices.
        '''
        new_self = type(self)()
        for i in index:
            new_self.append(self[i])
        return new_self

    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist selected by a boolean mask.
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_self.append(self[i])
        return new_self


    def color_by_instance(self):
        '''
        Assign colors according to instance.
        '''
        if len(self) == 0:
            return
         # If an instance color is defined, apply it directly to the point cloud.
        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        # Otherwise, assign each instance a color from the colormap.
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]


class MapObjectList(DetectionList):
    '''
    Store the complete list of point-cloud map objects.
    '''
    def compute_similarities(self, new_ft):
        '''
        Compute similarities using the features of a new point cloud.
        '''
        # Convert a NumPy array to a tensor when necessary.
        new_ft = to_tensor(new_ft)
        # Get features for all instances before computing cosine similarity.
        clip_fts = self.get_stacked_values_torch('ft')
        # Compute similarities.
        similarities = F.cosine_similarity(new_ft.unsqueeze(0), clip_fts)
        # Return similarity values.
        return similarities

    def to_serializable(self):
        '''
        Convert map objects to a simple NumPy-based representation for storage.
        '''
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            s_obj_dict['ft'] = to_numpy(s_obj_dict['ft'])
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['bbox_type'] = (
                'aabb'
                if isinstance(s_obj_dict['bbox'], o3d.geometry.AxisAlignedBoundingBox)
                else 'obb'
            )
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            # Remove pcd and bbox objects, retaining only their point representations.
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            s_obj_list.append(s_obj_dict)
        return s_obj_list

    def load_serializable(self, s_obj_list):
        '''
        Load serialized map objects.
        '''
        # The destination list must be empty before loading.
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)
            new_obj['ft'] = to_tensor(new_obj['ft'])
            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
            bbox_type = new_obj.pop('bbox_type', None)
            new_obj['bbox'] = _build_bbox(
                new_obj['bbox_np'], new_obj['pcd_np'], bbox_type=bbox_type
            )
            if len(new_obj['pcd_color_np']) > 0:
                new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            # Remove temporary serialized fields that are no longer needed.
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']
            self.append(new_obj)
