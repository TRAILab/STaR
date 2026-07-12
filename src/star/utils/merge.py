"""Utilities for computing object similarities and merging map detections."""
import sys
sys.path.append("/code1/dyn/github_repos/OpenGraph")
import torch
import torch.nn.functional as F
from star.some_class.map_class import DetectionList, MapObjectList
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import trange
# import for caption merging
from typing import List, Optional
import time
from .utils import get_bounding_box, process_pcd
from openai import OpenAI

def compute_spatial_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''Compute spatial similarities between detections and map objects.'''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')
    spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    return spatial_sim


def compute_iou_batch(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    '''
    Compute IoU between two sets of axis-aligned 3D bounding boxes.

    bbox1: (M, V, D), e.g. (M, 8, 3)
    bbox2: (N, V, D), e.g. (N, 8, 3)

    Returns: MxN spatial-similarity matrix.
    '''
    # Compute the axis-aligned bounds of each box.
    bbox1_min, _ = bbox1.min(dim=1) # Shape: (M, 3)
    bbox1_max, _ = bbox1.max(dim=1) # Shape: (M, 3)
    bbox2_min, _ = bbox2.min(dim=1) # Shape: (N, 3)
    bbox2_max, _ = bbox2.max(dim=1) # Shape: (N, 3)
    # Expand dimensions for pairwise operations.
    bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
    bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
    bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)
    # Compute intersection-box coordinates from the maximum minima and minimum maxima.
    inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
    inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)
    # Compute the intersection-box volumes.
    inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)
    # Compute the volumes of both bounding-box sets.
    bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
    bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)
    # Compute IoU, treating non-intersecting boxes as having zero intersection volume.
    iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)
    return iou

def compute_visual_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''Compute visual similarities between detections and map objects.'''
    det_fts = detection_list.get_stacked_values_torch('caption') # (M, D)
    obj_fts = objects.get_stacked_values_torch('caption') # (N, D)
    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    return visual_sim

def preprocess_text(text):
    '''Normalize text by removing common articles and prepositions.'''
    # remove common articles and prepositions
    stop_words = set(["a","with", "the", "in", "on", "ahead", "is", "to", "next", "down", "of", "along", "and"]) 
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def compute_caption_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''Compute TF-IDF caption similarities between detections and map objects.'''
    sentences1 = detection_list.get_stacked_str_torch('caption') # (M, D)
    sentences2 = objects.get_stacked_str_torch('caption') # (N, D)
    # preprocess text
    processed_sentences1 = [preprocess_text(sentence) for sentence in sentences1]
    processed_sentences2 = [preprocess_text(sentence) for sentence in sentences2]
    # # use bag-of-words representation
    # use TF-IDF vectorization (works better)
    vectorizer = TfidfVectorizer().fit(processed_sentences1 + processed_sentences2)
    vectorizer = vectorizer.transform(processed_sentences1 + processed_sentences2)

    # computecosine similarity matrix
    similarity_matrix = cosine_similarity(vectorizer[:len(sentences1)], vectorizer[len(sentences1):])
    # DEBUGinspect caption similarity
    return similarity_matrix


def compute_ft_similarities(detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''Compute feature similarities between detections and map objects.'''
    det_fts = detection_list.get_stacked_values_torch('ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('ft') # (N, D)
    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    ft_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    return ft_sim

def aggregate_similarities(cfg, spatial_sim: torch.Tensor,  ft_similarities: torch.Tensor, caption_similarities: torch.Tensor) -> torch.Tensor:
    '''Combine spatial, feature, and caption similarities using configured weights.'''
    # scoring method is configurable
    if len(spatial_sim) > 0:
        mask = spatial_sim == 0.0
        ft_similarities[mask] = float('-inf') # Meaningless if spatial_sim == 0
        caption_similarities[mask] = float('-inf') # Meaningless if spatial_sim == 0

    if caption_similarities is not None:
        sims = spatial_sim*cfg.spatial_weight + caption_similarities*cfg.caption_weight + ft_similarities*cfg.ft_weight # (M, N)
    else:
        sims = spatial_sim*cfg.spatial_weight + ft_similarities*cfg.ft_weight # (M, N)
    return sims

def merge_detections_to_objects_2(
    cfg,
    detection_list,  # List of detected objects
    objects,         # List of map objects
    agg_sim,          # torch.Tensor similarity matrix [num_detections, num_objects]
    valid_mask_indices,
    frame_idx=None,
):
    """
    Merge detections from the current frame into the global map object list.

    Args:
        cfg: Configuration object.
        detection_list (list): List of detected objects in the current frame.
        objects (list): Global map objects list.
        agg_sim (torch.Tensor): Aggregated similarity matrix between detections and map objects.

    Returns:
        objects (list): Updated map objects list.
        instance_id (dict): Mapping from detection index (1-based) to map object index.
    """

    instance_id = {}              # Map: detection idx -> object idx
    current_object_indices = []   # List of object idx corresponding to detections (optional, can return if needed)
    print_scenegraph_captions = bool(getattr(cfg, "print_scenegraph_captions", False))
    frame_prefix = f"frame={frame_idx} " if frame_idx is not None else ""
    # Iterate over each detection
    for i in range(agg_sim.shape[0]):
        if agg_sim[i].max() == float('-inf'):
            # No match found, add as a new object
            new_obj_idx = len(objects)
            instance_id[valid_mask_indices[i]] = new_obj_idx  # detection idx starts from 1
            objects.append(detection_list[i])
            current_object_indices.append(new_obj_idx)
            if print_scenegraph_captions:
                print(
                    f"Adding new detection {frame_prefix}list \033[96m{valid_mask_indices[i]},\033[0m as map object {new_obj_idx} "
                    f"with caption: \033[93m{detection_list[i]['caption']}\033[0m\n"
                )
        else:
            # Match found, merge with existing object
            j = agg_sim[i].argmax().item()
            matched_det = detection_list[i]
            matched_obj = objects[j]

            # Merge detection into matched object
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det)
            objects[j] = merged_obj

            instance_id[valid_mask_indices[i]] = j
            current_object_indices.append(j)
            if print_scenegraph_captions:
                print(
                    f"Merging detected object {frame_prefix}from\033[94m detection list {valid_mask_indices[i]},\033[0m "
                    f"with caption \033[92m{detection_list[i]['caption']}\033[0m: "
                    f"&& with map \033[94mobject {j}\033[0m with caption: \033[91m{matched_det['caption']}\033[0m\n"
                )

    return objects, instance_id

def merge_detections_to_objects(
    cfg,
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    '''Merge current-frame detections into the global map-object list.'''
    # iterate all detections and merge into objects
    for i in range(agg_sim.shape[0]):
        # if unmatched, add as new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det)
            objects[j] = merged_obj
    return objects



def merge_obj2_into_obj1(cfg, obj1, obj2, bg=False, class_name = None):
    '''Merge obj2 into obj1 and update the combined geometry and features.'''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    for k in obj1.keys():
        if k in ['caption']:
            obj1['caption'] += ', '
            obj1['caption'] += obj2['caption']
            if class_name is not None:
                # use unified background caption for background caption
                obj1['caption'] = class_name
        elif k not in ['pcd', 'bbox', 'ft', 'bg_class', "class_sk", "captions_ft", "img_bbox"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                # keep original object color
                obj1[k] = obj1[k] 
            else:
                raise NotImplementedError
        else:
            continue
    # Merge the point clouds and bounding boxes.
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'].remove_duplicated_points()

    obj1['pcd'] = process_pcd(cfg, obj1['pcd'], use_db=not bg)
    obj1['bbox'] = get_bounding_box(
        obj1['pcd'], mode=getattr(cfg, "bbox_mode", "obb")
    )
    obj1['bbox'].color = [0,1,0]
    # merge features
    obj1['ft'] = (obj1['ft'] * n_obj1_det +
                obj2['ft'] * n_obj2_det) / (
                n_obj1_det + n_obj2_det)
    obj1['ft'] = F.normalize(obj1['ft'], dim=0)
    return obj1


def caption_merge_llama_unquantization(cfg, objects: MapObjectList):
    '''Merge object captions using an unquantized Llama model.'''
    # LLAMAload pretrained model
    generator = Llama.build(
        ckpt_dir=cfg.llama_ckpt_dir,
        tokenizer_path=cfg.llama_tokenizer_path,
        max_seq_len=cfg.llama_max_seq_len,
        max_batch_size=cfg.llama_max_batch_size,
    )
    # example prompt
    caption_example1 = "a car parked on the street, a car parked on the street, a white car parked on the street, a car parked on the street, a black car parked on the street, a white car parked on the street, a white car on the road, a mirror of a white car"
    caption_example2 = "a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign"
    caption_example3 = "a triangular street sign, the back of a triangular sign"

    for i in trange(len(objects)):
        caption_obj = objects[i]["caption"]
        if ", " in caption_obj:
            # if too many observations make string too long, remove early observations
            comma_count = caption_obj.count(', ')
            if comma_count > cfg.max_caption_num:
                num_last_comma_index = 0
                for j in range(cfg.max_caption_num):
                    num_last_comma_index = caption_obj.rfind(', ', 0, num_last_comma_index - 1)
                # remove content before the third last comma
                caption_obj = caption_obj[num_last_comma_index + 2:]
            # build llama dialog
            dialogs: List[Dialog] = [
                [{"role": "system", 
                "content": "You are a phrase summarizer who can summarize a most complete phrase that best represents \
                them from a sequence of phrases separated by commas, including as much effective information, adjective \
                and elements as possible without severe conflicting. \
                You only need to produce a string of summarized phrase.\
                Please produce nothing else!!!!!!!! Only one phrase. \
                The output format is: 'Summarized parase: [[your summarized parase itself]]'"}
                ,{"role": "user", "content": caption_example1}
                ,{"role": "assistant", "content": "Summarized parase: [a white car parked on the street]"}
                ,{"role": "user", "content": caption_example2}
                ,{"role": "assistant", "content": "Summarized parase: [a red and white sign]"}
                ,{"role": "user", "content": caption_example3}
                ,{"role": "assistant", "content": "Summarized parase: [the back of a triangular street sign]"}
                ,{"role": "user", "content": caption_obj}],
            ]

            # run llama response
            results = generator.chat_completion(
                dialogs,  # type: ignore
                max_gen_len= None,
                temperature=0.6,
                top_p=0.9,
            )
            # read generation content from llama output as merged caption
            for dialog, result in zip(dialogs, results):
                input_text = result["generation"]["content"]
                pattern = r'\[([^]]+)\]'  # match content in brackets
                match = re.search(pattern, input_text)
                extracted_content = []
                if match:
                    extracted_content = match.group(1)
                objects[i]["caption"] = extracted_content
    return objects, generator

def caption_merge(cfg, objects: MapObjectList):
    '''Merge object captions using the configured language model.'''
    if cfg.class_methods == "llama":
        # LLAMAload pretrained model
        start = time.time()
        generator = Llama.build(
            ckpt_dir=cfg.llama_ckpt_dir,
            tokenizer_path=cfg.llama_tokenizer_path,
            max_seq_len=cfg.llama_max_seq_len,
            max_batch_size=cfg.llama_max_batch_size,
        )
        print(f'Loading model spends{time.time()-start}')
        # example prompt
        caption_example1 = "a car parked on the street, a car parked on the street, a white car parked on the street, a car parked on the street, a black car parked on the street, a white car parked on the street, a white car on the road, a mirror of a white car"
        caption_example2 = "a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign"
        caption_example3 = "a triangular street sign, the back of a triangular sign"

        for i in trange(len(objects)):
            t1 = time.time()
            caption_obj = objects[i]["caption"]
            if ", " in caption_obj:
                # if too many observations make string too long, remove early observations
                comma_count = caption_obj.count(', ')
                if comma_count > cfg.max_caption_num:
                    num_last_comma_index = 0
                    for j in range(cfg.max_caption_num):
                        num_last_comma_index = caption_obj.rfind(', ', 0, num_last_comma_index - 1)
                    # remove content before the third last comma
                    caption_obj = caption_obj[num_last_comma_index + 2:]
                # build llama dialog
                dialogs: List[Dialog] = [
                    [{"role": "system", 
                    "content": "You are a phrase summarizer who can summarize a most complete phrase that best represents \
                    them from a sequence of phrases separated by commas, including as much effective information, adjective \
                    and elements as possible without severe conflicting. \
                    You only need to produce a string of summarized phrase.\
                    Please produce nothing else!!!!!!!! Only one phrase. \
                    The output format is: 'Summarized parase: [[your summarized parase itself]]'"}
                    ,{"role": "user", "content": caption_example1}
                    ,{"role": "assistant", "content": "Summarized parase: [a white car parked on the street]"}
                    ,{"role": "user", "content": caption_example2}
                    ,{"role": "assistant", "content": "Summarized parase: [a red and white sign]"}
                    ,{"role": "user", "content": caption_example3}
                    ,{"role": "assistant", "content": "Summarized parase: [the back of a triangular street sign]"}
                    ,{"role": "user", "content": caption_obj}],
                ]

                # run llama response
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len= None,
                    temperature=0.6,
                    top_p=0.9,
                )
                print(f'for each response {time.time()-t1}')
                # read generation content from llama output as merged caption
                for dialog, result in zip(dialogs, results):
                    input_text = result["generation"]["content"]
                    pattern = r'\[([^]]+)\]'  # match content in brackets
                    match = re.search(pattern, input_text)
                    extracted_content = []
                    if match:
                        extracted_content = match.group(1)
                    objects[i]["caption"] = extracted_content
                
        print(f'processing the whole object list needs {time.time()-start}')
    elif cfg.class_methods == "chatgpt":
        # LLAMAload pretrained model
        start = time.time()
        client = OpenAI()
        # example prompt
        caption_example1 = "a car parked on the street, a car parked on the street, a white car parked on the street, a car parked on the street, a black car parked on the street, a white car parked on the street, a white car on the road, a mirror of a white car"
        caption_example2 = "a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign, a red and white sign"
        caption_example3 = "a triangular street sign, the back of a triangular sign"

        # # example prompt

        for i in trange(len(objects)):
            t1 = time.time()
            caption_obj = objects[i]["caption"]
            if ", " in caption_obj:
                # if too many observations make string too long, remove early observations
                comma_count = caption_obj.count(', ')
                if comma_count > cfg.max_caption_num:
                    num_last_comma_index = 0
                    for j in range(cfg.max_caption_num):
                        num_last_comma_index = caption_obj.rfind(', ', 0, num_last_comma_index - 1)
                    # remove content before the third last comma
                    caption_obj = caption_obj[num_last_comma_index + 2:]
                # # build llama dialog
                message = [
                        {"role": "system", 
                            "content": "You are a phrase summarizer who can summarize a most complete phrase that best represents \
                            them from a sequence of phrases separated by commas, including as much effective information, adjective \
                            and elements as possible without severe conflicting. \
                            You only need to produce a string of summarized phrase.\
                            Please produce nothing else!!!!!!!! Only one phrase. \
                            The output format is: 'Summarized parase: [[your summarized parase itself]]'"}
                        ,{"role": "user", "content": caption_example1}
                        ,{"role": "assistant", "content": "Summarized parase: [a white car parked on the street]"}
                        ,{"role": "user", "content": caption_example2}
                        ,{"role": "assistant", "content": "Summarized parase: [a red and white sign]"}
                        ,{"role": "user", "content": caption_example3}
                        ,{"role": "assistant", "content": "Summarized parase: [the back of a triangular street sign]"}
                        ,{"role": "user", "content": caption_obj}]
                
                # run llama response
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages = message
                    )
                
                input_text = completion.choices[0].message.content #completion['choices'][0]['message']['content']
                pattern = r'\[([^]]+)\]'
                match = re.search(pattern, input_text)
                extracted_content = ""
                if match:
                    extracted_content = match.group(1)
                objects[i]["caption"] = extracted_content
        generator =None
                # read generation content from llama output as merged caption
                # for dialog, result in zip(dialogs, results):
                #     input_text = result["generation"]["content"]
                # pattern = r'\[([^]]+)\]' # match content in brackets
                #     match = re.search(pattern, input_text)
                #     extracted_content = []
                #     if match:
                #         extracted_content = match.group(1)
                #     objects[i]["caption"] = extracted_content
                
    return objects, generator

        
def captions_ft(objects: MapObjectList, bg_objects: MapObjectList, sbert_model):
    '''Encode and normalize captions for foreground and background objects.'''
    for i in range(len(objects)):
        caption = objects[i]["caption"]
        caption_fts = sbert_model.encode(caption, convert_to_tensor=True)
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        objects[i]["captions_ft"] = caption_fts
    for i in range(len(bg_objects)):
        caption = bg_objects[i]["caption"]
        caption_fts = sbert_model.encode(caption, convert_to_tensor=True)
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        bg_objects[i]["captions_ft"] = caption_fts
    return objects, bg_objects
