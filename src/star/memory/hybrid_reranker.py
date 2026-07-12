import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

import torch
import torch.nn.functional as F

from utils.similarity_utils import cosine_similarity
from utils.rerank_utils import should_rerank

class HybridReranker:
    def __init__(self, memory, scene_graph: List[Dict], dataset_start_timestamp: float, fps: int = 10, sbert_model=None):
        self.memory = memory
        self.scene_graph = scene_graph  # list of object dicts
        self.dataset_start_timestamp = dataset_start_timestamp
        self.fps = fps  # frames per second
        self.sbert_model = sbert_model  # optional, for text embedding


    def _compute_object_iou(self, ids1: List[int], ids2: List[int]) -> float:
        """
        Compute Intersection over Union (IoU) between two lists of object ids.
        """
        set1 = set(ids1)
        set2 = set(ids2)
        intersection = len(set1.intersection(set2))

        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        return intersection / len(set2)


    def _enhanced_rerank(self, candidates: List[Dict], query: str, top_n: int, alpha=0.7, beta=0.0, threshold=0.5) -> List[Dict]:
        """
        Deduplicate and rerank candidates based on object id overlap and time diversity.
        """
        selected = []
        query_vec = self.memory.embedder.embed_query(query)

        for cand in candidates:
            if not selected:
                selected.append(cand)
                continue

            keep = True
            for sel in selected:
                # Compute Object ID IoU
                iou = self._compute_object_iou(cand['object_ids'], sel['object_ids'])

                # Compute normalized time difference (assume seconds unit)
                delta_t = abs(cand['time'] - sel['time']) / (60 * 5)  # normalized over 5 minutes
                delta_t = min(delta_t, 1.0)

                similarity = alpha * iou #alpha * (1 - iou) + beta * delta_t

                if similarity !=0: # < threshold:
                    keep = False
                    break

            if keep:
                selected.append(cand)

            if len(selected) >= top_n:
                break

        return selected

    def retrieve(self, query: str, top_k_video=20, top_k_scene=10, final_top_n=5, debug=True) -> List[Dict]:
        """
        Retrieve top relevant memories by combining video captions and scene objects,
        then perform deduplication and reranking to select a diverse set.
        """
        if debug:
            print(f"[HybridReranker] Start retrieval for query: {query}")
            
        # Step 1: Retrieve top-k video captions based on text similarity
        video_captions, memory_list = self.memory.search_by_text_hybrid(query, top_k_video)

        if False:
            for idx, doc in enumerate(memory_list):
                extracted = [int(x) for x in doc.metadata.get('object_id', '').split(',') if x.strip().isdigit()]
                print(f"[HybridReranker] Retrieved memory {idx} object_ids: {extracted}")
            print(f"[HybridReranker] Retrieved video captions: {video_captions}")

        # Step 2: Parse object ids and timestamps from memory_list
        video_items = []
        for doc in memory_list:
            obj_ids = [int(x) for x in doc.metadata.get('object_id', '').split(',') if x.strip().isdigit()]
            timestamp = doc.metadata.get('time', None)  # You should ensure timestamp is saved in metadata

            video_items.append({
                "text": doc.page_content,
                "object_ids": obj_ids,
                "time": timestamp[0],
                "position": doc.metadata.get('position', [0.0, 0.0, 0.0]),
                "source": "video_caption"
            })

        # Step 3: Retrieve top-k scene objects (optional, currently not used in rerank, but could be fused later)
        scene_objects, object_scores = self._search_scene_objects(query, top_k_scene)
        scene_items = []

        for idx, obj in enumerate(scene_objects):
            scene_items.append({
                "text": obj['caption'],
                "object_ids": [obj['obj_id']],  # assume each scene object has one id
                "score": object_scores[idx],
                # "timestamp": self._get_object_time(obj),
                # "position": np.mean(obj['pcd_np'], axis=0).tolist(),
                "source": "scene_graph"
            })

        matched_caption_to_objects = {}

        for mem in video_items:
            caption_text = mem.get('text', '')
            caption_obj_ids = mem.get('object_ids', [])
            matched_objects = []

            for scene_obj in scene_items:
                scene_obj_id = scene_obj.get('object_ids', [])[0]
                if scene_obj_id in caption_obj_ids:
                    matched_objects.append({
                        "object_id": scene_obj_id,
                        "object_caption": scene_obj.get('text', '')  # Save both id and caption
                    })

            if matched_objects:
                matched_caption_to_objects[caption_text] = matched_objects

        if False:
            for cap, objs in matched_caption_to_objects.items():
                print(f"[HybridReranker] Caption:\n {cap}")
                for obj in objs:
                    print(f"    Matched object id: {obj['object_id']} | caption: {obj['object_caption']}")
        # Step 4: Merge video_items and scene_items
        candidates = video_items# + scene_items

        # Step 5: Deduplication and reranking
        selected = self._enhanced_rerank(candidates, query, final_top_n)
        output = self._format_selected_memories(selected, ref_time=self.dataset_start_timestamp)
        print(f"[HybridReranker] Finished initial retrieval.\n Selected candidates:\n{output}")
            


            #     if matched_objs:
            #         for obj in matched_objs:
            #             print(f"-> Matched Object ID: {obj['object_id']} | Caption: {obj['object_caption']}")
            #     else:
            #         print("-> No matched scene objects found.")
        return output
    
    def _format_selected_memories(self, selected_list, ref_time=None):
        from time import localtime, strftime
        out_string = ""
        for mem in selected_list:
            # Convert time
            t = mem['time']
            if isinstance(t, (float, int)):
                if ref_time:
                    t += self.dataset_start_timestamp
                else:
                    raise ValueError("ref_time must be provided for time conversion.")
                t = localtime(t)
                t = strftime('%Y-%m-%d %H:%M:%S', t)
            else:
                t = str(t)  # fallback
            
            # Format position
            pos = np.array(mem.get('position', [0.0, 0.0, 0.0])).round(3).tolist()
            s = f"At time={t}, the robot was at an average position of {pos}. "
            s += f"The robot saw the following: {mem['text']}\n\n"
            out_string += s
        return out_string


    def _search_scene_objects(self, query: str, top_k_scene) -> List[Dict]:
        """
        Search scene objects using SBERT features for matching.
        """

        # use sbert model to encode the query
        text_query_ft = self.sbert_model.encode([query], convert_to_tensor=True)  # List -> Tensor
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)   # normalize

        scored_objects = []
        for obj in self.scene_graph:
            if 'ft' not in obj or obj['ft'] is None:
                print(f"Object {obj.get('id', 'unknown')} does not have ft, skipping.")
                continue
            
            # Convert to torch tensor
            obj_ft = torch.tensor(obj['ft'], device=text_query_ft.device)
            obj_ft = obj_ft / obj_ft.norm(dim=-1, keepdim=True)  # normalize

            # Cal cos similarity
            score = F.cosine_similarity(text_query_ft, obj_ft.unsqueeze(0), dim=-1).item()
            scored_objects.append((obj, score))

        scored_objects.sort(key=lambda x: -x[1])
        retrieved_objects = []
        retrieved_scores = []
        for obj, score in scored_objects[:top_k_scene]:
            retrieved_objects.append(obj)
            retrieved_scores.append(score)
        retrieved_objects = [obj for obj, score in scored_objects[:top_k_scene]]
        # show the top N objects with their scores
        return retrieved_objects, retrieved_scores


    def _align_caption_to_objects(self, video_captions, scene_objects, window_sec=3.0):
        """Map video captions to nearby scene objects based on time."""
        caption_object_map = {}
        for caption in video_captions:
            cap_time = self._extract_time_from_caption(caption)
            matched_objs = []
            for obj in scene_objects:
                obj_time = self._get_object_time(obj)
                if abs(obj_time - cap_time) <= window_sec:
                    matched_objs.append(obj)
            caption_object_map[caption] = matched_objs
        return caption_object_map

    def _build_memory_candidates(self, video_captions, scene_objects, caption_object_map):
        candidates = []
        for cap in video_captions:
            candidates.append({
                "text": cap.page_content if hasattr(cap, 'page_content') else cap,
                "position": self._get_position(cap, caption_object_map.get(cap, [])),
                "time": self._format_time(self._extract_time_from_caption(cap)),
                "source": "video_caption"
            })
        for obj in scene_objects:
            candidates.append({
                "text": obj['caption'],
                "position": self._get_position(obj),
                "time": self._format_time(self._get_object_time(obj)),
                "source": "scene_graph"
            })
        return candidates

    def _rerank(self, candidates: List[Dict], query: str, top_n: int) -> List[Dict]:
        selected = []
        query_vec = self.memory.embedder.embed_query(query)
        
        # score and rerank
        scored = []
        for mem in candidates:
            sim_score = cosine_similarity(query_vec, self.memory.embedder.embed_query(mem['text']))
            scored.append((mem, sim_score))
        scored.sort(key=lambda x: -x[1])  # sort descending by similarity

        for mem, score in scored:
            if all(cosine_similarity(self.memory.embedder.embed_query(mem['text']), 
                                     self.memory.embedder.embed_query(sel['text'])) < 0.8 for sel in selected):
                selected.append(mem)
            if len(selected) >= top_n:
                break

        return selected

    def _extract_time_from_caption(self, caption):
        """Extract time from caption metadata or estimate."""
        if hasattr(caption, 'metadata') and 'time' in caption.metadata:
            return self.dataset_start_timestamp + caption.metadata['time'][0]
        # fallback if no metadata
        return self.dataset_start_timestamp

    def _get_object_time(self, obj):
        return self.dataset_start_timestamp + (obj['image_idx'])# / self.fps

    def _get_position(self, entity, matched_objs=None):
        """Priority: scene object position > robot pose from caption."""
        if isinstance(entity, dict) and 'pcd_np' in entity:
            return np.mean(entity['pcd_np'], axis=0).tolist()  # mean of point cloud as position
        if matched_objs:
            # pick the first matched object
            obj = matched_objs[0]
            return np.mean(obj['pcd_np'], axis=0).tolist()
        # fallback: no object position, return None or random
        return [0.0, 0.0, 0.0]

    def _format_time(self, seconds_since_start):
        time_obj = datetime.utcfromtimestamp(seconds_since_start)
        return time_obj.strftime("%H:%M:%S")
