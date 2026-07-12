from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, List, Optional, Tuple, Dict
from langchain_core.documents import Document
import numpy as np



from .memory import Memory, MemoryItem

from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import torch
import torch.nn.functional as F
from termcolor import colored


FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970



class MilvusWrapper:

    def __init__(self, collection_name='test', ip_address='127.0.0.1', port=19530, drop_collection=False):
        self.collection_name = collection_name
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, address=ip_address, port=port, drop_collection=drop_collection) # 1024, 384


    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, address='127.0.0.1', port=19530, drop_collection=False):
        connections.connect(host=address, port=port)
        
        if drop_collection:
            utility.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000),
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
            FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),
            FieldSchema(name='object_id', dtype=DataType.VARCHAR, description='object id', max_length=3000),

        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            output_fields=["id", "text_embedding"]
        )

        return res




class MilvusMemory(Memory):


    def __init__(self, db_collection_name: str, db_ip='127.0.0.1', db_port=19530, time_offset=FIXED_SUBTRACT, embedder=None):

        self.db_collection_name = db_collection_name
        self.db_ip = db_ip
        self.db_port = db_port
        self.time_offset = time_offset

        self.embedder = embedder[0] or HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
        self.sbert_model = embedder[1]
        self.working_memory = []
        
        self.reset(drop_collection=True)
        self.scene_graph = None
        self.search_position = {
            "method_name": "position_search",
            "data": []
        }
        self.search_time = {
            "method_name": "time_search",
            "data": []
        }
        self.search_text = {}
        self.search_SG = {}

    def insert(self, item: MemoryItem, text_embedding=None):
        # Convert the dataclass item to a dictionary
        memory_dict = asdict(item)
        # Assign a unique ID based on current timestamp
        memory_dict['id'] = str(time.time())
        # If no embedding is provided, compute one using the embedder
        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])
            print("text embedding is none, therefore generate it online")
        # Adjust the timestamp by a time offset (e.g. the start time of the current memory collection)
        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def reset(self, drop_collection=True):

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, self.db_ip, self.db_port, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
        )
        
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5})


        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
        )

        self.time_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"host": self.db_ip, "port": self.db_port},
            collection_name=self.db_collection_name,
            vector_field='time',
            text_field='caption',
        )
    
    def search_by_position(self, query: tuple) -> str:
        docs_with_scores = similarity_search_with_score_by_vector(
            self.position_vector_db, np.array(query).astype(float)
        )
        self.working_memory += [doc for doc, _ in docs_with_scores]

        # extract the time and score from the documents
        data = []
        for doc, score in docs_with_scores:
            t = doc.metadata['time']
            data.append({
                "time": t + self.time_offset,
                "score": score,
            })
        self.search_position = {
            "method_name": "position_search",
            "data": data
        }

        docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores])
        print("docs for Position search: ", docs_str)
        return docs_str

    #     self.working_memory += docs 

    #     docs = self.memory_to_string(docs)
    #     print("docs for Time search: ", docs)

    #     """Look up things online."""

    #     return docs

    # def search_by_time(self, hms_time: str) -> str:

    #     # Input is time like 08:20:30
    #     # need to convert to searchable time
    #     t = localtime(self.time_offset)
    #     mdy_date = strftime('%m/%d/%Y', t)
    #     template = "%m/%d/%Y %H:%M:%S"

    #     # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
    #     # bad llms don't listen :(
    #     try:
    #         res = bool(datetime.datetime.strptime(hms_time, template))
    #     except ValueError:
    #         res = False

    #     hms_time = hms_time.strip()
    #     if not res: # convert to the right format then
    #         hms_time = mdy_date + ' ' + hms_time

    #     query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - self.time_offset
    #     # convert from hms_time to something searchable


    #     docs = similarity_search_with_score_by_vector(self.time_vector_db, np.array([query, 0]))

    #     self.working_memory += docs

    #     docs = self.memory_to_string(docs)
    #     # np.unique([doc.metadata['time'][0] for doc in docs])
    #     """Look up things online."""
    #     print("docs for time search: ", docs)
    #     return docs
    def search_by_time(self, hms_time: str) -> str:
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res:
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time, template).timetuple()) - self.time_offset

        docs_with_scores = similarity_search_with_score_by_vector(
            self.time_vector_db, np.array([query, 0])
        )

        self.working_memory += [doc for doc, _ in docs_with_scores]

        # Extract data for plotting
        data = []
        for doc, score in docs_with_scores:
            t = doc.metadata['time']
            data.append({
                "time": t + self.time_offset,
                "score": score,
            })
        self.search_time = {
            "method_name": "time_search",
            "data": data
        }

        docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores])
        print("docs for time search: ", docs_str)
        return docs_str


    # def search_by_text(self, query: str, k =5) -> str:

    #     docs = self.text_retriever.invoke(query, k=k)
    #     # docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k)

    #     self.working_memory += docs

    #     docs = self.memory_to_string(docs)

    #     """Look up things online."""        
    #     return docs

    # def search_by_text(self, query: str, k=5) -> str:
    #     # docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=k)
        
    #     # self.working_memory += [doc for doc, _ in docs_with_scores]
    #     docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=50)
    
    #     # ✅ Only add top-k to working memory
    #     self.working_memory += [doc for doc, _ in docs_with_scores[:k]]
    #     # Extract data for plotting
    #     data = []
    #     for doc, score in docs_with_scores:
    #         t = doc.metadata['time']
    #         data.append({
    #             "time": t + self.time_offset,
    #             "score": score,
    #         })
    #     self.latest_plot_data = {
    #         "method_name": "text_search",
    #         "data": data
    #     }

    #     docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores])
    #     return docs_str
    def search_by_text(self, query: str, k=6) -> str:
        # Retrieve more results (e.g., top-50) to inspect the overall distribution,
        # but only add the top-k results to the working memory for downstream tasks.
        docs_with_scores = self.text_retriever.vectorstore.similarity_search_with_score(query, k=50)

        # ✅ Add only the top-k documents to the working memory.
        self.working_memory += [doc for doc, _ in docs_with_scores[:k]]

        # ✅ Extract data for plotting.
        data = []
        scores = []
        for doc, score in docs_with_scores:
            t = doc.metadata['time']
            data.append({
                "time": t + self.time_offset,  # Shift time by the offset for real-time alignment.
                "score": score,
            })
            scores.append(score)

        # ✅ Normalize scores between 0 and 1.
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score == 0:
            normalized_scores = [0.0 for _ in scores]
        else:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        # ✅ Add normalized scores to the plot data.
        for i in range(len(data)):
            data[i]['score_normalized'] = normalized_scores[i]

        # ✅ Store the plot data for later visualization.
        if not hasattr(self, 'search_text'):
            self.search_text = {}

    
        self.search_text[query] = {
        "method_name": f"cue: {query}",
        "data": data,
        "score_min": min_score,
        "score_max": max_score,
        }
        # ✅ Return string representation of the top-k results for downstream use.
        docs_str = self.memory_to_string([doc for doc, _ in docs_with_scores[:k]])
        return docs_str

    def search_by_text_hybrid(self, query: str, k =5) -> str:

        docs = self.text_retriever.invoke(query, k=k)

        self.working_memory += docs
        memory_list = docs
        docs = self.memory_to_string(docs)

        """Look up things online."""        
        return docs, memory_list
    
    

    ### Doc formatting for the last LLM
    def memory_to_string(self, memory_list: list[MemoryItem], ref_time: float=None):
        if ref_time == None:
            ref_time = self.time_offset
        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        print(colored(f"Retrieved video captions: \n{out_string}", "yellow"))
        return out_string
    
    def format_scene_objects_for_llm(self, objects: List[Dict]) -> str:
        """
        Format retrieved scene objects into a readable string for LLM consumption.
        Each object will be described with its caption, ID, and approximate location from bbox center.
        """
        if not objects:
            return "No scene objects matched the query."

        lines = ["Here are the top matched scene objects:"]
        for idx, obj in enumerate(objects, start=1):
            caption = obj.get('caption', 'unknown')
            obj_id = obj.get('obj_id', 'N/A')
            # time info
            times = obj.get('time', 'unknown')
            time_dt = [datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') for t in times]
            intervals = self.merge_time_intervals(time_dt)
            description = self.format_intervals_compact(intervals)
            try:
                center = obj['bbox'].center  # Use OrientedBoundingBox center
                pos_str = f"at position [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
            except Exception:
                pos_str = "with unknown position"
            lines.append(f"Object ID {obj_id}: Located at {pos_str}; \"{caption}\". detected at times {description}.") #f"{idx}. Object ID {obj_id}: \"{caption}\" {pos_str}. detected at times {time_dt}.")
            print(colored(f"Object ID {obj_id}: Located at {pos_str}; \"{caption}\". detected at times {description}.", "yellow"))
    
        return "\n".join(lines)
    

    def merge_time_intervals(self, time_list_str, max_gap_sec=1):
        times = sorted([datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in time_list_str])
        if not times:
            return []

        intervals = []
        start = times[0]
        end = times[0]

        for curr in times[1:]:
            if (curr - end).total_seconds() <= max_gap_sec:
                end = curr
            else:
                intervals.append((start, end))
                start = curr
                end = curr
        intervals.append((start, end))

        return intervals

    def format_intervals_compact(self, intervals):
        if not intervals:
            return "No observation intervals found."

        first_day = intervals[0][0].strftime('%Y-%m-%d')
        lines = [f"The object was observed on {first_day} during:"]
        
        for s, e in intervals:
            if s.date() != e.date():
                # If it is cross-date then display separately
                lines.append(f" {s.strftime('%Y-%m-%d %H:%M:%S')} ~ {e.strftime('%Y-%m-%d %H:%M:%S')}")
            elif s == e:
                lines.append(f" {s.strftime('%H:%M:%S')}")
            else:
                lines.append(f" {s.strftime('%H:%M:%S')} ~ {e.strftime('%H:%M:%S')}")

        return ",".join(lines)

    #     lines = ["Here are the top matched scene objects:"]
    #     for idx, obj in enumerate(objects, start=1):
    #         caption = obj.get('caption', 'unknown')
    #         obj_id = obj.get('obj_id', 'N/A')
    #         try:
    #             pos = np.mean(obj['pcd_np'], axis=0).tolist()
    #             pos_str = f"at position {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}"
    #         except Exception:
    #             pos_str = "with unknown position"
    #         lines.append(f"{idx}. Object ID {obj_id}: \"{caption}\" {pos_str}.")

    #     return "\n".join(lines)

    
    def set_scene_graph(self, scene_graph):
        """Attach a SceneGraph for scene object retrieval."""
        self.scene_graph = scene_graph
    

    #     # use sbert model to encode the query
    #     text_query_ft = self.sbert_model.encode([query], convert_to_tensor=True)  # List -> Tensor
    #     text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)   # normalize

    #     scored_objects = []
    #     for obj in self.scene_graph:
    #         if 'ft' not in obj or obj['ft'] is None:
    #             print(f"Object {obj.get('id', 'unknown')} does not have ft, skipping.")
    #             continue
     
    #         obj_ft = torch.tensor(obj['ft'], device=text_query_ft.device)
    #         obj_ft = obj_ft / obj_ft.norm(dim=-1, keepdim=True)  # normalize

  
    #         score = F.cosine_similarity(text_query_ft, obj_ft.unsqueeze(0), dim=-1).item()
    #         scored_objects.append((obj, score))

    #     scored_objects.sort(key=lambda x: -x[1])
    #     retrieved_objects = []
    #     retrieved_scores = []
    #     for obj, score in scored_objects[:top_k_scene]:
    #         retrieved_objects.append(obj)
    #         retrieved_scores.append(score)
    #     retrieved_objects = [obj for obj, score in scored_objects[:top_k_scene]]
    #     doc = self.format_scene_objects_for_llm(retrieved_objects)
    #     # print("docs for scene graph search: ", doc)
    #     return doc
    
    def search_scenegraph(self, query: str, top_k_scene=10) -> str:
        """
        Search scene objects using SBERT features for matching and store plotting data.
        """
        text_query_ft = self.sbert_model.encode([query], convert_to_tensor=True)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)

        scored_objects = []
        for obj in self.scene_graph:
            if 'ft' not in obj or obj['ft'] is None:
                print(f"Object {obj.get('id', 'unknown')} does not have ft, skipping.")
                continue
            
            obj_ft = torch.tensor(obj['ft'], device=text_query_ft.device)
            obj_ft = obj_ft / obj_ft.norm(dim=-1, keepdim=True)

            score = F.cosine_similarity(text_query_ft, obj_ft.unsqueeze(0), dim=-1).item()
            scored_objects.append((obj, score))

        scored_objects.sort(key=lambda x: -x[1])

        # Store plot data
        plot_data = []
        for idx, (obj, score) in enumerate(scored_objects[:20]):
            times = obj.get('time', [])
            if not isinstance(times, list):
                times = [times]  # Make sure it's a list

            plot_data.append({
                "object_id": obj.get('obj_id', f"obj_{idx}"),
                "times": times,
                "score": score
            })
        # Save for downstream plotting
        if not hasattr(self, 'search_text'):
            self.search_text = {}
    
        self.search_SG[query] = {
        "method_name": f"SG_{query}",
        "data": plot_data,
        # "score_min": min_score,
        # "score_max": max_score,
        }

        retrieved_objects = [obj for obj, _ in scored_objects[:top_k_scene]]
        doc = self.format_scene_objects_for_llm(retrieved_objects)
        print(colored(f"docs for scene graph search: \n, {doc}", "yellow"))
        return doc






def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params

        # Determine result metadata fields with PK.
        output_fields = pos_db.fields[:]
        timeout = pos_db.timeout or timeout
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results.
        ret = []
        for result in res[0]:
            data = {x: result.entity.get(x) for x in output_fields}
            doc = pos_db._parse_document(data)
            pair = (doc, result.score)
            ret.append(pair)

        return ret #[doc for doc, _ in ret]


