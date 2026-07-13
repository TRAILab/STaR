
import re
import sys
import glob
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).resolve().parent))

import gzip
import pickle
import os
import argparse
import shutil
import traceback 
import distinctipy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from termcolor import colored
from dataclasses import asdict
from star.some_class.map_class import MapObjectList

from star.memory.memory import MemoryItem
from star.memory.text_memory import TextMemory

from star.agents.star_agent_aib import STaRAgent_AIB
from star.agents.star_agent_hybrid import STaRAgent_VANILA
from star.agents.star_agent_sg import STaRAgent_SG
from star.memory.milvus_memory_coda import MilvusMemory
from star.utils.util import get_caption, assign_frame_indices, assign_object_ids, extract_gt_times, plot_multi_method_scores, plot_scenegraph_scores
from star.utils.utils import print_to_cot_log

from langchain_huggingface import HuggingFaceEmbeddings

import torch
import open3d as o3d
import networkx as nx
from collections import Counter
from sentence_transformers import SentenceTransformer



# Vivid, high-contrast base palette (RGB in [0,1]); adapted from Tableau/Glasbey-like sets
_BASE_VIVID = [
    (0.121, 0.466, 0.705),  # blue
    (1.000, 0.498, 0.054),  # orange
    (0.172, 0.627, 0.172),  # green
    (0.839, 0.152, 0.156),  # red
    (0.580, 0.404, 0.741),  # purple
    (0.549, 0.337, 0.294),  # brown
    (0.890, 0.467, 0.761),  # pink
    (0.498, 0.498, 0.498),  # gray
    (0.737, 0.741, 0.133),  # olive
    (0.090, 0.745, 0.811),  # cyan
    (0.000, 0.000, 0.000),  # black
    (1.000, 0.000, 0.000),  # bright red
]


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def bbox_center(bbox):
    """Return a center vector for either an Open3D AABB or OBB."""
    get_center = getattr(bbox, "get_center", None)
    return np.asarray(get_center() if callable(get_center) else bbox.center)


def _normalize_single_object_id(object_id):
    if object_id is None:
        return None
    if isinstance(object_id, str):
        object_id = object_id.strip()
        if not object_id or object_id.lower() == "null":
            return None
        object_id = eval(object_id)
    if isinstance(object_id, (list, tuple)):
        if len(object_id) == 0:
            return None
        object_id = object_id[0]
    return int(object_id)


def _scene_object_by_id(scene_graph, object_id):
    if scene_graph is None or object_id is None:
        return None

    for obj in scene_graph:
        if isinstance(obj, dict):
            obj_id = obj.get("obj_id", None)
            if obj_id is not None and int(obj_id) == int(object_id):
                return obj

    try:
        return scene_graph[int(object_id)]
    except (IndexError, TypeError, ValueError):
        return None


def apply_object_id_position(parsed, scene_graph, log_file=None, announce=True):
    """Use the scene graph bbox center whenever the agent selected an object id."""
    try:
        object_id = _normalize_single_object_id(parsed.get("object_id"))
    except Exception as exc:
        print(colored(f"Could not parse predicted object id {parsed.get('object_id')}: {exc}", "yellow"))
        return parsed

    if object_id is None:
        return parsed

    obj = _scene_object_by_id(scene_graph, object_id)
    if obj is None or "bbox" not in obj:
        print(colored(f"Predicted object id {object_id} was not found in the scene graph; keeping model position.", "yellow"))
        return parsed

    pred_pos = bbox_center(obj["bbox"])
    parsed["object_id"] = object_id
    parsed["position"] = pred_pos.tolist()
    if announce:
        message = f"Using predicted object id {object_id} with position {pred_pos}"
        if log_file is not None:
            print_to_cot_log(message=f"System: {message}", log_file=log_file, color="yellow")
        else:
            print(colored(message, "yellow"))
    return parsed


# we can have binary, position-based, time-based, or description-based. let's answer accordingly
def evaluate_output(qa_instance, predicted, scenegrpahh=None):

    out_error = {}

    q_type = qa_instance['type']
    if 'position' in q_type:
        # answer = np.array(qa_instance['answers']['position'])
        answer = np.array(qa_instance['answers']['target_bbox']['center'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['position']) == str:
            predicted['position'] = eval(predicted['position'])
        try:
            if type(predicted['object_id']) == str:
                predicted['object_id'] = eval(predicted['object_id'])
            
            # check if object_id is valid, by default we set it to null
            if predicted['object_id'] is not None:
                pred_pos = bbox_center(scenegrpahh[int(predicted['object_id'])]['bbox'])
                print(colored(f"Using predicted object id {predicted['object_id']} with position {pred_pos}", "yellow"))
            else:
                pred_pos = np.array(predicted['position'])
                print(colored(f"Using predicted position {pred_pos}", "yellow"))
        except Exception as e:
            print(e)
            pred_pos = np.array(predicted['position'])
            print(colored(f"Using predicted position {pred_pos}", "yellow"))

        dist = np.linalg.norm(answer - pred_pos)

        out_error['position_error'] = dist

    elif 'binary' in q_type:

        answer = qa_instance['answers']['text'][1] # we made this assumption in other examples that binary answer is the second one

        if 'binary' in predicted and (predicted['binary'].lower() == "yes" or predicted['binary'].lower() == "no"):
            # get correct/incorrect label
            if predicted['binary'].lower() == answer.lower():
                correct = 1
            else:
                correct = 0

            out_error['binary_iscorrect'] = correct

    elif 'time' in q_type:

        answer = np.array(qa_instance['answers']['time'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['time']) == str:
            predicted['time'] = eval(predicted['time'])
        pred_time = np.array(predicted['time'])

        dist = abs(answer - pred_time)

        out_error['time_error'] = dist

    elif 'duration' in q_type:

        answer = np.array(qa_instance['answers']['duration'])

        # compute L2 loss between predicted['binary'] and answer
        if type(predicted['duration']) == str:
            predicted['duration'] = eval(predicted['duration'])
        pred_time = np.array(predicted['duration'])

        dist = abs(answer - pred_time)

        out_error['duration_error'] = dist

    elif 'text' in q_type:
        answer = qa_instance['answers']['text']
        out_error = {'answer': answer}
        print(colored(f"Ground truth answers: {answer}", "yellow"))
        print(colored(f"Robot's answer: {predicted['text']}", "yellow"))


    else:
        raise Exception("We do not support question type " + q_type)

    return out_error


def answer_squad_question(model, question, qa_instance=None, scenegrpahh=None):

    # print(f'Question: {question}')
    print(colored(f'Question: {question}', "green", attrs=["bold"]))

    parsed = None
    while True:
        try:

            start_time = time.time()
            response = model.query(question) # the key part of the code

            end_time = time.time()

            elapsed = end_time - start_time

            parsed = asdict(response)
            cot_log_file = getattr(model, "cot_log_file", None)
            if cot_log_file is None and getattr(model, "memory", None) is not None:
                cot_log_file = getattr(model.memory, "cot_log_file", None)
            parsed = apply_object_id_position(
                parsed,
                scenegrpahh,
                log_file=cot_log_file,
                announce=(qa_instance is None),
            )
            print(colored(f"Raw response: {parsed}", "blue"))

            out_error = {}
            if qa_instance is not None:
                out_error = evaluate_output(qa_instance, parsed, scenegrpahh)
            print(colored(f"Time taken to answer question: {elapsed} seconds", "green", attrs=["bold"]))

        except Exception as e:
            print(parsed)
            print(e)
            traceback.print_exception(*sys.exc_info()) 
            continue

        return_dict = {"response": parsed}
        return_dict.update(parsed)
        return_dict['error'] = out_error
        return_dict['elapsed'] = elapsed

        return return_dict


def print_eval_result(qa_instance, predicted, error_dict):
    q_type = qa_instance["type"]

    if q_type == "binary":
        gt = qa_instance["answers"]["text"][1]
        predicted_value = predicted.get("binary")
        correct = error_dict.get("binary_iscorrect")
        if correct is None:
            print(colored(f"[eval][result] type=binary predicted={predicted_value} gt={gt} correct=unknown", "yellow"))
        else:
            print(colored(f"[eval][result] type=binary predicted={predicted_value} gt={gt} correct={bool(correct)}", "green" if correct else "red"))
        return

    if q_type == "position":
        error = error_dict.get("position_error")
        if error is None:
            print(colored("[eval][result] type=position error=unknown correct=unknown", "yellow"))
            return
        correct = error < 20.0
        print(colored(f"[eval][result] type=position error={float(error):.3f}m correct={correct} threshold=20.0m", "green" if correct else "red"))
        return

    if q_type == "time":
        gt = qa_instance["answers"].get("time")
        predicted_value = predicted.get("time")
        error = error_dict.get("time_error")
        if error is None:
            print(colored(f"[eval][result] type=time predicted={predicted_value} gt={gt} error=unknown correct=unknown", "yellow"))
            return
        correct = error < 2.0
        print(colored(f"[eval][result] type=time predicted={predicted_value} gt={gt} error={float(error):.3f}s correct={correct} threshold=2.0s", "green" if correct else "red"))
        return

    if q_type == "duration":
        gt = qa_instance["answers"].get("duration")
        predicted_value = predicted.get("duration")
        error = error_dict.get("duration_error")
        if error is None:
            print(colored(f"[eval][result] type=duration predicted={predicted_value} gt={gt} error=unknown correct=unknown", "yellow"))
            return
        correct = error < 2.0
        print(colored(f"[eval][result] type=duration predicted={predicted_value} gt={gt} error={float(error):.3f}s correct={correct} threshold=2.0s", "green" if correct else "red"))
        return

    if q_type == "text":
        print(colored("[eval][result] type=text automatic correctness is not computed", "yellow"))
        return

    print(colored(f"[eval][result] type={q_type} automatic correctness is not computed", "yellow"))


def wait_for_gradio_command(instruction_path, poll_interval=0.2):
    """Wait for and consume one command submitted by the Gradio interface."""
    print(f"Waiting for a Gradio query at: {instruction_path}")

    while True:
        try:
            with open(instruction_path, "r", encoding="utf-8") as f:
                instruction = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            time.sleep(poll_interval)
            continue

        if not instruction.get("trigger", False):
            time.sleep(poll_interval)
            continue

        command = str(instruction.get("command", "")).strip()

        os.makedirs(os.path.dirname(instruction_path), exist_ok=True)
        temporary_path = f"{instruction_path}.eval_tmp"
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump({"trigger": False, "command": ""}, f)
        os.replace(temporary_path, instruction_path)

        if command:
            print(f"Received Gradio query: {command}")
            return command

        print("Ignoring an empty Gradio query.")


def prepare_gradio_logging(args, agent, memory, test_num, question):
    """Point the agent and Gradio UI at the same log/latest-index files."""
    latest_idx_path = args.latest_idx_path
    cot_log_file = os.path.join(
        args.results,
        str(args.sequence_id),
        "cot_log",
        args.postfix,
        f"cot_log_{test_num}.txt",
    )
    keyframe_dir = os.path.join(
        args.results,
        str(args.sequence_id),
        "images",
        args.postfix,
        str(test_num),
    )

    os.makedirs(os.path.dirname(latest_idx_path), exist_ok=True)
    with open(latest_idx_path, "w") as f:
        f.write(str(test_num))

    if os.path.isdir(keyframe_dir):
        for name in os.listdir(keyframe_dir):
            path = os.path.join(keyframe_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    else:
        os.makedirs(keyframe_dir, exist_ok=True)

    os.makedirs(os.path.dirname(cot_log_file), exist_ok=True)
    with open(cot_log_file, "w") as f:
        f.write("System: Chain of Thought Log\n")

    if hasattr(agent, "test_num"):
        agent.test_num = test_num
    if hasattr(agent, "cot_log_file"):
        agent.cot_log_file = cot_log_file
    if hasattr(memory, "cot_log_file"):
        memory.cot_log_file = cot_log_file
    if hasattr(agent, "memory") and hasattr(agent.memory, "cot_log_file"):
        agent.memory.cot_log_file = cot_log_file

    print_to_cot_log(message=f"User: {question}", log_file=cot_log_file)
    return cot_log_file


def plot_retrieval_for_gradio(args, agent, gt_info, test_num):
    """Save retrieval plots using the filename/folder convention Gradio polls."""
    plot_data = []
    search_text = getattr(agent.memory, "search_text", {})
    for _, content in search_text.items():
        plot_data.append(content)

    search_time = getattr(agent.memory, "search_time", None)
    search_position = getattr(agent.memory, "search_position", None)
    if search_time is not None:
        plot_data.append(search_time)
    if search_position is not None:
        plot_data.append(search_position)

    search_SG = getattr(agent.memory, "search_SG", {})
    sg_data = []
    for _, content in search_SG.items():
        sg_data.append(content)

    save_fig_path_db = os.path.join(
        args.results, str(args.sequence_id), args.VDB, args.postfix
    )
    save_fig_path_sg = os.path.join(
        args.results, str(args.sequence_id), args.SG, args.postfix
    )
    os.makedirs(save_fig_path_db, exist_ok=True)
    os.makedirs(save_fig_path_sg, exist_ok=True)

    if args.method == "scene_graph":
        plot_path = os.path.join(
            save_fig_path_sg, f"scenegraph_{test_num}_{args.postfix}.png"
        )
        ui_plot_path = os.path.join(
            save_fig_path_db, f"retrieval_DB_{test_num}_{args.postfix}.png"
        )
        print(colored(f"[eval][plot] scene graph retrieval plot: {plot_path}", "cyan"))
        plot_scenegraph_scores(sg_data, gt_info=gt_info, save_path=plot_path)
        plot_scenegraph_scores(sg_data, gt_info=gt_info, save_path=ui_plot_path)
    else:
        plot_path = os.path.join(
            save_fig_path_db, f"retrieval_DB_{test_num}_{args.postfix}.png"
        )
        print(colored(f"[eval][plot] retrieval plot: {plot_path}", "cyan"))
        plot_multi_method_scores(plot_data, k=args.topk, gt_info=gt_info, save_path=plot_path)

    return plot_path


def load_scenegraph(result, start_time=None, end_time=None, dataset_start_timestamp=None, fps=10):
    '''
    Load the full scene graph, and optionally filter objects by time window.

    Args:
        result_path (str): path to scene graph pkl
        start_time (float): task start time (absolute, e.g., seconds since epoch)
        end_time (float): task end time
        dataset_start_timestamp (float): the timestamp corresponding to image_idx=0
        fps (int): frames per second (default 10)
    '''
    objects, objects_all = result[0], result[1]

    if start_time is not None and end_time is not None and dataset_start_timestamp is not None:
        filtered_objects = []
        for obj in objects:
            if 'image_idx' not in obj:
                continue
            # Remove duplicates and sort indices
            idx_list = obj['image_idx']
            matched = False

            # Check if any index falls into the time window
            for idx in idx_list:
                obj_time = dataset_start_timestamp + (idx*10 / fps)
                if start_time <= obj_time <= end_time:
                    matched = True
                    break
            if matched:
                filtered_objects.append(obj)

        # print(f"[load_result] Total objects in scene graph: {len(objects)}")
        # print(f"[load_result] Objects kept after time filtering: {len(filtered_objects)}")
        print(colored(f"[load_result] Total objects in scene graph: {len(objects)}", "white", attrs=["dark"]))
        print(colored(f"[load_result] Objects kept after time filtering: {len(filtered_objects)}", "white", attrs=["dark"]))
        objects = MapObjectList(filtered_objects)  # only keep filtered objects

    # # Regenerate instance colors.
    # instance_colors = distinctipy.get_colors(len(objects) + (len(bg_objects) if bg_objects else 0), pastel_factor=0.5)
    # instance_colors = {str(i): c for i, c in enumerate(instance_colors)}

    return objects, objects_all#, bg_objects, instance_colors

def load_memory(args, qa_instance, use_milvus=True, ip_address='127.0.0.1', embedder=None, objects=None, objects_all=None):
    # Here we load everything needed to load a MilvusDB instance neatly
    captions_path = os.path.join(args.results, str(args.sequence_id), 'caption', f'{args.caption_file}.json') #_{str(args.sequence_id)}
    with open(captions_path, 'r') as f:
        out = json.load(f)

    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    all_end_times = np.array([float(x['file_end'][:-4]) for x in out])

    if args.all_mem or qa_instance is None:
        # if we want to use the full memory, we need to set the end time to the last frame
        start_time = all_start_times[0]
        end_time = all_end_times[-1]
    else:
        start_time = np.float64(qa_instance['start_time'])
        end_time = qa_instance['end_time']

    # scene_graph = args.scene_graph
    if use_milvus:
        # milv = MilvusWrapper(ip_address=ip_address)
        memory = MilvusMemory(f"eval_memory_{args.sequence_id}", db_ip=ip_address, time_offset=start_time, embedder=embedder, args=args)
    else:
        memory = TextMemory()

    memory.reset()

    outputs = []

    # Compute start idx
    diff = all_start_times - start_time
    start_idx = np.argmin(np.abs(diff))
    # Compute end idx
    diff = all_end_times - end_time
    end_idx = np.argmin(np.abs(diff))


    # Prefer frame indices saved during memory construction; fall back for old caption files.
    if all(item.get("frame_indices") for item in out):
        for item in out:
            item["frame_idx"] = item["frame_indices"]
    else:
        out = assign_frame_indices(out, all_start_times, all_end_times, duration=3.0)

    if args.method == "star" or qa_instance is None:
        objects, objects_all = load_scenegraph(
                # result_path=scene_graph_path,
                result=(objects, objects_all),
                start_time=all_start_times[0],
                end_time=all_end_times[-1],
                dataset_start_timestamp=all_start_times[0],
                fps=10  # or your real fps
            )
    else:
        objects, objects_all = load_scenegraph(
                # result_path=scene_graph_path,
                result=(objects, objects_all),
                start_time=qa_instance['start_time'],
                end_time=qa_instance['end_time'],
                dataset_start_timestamp=all_start_times[0],
                fps=10  # or your real fps
            )
    
    out = assign_object_ids(out, objects)
    # for i, x in enumerate(out):
    #     print(x['frame_idx'])
    #     print(x['object_id'])
    for i in range(start_idx, end_idx+1):
        # print(colored(f"Processing caption {i} with time {all_start_times[i]} to {all_end_times[i]}", "cyan"))
        item = out[i]
        obj_id = item.get('object_id', None)
        if isinstance(obj_id, list):
            object_id = ','.join(map(str, obj_id))
        elif isinstance(obj_id, int):
            object_id = str(obj_id)
        elif obj_id is None:
            object_id = ''

        entity = {
            'position': item['position'],
            'theta': item['theta'], # ignoring rotation
            'time': item['time'], 
            'caption': item['caption'],
            'object_id': object_id,
        }

        outputs.append(entity)

        entity = MemoryItem.from_dict(entity)

        if use_milvus:
            
            memory.insert(entity, text_embedding=item['text_embedding'])
            if args.all_mem:
                memory.set_scene_graph(objects_all)
            else:
                memory.set_scene_graph(objects)
        else:
            memory.insert(entity)

    return memory, outputs, all_start_times[0], objects

def load_SG_data(fps=10):
    '''
    Load the full scene graph, and optionally filter objects by time window.

    Args:
        result_path (str): path to scene graph pkl
        start_time (float): task start time (absolute, e.g., seconds since epoch)
        end_time (float): task end time
        dataset_start_timestamp (float): the timestamp corresponding to image_idx=0
        fps (int): frames per second (default 10)
    '''
    scene_graph_path = os.path.join(args.results, str(args.sequence_id), 'pcd', f'{args.scenegraph_file}.pkl.gz')
    print("scene_graph_path\n", scene_graph_path)
    captions_path = os.path.join(args.results, str(args.sequence_id), 'caption', f'{args.caption_file}.json')#_{str(args.sequence_id)}
    with open(captions_path, 'r') as f:
        out = json.load(f)
    all_start_times = np.array([float(x['file_start'][:-4]) for x in out])
    dataset_start_timestamp = all_start_times[0]

    with gzip.open(scene_graph_path, "rb") as f:
        results = pickle.load(f)
        # print(f"[load_result] Loaded {len(results)} objects from {result_path}")
        #print(colored(f"[load_result] Loaded {len(results)} objects from {result_path}", "grey"))
        
    if isinstance(results, dict):
        objects = MapObjectList()
        objects.load_serializable(results["objects"])
        
        if results.get('bg_objects') is None:
            bg_objects = None
        else:
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results["bg_objects"])
        
    elif isinstance(results, list):
        objects = MapObjectList()
        objects.load_serializable(results)
        bg_objects = None
    else:
        raise ValueError("Unknown results type: ", type(results))
    
    for i, obj in enumerate(objects):
        objects[i]['caption'] = get_caption(obj['caption'], method='majority')
        #print(f"[load_result] Object {i} has caption: {objects[i]['caption']}")
        objects[i]['image_idx'] = sorted(set(obj['image_idx']))
        objects[i]['obj_id'] = i
        objects[i]['time'] = [dataset_start_timestamp + (idx * 10 / fps) for idx in obj['image_idx']]
        # print(f"[load_result] Object {i} has {len(obj['image_idx'])} images, with time {obj['time']}")
    objects_all = objects.copy()
    timestamps = np.array(results['timestamps'])
    caption_image_records = [
        {
            'file_start': item['file_start'],
            'file_end': item['file_end'],
            'frame_indices': item.get('frame_indices', []),
        }
        for item in out
    ]
    return objects, objects_all, timestamps, caption_image_records

def set_run_name(args):
    args.run_name = f"{args.method}+{args.llm}"
    return args


def attach_memory_to_agent(args, agent, memory, scene_graph, global_starttime, sbert_model, test_num, timestamp_list=None, caption_image_records=None):
    """Attach memory with the method-specific context expected by each agent."""
    if isinstance(agent, STaRAgent_AIB):
        agent.set_memory(
            memory,
            scene_graph=scene_graph,
            dataset_start_timestamp=global_starttime,
            timestamp_list=np.array(timestamp_list) if timestamp_list is not None else None,
            caption_image_records=caption_image_records,
            sbert_model=sbert_model,
            test_num=test_num,
            args=args,
        )
    elif isinstance(agent, STaRAgent_SG):
        agent.set_memory(
            memory,
            scene_graph=scene_graph,
            dataset_start_timestamp=global_starttime,
            timestamp_list=np.array(timestamp_list) if timestamp_list is not None else None,
            sbert_model=sbert_model,
            test_num=test_num,
            args=args,
        )
    elif isinstance(agent, STaRAgent_VANILA):
        agent.set_memory(
            memory,
            scene_graph=scene_graph,
            dataset_start_timestamp=global_starttime,
            sbert_model=sbert_model,
            test_num=test_num,
            args=args,
        )
    else:
        agent.set_memory(memory)


def clear_search_state(memory):
    if hasattr(memory, "search_text"):
        memory.search_text = {}
    if hasattr(memory, "search_SG"):
        memory.search_SG = {}
    if hasattr(memory, "search_time"):
        memory.search_time = None
    if hasattr(memory, "search_position"):
        memory.search_position = None


# -----------------------------
# Main: step-by-step
# -----------------------------

def main(args):
    args = set_run_name(args)

    use_milvus = False
    if args.method == "star":
        base_llm = args.llm
        agent = STaRAgent_AIB(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True
        print(colored("Using STaR Agent", "white", attrs=["dark"]))
    elif args.method == "remembr":
        base_llm = args.llm
        agent = STaRAgent_VANILA(llm_type=base_llm, vlm_type='gpt-4.1', num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True
        print(colored("Using ReMEmbR Agent", "white", attrs=["dark"]))
    elif args.method == "scene_graph":
        base_llm = args.llm
        agent = STaRAgent_SG(llm_type=base_llm, num_ctx=args.num_ctx, temperature=args.temperature)
        use_milvus = True
        print(colored("Using Scene-Graph Agent", "white", attrs=["dark"]))

    # Load question data when evaluating an existing dataset.
    data = []
    if args.question_source == "dataset":
        data_path = os.path.join(args.data_dir, 'questions', str(args.sequence_id), args.qa_file+'.json')
        data = json.load(open(data_path, 'r'))
        data = data['data']

    # below is the variable to keep track of the evaluation
    running_successes = 0
    num_binary = 0
    sr_binary = 0

    running_pos_error = 0
    num_position = 0
    sr_position = 0

    running_time_error = 0
    num_time = 0
    sr_time = 0

    running_duration_error = 0
    num_duration = 0
    sr_duration = 0
    
    responses = []
    
    embedder = HuggingFaceEmbeddings(
        model_name='mixedbread-ai/mxbai-embed-large-v1',
        model_kwargs={
            "device": "cuda:0",  #  TITAN Xp
            #"torch_dtype": "auto",
        }
    )
    # object_vis = ObjectVisualizer()
    print("Initializing SBERT model...")
    #sbert_model = SentenceTransformer(cfg.sbert_path)
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sbert_model = sbert_model.to("cuda")
    print(colored("Done initializing SBERT model.", "white", attrs=["dark"]))
    # save the outputs
    out_path = os.path.join(args.out_dir, str(args.sequence_id), args.log_file)
    os.makedirs(out_path, exist_ok=True)
    # for i in tqdm.tqdm(range(0, len(data)), total=len(data)):
    i = 0
    objects, objects_all, timestamp_list, caption_image_records = load_SG_data(fps=10)
    #visualize_objects_org(objects)
    
    if args.evaluation_mode:
        print("Running in evaluation mode")
        if args.question_source == "gradio":
            memory, instance_captions, global_starttime, scene_graph = load_memory(
                args,
                None,
                use_milvus=use_milvus,
                ip_address=args.db_ip,
                embedder=[embedder, sbert_model],
                objects=objects,
                objects_all=objects_all,
            )
            print(colored(f"global_starttime: {global_starttime}", "white", attrs=["dark"]))
            print(colored(f"Video caption length: {len(instance_captions)}", "white", attrs=["dark"]))

            while True:
                test_num = i
                question = wait_for_gradio_command(args.instruction_path)
                attach_memory_to_agent(
                    args, agent, memory, scene_graph, global_starttime, sbert_model, test_num, timestamp_list, caption_image_records
                )
                prepare_gradio_logging(args, agent, memory, test_num, question)
                clear_search_state(agent.memory)

                out_dict = answer_squad_question(agent, question, None, scene_graph)
                plot_retrieval_for_gradio(args, agent, gt_info=None, test_num=test_num)

                out_dict['question'] = question
                out_dict['id'] = f"gradio_{test_num}"
                responses.append(out_dict)

                out_json = {
                    "version": 0.1,
                    "responses": responses
                }
                out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
                os.makedirs(out_path, exist_ok=True)
                name = args.run_name+'__'+args.caption_file+f"_{str(args.sequence_id)}"
                with open(os.path.join(out_path, f'{name}.json'), 'w') as f:
                    json.dump(out_json, f, indent=4)

                i += 1

        while i < len(data):
            # if i == 14 and i ==3:
            #     print("Skipping 14 and 3 for now, wrong ground truth")
            #     continue
            qa_instance = data[i]
            question = qa_instance['question']
            print(f"Evaluating {i} out of {len(data)} - {question}")
            if args.manual_evaluation:
                user_input = input("Type 'r' to run, 's' to skip, an index to jump, or 'q' to quit: ").strip().lower()
                
                if user_input in ("", "r", "run"):
                    pass
                elif user_input in ("s", "skip"):
                    responses.append({})
                    i += 1
                    continue
                elif user_input.isdigit():
                    jump_idx = int(user_input)
                    if 0 <= jump_idx < len(data):
                        i = jump_idx
                    else:
                        print(colored(f"Index {jump_idx} is out of range [0, {len(data) - 1}].", "red"))
                    continue
                elif user_input in ("q", "quit", "exit"):
                    break
                else:
                    print(colored(f"Unknown command '{user_input}'. Use r, s, q, or an index.", "yellow"))
                    continue
            test_num = i
            qa_instance = data[i]
            question = qa_instance['question']
            length_category = qa_instance['length_category']
            id = qa_instance['id']

            # if (qa_instance['type'] == 'text'):
            #     print("Skipping text questions for now")
            #     responses.append({}) # this means skipped!
            #     continue

            if (qa_instance['type'] not in args.categories or length_category not in args.lengths): #['binary', 'position', 'time', 'duration'] ['LONG', 'SHORT', 'MEDIUM']
                print(f"Skipping {qa_instance['type']} questions for now")
                responses.append({}) # this means skipped!
                i += 1
                continue

            memory, instance_captions, global_starttime, scene_graph = load_memory(args, data[i], use_milvus=use_milvus, ip_address=args.db_ip, embedder=[embedder, sbert_model], objects=objects, objects_all=objects_all)
            

            print(colored(f"global_starttime: {global_starttime}", "white", attrs=["dark"]))
            if len(instance_captions) == 0: # ISSUEset_scene_graph
                print("Length of Instance Captions is 0. It should not be")
                import pdb; pdb.set_trace()

            # print("HISTORY LENGTH", len(instance_captions))
            print(colored(f"Video caption length: {len(instance_captions)}", "white", attrs=["dark"]))

            # print("instance_captions", instance_captions)

            # model.update_for_instance(captions=instance_captions, ref_time=start_time)

            attach_memory_to_agent(
                args, agent, memory, scene_graph, global_starttime, sbert_model, test_num, timestamp_list, caption_image_records
            )
            prepare_gradio_logging(args, agent, memory, test_num, question)
            clear_search_state(agent.memory)

            out_dict = answer_squad_question(agent, question, qa_instance, scene_graph)
            
            gt_info = extract_gt_times(qa_instance)

            out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
            os.makedirs(out_path, exist_ok=True)
            plot_retrieval_for_gradio(args, agent, gt_info=gt_info, test_num=test_num)
            # plot_multi_method_scores(plot_data, gt_info=gt_info, save_path=os.path.join(args.out_dir, str(args.sequence_id), 'retrieval.png'))
            # plot_scenegraph_scores(search_SG, gt_info=gt_info, save_path=os.path.join(args.out_dir, str(args.sequence_id), 'scenegraph.png'))

            out_dict['question'] = qa_instance['question']
            out_dict['id'] = id

            error_dict = out_dict['error']
            print_eval_result(qa_instance, out_dict, error_dict)

            # keep track of how many of each. usually all CSVs are one type only
            if qa_instance['type'] == 'position':
                num_position += 1
                if 'position_error' in error_dict:
                    running_pos_error += error_dict['position_error']
                    if error_dict['position_error'] < 20.0:
                        sr_position += 1
                    else:
                        print(colored(f"Large Position Error {error_dict['position_error']} m", "red", attrs=["bold"]))
            
            elif qa_instance['type'] == 'binary':
                num_binary += 1
                if 'binary_iscorrect' in error_dict:
                    running_successes += error_dict['binary_iscorrect']
                    if error_dict['binary_iscorrect'] == 1:
                        sr_binary += 1
                    else:
                        print(colored("Wrong Binary Answer", "red", attrs=["bold"]))
            elif qa_instance['type'] == 'time':
                num_time += 1
                if 'time_error' in error_dict:
                    running_time_error += error_dict['time_error']
                    if error_dict['time_error'] < 2.0:
                        sr_time += 1
                    else:
                        print(colored(f"Large Temporal Error {error_dict['time_error']} mins", "red", attrs=["bold"]))
            elif qa_instance['type'] == 'duration':
                num_duration += 1
                if 'duration_error' in error_dict:
                    running_duration_error += error_dict['duration_error']
                    if error_dict['duration_error'] < 2.0:
                        sr_duration += 1
                    else:
                        print(colored(f"Large Duration Error {error_dict['duration_error']} mins", "red", attrs=["bold"]))

            print("Question:", question)
            if 'response' in out_dict:
                print("Response:", out_dict['response'])
            if num_binary > 0:
                print("Binary QA accuracy", running_successes/num_binary)
                print(f"Success Rate: {sr_binary/num_binary}, {sr_binary}/{num_binary}", )
            if num_position > 0:
                print("Position Error", running_pos_error/num_position)
                print(f"Success Rate: {sr_position/num_position}, {sr_position}/{num_position}", )
            if num_time > 0:
                print("Temporal Error", running_time_error/num_time)
                print(f"Success Rate: {sr_time/num_time}, {sr_time}/{num_time}")
            if num_duration > 0:
                print("Duration Error", running_duration_error/num_duration)
                print(f"Success Rate: {sr_duration/num_duration}, {sr_duration}/{num_duration}")

            print()

            responses.append(out_dict)
            i += 1
                
            # save all_questions into json
            out_json = {
                "version": 0.1,
                "responses": responses
            }

            # save the outputs
            out_path = os.path.join(args.out_dir, str(args.sequence_id), args.qa_file)
            os.makedirs(out_path, exist_ok=True)

            name = args.run_name+'__'+args.caption_file+f"_{str(args.sequence_id)}" #_+args.postfix
            with open(os.path.join(out_path, f'{name}.json'), 'w') as f:
                json.dump(out_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='Long Horizon Robot QA',
                        description='Runs various LLMs on the QA dataset',)
    
    
    parser.add_argument("--sequence_id", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8) 
    parser.add_argument("--all_mem", type=str2bool, default=False, help="Whether to use the full memory. Default is True.")
    parser.add_argument("--evaluation_mode", type=str2bool, default=True, help="Whether to run in evaluation mode. Default is False.")
    parser.add_argument("--manual_evaluation", type=str2bool, default=True, help="Whether to run in manual evaluation mode. Default is True.")
    parser.add_argument("--question_source", type=str, default="dataset", choices=["dataset", "gradio"], help="Use dataset questions or wait for questions submitted by run_gradio_interface.py.")
    parser.add_argument("--caption_file", type=str, default="captions_NVILA-Lite-2B") #captions_NVILA-8B
    parser.add_argument("--postfix", type=str, default="star", help="Display/output channel used by Gradio folders, e.g. CoDa or Isaacsim.")
    parser.add_argument("--method", type=str, default="star", choices=["star", "remembr", "scene_graph"], help="Evaluation method. Use star for our method, remembr for the vanilla ReMEmbR baseline, and scene_graph for the scene-graph baseline.")
    # ChatGPT models: gpt-4o, gpt-4o-mini, gpt-4.1-mini
    parser.add_argument("--llm", type=str, default="gpt-4.1", help="LLM backend used by the selected method.")
    parser.add_argument("--categories", nargs="+", default=["position","duration","time","binary","text"], choices=["position","duration","time","binary","text"])
    parser.add_argument("--lengths", nargs="+", default=["LONG","MEDIUM","SHORT"], choices=["LONG","MEDIUM","SHORT"])

    parser.add_argument("--qa_file", type=str, default="human_qa")
    parser.add_argument("--log_file", type=str, default="test_log")
    parser.add_argument("--VDB", type=str, default="search_DB")
    parser.add_argument("--SG", type=str, default="search_SG") 
    parser.add_argument("--coda_dir", type=str, default="/workspace/Local_data/CODa")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/coda")
    parser.add_argument("--scenegraph_file", type=str, default="full_pcd") 
    parser.add_argument("--root", type=str, default="/workspace/") 
    parser.add_argument("--results", type=str, default="/workspace/results") # sg 10
    parser.add_argument("--out_dir", type=str, default="/workspace/results/out") #out_time  out_descriptive_text  out_descriptive  out_spatial
    
    
    parser.add_argument("--instruction_path", type=str, default=None, help="Path to Gradio instruction.json. Defaults to results/sequence_id/instructions/instruction.json.")
    parser.add_argument("--latest_idx_path", type=str, default=None, help="Path to Gradio latest_frame.txt. Defaults to results/sequence_id/latest_frame/postfix/latest_frame.txt.")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to show debug info. Default is True.")
    parser.add_argument("--source_fps", type=float, default=10.0, help="Original RGB/LiDAR source frame rate.")
    parser.add_argument("--annotation_stride", type=float, default=10.0, help="Frame stride used when saving annotated RGB images.")
    parser.add_argument("--annotation_start_index", type=int, default=0, help="Starting index used in annotated_rgb filenames.")
    # llm-specific args
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_ctx", type=int, default=8192*4)

    # memory / retrieval specific args
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--db_name", type=str, default='test')
    parser.add_argument("--db_ip", type=str, default='127.0.0.1')

    args = parser.parse_args()
    if args.instruction_path is None:
        args.instruction_path = os.path.join(
            args.results, str(args.sequence_id), "instructions", "instruction.json"
        )
    if args.latest_idx_path is None:
        args.latest_idx_path = os.path.join(
            args.results,
            str(args.sequence_id),
            "latest_frame",
            args.postfix,
            "latest_frame.txt",
        )
    if args.source_fps <= 0.0:
        parser.error("--source_fps must be positive")
    if args.annotation_stride <= 0.0:
        parser.error("--annotation_stride must be positive")
    args.annotation_fps = args.source_fps / args.annotation_stride
    print(
        "Annotation cadence: "
        f"source_fps={args.source_fps:.3f}, "
        f"stride={args.annotation_stride:.3f}, "
        f"annotation_fps={args.annotation_fps:.3f}, "
        f"start_index={args.annotation_start_index}"
    )
    main(args)
