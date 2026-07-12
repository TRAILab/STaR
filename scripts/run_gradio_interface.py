import sys
import glob
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).resolve().parent))

import gradio as gr
import hydra
import os
import json
from functools import partial
from PIL import Image, ImageDraw, ImageFile, UnidentifiedImageError

def retrieve_fallback_image(fallback_image_path):
    if os.path.exists(fallback_image_path):
        print(f"cfg['fallback_image_path']: {fallback_image_path}")
        img = Image.open(fallback_image_path).convert("RGB")
        img.load()              # 🔥 force full read
        return img
    else:
        return get_null_image()


def get_null_image():
    # Dynamically create a fallback image with text
    img = Image.new("RGB", (256, 256), color="gray")
    draw = ImageDraw.Draw(img)
    draw.text((70, 120), "No Data", fill="white")
    return img


def init_folders(cfg):
    folder_list = [
        cfg["rgb_folder"],
        cfg["retrieve_annotated_folder"],
        cfg["retrieve_folder"],
        cfg["cot_log_folder"],
    ]

    for folder in folder_list:
        os.makedirs(folder, exist_ok=True)          
        delete_files(folder)          

    os.makedirs(os.path.dirname(cfg["instruction_path"]), exist_ok=True)
    with open(cfg["instruction_path"], "w") as f:
        json.dump({"trigger": False, "command": ""}, f)

    os.makedirs(os.path.dirname(cfg["latest_idx_path"]), exist_ok=True)
    with open(cfg["latest_idx_path"], "w") as f:
        f.write("-1")

    # Create Updated

def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def trigger_update_flag(update_flag_path):
    os.makedirs(os.path.dirname(update_flag_path), exist_ok=True)
    with open(update_flag_path, "w") as f:
        print("Update triggered!")
        json.dump({"trigger": True}, f)
    return "🔄 Update Triggered"


def load_retrieve_annotated_image(image_list, idx, fallback_image_path):
    try:
        image_path = image_list[idx]

        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            raise ValueError("Image file missing or empty")

        img = Image.open(image_path)
        img.load()
        return img

    except Exception as e:
        print(f"[WARN] Failed to load image: {e}")
        return Image.open(fallback_image_path)


def load_retrieved_image_pair(image_list, page, fallback_image_path):
    """Load two retrieved images for a zero-based carousel page."""
    total_pages = max(1, (len(image_list) + 1) // 2)
    page = max(0, min(int(page), total_pages - 1))
    first_image_idx = page * 2

    first_image = load_retrieve_annotated_image(
        image_list, first_image_idx, fallback_image_path
    )
    second_image = load_retrieve_annotated_image(
        image_list, first_image_idx + 1, fallback_image_path
    )
    page_label = f"Retrieved images: {page + 1} / {total_pages}"
    return first_image, second_image, page_label, page


def navigate_retrieved_images(
    direction,
    carousel_state,
    latest_idx_path,
    retrieve_annotated_folder,
    fallback_image_path,
):
    """Move the retrieved-image carousel one page while preserving its query index."""
    idx = get_latest_frame_idx(latest_idx_path)
    state = carousel_state or {"idx": idx, "page": 0}
    page = 0 if state.get("idx") != idx else int(state.get("page", 0))

    image_list = glob.glob(f"{retrieve_annotated_folder}/{idx}/*.png")
    image_list.sort(key=os.path.getmtime)
    page += direction

    first, second, label, page = load_retrieved_image_pair(
        image_list, page, fallback_image_path
    )
    return first, second, label, {"idx": idx, "page": page}
    

def get_latest_frame_idx(latest_idx_path):
    try:
        print("Reading latest idx from:", latest_idx_path)
        with open(latest_idx_path, "r") as f:
            return int(f.read().strip())
    except:
        return -1

# === Function to get latest image ===
def get_latest_som_image(idx, rgb_folder, fallback_image_path):
    image_path = os.path.join(rgb_folder, "captured_rgb_0.png")
    if os.path.exists(image_path):
        return Image.open(image_path)
        # img = Image.open(image_path)
        # img = img.load()
        # return img
    return retrieve_fallback_image(fallback_image_path)

    # img = Image.new("RGB", (256, 256), color="gray")
    # draw = ImageDraw.Draw(img)
    # draw.text((70, 120), "No Data", fill="white")
    # return img

# === Function to get retrieve image ===
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_retrieve_image(idx, retrieve_folder, postfix, fallback_image_path):
    image_path = os.path.join(
        retrieve_folder,
        f"retrieval_DB_{idx}_{postfix}.png"
    )

    try:
        # 1. File existence + size check
        if not os.path.exists(image_path):
            raise FileNotFoundError("retrieve image missing")

        if os.path.getsize(image_path) < 100:  # bytes
            raise ValueError("retrieve image too small (likely incomplete)")

        # 2. Try loading
        img = Image.open(image_path).convert("RGB")
        img.load()
        return img

    except (FileNotFoundError, ValueError, UnidentifiedImageError, OSError) as e:
        print(f"[WARN] Failed to load retrieve image ({image_path}): {e}")

        fallback = Image.open(fallback_image_path).convert("RGB")
        fallback.load()
        return fallback

# === Submit instruction from user ===
def submit_command(command, instruction_path):
    os.makedirs(os.path.dirname(instruction_path), exist_ok=True)
    with open(instruction_path, "w") as f:
        print(f"Submitting command: {command}")
        json.dump({"trigger": True, "command": command}, f)

    # if event:
    #     event.set()

    return "✅ Instruction Sent" #: {command}"

# === Main updater function (image + labels) ===
# def update_interface():
#     idx = get_latest_frame_idx()
#     image = get_latest_som_image(idx)
#     label_text = get_label_text(idx)
#     return image, label_text,

def update_interface(
    carousel_state,
    latest_idx_path, 
    retrieve_folder,
    postfix,
    fallback_image_path,
    cot_log_folder,
    retrieve_annotated_folder,
    rgb_folder
):
    idx = get_latest_frame_idx(latest_idx_path)
    image = get_latest_som_image(0, rgb_folder, fallback_image_path)
    # label_text = get_label_text(idx)

    retrieve_image = get_retrieve_image(idx, retrieve_folder, postfix, fallback_image_path)
    cot_messages = load_cot_chat(idx, cot_log_folder)  # this returns list of [role, content]

    # semantic_map = load_map_image(SEMANTIC_FOLDER, idx, "sem")
    # occupancy_map = load_map_image(OCCUPANCY_FOLDER, idx, "occ")
    # cost_map = load_map_image(COST_FOLDER, idx, "cost")
    image_list = glob.glob(f"{retrieve_annotated_folder}/{idx}/*.png")
    image_list.sort(key=os.path.getmtime)

    state = carousel_state or {"idx": idx, "page": 0}
    page = 0 if state.get("idx") != idx else int(state.get("page", 0))
    retrieve_1, retrieve_2, page_label, page = load_retrieved_image_pair(
        image_list, page, fallback_image_path
    )

    print("Latest idx:", idx)
    return (
        image,
        retrieve_image,
        cot_messages,
        retrieve_1,
        retrieve_2,
        page_label,
        {"idx": idx, "page": page},
    )

def load_cot_chat(idx, cot_log_folder):
    messages = []
    cot_path = os.path.join(cot_log_folder, f"cot_log_{idx}.txt")
    print("Loading CoT from:", cot_path)

    role = None
    buffer = []

    try:
        with open(cot_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Skip label-related lines
                if line.startswith("label-") or "3D bbox" in line or "extent" in line:
                    continue

                if line.startswith("User:"):
                    if role and buffer:
                        content = "\n".join(buffer).strip()
                        if role == "user":
                            messages.append([content, None])
                        else:
                            messages.append([None, content])
                        buffer = []
                    role = "user"
                    buffer.append(line[len("User:"):].strip())
                elif line.startswith("Assistant:") or line.startswith("System:"):
                    if role and buffer:
                        content = "\n".join(buffer).strip()
                        if role == "user":
                            messages.append([content, None])
                        else:
                            messages.append([None, content])
                        buffer = []
                    role = "assistant"
                    buffer.append(line.split(":", 1)[1].strip())
                else:
                    buffer.append(line)

            # Add last block
            if buffer: # and role:
                content = "\n".join(buffer).strip()
                if role == "user":
                    messages.append([content, None])
                else:
                    messages.append([None, content])

    except Exception:
        messages.append([None, "⌛ Waiting for CoT reasoning..."])

    if not messages:
        messages.append([None, "⌛ Waiting for CoT reasoning..."])

    return messages


# === Gradio Interface ===
DEMO_CSS = """
.section-title {
    margin: 0.25rem 0 0.1rem 0;
}
.carousel-nav {
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.carousel-nav button {
    max-width: 140px;
}
.carousel-page {
    min-width: 190px;
    min-height: 40px;
    text-align: center;
}
.carousel-page textarea {
    min-height: 40px !important;
    height: 40px !important;
    text-align: center;
    resize: none;
}
.retrieved-image {
    min-width: 0;
}
.retrieved-image img {
    object-fit: contain !important;
}
"""


def build_demo(cfg):
    with gr.Blocks(title="STaR", css=DEMO_CSS) as demo:
        carousel_state = gr.State({"idx": -1, "page": 0})
        gr.Markdown("## 🧠 STaR: Live Demo")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row():
                    command_box = gr.Textbox(
                        label="Enter Command",
                        lines=1,
                        placeholder="e.g., Find me a yellow hand truck",
                        show_label=True,
                        scale=8
                    )
                    submit_btn = gr.Button("Submit", min_width=10)
                    update_btn = gr.Button("Update", min_width=10)
                status_box = gr.Textbox(
                    label="Status",
                    lines=1,
                    max_lines=1,
                    interactive=False,
                    show_label=False,
                    container=False,
                )
                som_img = gr.Image(label="Captured RGB Image", height=180)
                # label_textbox = gr.Textbox(label="Object Labels", lines=11, max_lines=11, interactive=False)
            with gr.Column(scale=1):
            # 🟩 Status ABOVE CoT
                cot_chat = gr.Chatbot(label="🧠 Live Chain-of-Thought Reasoning", height=290)
        gr.Markdown("### Retrieval Overview", elem_classes=["section-title"])
        with gr.Row():
            retrieve_img = gr.Image(label="Retrieved Video Captions", height=140)
        gr.Markdown("### Retrieved Visual Evidence", elem_classes=["section-title"])
        with gr.Row(elem_classes=["carousel-nav"]):
            previous_btn = gr.Button("◀ Previous", min_width=120, scale=0)
            carousel_page_label = gr.Textbox(
                value="Retrieved images: 1 / 1",
                lines=1,
                max_lines=1,
                interactive=False,
                show_label=False,
                container=False,
                elem_classes=["carousel-page"],
            )
            next_btn = gr.Button("Next ▶", min_width=120, scale=0)
        with gr.Row(equal_height=True):
            retrieve_1 = gr.Image(
                label="Retrieved Image 1",
                height=160,
                scale=1,
                elem_classes=["retrieved-image"],
            )
            retrieve_2 = gr.Image(
                label="Retrieved Image 2",
                height=160,
                scale=1,
                elem_classes=["retrieved-image"],
            )

        submit_btn.click(fn=partial(submit_command, instruction_path=cfg["instruction_path"]), inputs=[command_box], outputs=status_box, api_name=False)
        update_btn.click(fn=partial(trigger_update_flag, update_flag_path=cfg['update_flag_path']), inputs=[], outputs=status_box, api_name=False) #
        update_fn = partial(
            update_interface,
            latest_idx_path=cfg['latest_idx_path'],
            postfix=cfg['postfix'],
            fallback_image_path=cfg['fallback_image_path'],
            cot_log_folder=cfg['cot_log_folder'],
            retrieve_folder=cfg['retrieve_folder'],
            retrieve_annotated_folder=cfg['retrieve_annotated_folder'],
            rgb_folder=cfg['rgb_folder']
        )
        carousel_outputs = [
            retrieve_1,
            retrieve_2,
            carousel_page_label,
            carousel_state,
        ]
        previous_btn.click(
            fn=partial(
                navigate_retrieved_images,
                -1,
                latest_idx_path=cfg['latest_idx_path'],
                retrieve_annotated_folder=cfg['retrieve_annotated_folder'],
                fallback_image_path=cfg['fallback_image_path'],
            ),
            inputs=[carousel_state],
            outputs=carousel_outputs,
            api_name=False,
        )
        next_btn.click(
            fn=partial(
                navigate_retrieved_images,
                1,
                latest_idx_path=cfg['latest_idx_path'],
                retrieve_annotated_folder=cfg['retrieve_annotated_folder'],
                fallback_image_path=cfg['fallback_image_path'],
            ),
            inputs=[carousel_state],
            outputs=carousel_outputs,
            api_name=False,
        )
        update_outputs = [
            som_img,
            retrieve_img,
            cot_chat,
            retrieve_1,
            retrieve_2,
            carousel_page_label,
            carousel_state,
        ]
        demo.load(
            fn=update_fn,
            inputs=[carousel_state],
            outputs=update_outputs,
            every=0.5,
            api_name=False
        )
    
    return demo

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    inference_cfg = cfg.inference

    demo = build_demo(inference_cfg)
    init_folders(inference_cfg)
    print("Initialized folders and cleared old data.")
    demo.queue()
    demo.launch(share=True)


if __name__ == "__main__":
    main()
