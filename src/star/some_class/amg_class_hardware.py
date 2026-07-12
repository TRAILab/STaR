"""
2024.01.16 
MyAutomaticMaskGenerator class for generating image masks and captions.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Optional, Any
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale
from tokenize_anything.utils.image import im_vstack
import time
import torchvision
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors

import argparse
import ast
from transformers import SamModel, SamProcessor
from dam import DescribeAnythingModel, disable_torch_init
import cv2


sys.path.append("/home/mfyuan/local_folder/GroundingDINO")

try:
    from Tag2Text.models import tag2text
    from recognize_anything import inference_tag2text, inference_ram
    import torchvision.transforms as TS
except ImportError as e:
    print("Tag2text sub-package not found. Please check your PATH. ")
    raise e

try: 
    from groundingdino.util.inference import Model
except ImportError as e:
    print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
    raise e

class MyAutomaticMaskGenerator:
    
    # initialize，pass input parameters
    def __init__(self, tagging_model, grounding_dino_model, tap_model, sbert_model, dam_model, sam_model, sam_processor):
        self.tagging_model = tagging_model
        self.grounding_dino_model = grounding_dino_model
        self.tap_model = tap_model
        self.sbert_model = sbert_model
        self.dam_model = dam_model
        self.sam_model = sam_model
        self.sam_processor = sam_processor
        # Tag2Texttransforms used by
        self.tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        
        self.specified_tags='None'
        self.classes = None
        # additional and removed classes
        self.add_classes = ["other item","pavement","grass","house","bicycle","motorcycle","person","parking",
                            "fence","sidewalk","tree","vegetation","sign","building","bush","rail","pole"]
        self.remove_classes = [
            "room", "kitchen", "office", "home", "corner",
            "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
            "image", "city", "blue", "skylight", "hallway", 
            "modern", "salon", "doorway", "wall lamp","floor"
        ]

    def apply_sam(self, image, input_points=None, input_boxes=None, input_labels=None):
        inputs = self.sam_processor(image, input_points=input_points, input_boxes=input_boxes,
                            input_labels=input_labels, return_tensors="pt").to('cuda')

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0]
        scores = outputs.iou_scores[0, 0]

        mask_selection_index = scores.argmax()

        mask_np = masks[mask_selection_index].numpy()

        return mask_np


    def add_contour(img, mask, input_points=None, input_boxes=None):
        img = img.copy()

        # Draw contour
        mask = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

        # Draw points if provided
        if input_points is not None:
            for points in input_points:  # Handle batch of points
                for x, y in points:
                    # Draw a filled circle for each point
                    cv2.circle(img, (int(x), int(y)), radius=10,
                            color=(1.0, 0.0, 0.0), thickness=-1)
                    # Draw a white border around the circle
                    cv2.circle(img, (int(x), int(y)), radius=10,
                            color=(1.0, 1.0, 1.0), thickness=2)

        # Draw boxes if provided
        if input_boxes is not None:
            for box_batch in input_boxes:  # Handle batch of boxes
                for box in box_batch:  # Iterate through boxes in the batch
                    x1, y1, x2, y2 = map(int, box)
                    # Draw rectangle with white color
                    cv2.rectangle(img, (x1, y1), (x2, y2),
                                color=(1.0, 1.0, 1.0), thickness=4)
                    # Draw inner rectangle with red color
                    cv2.rectangle(img, (x1, y1), (x2, y2),
                                color=(1.0, 0.0, 0.0), thickness=2)

        return img


    def denormalize_coordinates(coords, image_size, is_box=False):
        """Convert normalized coordinates (0-1) to pixel coordinates."""
        width, height = image_size
        if is_box:
            # For boxes: [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            return [
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            ]
        else:
            # For points: [x, y]
            x, y = coords
            return [int(x * width), int(y * height)]


    def print_streaming(text):
        """Helper function to print streaming text with flush"""
        print(text, end="", flush=True)

    @torch.no_grad()
    def generate_DAM(self, image: np.ndarray, cfg, save_path: str = None, save_vis: bool = True) -> List[Dict[str, Any]]:
        """
        Generate masks, concepts, and captions from an image using Tag2Text, Grounding DINO, and TAP models.
        Returns a list of dictionaries with segmentation results and an annotated image.
        """
        start_time = time.time()

        # Step 1: Generate tags using Tag2Text
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        raw_image_tensor = self.tagging_transform(raw_image).unsqueeze(0).to("cuda")
        res = inference_tag2text.inference(raw_image_tensor, self.tagging_model, self.specified_tags)
        caption = res[2]
        text_prompt = res[0].replace(' |', ',')
        classes = self.process_tag_classes(text_prompt, add_classes=self.add_classes, remove_classes=self.remove_classes)
        end_time1 = time.time()

        # Step 2: Generate bounding boxes using Grounding DINO
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=0.25,
            text_threshold=0.25,
        )

        # Apply Non-Maximum Suppression (NMS)
        if len(detections.class_id) > 0:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                0.5
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            valid_idx = detections.class_id != -1
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]
        end_time2 = time.time()
        if cfg.use_dam:
            masks, captions, caption_fts, concepts = [], [], [], []
            for box in detections.xyxy:
                x1, y1, x2, y2 = map(int, box)
                # DAM caption generation
                cropped_img = image_pil.crop((x1, y1, x2, y2))
                mask_np = self.apply_sam(image_pil, input_boxes=[[[x1, y1, x2, y2]]])
                mask = mask_np.astype(bool)

                caption = self.dam_model.get_description(
                    image_pil, Image.fromarray(mask_np.astype(np.uint8) * 255),
                    "<image>\nDescribe the masked region in detail.",
                    streaming=False, temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=512
                )

                concept = ""  # Placeholder for concepts if not generated
                caption_ft = self.sbert_model.encode(caption, convert_to_tensor=True, device="cuda")
                caption_ft = caption_ft / caption_ft.norm(dim=-1, keepdim=True)
                masks.append(mask)
                captions.append(caption)
                caption_fts.append(caption_ft.cpu())
                concepts.append(concept)
        else:
            # # Step 3: Generate masks and captions using TAP
            vis_img = image.copy()[:, :, ::-1]
            img_list, img_scales = im_rescale(image, scales=[1024], max_size=1024)
            input_size, original_size = img_list[0].shape, image.shape[:2]
            img_batch = im_vstack(img_list, fill_value=self.tap_model.pixel_mean_value, size=(1024, 1024))
            inputs = self.tap_model.get_inputs({"img": img_batch})
            inputs.update(self.tap_model.get_features(inputs))

            # Convert bounding boxes to batch_points format for TAP input
            batch_points = np.zeros((len(detections.xyxy), 2, 3), dtype=np.float32)
            for i in range(len(detections.xyxy)):
                batch_points[i, 0, :2] = detections.xyxy[i, :2]
                batch_points[i, 1, :2] = detections.xyxy[i, 2:]
                batch_points[i, 0, 2] = 2
                batch_points[i, 1, 2] = 3
            inputs["points"] = batch_points
            inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")

            # Run TAP model
            outputs = self.tap_model.get_outputs(inputs)
            iou_pred = outputs["iou_pred"].cpu().numpy()
            point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
            rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
            mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
            mask_pred = outputs["mask_pred"]
            masks = mask_pred[mask_index]
            masks = self.tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
            masks = masks[..., :input_size[0], :input_size[1]]
            masks = self.tap_model.upscale_masks(masks, original_size).gt(0).cpu().numpy()

            # Generate concepts and captions
            concepts, scores = self.tap_model.predict_concept(outputs["sem_embeds"][mask_index])
            sem_tokens = outputs["sem_tokens"][mask_index].squeeze(2)
            captions = self.tap_model.generate_text(sem_tokens)
            caption_fts = self.sbert_model.encode(captions, convert_to_tensor=True, device="cuda")
            caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
            end_time3 = time.time()

            print(f"In total: {end_time3 - start_time:.2f} secs, Tag2Text: {end_time1 - start_time:.2f} secs, DINO: {end_time2 - end_time1:.2f} secs, TAP: {end_time3 - end_time2:.2f} secs")

        # Generate visual output with contours
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data_plot = {}
        data_plot['image'] = image
        data_plot['masks'] = masks
        data_plot['detections'] = detections

        # Optional: Save visualization result
        if save_vis:
            plt.figure(figsize=(20, 8))
            plt.imshow(image)
            self.show_masks(masks, concepts, captions, plt.gca(), detections)
            plt.axis('off')
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()

        # Format output results
        results = []
        for i, (mask, concept, caption, caption_ft) in enumerate(zip(masks, concepts, captions, caption_fts)):
            results.append({
                "mask": mask,
                "concepts": concept,
                "caption": caption,
                "caption_ft": caption_ft.cpu(),
                "img_bbox": detections.xyxy[i],
            })

        return results, data_plot #annotated_image

    @torch.no_grad()
    def generate_OpenGraph(self, image: np.ndarray, save_path: str = None, save_vis: bool = True) -> List[Dict[str, Any]]:
        start_time = time.time()
        #####################################################
        # ############# 1. use tag2text generate tag ############
        #####################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 1.1 image preprocessing
        image_pil = Image.fromarray(image)
        raw_image = image_pil.resize((384, 384))
        raw_image_tensor = self.tagging_transform(raw_image).unsqueeze(0).to("cuda")
        # 1.2 run model inference on image
        res = inference_tag2text.inference(raw_image_tensor , self.tagging_model, self.specified_tags)
        # 1.3 get results and set required classes
        caption=res[2]
        
        text_prompt=res[0].replace(' |', ',')
        classes = self.process_tag_classes(
            text_prompt, 
            add_classes = self.add_classes,
            remove_classes = self.remove_classes,
        )
        end_time1 = time.time()
        execution_time = end_time1 - start_time
 
        
        #####################################################
        # ############# 2. usedinogenerate boxes for labels ############
        #####################################################
        # 2.1 model inference
        detections = self.grounding_dino_model.predict_with_classes(
            image=image, # This function expects a BGR image...
            classes=classes,
            box_threshold=0.25,
            text_threshold=0.25,
        )
        # 2.2 Remove a portion based on Non-Maximum Suppression (NMS) and class -1.
        if len(detections.class_id) > 0:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy), 
                torch.from_numpy(detections.confidence), 
                0.5
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            # remove class -1
            valid_idx = detections.class_id != -1
            detections.xyxy = detections.xyxy[valid_idx]
            detections.confidence = detections.confidence[valid_idx]
            detections.class_id = detections.class_id[valid_idx]
        end_time2 = time.time()
        execution_time = end_time2 - start_time

        
        #####################################################
        # ######### 3. usetapgenerate masks and captions for boxes ########
        #####################################################
        # 3.1 image preprocessing
        vis_img = image.copy()[:, :, ::-1]
        img_list, img_scales = im_rescale(image, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, image.shape[:2]
        img_batch = im_vstack(img_list, fill_value=self.tap_model.pixel_mean_value, size=(1024, 1024))
        inputs = self.tap_model.get_inputs({"img": img_batch})
        inputs.update(self.tap_model.get_features(inputs))
        # 3.2 convert masks above to required format
        batch_points = np.zeros((len(detections.xyxy), 2, 3), dtype=np.float32)
        for i in range(len(detections.xyxy)):
            batch_points[i, 0, :2] = detections.xyxy[i, :2]
            batch_points[i, 1, :2] = detections.xyxy[i, 2:]
            batch_points[i, 0, 2] = 2
            batch_points[i, 1, 2] = 3
        inputs["points"] = batch_points
        inputs["points"][:, :, :2] *= np.array(img_scales, dtype="float32")
        # 3.3 run model to get mask sizes
        outputs = self.tap_model.get_outputs(inputs)
        iou_pred = outputs["iou_pred"].cpu().numpy()
        point_score = batch_points[:, 0, 2].__eq__(2).__sub__(0.5)[:, None]
        rank_scores = iou_pred + point_score * ([1000] + [0] * (iou_pred.shape[1] - 1))
        mask_index = np.arange(rank_scores.shape[0]), rank_scores.argmax(1)
        mask_pred = outputs["mask_pred"]
        masks = mask_pred[mask_index]
        masks = self.tap_model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = self.tap_model.upscale_masks(masks, original_size).gt(0).cpu().numpy()
        # 3.4 run model to get concepts/captions
        # infer concepts
        concepts, scores = self.tap_model.predict_concept(outputs["sem_embeds"][mask_index])
        concepts, scores = [x for x in (concepts, scores)]
        # infer captions
        sem_tokens = outputs["sem_tokens"][mask_index].squeeze(2)
        # Dimensions: [1, 1, 20, 1, 256] For some reason, the model generates a 5D tensor????

        print(f"sem_tokens shape: {sem_tokens.shape}")
        captions = self.tap_model.generate_text(sem_tokens)
        caption_fts = self.sbert_model.encode(captions, convert_to_tensor=True, device="cuda")
        caption_fts = caption_fts / caption_fts.norm(dim=-1, keepdim=True)
        
        end_time3 = time.time()
        execution_time = end_time3 - start_time
        print(f"In total: {execution_time} secs, tag2text: {end_time1-start_time} secs, dino: {end_time2-end_time1} secs, tap: {end_time3-end_time2} secs")
        data_plot = {}
        data_plot['image'] = image
        data_plot['masks'] = masks
        data_plot['detections'] = detections
        # save
        # if save_path:
        # # create directory if it does not exist
        #     save_path.parent.mkdir(parents=True, exist_ok=True)
        #     plt.imsave(save_path, anotatated_image)
        
        #####################################################
        # ######### 4. final visualization and saving results ########
        #####################################################
        if save_vis:
            plt.figure(figsize=(20,8))
            plt.imshow(image)
            #__VLT__modify
            self.show_masks(masks, concepts, captions, plt.gca(), detections)
            plt.axis('off')
            # save image if output path is provided
            if save_path:
                # create directory if it does not exist
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # get bbox of non-zero pixels in image
                '''non_zero_pixels = cv2.findNonZero(cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY))
                x, y, w, h = cv2.boundingRect(non_zero_pixels)
                # crop image
                cropped_img = vis_img[y:y+h, x:x+w]'''
                # save cropped image
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()  # close figure to avoid notebook display
                
        # in loop, get bbox coords from detections for each mask and save source image
        # for i, (mask, concept, caption, caption_ft) in enumerate(zip(masks, concepts, captions, caption_fts)):
        # # find coordinates where mask is True
        #     true_coords = np.argwhere(mask)
        #     if len(true_coords) > 0:
        # # get corresponding bbox coordinates from detections
        #         box = detections.xyxy[i]
        #         x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # # crop region from source image using bbox coordinates
        #         cropped_image = image_rgb.copy()
        # # fill regions outside mask with white
        #         mask = mask[0]
        # cropped_image[~mask] = [255, 255, 255] # white RGB value
        #         cropped_image = cropped_image[y_min:y_max, x_min:x_max]
        # # save cropped image to disk
        #         mask_image_path = save_path.parent / f"mask_{i+1}_image.jpg"
        #         cv2.imwrite(str(mask_image_path), cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        #         print(caption)
        #         print(f"Mask {i+1} image saved at: {mask_image_path}")

        # return results
        result = []
        for i, (mask, concept, caption, caption_ft) in enumerate(zip(masks, concepts, captions, caption_fts)):
            
            result.append({
                "mask": mask,
                "concepts": concept,
                "caption": caption,
                "caption_ft": caption_ft.cpu(),
                "img_bbox": detections.xyxy[i],
            })
        return result, data_plot#anotatated_image
    

    def process_tag_classes(self, text_prompt:str, add_classes:List[str]=[], remove_classes:List[str]=[]) -> list[str]:
        '''
        Convert a Tag2Text text prompt into a class list for use by DINO.
        '''
        classes = text_prompt.split(',')
        classes = [obj_class.strip() for obj_class in classes]
        classes = [obj_class for obj_class in classes if obj_class != '']
        for c in add_classes:
            if c not in classes:
                classes.append(c)
        for c in remove_classes:
            classes = [obj_class for obj_class in classes if c not in obj_class.lower()]
        return classes
    
    
    def show_masks(self, masks, concepts, captions, ax, detections):
        '''
        Render masks for image visualization.
        '''
        for i, (mask, concept, caption) in enumerate(zip(masks, concepts, captions)):
            # find coordinates where mask is True
            true_coords = np.argwhere(mask)
            if len(true_coords) > 0:
                # show mask
                color = np.concatenate([np.random.random(3), np.array([1])], axis=0)  # adjust color alpha
                ax.imshow(mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1), alpha=0.9, label=f'Mask {i+1}')
                # show box
                box = detections.xyxy[i]
                rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                # show text
                center_x = (box[0]+box[2])/2
                center_y = (box[1]+box[3])/2
                caption_width = len(caption) * 5  # adjust based on font size
                caption_x = center_x - caption_width / 2
                caption_y = center_y
                # show caption
                ax.text(caption_x, caption_y, f"{concept}:{caption}", color='black', fontsize=8, bbox=dict(facecolor=color, alpha=1.0))

    def show_labeled_masks(self, masks, detections, ax, image_width, image_height):
        """
        Display bounding boxes with numbered labels, dynamically repositioned to avoid overlap
        and placed outside for small bounding boxes if needed. Labels match the color of their respective bounding boxes.

        Args:
            masks (list of np.ndarray): Binary masks for each object.
            detections (list or np.ndarray): Bounding boxes with coordinates [x_min, y_min, x_max, y_max].
            ax (matplotlib.axes.Axes): Matplotlib axes for visualization.
            image_width (int): Width of the image.
            image_height (int): Height of the image.
        """
        # Predefined distinct colors for labels and bounding boxes
        distinct_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00',
            '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80',
            '#A52A2A', '#5F9EA0', '#D2691E', '#9ACD32', '#DA70D6', '#7FFFD4',
            '#FF4500', '#2E8B57'
        ]

        # Track existing label positions to avoid overlap
        label_positions = []
        offset_step = 10  # Step size to prevent overlap

        # Corner options for small and large bounding boxes
        def clamp(value, min_value, max_value):
            return max(min_value, min(value, max_value))

        def safe_corners(box, is_small):
            """ Generate safe label positions for bounding box corners within image bounds. """
            x_min, y_min, x_max, y_max = box
            if is_small:
                # Label positions outside the bounding box
                return {
                    "outside-top-left": (clamp(x_min - offset_step, 0, image_width), clamp(y_min - offset_step, 0, image_height)),
                    "outside-top-right": (clamp(x_max + offset_step, 0, image_width), clamp(y_min - offset_step, 0, image_height)),
                    "outside-bottom-left": (clamp(x_min - offset_step, 0, image_width), clamp(y_max + offset_step, 0, image_height)),
                    "outside-bottom-right": (clamp(x_max + offset_step, 0, image_width), clamp(y_max + offset_step, 0, image_height)),
                }
            else:
                # Label positions inside the bounding box corners
                return {
                    "top-left": (clamp(x_min, 0, image_width), clamp(y_min, 0, image_height)),
                    "top-right": (clamp(x_max, 0, image_width), clamp(y_min, 0, image_height)),
                    "bottom-left": (clamp(x_min, 0, image_width), clamp(y_max, 0, image_height)),
                    "bottom-right": (clamp(x_max, 0, image_width), clamp(y_max, 0, image_height)),
                }

        # Iterate through masks and detections
        for i, mask in enumerate(masks):
            color = distinct_colors[i % len(distinct_colors)]  # Cycle through distinct colors

            # Get the bounding box
            box = detections.xyxy[i]
            bbox_width = box[2] - box[0]
            bbox_height = box[3] - box[1]
            is_small = bbox_width < 30 or bbox_height < 30

            # Draw the bounding box
            rect = plt.Rectangle(
                (box[0], box[1]),
                bbox_width,
                bbox_height,
                linewidth=1.0,
                edgecolor=color,
                facecolor='none',
            )
            ax.add_patch(rect)

            # Determine safe label positions
            corners = safe_corners(box, is_small)
            corner_order = list(corners.keys())
            label_position = None

            # Check for overlap and adjust label position
            for corner in corner_order:
                pos_x, pos_y = corners[corner]
                overlap = any(
                    abs(pos_x - prev_x) < offset_step and abs(pos_y - prev_y) < offset_step
                    for prev_x, prev_y in label_positions
                )
                if not overlap:
                    label_position = (pos_x, pos_y)
                    break

            # If no non-overlapping position is found, use the last corner
            if label_position is None:
                label_position = (corners[corner_order[-1]])

            # Store the new label position
            label_positions.append(label_position)

            # Adjust color for overlapping small bounding boxes
            for j, other_box in enumerate(detections.xyxy):
                if i != j and is_small and (
                    box[0] < other_box[2] and box[2] > other_box[0] and  # x-axis overlap
                    box[1] < other_box[3] and box[3] > other_box[1]      # y-axis overlap
                ):
                    overlapping_color = distinct_colors[(i + j) % len(distinct_colors)]
                    rect.set_edgecolor(overlapping_color)
                    color = overlapping_color  # Update label color to match overlapping box

            # Draw the label at the chosen position with matching color
            ax.text(
                label_position[0],
                label_position[1],
                str(i + 1),  # Label number
                color='black',  # Text color for contrast
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='round, pad=0.2'),
            )

    '''def prepare_labeled_masks(self, image, masks, detections, image_width, image_height):
        """
        Display bounding boxes with numbered labels, dynamically repositioned to avoid overlap
        and placed outside for small bounding boxes if needed. Labels match the color of their respective bounding boxes.
        Returns the annotated image.

        Args:
            image (np.ndarray): The original RGB image as a NumPy array.
            masks (list of np.ndarray): Binary masks for each object.
            detections (list or np.ndarray): Bounding boxes with coordinates [x_min, y_min, x_max, y_max].
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            np.ndarray: Annotated RGB image with bounding boxes and labels drawn.
        """
        # Predefined distinct colors for labels and bounding boxes
        distinct_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00',
            '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80',
            '#A52A2A', '#5F9EA0', '#D2691E', '#9ACD32', '#DA70D6', '#7FFFD4',
            '#FF4500', '#2E8B57'
        ]

        # Convert distinct_colors to RGB for OpenCV
        distinct_colors_rgb = [
            tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for color in distinct_colors
        ]

        # Create a copy of the image for annotation
        annotated_image = image.copy()

        # Track existing label positions to avoid overlap
        label_positions = []
        offset_step = 10  # Step size to prevent overlap

        def clamp(value, min_value, max_value):
            """Clamp a value to ensure it stays within min and max bounds."""
            return max(min_value, min(value, max_value))

        def safe_corners(box, is_small):
            """ Generate safe label positions for bounding box corners within image bounds. """
            x_min, y_min, x_max, y_max = box
            if is_small:
                # Label positions outside the bounding box
                return {
                    "outside-top-left": (clamp(x_min - offset_step, 0, image_width), clamp(y_min - offset_step, 0, image_height)),
                    "outside-top-right": (clamp(x_max + offset_step, 0, image_width), clamp(y_min - offset_step, 0, image_height)),
                    "outside-bottom-left": (clamp(x_min - offset_step, 0, image_width), clamp(y_max + offset_step, 0, image_height)),
                    "outside-bottom-right": (clamp(x_max + offset_step, 0, image_width), clamp(y_max + offset_step, 0, image_height)),
                }
            else:
                # Label positions inside the bounding box corners
                return {
                    "top-left": (clamp(x_min, 0, image_width), clamp(y_min, 0, image_height)),
                    "top-right": (clamp(x_max, 0, image_width), clamp(y_min, 0, image_height)),
                    "bottom-left": (clamp(x_min, 0, image_width), clamp(y_max, 0, image_height)),
                    "bottom-right": (clamp(x_max, 0, image_width), clamp(y_max, 0, image_height)),
                }

        # Iterate through each detection
        for i, box in enumerate(detections.xyxy):
            color = distinct_colors_rgb[i % len(distinct_colors_rgb)]  # Cycle through colors

            # Get the bounding box dimensions
            x_min, y_min, x_max, y_max = map(int, box)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            is_small = bbox_width < 30 or bbox_height < 30

            # Draw the bounding box
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, 1)

            # Determine safe label positions
            corners = safe_corners((x_min, y_min, x_max, y_max), is_small)
            corner_order = list(corners.keys())
            label_position = None

            # Check for overlap and adjust label position
            for corner in corner_order:
                pos_x, pos_y = corners[corner]
                overlap = any(
                    abs(pos_x - prev_x) < offset_step and abs(pos_y - prev_y) < offset_step
                    for prev_x, prev_y in label_positions
                )
                if not overlap:
                    label_position = (pos_x, pos_y)
                    break

            # If no non-overlapping position is found, use the last corner
            if label_position is None:
                label_position = corners[corner_order[-1]]

            # Store the new label position
            label_positions.append(label_position)
            font_size =0.3
            # Draw a filled rectangle for label background
            label_text = f"{i + 1}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
            label_width, label_height = label_size
            label_pos_x, label_pos_y = label_position
            cv2.rectangle(
                annotated_image,
                (label_pos_x, label_pos_y - label_height - 4),
                (label_pos_x + label_width, label_pos_y),
                color,
                -1  # Filled rectangle
            )

            # Put the label text
            cv2.putText(
                annotated_image,
                label_text,
                (label_pos_x, label_pos_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 0),  # White text
                1
            )

        return annotated_image'''
    def prepare_labeled_masks(self, image, masks, detections, image_width, image_height):
        """
        Display bounding boxes with numbered labels, dynamically repositioned to avoid overlap
        and ensure labels are inside the image. Returns the annotated image.

        Args:
            image (np.ndarray): The original RGB image as a NumPy array.
            masks (list of np.ndarray): Binary masks for each object.
            detections (list or np.ndarray): Bounding boxes with coordinates [x_min, y_min, x_max, y_max].
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            np.ndarray: Annotated RGB image with bounding boxes and labels drawn.
        """
        # Predefined distinct colors for labels and bounding boxes
        distinct_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00',
            '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80',
            '#A52A2A', '#5F9EA0', '#D2691E', '#9ACD32', '#DA70D6', '#7FFFD4',
            '#FF4500', '#2E8B57'
        ]

        # Convert distinct_colors to RGB for OpenCV
        distinct_colors_rgb = [
            tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for color in distinct_colors
        ]

        # Create a copy of the image for annotation
        annotated_image = image.copy()

        # Track existing label positions to avoid overlap
        label_positions = []
        offset_step = 5  # Step size to prevent overlap

        def clamp(value, min_value, max_value):
            """Clamp a value to ensure it stays within min and max bounds."""
            return max(min_value, min(value, max_value))

        def adjust_label_position(label_pos_x, label_pos_y, label_width, label_height):
            """Adjust label position to keep it inside the image."""
            label_pos_x = clamp(label_pos_x, 0, image_width - label_width)
            label_pos_y = clamp(label_pos_y, label_height + 4, image_height)
            return label_pos_x, label_pos_y

        def safe_corners(box, is_small):
            """ Generate safe label positions for bounding box corners within image bounds. """
            x_min, y_min, x_max, y_max = box
            if is_small:
                # Label positions outside the bounding box
                return {
                    "outside-top-left": (clamp(x_min - offset_step, 0, image_width), clamp(y_min - offset_step, 0, image_height)),
                    "outside-top-right": (clamp(x_max + offset_step, 0, image_width), clamp(y_min - offset_step, 0, image_height)),
                    "outside-bottom-left": (clamp(x_min - offset_step, 0, image_width), clamp(y_max + offset_step, 0, image_height)),
                    "outside-bottom-right": (clamp(x_max + offset_step, 0, image_width), clamp(y_max + offset_step, 0, image_height)),
                }
            else:
                # Label positions inside the bounding box corners
                return {
                    "top-left": (clamp(x_min, 0, image_width), clamp(y_min, 0, image_height)),
                    "top-right": (clamp(x_max, 0, image_width), clamp(y_min, 0, image_height)),
                    "bottom-left": (clamp(x_min, 0, image_width), clamp(y_max, 0, image_height)),
                    "bottom-right": (clamp(x_max, 0, image_width), clamp(y_max, 0, image_height)),
                }

        # Iterate through each detection
        label_dict = {}
        for i, box in enumerate(detections.xyxy):
            color = distinct_colors_rgb[i % len(distinct_colors_rgb)]  # Cycle through colors

            # Get the bounding box dimensions
            x_min, y_min, x_max, y_max = map(int, box)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            is_small = bbox_width < 30 or bbox_height < 30

            # Draw the bounding box
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), color, 1)

            # Determine safe label positions
            corners = safe_corners((x_min, y_min, x_max, y_max), is_small)
            corner_order = list(corners.keys())
            label_position = None

            # Check for overlap and adjust label position
            for corner in corner_order:
                pos_x, pos_y = corners[corner]
                overlap = any(
                    abs(pos_x - prev_x) < 20 and abs(pos_y - prev_y) < 20
                    for prev_x, prev_y in label_positions
                )
                if not overlap:
                    label_position = (pos_x, pos_y)
                    break

            # If no non-overlapping position is found, use the last corner
            if label_position is None:
                label_position = corners[corner_order[-1]]

            # Adjust label position to ensure it remains inside the image
            font_size = 0.3
            label_text = f"{i + 1}"
            label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
            label_width, label_height = label_size
            label_pos_x, label_pos_y = adjust_label_position(label_position[0], label_position[1], label_width, label_height)

            # Store the new label position
            label_positions.append((label_pos_x, label_pos_y))
            label_dict[label_text] = (label_pos_x, label_pos_y - label_height - 4, label_pos_x + label_width, label_pos_y, color, label_pos_x, label_pos_y - 2)
            # Draw a filled rectangle for label background

            # # Put the label text
        for key in label_dict.keys():
            cv2.rectangle(
                annotated_image,
                (label_dict[key][0], label_dict[key][1]),
                (label_dict[key][2], label_dict[key][3]),
                label_dict[key][4],
                -1  # Filled rectangle
            )
            cv2.putText(
                annotated_image,
                key,
                (label_dict[key][5], label_dict[key][6]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),  # Black text
                1
            )

        return annotated_image
    

    def prepare_labeled_contours(self, image, masks, detections, image_width, image_height):
        """
        Display object contours with numbered labels, dynamically positioning labels based on contour size.

        Args:
            image (np.ndarray): The original RGB image as a NumPy array.
            masks (list of np.ndarray): Binary masks for each object.
            detections (list or np.ndarray): Bounding boxes with coordinates [x_min, y_min, x_max, y_max].
            image_width (int): Width of the image.
            image_height (int): Height of the image.

        Returns:
            np.ndarray: Annotated RGB image with contours and labels drawn.
        """
        # Define distinct colors for each object
        distinct_colors = [
            '#A52A2A', '#5F9EA0', '#D2691E', '#9ACD32', '#DA70D6', '#7FFFD4',
            '#FF8000', '#8000FF', '#0080FF', '#80FF00', '#FF0080', '#00FF80',
            '#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFFF00',
            '#FF4500', '#2E8B57'
        ]

        # Convert color hex to RGB for OpenCV
        distinct_colors_rgb = [
            tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) for color in distinct_colors
        ]

        # Copy the original image for annotation
        annotated_image = image.copy()

        # Track label positions to prevent overlap
        label_positions = []
        offset_step = 5  # Offset distance for small contours

        def clamp(value, min_value, max_value):
            """Clamp a value to ensure it stays within min and max bounds."""
            return max(min_value, min(value, max_value))

        def find_safe_label_position(contour, bbox, is_small):
            """Find a safe label position inside or near the object contour."""
            x_min, y_min, x_max, y_max = bbox
            bbox_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))

            if not is_small:
                # Try placing label inside the contour (center)
                return bbox_center

            # If small, find a nearby position
            return (
                clamp(x_max, 0, image_width),
                clamp(y_min - offset_step, 0, image_height)
            )

        # Iterate through each detection and mask
        label_dict = {}
        for i, (mask, box) in enumerate(zip(masks, detections.xyxy)):
            color = distinct_colors_rgb[i % len(distinct_colors_rgb)]  # Cycle through colors
            # Validate mask (Ensure it's not empty or incorrectly shaped)
            if mask is None or mask.size == 0:
                print(f"Warning: Empty mask detected at index {i}")
                continue
            
            # Convert mask to uint8 format
            mask_uint8 = (mask.astype(np.uint8) * 255)
            # Ensure mask is 2D (Remove extra channel if present)
            if mask_uint8.ndim == 3 and mask_uint8.shape[0] == 1:
                mask_uint8 = mask_uint8.squeeze(0)  # Remove the first dimension
            
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If no valid contour, skip
            if not contours:
                print(f"Warning: No contours found for mask at index {i}")
                continue

            # Determine if the contour is small
            x_min, y_min, x_max, y_max = map(int, box)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            is_small = bbox_width < 30 or bbox_height < 30

            # Draw contours on the image
            cv2.drawContours(annotated_image, contours, -1, color, 2)

            # Find a safe label position
            label_pos_x, label_pos_y = find_safe_label_position(contours[0], (x_min, y_min, x_max, y_max), is_small)

            # Define label text
            label_text = f"{i + 1}"
            font_size = 0.35
            #label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
            label_width, label_height = 14, 14

            # Adjust label position to prevent out-of-bounds issues
            # Store the new label position
            label_positions.append((label_pos_x, label_pos_y))
            label_dict[label_text] = (label_pos_x, label_pos_y - label_height, label_pos_x + label_width, label_pos_y, color, label_pos_x, label_pos_y-3)
            # Draw label background rectangle
        for key in label_dict.keys():
            cv2.rectangle(
                annotated_image,
                (label_dict[key][0], label_dict[key][1]),
                (label_dict[key][2], label_dict[key][3]),
                label_dict[key][4],
                -1  # Filled rectangle
            )
            cv2.putText(
                annotated_image,
                key,
                (label_dict[key][5], label_dict[key][6]),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 0),  # Black text
                1
            )

            # # Put label text

        return annotated_image


