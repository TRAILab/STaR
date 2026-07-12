import cv2

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
