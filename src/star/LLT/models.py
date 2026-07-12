import numpy as np
import matplotlib.pyplot as plt
import sys
from openai import OpenAI
from PIL import Image
import base64

import cv2
import numpy as np
import base64
from PIL import Image
import io

def encode_image(image):
    # Case 1: If the input is a NumPy array (OpenCV image)
    if isinstance(image, np.ndarray):
        
        _, buffer = cv2.imencode('.png', image)
        byte_data = buffer.tobytes()

    # Case 2: If the input is a PIL Image
    elif isinstance(image, Image.Image):
        # Convert the PIL Image to bytes
        with io.BytesIO() as byte_io:
            image.save(byte_io, format="PNG")  # Save the image in PNG format
            byte_data = byte_io.getvalue()

    # Case 3: If the input is raw bytes
    elif isinstance(image, bytes):
        byte_data = image

    # Case 4: If the input is a file path
    elif isinstance(image, str):
        with open(image, "rb") as image_file:
            byte_data = image_file.read()

    else:
        raise ValueError("Input should be a NumPy array, PIL Image, bytes, or file path.")
    
    # Encode the byte data to base64
    return base64.b64encode(byte_data).decode('utf-8')


def get_chatgpt_output(model, new_prompt, messages, role, COT_LOG_PATH, image_path1, image_path2=None, file=sys.stdout):

    print(role + ":", file=file)
    print(new_prompt, file=file)
    if model == "gpt-4o" or model == "gpt-4o-mini" or model =="gpt-4.1":
        base64_rgb = encode_image(image_path1)
        messages.append({   
                "role": role,
                "content": new_prompt,
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": new_prompt,
                },
                # {
                # "type": "image_url",
                # "image_url": {
                #     "url":  f"data:image/jpeg;base64,{base64_image}"
                # },
                # },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_rgb}", "detail": "high"
                },
                },
                # {
                # "type": "image_url",
                # "image_url": {
                #     "url":  f"data:image/jpeg;base64,{based64_semantic}", "detail": "high"
                # },
                # },
            ],
            })
    elif model == "gpt-4":
        messages.append({"role":role, "content":new_prompt})
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""
    with open(COT_LOG_PATH, "a") as f_log:
        f_log.write(f"Assistant:\n")
        f_log.flush()

        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content
            finish_reason = chunk.choices[0].finish_reason
            if chunk_content is not None:
                print(chunk_content, end="", file=file)
                print(chunk_content, end="", file=f_log)
                f_log.flush()
                new_output += chunk_content
            else:
                print("finish_reason:", finish_reason, file=file)

    messages.append({"role":"assistant", "content":new_output})

    return messages
