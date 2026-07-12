import os
import base64
from typing import List

import numpy as np
from openai import OpenAI

from .captioner import Captioner

class OpenAICaptioner(Captioner):
    def __init__(self, model_name: str = 'gpt-4o') -> None:
        self.model_name: str = model_name 
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def __encode_image(self, image_path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def caption(self, image_paths: List[str], query: str) -> str:
        content = [{"type": "text", "text": query}]

        for image_path in image_paths:
            base64_img = self.__encode_image(image_path)
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            else:
                mime_type = "application/octet-stream"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_img}"
                }
            })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=1000
        )

        return response.choices[0].message