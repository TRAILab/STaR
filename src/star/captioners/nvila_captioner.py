import argparse
import importlib.util
import json
import os
import sys
sys.path.append('/workspace/third_parties/VILA')
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from .captioner import Captioner

from pydantic import BaseModel
from termcolor import colored

from PIL import Image as PILImage
from io import BytesIO
import tempfile

import llava
from llava import conversation as clib
from llava.media import Image as LlavaImage, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat

def get_schema_from_python_path(path: str) -> str:
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


class NVILACaptioner(Captioner):
    def __init__(self, args):
        model_name = getattr(args, "model_name", "NVILA")
        print("\n[captioner] Loading video caption model")
        print(f"  name                    {model_name}")
        print(f"  path                    {args.model_path}")
        # Convert json mode to response format
        if not args.json_mode:
            self.response_format = None
        elif args.json_schema is None:
            self.response_format = ResponseFormat(type="json_object")
        else:
            schema_str = get_schema_from_python_path(args.json_schema)
            self.response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

        self.model = llava.load(args.model_path).to("cuda")
        self.model.eval()
        # Set conversation mode
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
        self.args = args
        print("[captioner] Video caption model ready\n")

    def caption(self, images=None, query: str = None, max_new_tokens: int = 40) -> str:
        # Prepare multi-modal prompt
        prompt = []
        if self.args.media is not None or images is not None:
            inputs = images if images is not None else self.args.media
            idx = 0
            for media in inputs or []:
                # New logic: Support strings, PIL images, and llava.media.Image objects.
                if isinstance(media, (LlavaImage, Video)):
                    pass  # already wrapped
                elif isinstance(media, str):
                    if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        media = LlavaImage(media)
                    elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                        media = Video(media)
                    else:
                        raise ValueError(f"Unsupported media path: {media}")
                elif isinstance(media, PILImage.Image):
                    # Save PIL image to temp file and wrap
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        media.save(tmp.name, format="JPEG")
                        media = LlavaImage(tmp.name)
                else:
                    raise TypeError(f"Unsupported media input type: {type(media)}")

                prompt.append(f"the label of this image is: {idx}")
                prompt.append(media)
                idx += 1
        # Add text input

        if query is not None:
            prompt.append(query)
        else:
            prompt.append(self.args.query)

        # `default_generation_config` returns a fresh copy each access, so mutate
        # one local config object and pass that same instance to generation.
        generation_config = self.model.default_generation_config
        generation_config.max_new_tokens = max_new_tokens
        generation_config.temperature = 0
        response = self.model.generate_content(
            prompt, 
            response_format=self.response_format, 
            generation_config=generation_config
        )

        return response
