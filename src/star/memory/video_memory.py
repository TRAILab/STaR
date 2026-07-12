from dataclasses import dataclass, asdict

import datetime, time
from time import strftime, localtime
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from .memory import Memory, MemoryItem

FIXED_SUBTRACT=1721761000 # this is just a large value that brings us closed to 1970


@dataclass
class ImageMemoryItem(MemoryItem):
    time: float
    position: list
    theta: float
    image: Image.Image


class VideoMemory(Memory):

    def __init__(self, fps=1):
        self.memory = []
        self.last_memory_time = 0
        self.fps = fps

    def insert(self, item: ImageMemoryItem):
        self.memory.append(item)

    def reset(self):
        self.memory = []

    def get_working_memory(self) -> list[ImageMemoryItem]:
        return self.memory



def format_memory(memory_item_list: list[MemoryItem]):
    return memory_item_list
