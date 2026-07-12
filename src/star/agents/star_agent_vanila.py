from typing import Annotated, Sequence, TypedDict
import traceback
import sys, re
import os

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

warnings.filterwarnings(
    "ignore",
    message="The class `Milvus` was deprecated in LangChain 0.2.0 and will be removed in 1.0*",
)
sys.path.append(sys.path[0] + '/..')


from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from termcolor import colored

from ..utils.util import file_to_string
from .functions_wrapper import FunctionsWrapper
from ..memory.memory import Memory
from .agent import Agent, AgentOutput

import datetime

def extract_robust_full_timestamps(text: str) -> list[str]:
    """
    extract full timestamps in the format of 'YYYY-MM-DD HH:MM:SS' from the text
    """
    results = []

    # capture YYYY-MM-DD HH:MM:SS
    full_dt_matches = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
    results.extend(full_dt_matches)

    # then find expressions like 'YYYY-MM-DD HH:MM:SS to HH:MM:SS'
    range_matches = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*(?:to|and|until|-|~)\s*(\d{2}:\d{2}:\d{2})", text)
    for date_part, time_only in range_matches:
        date_str = date_part.split(" ")[0]
        results.append(f"{date_str} {time_only}")

    # deduplicate results and # maintain order
    seen = set()
    unique_results = []
    for t in results:
        if t not in seen:
            seen.add(t)
            unique_results.append(t)

    return unique_results

def extract_full_timestamps(text: str) -> list[str]:
    """
    extract full timestamps in the format of 'YYYY-MM-DD HH:MM:SS' from the text
    """
    return re.findall(r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b", text)

def datetime_string_to_frame_index(dt_str: str, global_starttime: float, fps: int = 10) -> int:
    """
    convert a full datetime string to a frame index
    """
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    delta_seconds = (dt.timestamp() - global_starttime)

    if delta_seconds < 0:
        raise ValueError(f"{dt_str} is before the video start time.")

    return int(delta_seconds * fps)

def build_image_paths_from_full_timestamps(dt_list: list , global_starttime: float, fps: int = 10, frame_dir: str = "frames") -> list[str]:
    """
    extract full timestamps from the text and convert them to image paths
    """
    seen = set()
    image_paths = []
    frame_ids = []

    for dt_str in dt_list:
        if dt_str not in seen:
            seen.add(dt_str)
            frame_index = datetime_string_to_frame_index(dt_str, global_starttime, fps)

            path = f"{frame_dir}{frame_index}.png"
            image_paths.append(path)
            frame_ids.append(frame_index)
    return image_paths, frame_ids

### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)

        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(item)

    return state


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

def after_human_feedback(state: AgentState) -> str:
    last_msg = state["messages"][-1].content.lower()
    print(f"Human feedback received: {last_msg}")

    if any(kw in last_msg for kw in ["yes", "approved", "looks good", "correct", "that's right"]):
        return "end"
    else:
        print("Human provided feedback. Returning to agent for refinement.")
        return "back2agent"

def should_continue(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

def should_continue_new(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "generate"
    else:
        return "continue"

def try_except_continue(state, func):
    while True:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue

class STaRAgent_VANILA(Agent):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0):

        # Wrapper that handles everything
        llm = self.llm_selector(llm_type, temperature, num_ctx)
        chat = FunctionsWrapper(llm)

        self.num_ctx = num_ctx
        self.temperature = temperature

        self.chat = chat
        self.llm_type = llm_type
        ### Load vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        top_level_path = str(os.path.dirname(__file__)) + '/../'
        # Different port #1
        self.agent_prompt = file_to_string(top_level_path+'prompts_isaacsim/agent_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts_isaacsim/generate_system_prompt.txt')
        self.agent_gen_only_prompt = file_to_string(top_level_path+'prompts_isaacsim/agent_gen_system_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        self.human_review = {}

        self.chat_history = ChatMessageHistory()


    def llm_selector(self, llm_type, temperature, num_ctx):
        llm = None
        # Support for LLM Gateway
        if 'gpt-4' in llm_type:
            llm = ChatOpenAI(model=llm_type, temperature=temperature, max_tokens=num_ctx)

        elif 'o4' in llm_type:
            # For o4-mini (GPT-4o-mini), use the correct model name
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
            ).bind(
                max_completion_tokens=num_ctx  # ✅ Correct param for GPT-4o style models
            )

        # Support for NIMs
        elif 'nim/' in llm_type:
            print("Using NVIDIA LLMs!")
            llm_name = llm_type[4:]
            llm = ChatNVIDIA(model=llm_name)

        # Support for Ollama functions
        elif llm_type == 'command-r':
            print("Using Command-R LLMs!")
            llm = ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx)
        else:
            print("Using Ollama LLMs!")
            llm = ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

        if llm is None:
            raise Exception("No correct LLM provided")
        print(colored(f"Using LLM: {llm_type} with temperature {temperature} and num_ctx {num_ctx}", "white", attrs=["dark"]))
        return llm

    def set_memory(self, memory: Memory, scene_graph=None, dataset_start_timestamp=None, sbert_model=None, test_num=None, args=None, timestamp_list=None):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()
        self.timestamp_list = timestamp_list
        # Different port #2

    def create_tools(self, memory):

        template = "At time={{time}} seconds, the robot was at an average position of {{position}} with an average orientation of {{theta}} radians. "
        template += "The robot saw the following: {{page_content}}"


        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_text_org(x),
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: tuple = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                                The query must be an (x,y,z) array with floating point values \
                                Based on the question and your context, decide what position to search for in the database. \
                                This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. \
                                The query will then search your memories for you.")
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                                The query must be a string containing only time. \
                                Based on the question and your context, decide what time to search for in the database. \
                                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        # position-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_time(x),
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an H:M:S time.",
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )
        '''
        # Define input format
        class SceneGraphRetrieverInput(BaseModel):
            x: str = Field(description="The query to retrieve a scene object by semantic similarity. \
                                        This should be a noun phrase or description like 'red couch', 'yellow police pole', or 'white bicycle'. \
                                        Use this when you want a precise spatial location or a known object.")

        # Create the tool
        self.scenegraph_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_scenegraph(x),
            name="retrieve_from_scenegraph",
            description="Search and return precise object information from the Scene Graph memory.",
            args_schema=SceneGraphRetrieverInput
        )'''

        class NearestObjectInput(BaseModel):
            robot_position: list = Field(
                description="The [x, y, z] position of the robot."
            )
            object_positions: list = Field(
                description="A list of [[x1, y1, z1], [x2, y2, z2]] positions of candidate objects."
            )

        #     if not object_positions:
        #         return "No objects provided to compare."

        #     distances = [np.linalg.norm(robot_pos - obj_pos) for obj_pos in object_positions]
        #     min_idx = np.argmin(distances)
        #     closest_pos = object_positions[min_idx]
        #     min_distance = distances[min_idx]

        #     return f"The closest object is at position {closest_pos.tolist()} with a distance of {min_distance:.2f} meters."

        def find_nearest_object(robot_position: list, object_positions: list) -> str:
            import numpy as np

            # Convert the robot position to a NumPy array
            robot_pos = np.array(robot_position)

            # Return message if no objects are provided
            if not object_positions:
                return "No objects were provided to compare distances."

            distances_info = []
            # Compute the distance from the robot to each object
            for pos in object_positions:
                obj_pos = np.array(pos)
                distance = np.linalg.norm(robot_pos - obj_pos)
                distances_info.append((pos, distance))

            # Sort objects by distance (ascending)
            distances_info.sort(key=lambda x: x[1])

            # Format the response in a readable, natural language style
            response_lines = []
            for pos, dist in distances_info:
                response_lines.append(f"The object at position {pos} is {dist:.2f} meters away.")

            return "Here are the distances to the objects:\n" + " ".join(response_lines)

        self.nearest_object_tool = StructuredTool.from_function(
            func=find_nearest_object,
            name="find_nearest_object",
            description="Given the robot's position and a list of object positions, find the nearest object and summarize the distance.",
            args_schema=NearestObjectInput
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool, self.nearest_object_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]


    def human_feedback_node(self, state):
        last_message = state["messages"][-1]

        # Check: if the last message is human input, skip interrupt
        if isinstance(last_message, HumanMessage):
            print("[Debug] Human feedback already received, skipping interrupt.")
            return {"back2agent": "agent"}

        # Otherwise, send the prompt to human
        question = state["messages"][0].content
        last_model_output = state["messages"][-1].content
        # Parse
        try:
            if isinstance(last_model_output, str):
                if '```json' in last_model_output:
                    parsed_output = parse_json(last_model_output)
                else:
                    parsed_output = eval(last_model_output)
            else:
                parsed_output = last_model_output  # already a dict
        except Exception:
            print("Failed to parse last_model_output, using raw text.")
            parsed_output = {"text": last_model_output}

        # Build human-friendly message
        reasoning = parsed_output.get("answer_reasoning", "")
        final_answer = parsed_output.get("text", last_model_output)

        prompt_msg = (
            f"✅ I tried to answer your question: \"{question}\"\n\n"
            f"📝 Here is my reasoning:\n{reasoning}\n\n"
            f"➡️ My current answer is: {final_answer}\n\n"
            "Does this look correct? If yes, say 'yes'. Otherwise, please specify what’s missing or unclear."
        )
        print(colored(f"[Review] message: \n{prompt_msg}", "cyan", attrs=["bold"]))

        # Collect human feedback
        action = input(colored("Please provide your action (yes/feedback):\n", "cyan", attrs=["bold"]))
        if action != "yes":
            feedback = action + "; Based on the user feedback, please try the second round of searching to improve your answer."
        else:
            feedback = "Yes, confirmed!"
        self.agent_call_count = 0
        return {"messages": state["messages"] + [HumanMessage(content=feedback)]}


    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.chat

        if self.agent_call_count < 3:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_gen_only_prompt

        agent_prompt = ChatPromptTemplate.from_messages(
            [
                #("system", prompt),
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),

            ]
        )

        model = agent_prompt | model

        question = f"The question is: {messages[0]}"

        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type) and ('o4' not in self.llm_type):
            print("Making sure not using gpt-4 or nim!")
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id

        response = model.invoke({"question": question, "chat_history": messages[:]})

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"

        self.agent_call_count += 1
        print(colored(f"Agent response: {response}", "red", attrs=["bold"]))
        return {"messages": [response]}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please responsed in the desired format."


        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})


        gen_prompt = ChatPromptTemplate.from_messages(
            [
                # ("human", "What do you do?"),
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                # ("ai", filled_prompt.text),
                ("human", "{question}"),

            ]
        )

        model = gen_prompt | self.chat

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = eval(response)
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["time", "text", "binary", "position", "duration"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")

            if type(parsed['position']) == str:
                parsed['position'] = eval(parsed['position'])

            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except:
            raise ValueError("Generate call failed. Retrying...")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}

    def generate_new(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please responsed in the desired format."


        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})


        gen_prompt = ChatPromptTemplate.from_messages(
            [
                # ("human", "What do you do?"),
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                # ("ai", filled_prompt.text),
                ("human", "{question}"),

            ]
        )

        model = gen_prompt | self.chat

        response = model.invoke({"question": question, "chat_history": messages[1:]})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())

        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = eval(response)
            else:
                parsed = parse_json(response)

            # then check it has all the required keys
            keys_to_check_for = ["time", "text", "binary", "position", "duration"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")

            if type(parsed['position']) == str:
                parsed['position'] = eval(parsed['position'])

            if (parsed['position'] is not None) and len(parsed['position']) != 3:
                raise ValueError(f"Shape of position was incorrect. {parsed['position']}. Retrying...")

        except:
            raise ValueError("Generate call failed. Retrying...")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"

        # return {"messages": [str(parsed)]}

        return {"messages": state["messages"] + [AIMessage(content=str(parsed))]}

    def generate_with_timestamp(self, state):
        """
        Generate answer using multimodal input (text + images) if available,
        or fallback to text-only if no images found.
        """
        import base64, io
        from PIL import Image, ImageDraw, ImageFont

        #     return image
        def annotate_image_with_timestamp(image_path: str, timestamp: str) -> Image.Image:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            try:
                font_path = "/home/mfyuan/local_folder/OpenNav_v2/remembr/remembr/config/arial.ttf"
                font = ImageFont.truetype(font_path, size=48)
            except:
                font = ImageFont.load_default()
                print(colored("Using default font", "yellow", attrs=["bold"]))

            text_position = (10, 10)
            text_color = "white"
            background_color = "black"

            # Calculate size of the text box
            text_size = draw.textbbox(text_position, timestamp, font=font)
            background_box = (text_size[0] - 10, text_size[1] - 5, text_size[2] + 10, text_size[3] + 5)

            # Draw background rectangle
            draw.rectangle(background_box, fill=background_color)

            # Draw the text over the background
            draw.text(text_position, timestamp, fill=text_color, font=font)

            return image
        def image_to_base64_data_url(image: Image.Image) -> str:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"

        def image_to_base64_data(image_path: str, ts: str) -> str:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                # Determine MIME type based on file extension
                if path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
                    mime_type = "image/jpeg"
                else:
                    mime_type = "application/octet-stream"

                # Add text label before each image (e.g., Image 1, Image 2, ...)
                txt = {"type": "text", "text": f"Image at timestamp {ts}:"}
                img = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
                }

            return txt, img

        messages = state["messages"]
        question = messages[0].content + "\n Please respond in the desired format."
        last_message = messages[-1]
        docs = last_message.content
        print(colored(f"Inside generate for the last content-check if it is the response:\n {docs}", "red", attrs=["bold"]))
        # Extract images
        multimodal_inputs = []
        try:
            dt_list = extract_robust_full_timestamps(docs)
            print(f"Extracted timestamps: {dt_list}")
            image_paths, frame_ids = build_image_paths_from_full_timestamps(dt_list, self.start_timestamp, fps=10, frame_dir=f"/media/ssd/Local_data/CODa_dataset/data/2d_rect/cam0/{str(self.args.sequence_id)}/2d_rect_cam0_{str(self.args.sequence_id)}_")
            use_existing_images = True
            for ts, path, img_id in zip(dt_list, image_paths, frame_ids):
                try:

                    if use_existing_images:
                        # Use existing images if available
                        output_path = f"/home/mfyuan/local_folder/OpenNav_v2/remembr/remembr/data/scene_graphs/{str(self.args.sequence_id)}/annotated_rgb/annotated_rgb_{str(round(img_id / 10) * 10)}.png"
                        if not os.path.exists(output_path):
                            print(colored(f"Image {output_path} does not exist, using generated image.", "yellow"))
                    else:
                        img = annotate_image_with_timestamp(path, ts)
                        # save the image
                        output_path = f"/home/mfyuan/local_folder/OpenNav_v2/remembr/remembr/data/images/{str(self.args.sequence_id)}/{str(self.test_num)}/{ts.replace(':', '-')}.png"
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        img.save(output_path)  # Optional: save the annotated image
                    print(colored(f"Using IMG: {output_path}", "green", attrs=["bold"]))
                    txt_prompt, img_prompt = image_to_base64_data(output_path, ts)
                    multimodal_inputs.append(txt_prompt)
                    multimodal_inputs.append(img_prompt)
                    print(colored(f"Processed image {txt_prompt}", "green", attrs=["bold"]))
                except Exception as e:
                    print(colored(f"Failed to process image {path}: {e}", "yellow"))
        except Exception as e:
            print(colored(f"Image processing failed: {e}", "red"))
            multimodal_inputs = []

        # # Prompt template

        # # Construct input for GPT-4o

        # # Run model

        # Construct multimodal prompt using ChatPromptTemplate if images are available
        if multimodal_inputs:

            prompt = PromptTemplate(template=self.generate_prompt, input_variables=["context", "question"])
            filled_prompt = prompt.invoke({'question': question})

            prompt_img = "Objects in the image are outlined with contours in different colors, and each has a corresponding numerical label in the same color representing its identifier in the scene graph. Use the retrieved information together with these identifiers to help you more accurately obtain the target object's location. If the retrieved historical information contains the target object’s center coordinates, you must prioritize using those coordinates. If not, use the robot’s position coordinates as the answer."
            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                ("human", [
                    {"type": "text", "text":  f"{prompt_img}; Which of the following images best matches the question: '{question}'?"}
                ] + multimodal_inputs)
            ])

            model = gen_prompt | self.chat
            response = model.invoke({"question": question, "chat_history": messages[1:]})

        else:
            # fallback to text-only chain if no images found
            prompt = PromptTemplate(template=self.generate_prompt, input_variables=["context", "question"])
            filled_prompt = prompt.invoke({'question': question})

            gen_prompt = ChatPromptTemplate.from_messages([
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ])

            model = gen_prompt | self.chat
            response = model.invoke({"question": question, "chat_history": messages[1:]})
        # Post-process output
        response_text = ''.join(response.content.splitlines())
        try:
            parsed = eval(response_text) if '```json' not in response_text else parse_json(response_text)
            for key in ["time", "text", "binary", "position", "duration"]:
                if key not in parsed:
                    raise ValueError("Missing keys during generate.")
            if isinstance(parsed["position"], str):
                parsed["position"] = eval(parsed["position"])
            if parsed["position"] and len(parsed["position"]) != 3:
                raise ValueError("Invalid position format.")
        except Exception as e:
            raise ValueError(f"Generate call failed: {e}")

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        return {"messages": state["messages"] + [AIMessage(content=str(parsed))]}


    def build_graph(self):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)

        workflow.add_node(
            "generate", lambda state: try_except_continue(state, self.generate)
        )  # Generating a response after we know the documents are relevant
        # Call agent node to decide to retrieve or not


        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "end": "generate",
            },
        )

        workflow.add_edge('action', 'agent')
        workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()

    def build_graph_new(self):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        tool_node = ToolNode(self.tool_list)
        workflow.add_node("action", tool_node)
        workflow.add_node("generate", lambda state: try_except_continue(state, self.generate_with_timestamp))
        workflow.add_node("human_feedback", self.human_feedback_node)

        workflow.set_entry_point("agent")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "agent",
            # Assess agent decision
            should_continue,
            {
                # Translate the condition outputs to nodes in our graph
                "continue": "action",
                "generate": "generate",
            },
        )

        workflow.add_conditional_edges(
        "human_feedback",
        after_human_feedback,
        {"end": END, "back2agent": "agent"},
       )


        workflow.add_edge('action', 'agent')
        workflow.add_edge("generate", "human_feedback")
        checkpoint = MemorySaver()
        # Compile
        self.graph = workflow.compile(checkpointer=checkpoint)
        self.config = {
            "configurable": {
                "thread_id": uuid.uuid4(),
            }
            }
        # View
        if False:
            img_bytes = self.graph.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(img_bytes)
            print("Graph saved as graph.png")


    def query(self, question: str):

        inputs = { "messages": [
                                (("user", question)),
            ]
        }

        out = self.graph.invoke(inputs)
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = eval(response)
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)


        return response

    def query_new(self, question: str):

        inputs = { "messages": [
            (("user", question)),
            ]
        }

        if GraphInterrupt:
            out = self.graph.invoke(inputs, config=self.config)
        else:
            out = self.graph.invoke(inputs)
        response = out['messages'][-2]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = eval(response)
        else:
            parsed = parse_json(response)

        response = AgentOutput.from_dict(parsed)
        return response


if __name__ == "__main__":

    from memory.milvus_memory import MilvusMemory

    # llm_name =
    # Options: 'nim/meta/llama-3.1-405b-instruct', 'gpt-4o', or any Ollama LLMs (such as 'codestral')
    memory = MilvusMemory("test", db_ip='127.0.0.1')

    llm_name = 'gpt-4o'
    agent = STaRAgent_VANILA(llm_type=llm_name)

    agent.set_memory(memory)

    response = agent.query("Where can I sit?")
    response = agent.query_position("Where can I sit?")
