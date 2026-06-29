# Running
STaR is a project that utilize LLM, VLM, and scene graph to build and reason over long term spatio-temporal memories.

## Setup
1. It is recommended to use docker to run the project as we use the config files for docker by default, check [here](INSTALL.md) to learn how to run on docker container.
2. Install MilvusDB
    ```
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o launch_milvus_container.sh
    ```

    docker must be installed on the system to easily use Milvus by simply running the command below. This script will automatically launch MilvusDB on a docker container. Otherwise, the user must install MilvusDB from scratch themselves

    ```
    bash launch_milvus_container.sh start
    ```
3. Install OLLama (Optional)
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ```

## Usage
### Step 1- Create a Memory database and the Scene Graph
- Before you can use the STaR agent, you need to build robot memory.
- Read [here](BuildMemory.md) for more details about how to collect memory data for the agent.
- Read [here](INSTALL.md) to learn more about how to configure the system.

### Step 2 - Open the User Interface
STaR is designed to interact with users via a website. <br>
To start the user interface, run `python3 scripts/run_gradio_interface.py`, then you can open the brower and go to `127.0.0.1:<your port number>` (Note: you can find the generated link directly in the terminal window and click it to open the interface automatically).<br>
In order to allow the agent to execute multi-modal tasks, you also have to run `python3 scripts/run_camera_subscriber.py` to get the image from the publisher.

### Step 3 - Create the Agent
Now that you have created the memory and the scene graph for the agent. <br>
You can run `python3 scripts/eval_agent.py` to create the agent, the agent will wait for the user's question.

### Step 4 - Run the Agent!
Enter questions in the text area, and optinally you can press update button to get the current image of the robot. <br>
Then press submit button, the agent will then search from it's memory and then navigate to the position or answer the question.

## Note
It is recommended to use the ROS2 Node to publish the data when validating the agent's performance. We validated our framework using the CODA dataset, you can check `scripts/write_coda_rosbag.py`.
