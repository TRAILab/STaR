# STaR Agentic RAG

STaR supports an **Agentic RAG** workflow built on the robot's multimodal memory. Given an open-ended user query, the STaR agent:
- Plans an effective memory retrieval strategy.
- Autonomously invokes the appropriate retrieval tools.
- Retrieves the most relevant multimodal memories from the robot's long-term memory.
- Performs cross-modal contextual reasoning over the retrieved memories.
- Generates an accurate and context-aware response.

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

## Run 
Now that you have created the memory for the agent. <br>
You can run `python3 scripts/eval_agent.py` to create the agent, the agent will wait for the user's question.
