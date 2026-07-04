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

Running QA and Retrieval
========================

STaR supports two common ways to run question answering:

1. Evaluate existing questions from a QA JSON file.
2. Ask live questions through the Gradio interface.

Make sure Milvus is running before starting the agent:

    bash launch_milvus_container.sh start


Important Config
----------------

The main config is:

    configs/config.yaml

Key fields:

    sequence: "0"
    postfix: "CoDa"

`sequence` selects the dataset/run ID.
`postfix` selects the Gradio output channel and must match the runner argument
`--postfix`.

The default Gradio paths are defined in:

    configs/inference/docker.yaml


Expected Data Layout
--------------------

By default, the scripts expect:

    /workspace/results/<sequence>/caption/<caption_file>.json
    /workspace/results/<sequence>/pcd/<scenegraph_file>.pkl.gz
    /workspace/results/<sequence>/annotated_rgb/annotated_rgb_<idx>.png
    /workspace/Local_data/CODa/timestamps/<sequence>.txt
    /workspace/data/coda/questions/<sequence>/<qa_file>.json


Run eval_AIB.py
---------------

Evaluate existing questions:

    python scripts/eval_AIB.py \
      --question_source dataset \
      --method star \
      --llm gpt-4.1 \
      --sequence_id 0 \
      --postfix CoDa

Available methods:

    --method star
    --method remembr
    --method scene_graph

Useful options:

    --qa_file human_qa
    --caption_file captions_NVILA-Lite-2B
    --scenegraph_file full_pcd
    --results /workspace/results
    --data_dir /workspace/data/coda
    --coda_dir /workspace/Local_data/CODa


Ask Questions From Gradio
-------------------------

Terminal 1:

    python scripts/run_gradio_interface.py

Terminal 2:

    python scripts/eval_AIB.py \
      --question_source gradio \
      --method star \
      --llm gpt-4.1 \
      --sequence_id 0 \
      --postfix CoDa

Then open the Gradio URL and submit a question.

Gradio will show:

    /workspace/results/<sequence>/search_DB/<postfix>/retrieval_DB_<idx>_<postfix>.png
    /workspace/results/<sequence>/images/<postfix>/<idx>/
    /workspace/results/<sequence>/cot_log/<postfix>/cot_log_<idx>.txt

