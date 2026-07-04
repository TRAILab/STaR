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

STaR QA Quick Start
===================

This guide covers two common ways to run STaR question answering:

1. Run evaluation on an existing QA file.
2. Ask live questions from the Gradio web interface.


**0. Start Milvus**

Start Milvus before running the agent:

    bash launch_milvus_container.sh start


**1. Check The Config**

Main config:

    configs/config.yaml

Important fields:

    sequence: "0"
    postfix: "CoDa"

Meaning:

    sequence
        Dataset / run ID.

    postfix
        Gradio output channel. This must match the `--postfix` argument used by
        eval_AIB.py.

Gradio path config:

    configs/inference/docker.yaml


**2. Prepare The Expected Files**

The default Docker paths are:

    Video captions
        /workspace/results/<sequence>/caption/<caption_file>.json

    Scene graph / point-cloud memory
        /workspace/results/<sequence>/pcd/<scenegraph_file>.pkl.gz

    Annotated RGB keyframes
        /workspace/results/<sequence>/annotated_rgb/annotated_rgb_<idx>.png

    Raw frame timestamps
        /workspace/Local_data/CODa/timestamps/<sequence>.txt

    QA questions
        /workspace/data/coda/questions/<sequence>/<qa_file>.json


**3. Run eval_AIB.py On Existing Questions**

Base command:

    python scripts/eval_AIB.py

By default, this runs dataset evaluation with:

    --question_source dataset
    --manual_evaluation True
    --method star
    --llm gpt-4.1
    --sequence_id 0
    --postfix CoDa

Override only the options you need. For example:

    --manual_evaluation False
        Run all selected questions automatically.

    --manual_evaluation True
        Step through questions manually. The script asks before each question,
        and you can run, skip, jump to an index, or quit.

Available methods:

    star
        STaR / AIB retrieval agent.

    remembr
        Vanilla ReMEmbR-style baseline.

    scene_graph
        Scene-graph baseline.

Common options:

    --sequence_id
        Dataset / run ID. Default: 0.

    --postfix
        Gradio output channel. Default: CoDa.

    --qa_file
        QA file name without .json. Default: human_qa.

    --caption_file
        Caption file name without .json. Default: captions_NVILA-Lite-2B.

    --scenegraph_file
        Scene graph file name without .pkl.gz. Default: full_pcd.

    --results
        Result folder containing captions, scene graph, and annotated images.
        Default: /workspace/results.

    --data_dir
        Dataset folder containing QA files. Default: /workspace/data/coda.

    --coda_dir
        CODa folder containing timestamp files. Default: /workspace/Local_data/CODa.


**4. Ask Live Questions From Gradio With eval_AIB.py**

Terminal 1: start the web UI.

    python scripts/run_gradio_interface.py

Terminal 2: start the QA agent.

    python scripts/eval_AIB.py \
      --question_source gradio

Then open the Gradio URL, type a question, and press Submit.

If your config uses a different sequence or postfix, pass only those overrides:

    python scripts/eval_AIB.py \
      --question_source gradio \
      --sequence_id <sequence> \
      --postfix <postfix>


**5. What Gradio Displays**

Retrieval score plot

    Path:
        /workspace/results/<sequence>/search_DB/<postfix>/retrieval_DB_<idx>_<postfix>.png

    Meaning:
        Shows which video-caption memories were retrieved from the vector
        database and how relevant they were over time.

Retrieved keyframes

    Path:
        /workspace/results/<sequence>/images/<postfix>/<idx>/

    Meaning:
        Visual evidence images selected from the timestamps suggested by the
        agent.

Agent log

    Path:
        /workspace/results/<sequence>/cot_log/<postfix>/cot_log_<idx>.txt

    Meaning:
        Records the user question, retrieval steps, selected timestamps/images,
        and final reasoning output shown in the Gradio chat.
