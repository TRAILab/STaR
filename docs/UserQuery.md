# STaR Agentic RAG

STaR supports an **Agentic Retrieval-Augmented Generation (Agentic RAG)** workflow built upon the robot's multimodal long-term memory.

Given an open-ended user query, the STaR agent will:

- 🧠Plan an effective memory retrieval strategy.
- 🔧 Autonomously invoke the required retrieval tools.
- 📚 Retrieve the most relevant multimodal memories.
- 🔍 Perform cross-modal contextual reasoning.
- 💬 Generate an accurate and context-aware response.

---

# Prerequisites

## 1. Launch Milvus

Milvus is required for multimodal memory retrieval.

In a host terminal, run this from the STaR project root:

```bash
bash scripts/bash/launch_milvus_container.sh start
```

> **Note**
>
> Docker must be installed. The script automatically launches Milvus inside a Docker container.

---

## 2. Install Ollama (Optional)

Only required when using local LLMs.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

# Quick Start

STaR supports two common QA modes:

| Mode | Description |
|-------|-------------|
| **Interactive Gradio** | Ask live questions through the web interface. |
| **Dataset Evaluation** | Evaluate an existing NaVQA question file. |

---

# Step 1. Start Milvus

In a host terminal, run this from the STaR project root:

```bash
bash scripts/bash/launch_milvus_container.sh start
```

---

# Step 2. Check Configuration

Main configuration:

```
configs/config.yaml
```

Important fields:

```yaml
sequence: "0"
postfix: "star"
```

### sequence

Set this to the same sequence used when building the
memory; otherwise, the query workflow will look for a different memory.

### postfix

Used as the Gradio output channel.

> **Important**
>
> This value **must match** the `--postfix` argument passed to `eval_AIB.py`.

Docker path configuration:

```
configs/inference/docker.yaml
```

---

# Step 3. Prepare Required Files

By default, STaR expects the following files:

| Data | Default Path |
|------|--------------|
| Video captions | `/workspace/results/<sequence>/caption/<caption_file>.json` |
| Scene graph memory | `/workspace/results/<sequence>/pcd/<scenegraph_file>.pkl.gz` |
| Annotated RGB keyframes | `/workspace/results/<sequence>/annotated_rgb/annotated_rgb_<idx>.png` |
| QA file (for NaVQA only) | `/workspace/data/coda/questions/<sequence>/<qa_file>.json` |

---

## OpenAI API Key

Configure your OpenAI API key before testing either workflow below. It is required for both Gradio and dataset evaluation:

```bash
nano ~/.bashrc
```

Add the following line, replacing the empty value with your key:

```bash
export OPENAI_API_KEY=""
```

Then reload the shell configuration:

```bash
source ~/.bashrc
```

---

# Option 1 — Ask Live Questions (Gradio)

Run both Gradio and `eval_AIB.py` inside the Docker container.

## Terminal 1

Launch the web interface:

```bash
python scripts/run_gradio_interface.py
```

Open the Gradio URL shown in the terminal in your browser before starting the QA agent.

---

## Terminal 2

Launch the QA agent:

```bash
python scripts/eval_AIB.py \
    --question_source gradio \
    --all_mem True
```

Return to the open Gradio page and start asking questions.

If using a different dataset:

```bash
python scripts/eval_AIB.py \
    --question_source gradio \
    --all_mem True \
    --sequence_id <sequence> \
    --postfix <postfix>
```

---

# Testing the Agentic Workflow

Before testing a sequence, review its annotated keyframes to become familiar with the environment and to design grounded questions:

```text
/workspace/results/<sequence>/annotated_rgb/
```

With Gradio running in Terminal 1 and `eval_AIB.py --question_source gradio` running in Terminal 2, submit questions through the web interface. For example:

- “Where can I park my bike?”

After each response, inspect the retrieved keyframes and reasoning log below to check whether the agent selected relevant visual evidence.

---

# Gradio Visualization

After each query, STaR generates several visualizations.

## Retrieval Score Timeline

**Purpose**

Shows which video-caption memories were retrieved and how relevant they are over time.

**Output**

```
/workspace/results/<sequence>/search_DB/<postfix>/
    retrieval_DB_<idx>_<postfix>.png
```

---

## Retrieved Keyframes

**Purpose**

Displays the visual evidence selected by the agent.

**Output**

```
/workspace/results/<sequence>/images/<postfix>/<idx>/
```

---

## Agent Reasoning Log

**Purpose**

Records the complete reasoning process, including:

- User question
- Retrieval actions
- Selected timestamps
- Retrieved images
- Final reasoning
- Generated answer

The generated answer also reports the timestamp of task-relevant memories and, when available, the associated object ID. Pass these values to the 3D primitive map to retrieve the object's caption and location.

You can also use the predicted object index with `scripts/vis_3D.py` to locate the target object in the 3D visualization.

**Output**

```
/workspace/results/<sequence>/cot_log/<postfix>/
    cot_log_<idx>.txt
```

---

# Option 2 — Run Dataset Evaluation

Before running NaVQA evaluation, follow [CODaDATA.md](CODaDATA.md) to generate the processed NaVQA question list.

Simply run:

```bash
python scripts/eval_AIB.py \
    --question_source dataset \
    --all_mem False
```

For NaVQA dataset evaluation, use `--all_mem False` so retrieval is limited to the question's relevant time range. Gradio uses `--all_mem True` to search the complete memory.

Recommended NaVQA configuration:

```text
--question_source dataset
--all_mem False
--manual_evaluation True
--method star
--llm gpt-4.1
--sequence_id 0
--postfix star
```

---

## Manual vs Automatic Evaluation

### Manual

```bash
--manual_evaluation True
```

The script pauses before each question and lets you:

- Run
- Skip
- Jump to another index
- Quit

### Automatic

```bash
--manual_evaluation False
```

Runs every selected question automatically.

---

## Retrieval Methods

| Method | Description |
|---------|-------------|
| `star` | Fine-grained multimodal RAG |
| `remembr` | Medium-grained text-only RAG |
| `scene_graph` | Coarse-grained scene graph RAG |

---

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--sequence_id` | Dataset / experiment ID | `0` |
| `--all_mem` | Search all memory (`True`) or only the question's relevant time range (`False`) | `True` |
| `--postfix` | Gradio output channel | `CoDa` |
| `--qa_file` | QA filename (without `.json`) | `human_qa` |
| `--caption_file` | Caption filename | `captions_NVILA-Lite-2B` |
| `--scenegraph_file` | Scene graph filename | `full_pcd` |
| `--results` | Results directory | `/workspace/results` |
| `--data_dir` | Dataset directory | `/workspace/data/coda` |
| `--coda_dir` | CODa timestamps | `/workspace/Local_data/CODa` |
