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
| **Dataset Evaluation** | Evaluate an existing NaVQA question file. |
| **Interactive Gradio** | Ask live questions through the web interface. |

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
postfix: "CoDa"
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
| Frame timestamps | `/workspace/Local_data/CODa/timestamps/<sequence>.txt` |
| QA file | `/workspace/data/coda/questions/<sequence>/<qa_file>.json` |

---

# Option 1 — Run Dataset Evaluation

Simply run:

```bash
python scripts/eval_AIB.py
```

Default configuration:

```text
--question_source dataset
--manual_evaluation True
--method star
--llm gpt-4.1
--sequence_id 0
--postfix CoDa
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
| `star` | STaR Agentic RAG (recommended) |
| `remembr` | Vanilla ReMEmbR baseline |
| `scene_graph` | Scene graph retrieval baseline |

---

## Common Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--sequence_id` | Dataset / experiment ID | `0` |
| `--postfix` | Gradio output channel | `CoDa` |
| `--qa_file` | QA filename (without `.json`) | `human_qa` |
| `--caption_file` | Caption filename | `captions_NVILA-Lite-2B` |
| `--scenegraph_file` | Scene graph filename | `full_pcd` |
| `--results` | Results directory | `/workspace/results` |
| `--data_dir` | Dataset directory | `/workspace/data/coda` |
| `--coda_dir` | CODa timestamps | `/workspace/Local_data/CODa` |

---

# Option 2 — Ask Live Questions (Gradio)

## Terminal 1

Launch the web interface:

```bash
python scripts/run_gradio_interface.py
```

---

## Terminal 2

Launch the QA agent:

```bash
python scripts/eval_AIB.py \
    --question_source gradio
```

Then open the Gradio URL in your browser and start asking questions.

If using a different dataset:

```bash
python scripts/eval_AIB.py \
    --question_source gradio \
    --sequence_id <sequence> \
    --postfix <postfix>
```

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

**Output**

```
/workspace/results/<sequence>/cot_log/<postfix>/
    cot_log_<idx>.txt
```
