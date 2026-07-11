For evaluation on NaVQA, we provide the following download instructions and an overview of the overall data structure.

### Download the Dataset

Follow the instructions on the [CODa dataset](https://amrl.cs.utexas.edu/coda/download.html) website to set up the Conda environment and download the required data sequences.

The full CODa dataset contains 22 sequences, but this project only requires the following seven sequences:

```text
0, 3, 4, 6, 16, 21, 22
```

Throughout this repository, these numbers are referred to as **sequence IDs**. Each sequence ID is associated with **30 question-answer pairs** for evaluation.

> Because of the number of videos, be sure to have a large amount of storage. The processed dataset is ~335GB, but since the pre-processing phase also downloads LiDAR and other outputs, we would recommend having ~500GB extra storage.


Clone the [CODa-devkit](https://github.com/ut-amrl/coda-devkit) and download a sequence:

```bash
# Download sequence 0 (≈15 GB)
conda activate coda

python scripts/download_split.py -d /path/to/CODa -t sequence -se 0
```

Set the dataset root for convenience:

```bash
export CODA_ROOT_DIR=/path/to/CODa
```

Expected directory structure after download:

```
CODa/
├── 2d_rect/          # Rectified camera images
│   ├── cam0/
│   │   └── <seq>/    # Numbered frames: 000000.png, 000001.png, ...
│   └── cam1/
├── 3d_comp/          
│   ├── os1/
│   │   └── <seq>/    # Numbered pcds: 3d_comp_os1_0_0.bin, 3d_comp_os1_0_1.bin
├── calibrations/     
│   └── <seq>/        # Per-sequence camera intrinsics & extrinsics
├── poses/
│   └── dense_global/ # Global pose estimates (used for TF)
└── timestamps/       # Per-sequence timestamp files
```

## Prepare the NaVQA Evaluation Dataset

> **Note**
>
> This step is only required if you plan to evaluate on **NaVQA**.
>
> Complete the [**Memory Construction**](BuildMemory.md) pipeline first, then return to this section. The preprocessing script requires the generated **video captions**.

### 1. Verify the QA annotation file

Ensure the following file exists:

```text
star_uv/data/coda/navqa/data.csv
```

This file contains the original NaVQA questions and answers that will be converted into the format required by STaR.

### 2. Convert the questions to the STaR format

Run the preprocessing script using the video caption file generated during memory construction:

```bash
python scripts/question_scripts/form_question_jsons.py \
    --caption_file caption_<captioner_name>
```

For example:

```bash
python scripts/question_scripts/form_question_jsons.py \
    --caption_file caption_NVILA-Lite-2B
```

This script also computes the **optimal context** for each question based on the selected captioner and caption interval. We recommend generating video captions every **3 seconds**.

### 3. Verify the output

After preprocessing completes, the following directory should be created:

```text
data/questions/
```

This directory contains the processed NaVQA questions used for evaluation.
