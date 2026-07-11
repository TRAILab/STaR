# NaVQA Dataset Preparation

This guide describes how to prepare the **NaVQA** evaluation dataset for STaR.

## Prerequisites

Before preparing the evaluation dataset, complete the following steps in order:

1. Set up the project environment by following the [**Installation Guide**](INSTALL.md).
2. Complete the [**Memory Construction**](BuildMemory.md) pipeline to generate the required **video captions**.

---

## 1. Download the CODa Dataset

NaVQA is built on the **CODa** dataset. Follow the official [CODa download instructions](https://amrl.cs.utexas.edu/coda/download.html) to install the CODa toolkit and download the dataset.

### Install the CODa Toolkit

```bash
git clone git@github.com:ut-amrl/coda-devkit.git
cd coda-devkit

conda env create -f environment.yml
conda activate coda
```

### Download the Required Sequences

The complete CODa dataset contains **22 sequences**, but STaR only requires the following **7 sequences**:

```text
0, 3, 4, 6, 16, 21, 22
```

Download each sequence using:

```bash
python scripts/download_split.py -d /path/to/CODa -t sequence -se <sequence_id>
```

For example:

```bash
python scripts/download_split.py -d /path/to/CODa -t sequence -se 0
```

For convenience, set the dataset root:

```bash
export CODA_ROOT_DIR=/path/to/CODa
```

> **Storage Requirement**
>
> The processed dataset occupies approximately **335 GB**. During preprocessing, additional LiDAR and intermediate files are downloaded, so we recommend reserving **at least 500 GB** of free disk space.

### Expected Directory Structure

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


## 2 Prepare the NaVQA Evaluation Dataset

> **Note**
>
> This step is only required if you plan to evaluate on **NaVQA**.
>
> > Before proceeding, complete the following prerequisites in order:
> 1. Set up the project Docker environment by following the instructions in [**Installation**](INSTALL.md).
> 2. Complete the [**Memory Construction**](BuildMemory.md) pipeline.

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
