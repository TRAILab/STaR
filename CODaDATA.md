For evaluation on NaVQA, we provide the following download instructions and an overview of the overall data structure.

## Download and preprocess the CODa dataset
First download the relevant subsets of the [CODa dataset](https://amrl.cs.utexas.edu/coda/), which consists of 22 sequences. 

We only need 7 of them which are `0, 3, 4, 6, 16, 21, 22`. These numbers will be referred to as sequence IDs. Each sequence ID has 30 questions associated with it.

> Because of the number of videos, be sure to have a large amount of storage. The processed dataset is ~335GB, but since the pre-processing phase also downloads LiDAR and other outputs, we would recommend having ~500GB extra storage.


Clone the [CODa-devkit](https://github.com/ut-amrl/coda-devkit) and download a sequence:

```bash
git clone git@github.com:ut-amrl/coda-devkit.git
cd coda-devkit

# Download sequence 0 (≈15 GB)
python scripts/download_split.py -d /path/to/CODa -t sequence -se 0
```

Pre-defined splits (`tiny`, `small`, `medium`, `full`) are also available — see the devkit README.

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
├── 3d_raw/          
│   ├── cam0/
│   └── cam0_undist/
├── calibrations/     # Per-sequence camera intrinsics & extrinsics
│   └── <seq>/
├── poses/
│   └── dense_global/ # Global pose estimates (used for TF)
└── timestamps/       # Per-sequence timestamp files
```
