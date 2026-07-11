# Install
This project requires several libraries and model weights.

If you use the provided Docker environment, they will be installed automatically.
Otherwise, you will need to install them manually on the host machine.

The required Python dependencies are listed in `pyproject.toml`.

### Install TAG2TEXT Model
```bash
mkdir third_parties & cd third_parties
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

Download pretrained weights
```bash
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```
### Install Grounding DINO Model
```
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install --no-build-isolation -e GroundingDINO
```
Download pretrained weights
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Install TAP Model
Follow the [instructions](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#installation) to install the TAP model and download the pretrained weights [here](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#models).
### Install SBERT Model
```bash
pip install -U sentence-transformers
```
Download pretrained weights
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```
### Install 4DMOS
Follow the [instruction](https://github.com/PRBonn/4DMOS/tree/96a40a189c37eb03e17f7666a88485064b766c45) of 4DMOS to install it.
We use the old version of 4DMOS. We use the [weight 10_scans.ckpt](https://www.ipb.uni-bonn.de/html/projects/4DMOS/10_scans.zip)
## Required Libraries for Memory Retrieval
### Install Flash Attention
Visit the [FlashAttention release page](https://github.com/Dao-AILab/flash-attention/releases/) and download the version compatible with your CUDA and PyTorch setup.
<br>The following command is just a referenece:
```
python -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Prepare NVILA
Clone the repository [here](https://github.com/NVlabs/VILA), then navigate to NVILA directory
```bash
pip install -e ".[train,eval]"

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

pip install protobuf==3.20.*
```
### Other Required Libraries
Run `pip install -r requirements.txt` to install the remaining libraries.

## Environment Note
Some dependencies across different modules may have version conflicts. <br>

On certain systems, it may still work when installed in a single environment, but this is not guaranteed.

