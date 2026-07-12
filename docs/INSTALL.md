# Run with Docker Container

## Compatibility

The current Docker environment has been tested on:

- **Operating System:** Ubuntu (x86_64)
- **ROS:** ROS 2 Humble
- **CUDA:** 12.8
- **GPU:** NVIDIA RTX 4090 / RTX 5090
- **Docker:** Docker Compose v2

> **Note**
>
> A ROS 2 Jazzy branch for NVIDIA DGX Spark (ARM) will be released in a future update.

---

# 1. Prepare the Docker Environment

## Install Docker

Follow the official installation guides:

- Docker: https://docs.docker.com/get-started/
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

---

## Build the Base Image

From the `docker` directory:

```bash
cd docker
docker build -t base-uv-dev:latest -f Dockerfile.cuda128_humble_uv .
```

The first build installs:

- ROS 2
- CUDA-enabled PyTorch
- Core Python dependencies

> **Note**
>
> The initial build may take around **one hour**, depending on your internet connection and system performance.

---

## Launch the Development Container

```bash
./run_star.sh
```

The script will automatically:

- Use the reusable base image
- Build the STaR image if necessary
- Start the container
- Open an interactive shell

---

# 2. Project Directory Layout

The default `docker-compose.yml` assumes the following workspace structure:

```text
<workspace>/
├── STaR/
│   ├── docker/
│   ├── scripts/
│   │   └── weights/
│   ├── data/
│   └── results/
└── CODa/
```

### Default Volume Mounts

| Host | Container | Purpose |
|------|-----------|----------|
| `STaR/` | `/workspace/star` | Project source code |
| `STaR/scripts/weights/` | `/workspace/star/weights` | Model checkpoints |
| `<workspace>/` | `/workspace/Local_data` | Dataset parent directory |
| `STaR/results/` | `/workspace/results` | Generated outputs |

With this layout, the default `basedir` in `configs/dataset/CODa_docker.yaml` is:

```text
/workspace/Local_data/CODa
```

### Using a Different Directory Layout

If you store the project, dataset, weights, or output directory elsewhere:

1. Edit the **host-side** paths under `volumes:` in:

   ```text
   docker/docker-compose.yml
   ```

2. Keep all **container-side** paths unchanged.

3. Update the `basedir` in:

   ```text
   configs/dataset/CODa_docker.yaml
   ```

4. Recreate the container:

   ```bash
   cd docker
   ./run_star.sh --reset
   ```

---

# 3. Download Third-Party Models

From the repository root:

```bash
./scripts/bash/download_weights.sh
```

The script downloads all required model checkpoints into:

```text
scripts/weights/
```

which is automatically mounted inside the container as:

```text
/workspace/star/weights
```

See `SETUP.md` for more information about the required third-party models.

---

# 4. Container Management

### Reset the Container

```bash
./run_star.sh --reset
```

### Rebuild the Docker Image

```bash
./run_star.sh --rebuild
```

---

# 5. Usage

Before running the **STaR Agentic RAG** evaluation, you must first build the robot's multimodal memory.

Please follow the instructions in:

```text
BuildMemory.md
```

After data collection and memory construction, the generated outputs will be saved in:

```text
results/
```

---

# 6. ROS Configuration & Troubleshooting

## ROS topics are not visible inside the container

Make sure the **host** and the **container** use the same `ROS_DOMAIN_ID`.

Check the current value:

```bash
# On the host
echo $ROS_DOMAIN_ID

# Inside the container
echo $ROS_DOMAIN_ID
```

If necessary, set it on the host before starting the container:

```bash
export ROS_DOMAIN_ID=90
```

Then recreate the container:

```bash
cd docker
./run_star.sh --reset
```

---

## ROS topics are visible but no data is received

Try forcing Fast DDS to use UDP transport:

```bash
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

---

## Unable to visualize the scene graph

On the **host machine** (not inside the container), allow local Docker containers to access the X server:

```bash
xhost +local:
```