# Run with Docker Container
## Prepare Docker Container
1. Follow the instructions [here](https://docs.docker.com/get-started/) to install docker on your device.
2. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install the NVIDIA container toolkit.
3. Build and start the development container in two steps. First, build the
   reusable base image from the repository root:

   ```bash
   cd docker
   docker build -t base-uv-dev:latest -f Dockerfile.cuda128_humble_uv .
   ```

   This initial build installs ROS, CUDA-enabled PyTorch, and core Python
   dependencies. It can take about one hour, depending on your internet
   connection and system performance.

   Then, from the `docker` directory, start the STaR development container:

   ```bash
   ./run_star.sh
   ```

   With the default directory layout below, no changes to
   `docker/docker-compose.yml` are required. The script uses the base image,
   builds the STaR image when necessary, starts the container, and opens an
   interactive shell.

## Default host layout and mounts

The supplied Compose configuration uses paths relative to
`STaR/docker/docker-compose.yml`. Keep the repository and dataset in this
layout to run without editing the Compose file:

```text
<workspace>/
├── STaR/                         # this repository
│   ├── docker/
│   ├── scripts/weights/           # created by download_weights.sh
│   ├── data/                      # optional project data
│   └── results/                   # generated outputs
└── CODa/                          # dataset directory
```

The default mounts are:

| Host path | Container path | Purpose |
| --- | --- | --- |
| `STaR/` | `/workspace/star` | Project source code |
| `STaR/scripts/weights/` | `/workspace/star/weights` | Model checkpoints |
| `<workspace>/` | `/workspace/Local_data` | Dataset parent directory |
| `STaR/results/` | `/workspace/results` | Generated results |

With this layout, the default `basedir` in
`configs/dataset/CODa_docker.yaml` is `/workspace/Local_data/CODa`.

If you store the project, weights, dataset, or results elsewhere, edit only
the host-side (left-hand) paths under `volumes:` in
`docker/docker-compose.yml`. Keep the container-side paths unchanged, update
`basedir` in the dataset YAML if needed, then recreate the container:

```bash
cd docker
./run_star.sh --reset
```

## Prepare Third Party Models
The download script is located at `scripts/bash/download_weights.sh`. From the
repository root, run:

```bash
./scripts/bash/download_weights.sh
```

The script downloads weights to `scripts/weights/`, which is already mounted
at `/workspace/star/weights` by the default Compose configuration. See the
[installation instructions](SETUP.md) for details about the required models.
### Reset the container
If you want to reset the container, you can run 
```bash id="m5o5mc"
./run_star.sh --reset
```
### Rebuild the image
If you want to rebuild the image, you can run 
```bash id="m5o5mc"
./run_star.sh --rebuild
```
## Usage
Before running the [STaR Agentic RAG](UserQuery.md) evaluation, you need to first build the robot's multimodal memory. Please follow the instructions in [Memory Construction](BuildMemory.md).

After you collect the data with the code run in container, you should be about to see a folder called `result`.
## Before Running: ROS Configuration and Troubleshooting

1. **ROS 2 topics are not visible inside the container**

   Make sure the host and the container use the same `ROS_DOMAIN_ID`. The
   Compose file passes the host value into the container and defaults to `0`
   when it is unset.

   Check the domain ID:
   ```bash
   # On the host
   echo $ROS_DOMAIN_ID

   # Inside the container
   echo $ROS_DOMAIN_ID
   ```

   If necessary, set it on the host before starting the container (replace
   `90` with your desired domain ID):
   ```bash
   export ROS_DOMAIN_ID=90
   ```

   Recreate the container after changing this value:
   ```bash
   cd docker
   docker compose up -d --force-recreate
   ```

2. **ROS 2 topics are visible, but no data is received**

   Try forcing Fast DDS to use UDP transport:
   ```bash
   export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
   ```

3. **Unable to visualize the scene graph from the Docker container**

   On the **host machine**, allow local Docker containers to access the X server:
   ```bash
   xhost +local:
   ```

4. **Compatibility**

   This Docker image has been tested with:
   - CUDA 12.8
   - Docker Compose v2
