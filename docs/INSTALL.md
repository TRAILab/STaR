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

   Before running the script, configure the project, model, and dataset mounts
   in `docker-compose.yml` as described below. The script then uses the base
   image, builds the STaR image when necessary, starts the container, and opens
   an interactive shell.

## Configure host mounts

The container accesses project files, model weights, and datasets through the
volume mounts in `docker-compose.yml`. Update the host-side paths there to
match your local setup.
## Prepare Third Party Models
The download script is located at `scripts/bash/download_weights.sh`. From the
repository root, run:

```bash
chmod +x scripts/bash/download_weights.sh
./scripts/bash/download_weights.sh
```

By default, the script downloads all weights to `scripts/weights/`. Update the
model-weight volume mount in `docker-compose.yml` to use that directory. See
the [installation instructions](SETUP.md) for details about the required models.
## After Preparation
After you prepared the container and the models, the file structure should look like this:
```
/workspace/
|
+---star
|   |
|   +--- <source code of the project>
|
+---third_parties
    |
    +---4DMOS
    |
    +---VILA
    |
    +---GroundingDINO
    |
    +---recognize-anything
    |
    +---tokenize-anything
```
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
## Notes

1. **ROS 2 topics are not visible inside the container**

   Make sure the host and the container are using the same `ROS_DOMAIN_ID`.

   Check the domain ID:
   ```bash
   # On the host
   echo $ROS_DOMAIN_ID

   # Inside the container
   echo $ROS_DOMAIN_ID
   ```

   If necessary, set the same value in both environments (replace `90` with your desired domain ID):
   ```bash
   export ROS_DOMAIN_ID=90
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
