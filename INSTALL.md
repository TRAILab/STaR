# Run with Docker Container
## Prepare Docker Container
1. Follow the instructions [here](https://docs.docker.com/get-started/) to install docker on your device.
2. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install the NVIDIA container toolkit.
3. The project source code and third-party models need to be mounted into the container. You can navigate to the docker folder, then run `./run_star.sh` to start a container with code mounted.
## Mount your files to the container
You can navigate to `docker-compose.yml` to modify the files that you want to mount to the container.
## Prepare Third Party Models
You can check the [install instruction](SETUP.md) for more details about the required models. You can download the weights on your host machine using `scripts/bash/download_weights.sh` and then mount to the container, modify the docker-compose.yml to configure the mount.
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
Check [here](BuildMemory.md) for more details about the data collection pipeline. 

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
