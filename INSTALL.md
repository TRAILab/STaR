# Run with Docker Container
## Prepare Docker Container
1. Follow the instructions [here](https://docs.docker.com/get-started/) to install docker on your device.
2. Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install the NVIDIA container toolkit.
3. The project source code and third-party models need to be mounted into the container. You can navigate to the docker folder, then run `./run_star.sh` to start a container with code mounted.
## Mount your files to the container
You can navigate to `docker-compose.yml` to modify the files that you want to mount to the container.
## Prepare Third Party Models
You can check the [install instruction](./INSTALL.md) for more details about the required models. You can download the weights on your host machine using `scripts/bash/download_weights.sh` and then mount to the container, modify the docker-compose.yml to configure the mount.
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
Check [here](DATA_COLLECTION.md) for more details about the data collection pipeline. 

After you collect the data with the code run in container, you should be about to see a folder called `result`.
## Note
1. If you cannot see the ros2 topic in the container, check ROS_DOMIAN_ID on both host and the container, the host and the container should have the same domain id
2. If you can see the topic but cannot get the actually data from the topic, you can try `export FASTDDS_BUILTIN_TRANSPORTS=UDPv4`
3. If you want to visualize the scenegraph when using the container, got to the host machine and then run `xhost +local:`
4. This docker image is designed for CUDA12.8 and docker-compose 2
