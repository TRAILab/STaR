# RUNNING
# Step 1: Multimodal Structured Memory Construction

## 1. Configure the Memory Save Directory

Set the output directory for the constructed memory in:

```bash
/star_uv/configs/config.yaml
```

Modify the memory folder name (example using sequence `3`):

```yaml
sequence: "0" 
```

It is recommended to keep the memory folder name consistent with the dataset sequence ID.

---

## 2. Launch the Docker Container

Open a new terminal and run:

```bash
cd docker/
./run_star.sh
```

---

## 3. Start the Memory Construction Pipeline

Inside the Docker container:

```bash
cd star
python scripts/run_data_collection_lidar.py
```

This script constructs the multimodal structured memory from ROS topics.

---

## 4. Configure Dataset Path and Sequence

Set the dataset configuration in:

```bash
/star_uv/configs/dataset/CODa_docker.yaml
```

### Configure the dataset root path

```yaml
basedir: /path/to/CODa
```

> Make sure your local dataset directory is mounted inside the Docker container through:
>
> ```bash
> /star_uv/docker/docker-compose.yml
> ```

### Configure the sequence ID

```yaml
sequence: !!str 0
```

The `sequence` corresponds to the ROS topics published by:

```bash
write_coda_rosbag.py
```

---

## 5. Publish ROS Topics from the Dataset

Open another terminal and launch the Docker container again:

```bash
cd docker/
./run_star.sh
```

Then publish the dataset as ROS topics:

```bash
python scripts/write_coda_rosbag.py
```

This will replay the selected CODa sequence and publish sensor data through ROS topics for memory construction.
