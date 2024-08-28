
# VoxelNext on ONCE Dataset

This project utilizes the OpenPCDet framework to train a 3D object detection model on the ONCE dataset using the VoxelNext neural network model.

## Overview

The aim of this project is to leverage state-of-the-art deep learning techniques for LiDAR-based 3D object detection. By using the OpenPCDet framework, we can train and evaluate models on the ONCE dataset to achieve high-accuracy detection results.

## Setup Instructions

### Build Docker Image

Before starting the Docker application, you need to build the Docker image. Here’s how you do it:

```bash
docker build -t once_pcdet .
```

### Start Docker Application in Detached Mode

```bash
docker compose up -d
```

### Open Bash Shell in App Container

```bash
docker compose exec app /bin/bash
```

### Create and Activate Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### Install the Package in Development Mode

```bash
python setup.py develop
```

### Training the Model

Navigate to the `tools` directory and start the training process:

```bash
cd tools
```

It's important to run the training and testing commands from within this directory because the script uses the script's location as the path reference.

> **Note:** Always ensure you're in the `tools` directory before starting the training or testing process. This is crucial for the correct functioning of the scripts.

## Run Bash Script to Start the Training Process

```bash
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml
```

## Usage

### Run Demo

```bash
python demo.py --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml --ckpt ../output/once_models/voxelnext_ioubranch_large/default/ckpt/latest_model.pth --data_path ${DATA_PATH_BIN_FILE}
```

### Test All Epochs

```bash
python test.py --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml --batch_size ${BATCH_SIZE} --eval_all
```

### Export to PyTorch Format

```bash
python pytorch_export.py --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml --ckpt ../output/once_models/voxelnext_ioubranch_large/default/ckpt/latest_model.pth --data_path ${DATA_PATH_BIN_FILE} --output_dir ${OUTPUT_DIR}
```

## Licensing Information

### Project License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). For more details, see the [LICENSE](LICENSE.md) file.

### Third-Party Licenses

- **OpenPCDet Framework:** Licensed under the Apache License Version 2.0. More information can be found at [OpenPCDet GitHub Repository](https://github.com/open-mmlab/OpenPCDet).
- **VoxelNext Model:** Also licensed under the Apache License Version 2.0. More information is available at [VoxelNeXt GitHub Repository](https://github.com/dvlab-research/VoxelNeXt).
- **ONCE Dataset:** Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). More details at [ONCE Dataset Website](https://once-for-auto-driving.github.io/).

### Attribution

This project builds upon the following works:
- [OpenPCDet Framework](https://github.com/open-mmlab/OpenPCDet)
- [VoxelNeXt Neural Network](https://github.com/dvlab-research/VoxelNeXt)
- [ONCE Dataset](https://once-for-auto-driving.github.io/)
