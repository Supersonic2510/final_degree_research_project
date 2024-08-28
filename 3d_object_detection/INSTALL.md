# Model Training Instructions

Follow these steps to train your models:

## Build Docker Image

Before starting the Docker application, you need to build the Docker image. Here’s how you do it:

```
docker build -t once_pcdet .
```
This command builds a Docker image named `once_pcdet` from the Dockerfile in the current directory.

## Start Docker Application in Detached Mode

```
docker compose up -d
```

This command starts your Docker application in detached mode. The `-d` flag allows the containers to run in the background.

## Open Bash Shell in App Container

```
docker compose exec app /bin/bash
```

This command opens a bash shell in the `app` container. You can execute commands directly in the container using this shell.

## Create a Virtual Environment for Python

```
python -m venv .venv
```

This command creates a virtual environment for Python in the current directory. The virtual environment is named `.venv`.

## Activate the Virtual Environment

```
source .venv/bin/activate
```

This command activates the virtual environment. Once activated, any Python packages installed will be installed to this environment.

## Update tools

```
pip install -U pip setuptools wheel
```

This command updates the dependencies to build the other tools.

## Install PyTorch, torchvision, and torchaudio Packages

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This command installs the PyTorch, torchvision, and torchaudio packages from a specific index URL. The URL points to a repository containing precompiled PyTorch wheels for CUDA 11.8.

## Install Python Packages Listed in requirements.txt

```
pip install -r requirements.txt
```

## Install Current Package in Development Mode

```
python setup.py develop
```

This command installs the current package in 'development mode'. This means you can modify the source code and see the changes without having to reinstall the package.

## Change Current Directory to Tools Subdirectory

```
cd tools
```

This command changes the current directory to the `tools` subdirectory.

> **Note**: It's important to run the training and testing commands from inside the `tools` directory. If not, the script may not work properly as it uses the script's location as the path reference. So, always ensure you're in the `tools` directory before starting the training or testing process.

## Run Bash Script to Start Training Process

```
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml
```

This command runs a bash script that starts the training process. `${NUM_GPUS}` should be replaced with the number of GPUs you want to use for training. The `--cfg_file` option specifies the configuration file for the model.

Remember to replace `${NUM_GPUS}` with the actual number of GPUs you want to use for training.

```
python demo.py --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml --ckpt ../output/once_models/voxelnext_ioubranch_large/default/ckpt/latest_model.pth --data_path ../data/once/data/000027/lidar_roof/1616100800400.bin 
```

```
python temporal_visualizer_ONCE.py --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml --ckpt ../output/once_models/voxelnext_ioubranch_large/default/ckpt/latest_model.pth --data_path ../data/once/data/000027  
```

```
python test.py --cfg_file cfgs/once_models/voxelnext_ioubranch_large.yaml --batch_size 8 --eval_all
```
