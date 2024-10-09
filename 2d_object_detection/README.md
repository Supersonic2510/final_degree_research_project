# YOLOv8 on Mapillary Traffic Sign Dataset

This project utilizes the Ultralytics YOLOv8 Library to train a 2D object detection model on the Mapillary Traffic Sign dataset.

## Project Overview

This project leverages the **Ultralytics YOLOv8 Library** to train a high-accuracy 2D object detection model specifically for traffic signs. Using the **Mapillary Traffic Sign dataset**, the project aims to detect and classify various traffic signs, contributing to advancements in autonomous driving systems and traffic management.

## Setup Instructions

### Build Docker Image

Before running the Docker application, build the Docker image with the following command:

```bash
docker build -t yolov8 .
```

### Start Docker Application in Detached Mode

Start the Docker application using Docker Compose:

```bash
docker compose up -d
```

### Open Bash Shell in App Container

To interact with the Docker container, open a bash shell:

```bash
docker compose exec app /bin/bash
```

### Create and Activate Python Virtual Environment

Within the container, create a Python virtual environment to manage dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Python Dependencies

Install the necessary Python libraries and dependencies:

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Transform Dataset for YOLOv8 Compatibility

To convert the **Mapillary Traffic Sign Dataset** into a format compatible with YOLOv8, use the provided Jupyter notebook **`mapillary-to-yolo.ipynb`**. This notebook guides you through transforming the dataset to fit the YOLOv8 input requirements.

You can launch this notebook in Jupyter by executing the following commands in the shell:

```bash
jupyter notebook mapillary-to-yolo.ipynb
```

### (Optional) Reduce Dataset

If you wish to reduce the dataset size for faster experimentation or specific use cases, modify the percentage of data used in the code. The script **`delete_and_reduce_dataset.pl`** can help you achieve this.

To execute it, run the following command:

```bash
perl delete_and_reduce_dataset.pl
```

## Training the Model

Train your model using the **`YOLOv8-train.ipynb`** Jupyter notebook. This notebook includes all necessary training commands and configuration options.

Run the following command to launch the training notebook:

```bash
jupyter notebook YOLOv8-train.ipynb
```

## Licensing Information

### Project License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). For more details, see the [LICENSE](LICENSE.md) file.

### Third-Party Licenses

- **YOLOv8 Model:** This project uses the YOLOv8 model, which is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). Any use of this project must comply with the terms of the AGPL-3.0. More information can be found at [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics).
- **Mapillary Traffic Sign Dataset:** This project uses the Mapillary Traffic Sign Dataset, which is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). The dataset itself is not included in this repository. The dataset can be accessed [here](https://www.mapillary.com/dataset/trafficsign).
