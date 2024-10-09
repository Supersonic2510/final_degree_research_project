
# Real-Time Simulation Visualizer with Carla

![Real-Time Simulation GIF](docs/simulation_snippet.gif)

This project utilizes the Carla Open Simulator to test and visualize model performance in real time using a Flask server and Websockets.

## Overview

The goal of this project is to use [Carla Open Simulator](https://carla.org/) as a platform for testing deep learning models in real-time environments. By using a Flask server and Websockets, we can achieve real-time performance visualization and interaction. The system facilitates the integration between Carla simulation data and the models, allowing for dynamic testing and result display.

## Setup Instructions

### Install Python Dependencies

To set up the environment, first install the required Python dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

There are 5 main Jupyter notebooks that are used for different tasks in this project:

- `gather_lidar_data.ipynb`: Used to retrieve data from the Carla Simulator for training purposes.
- `analyze_lidar_data.ipynb`: Used for data analysis of the gathered LiDAR data.
- `display_lidar_data.ipynb`: Contains extra functions to plot and visualize more data from LiDAR.
- `display_voxelnext_model.ipynb`: Used to test the VoxelNeXt model in real-time and evaluate performance metrics.

To run any of these notebooks, start a Jupyter notebook server:

```bash
jupyter notebook
```

Then, open and run the desired notebook for real-time simulation, data analysis, or model evaluation.

### Running the Flask Server

You can also start the Flask server to enable real-time data streaming:

```bash
python app.py
```

The Flask server will handle the communication between Carla and the front-end using Websockets, ensuring real-time updates are displayed.

## Usage

Ensure the Carla Open Simulator is running on your system. Then, you can run the Jupyter notebooks to gather data or visualize the model performance in real time. You can also access the real-time visualization through the Flask server.

## Visualize in Real Time

Access the visualization through the browser by navigating to the following address after the server starts:

```bash
http://localhost:5000
```

You will be able to see the real-time visualization of the model’s performance and make adjustments as necessary.

## Licensing Information

### Project License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). For more details, see the LICENSE file.

### Third-Party Licenses

- Carla Open Simulator: Licensed under the MIT License. More information can be found at [Carla GitHub Repository](https://github.com/carla-simulator/carla).
- Flask: Licensed under the BSD License. More details can be found at [Flask GitHub Repository](https://github.com/pallets/flask).
