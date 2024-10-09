import numpy as np
import plotly
from dash_canvas.utils import array_to_data_url
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.io as pio
import json
import os
from visual_utils import visualize_lidar_data_3D_real_time
from threading import Thread


class DisplayServer:
    def __init__(self, template_folder='assets', static_folder='assets'):
        # Initialize the Flask app
        self.app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)

        # Initialize SocketIO with the Flask app
        self.socketio = SocketIO(self.app)

        # Register the routes and events
        self.register_routes()
        self.register_events()

        # Initialize with some default data
        self.points = np.zeros((1, 4))  # Initial point cloud data
        self.ref_boxes = np.zeros((1, 7))  # Initial reference boxes
        self.ref_labels = np.zeros(1)  # Initial labels
        self.ref_scores = np.zeros(1)  # Initial scores

        # Initialize images as empty (n, m, 4) ndarrays
        self.front_image = np.zeros((1080, 1920, 4))  # Default empty image
        self.rear_image = np.zeros((1080, 1920, 4))  # Default empty image
        self.left_image = np.zeros((1080, 1920, 4))  # Default empty image
        self.right_image = np.zeros((1080, 1920, 4))  # Default empty image

    def update_data(self, points, ref_boxes, ref_labels, ref_scores, front_image, rear_image, left_image, right_image):
        # Extract the data
        self.points = points
        self.ref_boxes = ref_boxes
        self.ref_labels = ref_labels
        self.ref_scores = ref_scores
        self.front_image = front_image
        self.rear_image = rear_image
        self.left_image = left_image
        self.right_image = right_image

    def register_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

    def register_events(self):
        @self.socketio.on('update_data')
        def handle_update_data():
            # Call the function that returns the full Plotly figure
            fig = visualize_lidar_data_3D_real_time(
                self.points,
                self.ref_boxes,
                self.ref_labels,
                self.ref_scores,
                True
            )
            # Convert the plot to HTML
            fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Send the HTML to update the chart
            emit(
                'update_chart',
                {
                    'lidar_fig': fig_json,
                    'front_image': array_to_data_url(self.front_image, dtype=np.uint8),
                    'rear_image': array_to_data_url(self.rear_image, dtype=np.uint8),
                    'left_image': array_to_data_url(self.left_image, dtype=np.uint8),
                    'right_image': array_to_data_url(self.right_image, dtype=np.uint8),
                },
                broadcast=True)

        @self.socketio.on('client_ok')
        def handle_client_ok(data):
            if data['status'] == 'OK':
                # Emit 'ok' to the Python client
                self.socketio.emit('server_ok', {'status': 'OK'}, broadcast=True)

    def start(self, debug=True):
        # Start the Flask server with SocketIO in a separate thread
        self.server_thread = Thread(target=self.socketio.run, args=(self.app,),
                                    kwargs={'debug': debug, 'use_reloader': False})
        self.server_thread.start()

    def stop(self):
        # Stop the server thread if necessary
        func = request.environ.get('werkzeug.server.shutdown')
        if func is not None:
            func()
        self.server_thread.join()


# Example usage in a Jupyter Notebook
if __name__ == '__main__':
    server = DisplayServer()
    server.start()