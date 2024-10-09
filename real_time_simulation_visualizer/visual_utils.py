import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math

# Initialize a global figure object
global_fig = None


def create_custom_camera():
    # 135 degrees rotation along Z-axis
    z_rotation_angle = math.radians(180)

    # 45 degrees downward tilt (affects z value)
    downward_tilt_angle = math.radians(25)

    # Define the distance from the center for the zoom effect
    distance = 0.8  # Updated zoom distance

    # Calculate the x, y, and z values based on the tilt and rotation
    x = distance * math.cos(z_rotation_angle) * math.cos(downward_tilt_angle)
    y = distance * math.sin(z_rotation_angle) * math.cos(downward_tilt_angle)
    z = distance * math.sin(downward_tilt_angle)

    return dict(
        up=dict(x=0, y=0, z=1),  # Z-axis remains upward
        eye=dict(x=x, y=y, z=z),  # Camera position calculated based on distance and angles
        center=dict(x=0, y=0, z=0)  # Focus on the center
    )

def create_3D_bounding_box(box):
    cx, cy, cz, dx, dy, dz, heading = box
    R = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    corners = np.array([
        [dx / 2, dy / 2, dz / 2],
        [dx / 2, dy / 2, -dz / 2],
        [dx / 2, -dy / 2, dz / 2],
        [dx / 2, -dy / 2, -dz / 2],
        [-dx / 2, dy / 2, dz / 2],
        [-dx / 2, dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, dz / 2],
        [-dx / 2, -dy / 2, -dz / 2]
    ])
    rotated_corners = np.dot(corners, R.T)
    translated_corners = rotated_corners + np.array([cx, cy, cz])
    return translated_corners


def calculate_iou(corners1, corners2):
    def calculate_min_max_coordinates(corners):
        min_coords = np.min(corners, axis=0)
        max_coords = np.max(corners, axis=0)
        return min_coords, max_coords

    def calculate_intersection_volume(min1, max1, min2, max2):
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        if np.any(intersection_min > intersection_max):
            return 0
        intersection_dims = intersection_max - intersection_min
        return np.prod(intersection_dims)

    def calculate_box_volume(min_coords, max_coords):
        dims = max_coords - min_coords
        return np.prod(dims)

    min1, max1 = calculate_min_max_coordinates(corners1)
    min2, max2 = calculate_min_max_coordinates(corners2)

    intersection_volume = calculate_intersection_volume(min1, max1, min2, max2)

    volume1 = calculate_box_volume(min1, max1)
    volume2 = calculate_box_volume(min2, max2)

    union_volume = volume1 + volume2 - intersection_volume

    if union_volume == 0:
        return 0

    iou = intersection_volume / union_volume
    return iou


def filter_boxes(boxes, labels, scores, threshold=0.5):
    corners = list(map(create_3D_bounding_box, boxes))
    N = len(corners)
    indices = list(range(N))
    to_remove = set()

    for i in range(N):
        for j in range(i + 1, N):
            if i in to_remove or j in to_remove:
                continue
            iou = calculate_iou(corners[i], corners[j])
            if iou > threshold:
                if scores[i] > scores[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    remaining_indices = [i for i in indices if i not in to_remove]

    filtered_boxes = [boxes[i] for i in remaining_indices]
    filtered_corners = [corners[i] for i in remaining_indices]
    filtered_labels = [labels[i] for i in remaining_indices]
    filtered_scores = [scores[i] for i in remaining_indices]

    return filtered_boxes, filtered_corners, filtered_labels, filtered_scores


def camera_coord_to_world(camera_coord):
    # Define the transformation matrix
    # X-axis: right
    # Y-axis: up
    # Z-axis: forward
    R_points = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    return camera_coord @ R_points


def world_coord_to_camera(world_coord):
    # Define the transformation matrix
    # X-axis: forward
    # Y-axis: right
    # Z-axis: up
    R_camera = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    return world_coord @ R_camera


def transform_to_2d_system(coord):
    """
    Transform 3D coordinates to 2D system where:
    - x (original) -> nothing
    - y (original) -> x (new)
    - z (original) -> y (new)

    Parameters:
    coord (numpy array): 3D coordinates with shape (n, 3) or (3,)

    Returns:
    numpy array: 2D coordinates with shape (n, 2)
    """
    # Check if the input is a 1D array with shape (3,)
    # Check if the input is empty
    if coord.size == 0:
        return np.empty((0, 2))

    if coord.ndim == 1 and coord.shape[0] == 3:
        return np.array([-coord[1], coord[2]])

    return np.array([-coord[:, 1], coord[:, 2]]).T


# Define the camera frustum vertices
def compute_frustum_vertices(K, z_near, z_far, camera_pos):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Compute frustum vertices in the camera coordinate system
    vertices_near_camera = np.array([
        [-cx * z_near / fx, -cy * z_near / fy, z_near],
        [cx * z_near / fx, -cy * z_near / fy, z_near],
        [cx * z_near / fx, cy * z_near / fy, z_near],
        [-cx * z_near / fx, cy * z_near / fy, z_near]
    ])

    vertices_far_camera = np.array([
        [-cx * z_far / fx, -cy * z_far / fy, z_far],
        [cx * z_far / fx, -cy * z_far / fy, z_far],
        [cx * z_far / fx, cy * z_far / fy, z_far],
        [-cx * z_far / fx, cy * z_far / fy, z_far]
    ])

    # Apply rotation matrix to the vertices
    vertices_near_world = camera_coord_to_world(vertices_near_camera) + camera_pos
    vertices_far_world = camera_coord_to_world(vertices_far_camera) + camera_pos

    return vertices_near_world, vertices_far_world


def filter_points_in_frustum(points, K, z_near, z_far):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    filtered_points = []
    for point in points:
        point_camera = world_coord_to_camera(point[:3])  # Ignore intensity for transformation
        x, y, z = point_camera
        if z_near < z < z_far:
            x_proj = (x * z_near) / z
            y_proj = (y * z_near) / z
            if -cx * z_near / fx < x_proj < cx * z_near / fx and -cy * z_near / fy < y_proj < cy * z_near / fy:
                filtered_points.append(point)

    return np.array(filtered_points)


def filter_boxes_in_frustum(ref_boxes, ref_labels, ref_scores, K, z_near, z_far):
    filtered_boxes = []
    filtered_corners = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(ref_boxes, ref_labels, ref_scores):
        corners = create_3D_bounding_box(box)
        if len(filter_points_in_frustum(corners, K, z_near, z_far)) > 0:
            filtered_boxes.append(box)
            filtered_corners.append(corners)
            filtered_labels.append(label)
            filtered_scores.append(score)

    return np.array(filtered_boxes), np.array(filtered_corners), np.array(filtered_labels), np.array(filtered_scores)


def visualize_lidar_data_3D_real_time(points, ref_boxes, ref_labels, ref_scores, window=False):
    class_info = {
        0: ('', (0, 0, 0)), # Empty for ERROR
        1: ('Car', (255, 0, 0)),  # Red for Car
        2: ('Bus', (0, 255, 0)),  # Green for Bus
        3: ('Truck', (0, 0, 255)),  # Blue for Truck
        4: ('Pedestrian', (255, 255, 0)),  # Yellow for Pedestrian
        5: ('Cyclist', (255, 0, 255))  # Magenta for Cyclist
    }

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    rs = points[:, 3]

    # Find the range for each axis
    x_range = [-75, 75]
    y_range = [-75, 75]
    z_range = [0, 30]

    # Calculate the maximum range for equal scaling
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])

    # Center the ranges around the data mean to ensure equal scaling
    x_center = (x_range[1] + x_range[0]) / 2
    y_center = (y_range[1] + y_range[0]) / 2
    z_center = (z_range[1] + z_range[0]) / 2

    x_range = [x_center - max_range / 2, x_center + max_range / 2]
    y_range = [y_center - max_range / 2, y_center + max_range / 2]
    z_range = [z_center - max_range / 2, z_center + max_range / 2]

    filtered_boxes, filtered_corners, filtered_labels, filtered_scores = filter_boxes(ref_boxes, ref_labels, ref_scores,
                                                                                      threshold=0.3)

    def create_scatter3d(color_scheme):
        if color_scheme == 'monocolor':
            colors = 'rgb(255, 255, 255)'  # All points white
            colorscale = 'Greys'
            colorbar_title = 'Monocolor'
            tickvals = [0, 64, 128, 192, 255]
            ticktext = ['0%', '25%', '50%', '75%', '100%']
        elif color_scheme == 'heightmap':
            colors = zs  # Use height for coloring
            colorscale = 'Turbo'
            colorbar_title = 'Height'
            tickvals = None
            ticktext = None
        else:  # Default to 'default'
            colors = rs  # Use reflection intensity
            colorscale = 'Viridis'
            colorbar_title = 'Reflection Intensity'
            tickvals = None
            ticktext = None

        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(
                size=0.5,
                color=colors,  # Use the selected color scheme
                colorscale=colorscale,  # Use the selected color scale
                colorbar=dict(
                    title=colorbar_title,
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
                opacity=0.6
            ),
            name='LIDAR Points'
        )

    def create_full_figure(color_scheme):
        data = [create_scatter3d(color_scheme)]

        for box, corners, label, score in zip(filtered_boxes, filtered_corners, filtered_labels, filtered_scores):
            label_name, label_color = class_info[label]
            label_color_rgb = f'rgb({label_color[0]}, {label_color[1]}, {label_color[2]})'
            label_text = f"{label_name} ({score:.2f})"

            # Add the edges of the bounding box
            for i in range(4):
                data.append(go.Scatter3d(
                    x=[corners[i][0], corners[(i + 1) % 4][0], corners[(i + 1) % 4 + 4][0], corners[i + 4][0],
                       corners[i][0]],
                    y=[corners[i][1], corners[(i + 1) % 4][1], corners[(i + 1) % 4 + 4][1], corners[i + 4][1],
                       corners[i][1]],
                    z=[corners[i][2], corners[(i + 1) % 4][2], corners[(i + 1) % 4 + 4][2], corners[i + 4][2],
                       corners[i][2]],
                    mode='lines',
                    line=dict(color=label_color_rgb, width=4),
                    showlegend=False
                ))

            # Add label text
            data.append(go.Scatter3d(
                x=[box[0]],
                y=[box[1]],
                z=[box[2]],
                mode='text',
                text=[label_text],
                textposition="top center",
                textfont=dict(size=12, color=label_color_rgb),
                showlegend=False
            ))

        return data

    fig = go.Figure(data=create_full_figure('default'))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='cube'  # Ensures the axes are equally scaled
        ),
        title=dict(
            text='LiDAR Point Cloud',
            font=dict(size=24)  # Increase title font size
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            font=dict(size=12)
        ),
        updatemenus=[dict(
            buttons=list([
                dict(
                    args=[
                        {'marker.color': ['rgb(255, 255, 255)'],
                         'marker.colorscale': 'Greys',
                         'marker.colorbar.title': 'Monocolor',
                         'marker.colorbar.tickvals': [0, 64, 128, 192, 256],
                         'marker.colorbar.ticktext': ['0%', '25%', '50%', '75%', '100%']}
                    ],
                    label="Monocolor",
                    method="restyle"
                ),
                dict(
                    args=[
                        {'marker.color': [zs],
                         'marker.colorscale': 'Turbo',
                         'marker.colorbar.title': 'Height',
                         'marker.colorbar.tickvals': None,
                         'marker.colorbar.ticktext': None}
                    ],
                    label="Heightmap",
                    method="restyle"
                ),
                dict(
                    args=[
                        {'marker.color': [rs],
                         'marker.colorscale': 'Viridis',
                         'marker.colorbar.title': 'Reflection Intensity',
                         'marker.colorbar.tickvals': None,
                         'marker.colorbar.ticktext': None}
                    ],
                    label="Reflection Intensity",
                    method="restyle"
                ),
            ]),
            direction="down",
            showactive=True,
            x=0,
            xanchor="left",
            y=0.95,
            yanchor="top",
            active=2
        )]
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Colorscale",
                x=0,
                xref="paper",
                y=1,
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top"
            )
        ]
    )

    if window:
        pio.renderers.default = 'browser'
    else:
        pio.renderers.default = 'png'  # Use 'png' for PyCharm in-line image

    return fig


def visualize_lidar_data_3D(points, ref_boxes, ref_labels, ref_scores, window=False):
    class_info = {
        0: ('', (0, 0, 0)),  # Empty for ERROR
        1: ('Car', (255, 0, 0)),  # Red for Car
        2: ('Bus', (0, 255, 0)),  # Green for Bus
        3: ('Truck', (0, 0, 255)),  # Blue for Truck
        4: ('Pedestrian', (255, 255, 0)),  # Yellow for Pedestrian
        5: ('Cyclist', (255, 0, 255))  # Magenta for Cyclist
    }

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    rs = points[:, 3]

    # Find the range for each axis
    x_range = [xs.min(), xs.max()]
    y_range = [ys.min(), ys.max()]
    z_range = [zs.min(), zs.max()]

    # Calculate the maximum range for equal scaling
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])

    # Center the ranges around the data mean to ensure equal scaling
    x_center = (x_range[1] + x_range[0]) / 2
    y_center = (y_range[1] + y_range[0]) / 2
    z_center = (z_range[1] + z_range[0]) / 2

    x_range = [x_center - max_range / 2, x_center + max_range / 2]
    y_range = [y_center - max_range / 2, y_center + max_range / 2]
    z_range = [z_center - max_range / 2, z_center + max_range / 2]

    filtered_boxes, filtered_corners, filtered_labels, filtered_scores = filter_boxes(ref_boxes, ref_labels, ref_scores,
                                                                                      threshold=0.3)

    def create_scatter3d(color_scheme):
        if color_scheme == 'monocolor':
            colors = 'rgb(255, 255, 255)'  # All points white
            colorscale = 'Greys'
            colorbar_title = 'Monocolor'
            tickvals = [0, 64, 128, 192, 255]
            ticktext = ['0%', '25%', '50%', '75%', '100%']
        elif color_scheme == 'heightmap':
            colors = zs  # Use height for coloring
            colorscale = 'Turbo'
            colorbar_title = 'Height'
            tickvals = None
            ticktext = None
        else:  # Default to 'default'
            colors = rs  # Use reflection intensity
            colorscale = 'Viridis'
            colorbar_title = 'Reflection Intensity'
            tickvals = None
            ticktext = None

        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(
                size=0.5,
                color=colors,  # Use the selected color scheme
                colorscale=colorscale,  # Use the selected color scale
                colorbar=dict(
                    title=colorbar_title,
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
                opacity=0.6
            ),
            name='LIDAR Points'
        )

    def create_full_figure(color_scheme):
        data = [create_scatter3d(color_scheme)]

        for box, corners, label, score in zip(filtered_boxes, filtered_corners, filtered_labels, filtered_scores):
            label_name, label_color = class_info[label]
            label_color_rgb = f'rgb({label_color[0]}, {label_color[1]}, {label_color[2]})'
            label_text = f"{label_name} ({score:.2f})"

            # Add the edges of the bounding box
            for i in range(4):
                data.append(go.Scatter3d(
                    x=[corners[i][0], corners[(i + 1) % 4][0], corners[(i + 1) % 4 + 4][0], corners[i + 4][0],
                       corners[i][0]],
                    y=[corners[i][1], corners[(i + 1) % 4][1], corners[(i + 1) % 4 + 4][1], corners[i + 4][1],
                       corners[i][1]],
                    z=[corners[i][2], corners[(i + 1) % 4][2], corners[(i + 1) % 4 + 4][2], corners[i + 4][2],
                       corners[i][2]],
                    mode='lines',
                    line=dict(color=label_color_rgb, width=4),
                    showlegend=False
                ))

            # Add label text
            data.append(go.Scatter3d(
                x=[box[0]],
                y=[box[1]],
                z=[box[2]],
                mode='text',
                text=[label_text],
                textposition="top center",
                textfont=dict(size=12, color=label_color_rgb),
                showlegend=False
            ))

        return data

    fig = go.Figure(data=create_full_figure('default'))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='cube'  # Ensures the axes are equally scaled
        ),
        title=dict(
            text='LiDAR Point Cloud',
            font=dict(size=24)  # Increase title font size
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            font=dict(size=12)
        ),
        updatemenus=[dict(
            buttons=list([
                dict(
                    args=[
                        {'marker.color': ['rgb(255, 255, 255)'],
                         'marker.colorscale': 'Greys',
                         'marker.colorbar.title': 'Monocolor',
                         'marker.colorbar.tickvals': [0, 64, 128, 192, 256],
                         'marker.colorbar.ticktext': ['0%', '25%', '50%', '75%', '100%']}
                    ],
                    label="Monocolor",
                    method="restyle"
                ),
                dict(
                    args=[
                        {'marker.color': [zs],
                         'marker.colorscale': 'Turbo',
                         'marker.colorbar.title': 'Height',
                         'marker.colorbar.tickvals': None,
                         'marker.colorbar.ticktext': None}
                    ],
                    label="Heightmap",
                    method="restyle"
                ),
                dict(
                    args=[
                        {'marker.color': [rs],
                         'marker.colorscale': 'Viridis',
                         'marker.colorbar.title': 'Reflection Intensity',
                         'marker.colorbar.tickvals': None,
                         'marker.colorbar.ticktext': None}
                    ],
                    label="Reflection Intensity",
                    method="restyle"
                ),
            ]),
            direction="down",
            showactive=True,
            x=0,
            xanchor="left",
            y=0.95,
            yanchor="top",
            active=2
        )]
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Colorscale",
                x=0,
                xref="paper",
                y=1,
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top"
            )
        ]
    )

    if window:
        pio.renderers.default = 'browser'
    else:
        pio.renderers.default = 'png'  # Use 'png' for PyCharm in-line image

    fig.show()

def visualize_lidar_data_3D_3rd_person_view(points, ref_boxes, ref_labels, ref_scores, window=False, save_path=None):
    class_info = {
        0: ('', (0, 0, 0)),  # Empty for ERROR
        1: ('Car', (255, 0, 0)),  # Red for Car
        2: ('Bus', (0, 255, 0)),  # Green for Bus
        3: ('Truck', (0, 0, 255)),  # Blue for Truck
        4: ('Pedestrian', (255, 255, 0)),  # Yellow for Pedestrian
        5: ('Cyclist', (255, 0, 255))  # Magenta for Cyclist
    }

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    rs = points[:, 3]

    # Find the range for each axis
    x_range = [xs.min(), xs.max()]
    y_range = [ys.min(), ys.max()]
    z_range = [zs.min(), zs.max()]

    # Calculate the maximum range for equal scaling
    max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])

    # Center the ranges around the data mean to ensure equal scaling
    x_center = (x_range[1] + x_range[0]) / 2
    y_center = (y_range[1] + y_range[0]) / 2
    z_center = (z_range[1] + z_range[0]) / 2

    x_range = [x_center - max_range / 2, x_center + max_range / 2]
    y_range = [y_center - max_range / 2, y_center + max_range / 2]
    z_range = [z_center - max_range / 2, z_center + max_range / 2]

    filtered_boxes, filtered_corners, filtered_labels, filtered_scores = filter_boxes(ref_boxes, ref_labels, ref_scores, threshold=0.3)

    def create_scatter3d(color_scheme):
        if color_scheme == 'monocolor':
            colors = 'rgb(255, 255, 255)'  # All points white
            colorscale = 'Greys'
            colorbar_title = 'Monocolor'
            tickvals = [0, 64, 128, 192, 255]
            ticktext = ['0%', '25%', '50%', '75%', '100%']
        elif color_scheme == 'heightmap':
            colors = zs  # Use height for coloring
            colorscale = 'Turbo'
            colorbar_title = 'Height'
            tickvals = None
            ticktext = None
        else:  # Default to 'default'
            colors = rs  # Use reflection intensity
            colorscale = 'Viridis'
            colorbar_title = 'Reflection Intensity'
            tickvals = None
            ticktext = None

        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(
                size=0.5,
                color=colors,  # Use the selected color scheme
                colorscale=colorscale,  # Use the selected color scale
                colorbar=dict(
                    title=colorbar_title,
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
                opacity=0.6
            ),
            name='LIDAR Points'
        )

    def create_full_figure(color_scheme):
        data = [create_scatter3d(color_scheme)]

        for box, corners, label, score in zip(filtered_boxes, filtered_corners, filtered_labels, filtered_scores):
            label_name, label_color = class_info[label]
            label_color_rgb = f'rgb({label_color[0]}, {label_color[1]}, {label_color[2]})'
            label_text = f"{label_name} ({score:.2f})"

            # Add the edges of the bounding box
            for i in range(4):
                data.append(go.Scatter3d(
                    x=[corners[i][0], corners[(i + 1) % 4][0], corners[(i + 1) % 4 + 4][0], corners[i + 4][0], corners[i][0]],
                    y=[corners[i][1], corners[(i + 1) % 4][1], corners[(i + 1) % 4 + 4][1], corners[i + 4][1], corners[i][1]],
                    z=[corners[i][2], corners[(i + 1) % 4][2], corners[(i + 1) % 4 + 4][2], corners[i + 4][2], corners[i][2]],
                    mode='lines',
                    line=dict(color=label_color_rgb, width=4),
                    showlegend=False
                ))

            # Add label text
            data.append(go.Scatter3d(
                x=[box[0]],
                y=[box[1]],
                z=[box[2]],
                mode='text',
                text=[label_text],
                textposition="top center",
                textfont=dict(size=12, color=label_color_rgb),
                showlegend=False
            ))

        return data

    fig = go.Figure(data=create_full_figure('default'))

    # Update layout with custom camera settings
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title='X', range=x_range),
            yaxis=dict(title='Y', range=y_range),
            zaxis=dict(title='Z', range=z_range),
            aspectmode='cube',  # Ensures the axes are equally scaled
            camera=create_custom_camera()  # Custom camera settings
        ),
        title=dict(
            text='LiDAR Point Cloud',
            font=dict(size=24)  # Increase title font size
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            font=dict(size=12)
        ),
        updatemenus=[dict(
            buttons=list([
                dict(
                    args=[
                        {'marker.color': ['rgb(255, 255, 255)'],
                         'marker.colorscale': 'Greys',
                         'marker.colorbar.title': 'Monocolor',
                         'marker.colorbar.tickvals': [0, 64, 128, 192, 256],
                         'marker.colorbar.ticktext': ['0%', '25%', '50%', '75%', '100%']}
                    ],
                    label="Monocolor",
                    method="restyle"
                ),
                dict(
                    args=[
                        {'marker.color': [zs],
                         'marker.colorscale': 'Turbo',
                         'marker.colorbar.title': 'Height',
                         'marker.colorbar.tickvals': None,
                         'marker.colorbar.ticktext': None}
                    ],
                    label="Heightmap",
                    method="restyle"
                ),
                dict(
                    args=[
                        {'marker.color': [rs],
                         'marker.colorscale': 'Viridis',
                         'marker.colorbar.title': 'Reflection Intensity',
                         'marker.colorbar.tickvals': None,
                         'marker.colorbar.ticktext': None}
                    ],
                    label="Reflection Intensity",
                    method="restyle"
                ),
            ]),
            direction="down",
            showactive=True,
            x=0,
            xanchor="left",
            y=0.95,
            yanchor="top",
            active=2
        )]
    )

    fig.update_layout(
        annotations=[
            dict(
                text="Colorscale",
                x=0,
                xref="paper",
                y=1,
                yref="paper",
                showarrow=False,
                xanchor="left",
                yanchor="top"
            )
        ]
    )

    if window:
        pio.renderers.default = 'browser'
    else:
        pio.renderers.default = 'png'  # Use 'png' for PyCharm in-line image

    fig.show()

    if save_path:
        # Save the figure directly as a PNG using Kaleido
        fig.write_image(save_path, engine='kaleido')
        print(f"Plot saved at {save_path}")

def visualize_lidar_data_2D(image, ref_boxes, ref_labels, ref_scores, K, z_near, z_far, window=False):
    class_info = {
        0: ('', (0, 0, 0)),  # Empty for ERROR
        1: ('Car', (255, 0, 0)),  # Red for Car
        2: ('Bus', (0, 255, 0)),  # Green for Bus
        3: ('Truck', (0, 0, 255)),  # Blue for Truck
        4: ('Pedestrian', (255, 255, 0)),  # Yellow for Pedestrian
        5: ('Cyclist', (255, 0, 255))  # Magenta for Cyclist
    }

    filtered_boxes, filtered_corners, filtered_labels, filtered_scores = filter_boxes(ref_boxes, ref_labels, ref_scores,threshold=0.3)

    filtered_boxes, filtered_corners, filtered_labels, filtered_scores = filter_boxes_in_frustum(filtered_boxes, filtered_labels, filtered_scores, K, z_near, z_far)

    # Define image size
    image_height, image_width, _ = image.shape

    def create_image_trace():
        return go.Image(z=image, x0=0, y0=0, dx=1, dy=1)

    def create_full_figure():
        data = [create_image_trace()]

        for box, corners, label, score in zip(filtered_boxes, filtered_corners, filtered_labels, filtered_scores):
            label_name, label_color = class_info[label]
            label_color_rgb = f'rgb({label_color[0]}, {label_color[1]}, {label_color[2]})'
            label_text = f"{label_name} ({score:.2f})"

            # Transform corners to 2D
            corners_2d = transform_to_2d_system(corners)

            # Add the edges of the bounding box
            for i in range(4):
                data.append(go.Scatter(
                    x=[corners_2d[i][0], corners_2d[(i + 1) % 4][0], corners_2d[(i + 1) % 4 + 4][0],
                       corners_2d[i + 4][0], corners_2d[i][0]],
                    y=[corners_2d[i][1], corners_2d[(i + 1) % 4][1], corners_2d[(i + 1) % 4 + 4][1],
                       corners_2d[i + 4][1], corners_2d[i][1]],
                    mode='lines',
                    line=dict(color=label_color_rgb, width=2),
                    showlegend=False
                ))

            # Add label text
            box_2d = transform_to_2d_system(box)
            data.append(go.Scatter(
                x=[box_2d[0]],
                y=[box_2d[1]],
                mode='text',
                text=[label_text],
                textposition="top center",
                textfont=dict(size=12, color=label_color_rgb),
                showlegend=False
            ))

        return data

    fig = go.Figure(data=create_full_figure())

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(title='X', range=[0, image_width], showgrid=False, zeroline=False),
        yaxis=dict(title='Y', range=[image_height, 0], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        title=dict(
            text='LiDAR Point Cloud (2D with Image)',
            font=dict(size=24)  # Increase title font size
        ),
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            font=dict(size=12)
        ),
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )

    if window:
        pio.renderers.default = 'browser'
    else:
        pio.renderers.default = 'png'  # Use 'png' for PyCharm in-line image

    fig.show()


def visualize_lidar_data_3D_static(points, ref_boxes, ref_labels, ref_scores):
    class_info = {
        0: ('', (0, 0, 0)),  # Empty for ERROR
        1: ('Car', (255, 0, 0)),  # Red for Car
        2: ('Bus', (0, 255, 0)),  # Green for Bus
        3: ('Truck', (0, 0, 255)),  # Blue for Truck
        4: ('Pedestrian', (255, 255, 0)),  # Yellow for Pedestrian
        5: ('Cyclist', (255, 0, 255))  # Magenta for Cyclist
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    rs = points[:, 3]

    sc = ax.scatter(xs, ys, zs, c=rs, cmap='viridis', alpha=0.6, s=0.1)
    plt.colorbar(sc, label='Reflection Intensity')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LiDAR Point Cloud (Static 3D)')

    filteres_boxes, filtered_corners, filtered_labels, filtered_scores = filter_boxes(ref_boxes, ref_labels, ref_scores,
                                                                                      threshold=0.3)

    # Adding bounding boxes and labels
    for box, corners, label, score in zip(filteres_boxes, filtered_corners, filtered_labels, filtered_scores):
        label_name, label_color = class_info[label]
        label_text = f"{label_name} ({score:.2f})"

        # Convert label color to matplotlib format
        label_color = tuple([c / 255.0 for c in label_color])

        # Add the edges of the bounding box
        edges = [
            [corners[0], corners[1]], [corners[1], corners[3]], [corners[3], corners[2]], [corners[2], corners[0]],
            # Bottom face
            [corners[4], corners[5]], [corners[5], corners[7]], [corners[7], corners[6]], [corners[6], corners[4]],
            # Top face
            [corners[0], corners[4]], [corners[1], corners[5]], [corners[2], corners[6]], [corners[3], corners[7]]
            # Side edges
        ]

        for edge in edges:
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], color=label_color,
                    linewidth=2)

        # Add label text
        ax.text(box[0], box[1], box[2], label_text, color=label_color, fontsize=3, ha='center')

    # Ensure the axes are equally scaled
    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max()
    Xb = 0.5 * max_range * np.array([-1, 1])
    Yb = 0.5 * max_range * np.array([-1, 1])
    Zb = 0.5 * max_range * np.array([-1, 1])
    ax.set_xlim(Xb + xs.mean())
    ax.set_ylim(Yb + ys.mean())
    ax.set_zlim(Zb + zs.mean())

    plt.show()
