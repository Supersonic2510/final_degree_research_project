import numpy as np


def project_point_to_image(x, y, z, image_width=1280, image_height=720, fov=90.0):
    # Camera position translation (camera at (0, 0, -0.5))
    x_c = x
    y_c = y
    z_c = z + 0.5

    # Perspective projection
    fov_rad = np.radians(fov)
    aspect_ratio = image_width / image_height
    half_width = np.tan(fov_rad / 2)
    half_height = half_width / aspect_ratio

    # Normalized device coordinates (NDC)
    x_ndc = x_c / (z_c * half_width)
    y_ndc = y_c / (z_c * half_height)

    # Convert NDC to pixel coordinates
    pixel_x = (x_ndc + 1) / 2 * image_width
    pixel_y = (1 - y_ndc) / 2 * image_height

    return pixel_x, pixel_y


# Example 3D point
x, y, z = 1, 0, 0
pixel_x, pixel_y = project_point_to_image(x, y, z)
print(f'2D Projection: ({pixel_x:.2f}, {pixel_y:.2f})')
