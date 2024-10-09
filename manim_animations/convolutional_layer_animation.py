from manim import *
from PIL import Image
import numpy as np

# Define a Gaussian-like kernel (approximating an RBF kernel)
def gaussian_kernel(size, sigma=1.0):
    ax = np.linspace(-(size - 1) // 2, (size - 1) // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

# Define a Ridge (Laplacian) filter
ridge_kernel = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]])

# Function to apply convolution with the given kernel
def convolve_2d(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2  # Padding size
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
    result = np.zeros_like(image)

    # Apply the kernel to each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = np.sum(padded_image[i:i + kernel_size, j:j + kernel_size] * kernel)
    
    return result

# Function to perform Gaussian convolution on an image with variable kernel size
def gaussian_convolution(img_array, kernel_size=3):
    # Create a Gaussian kernel of the specified size
    kernel = gaussian_kernel(kernel_size, sigma=1.0)

    # Apply convolution on each channel separately (R, G, B)
    convolved_img = np.zeros_like(img_array)
    for channel in range(3):  # R, G, B channels
        convolved_img[:, :, channel] = convolve_2d(img_array[:, :, channel], kernel)
    
    return convolved_img.astype(np.uint8)

# Function to perform Ridge convolution on an image using the predefined ridge kernel
def ridge_convolution(img_array):
    # Apply convolution on each channel separately (R, G, B)
    convolved_img = np.zeros_like(img_array)
    for channel in range(3):  # R, G, B channels
        convolved_img[:, :, channel] = convolve_2d(img_array[:, :, channel], ridge_kernel)
    
    return convolved_img.astype(np.uint8)

# Function to perform average pooling
def average_pooling_image(convolved_img, pool_size=2):
    output_shape = (convolved_img.shape[0] // pool_size, convolved_img.shape[1] // pool_size, convolved_img.shape[2])
    pooled_img = np.zeros(output_shape)
    
    for i in range(0, convolved_img.shape[0], pool_size):
        for j in range(0, convolved_img.shape[1], pool_size):
            for channel in range(3):  # For each channel R, G, B
                pooled_img[i//pool_size, j//pool_size, channel] = np.mean(convolved_img[i:i + pool_size, j:j + pool_size, channel])
    
    return pooled_img.astype(np.uint8)

# Function to perform max pooling
def max_pooling_image(convolved_img, pool_size=2):
    output_shape = (convolved_img.shape[0] // pool_size, convolved_img.shape[1] // pool_size, convolved_img.shape[2])
    pooled_img = np.zeros(output_shape)
    
    for i in range(0, convolved_img.shape[0], pool_size):
        for j in range(0, convolved_img.shape[1], pool_size):
            for channel in range(3):  # For each channel R, G, B
                pooled_img[i//pool_size, j//pool_size, channel] = np.max(convolved_img[i:i + pool_size, j:j + pool_size, channel])
    
    return pooled_img.astype(np.uint8)



class ImageGrid(Mobject):
    def __init__(self, image, pixel_size, **kwargs):
        super().__init__(**kwargs)
        self.squares = []
        self.image_width = image.width
        self.image_height = image.height

        # Create the pixel grid
        for i in range(image.width):
            for j in range(image.height):
                # Get the color of the pixel
                pixel_color = image.getpixel((i, j))
                
                # Create a square for each pixel
                square = Square(
                    side_length=pixel_size,
                    stroke_opacity=0,  # Opacity of the stroke between pixels
                    stroke_width=(32 / max(image.width, image.height)),  # Adjust stroke width for thick borders
                    stroke_color=WHITE,  # Color of the stroke between pixels
                    fill_opacity=0,
                    fill_color=ManimColor.from_rgb([c / 255.0 for c in pixel_color[:3]]),  # Convert RGB to Manim color
                )
                
                # Position each square in the grid
                square.move_to(np.array([
                    (i - image.width / 2) * pixel_size,
                    -(j - image.height / 2) * pixel_size,
                    0
                ]))
                
                # Add the square to the scene
                self.add(square)
                self.squares.append((i, j, square))

    def create(self, run_time=1.0):
        animations = []
        diagonals = {}

        for i, j, square in self.squares:
            diagonal_index = i + j
            if diagonal_index not in diagonals:
                diagonals[diagonal_index] = []
            diagonals[diagonal_index].append(square)

        total_diagonals = self.image_width + self.image_height - 1
        diagonal_time_allocation = run_time / total_diagonals

        for _, _, square in self.squares:
            square.rotate(90 * DEGREES, axis=UP)

        for diagonal_index, diagonal_squares in diagonals.items():
            diagonal_animations = []
            for square in diagonal_squares:
                diagonal_animations.append(
                    square.animate
                    .rotate(angle=-90 * DEGREES, axis=UP)
                    .set_fill(opacity=1)
                    .set_stroke(opacity=1),
                )
            animations.append(
                AnimationGroup(*diagonal_animations, lag_ratio=0.01)
                .set_run_time(diagonal_time_allocation)
            )

        return Succession(*animations)
    
    def create_from_grid(self, original, filter_size=3, run_time=1.0, color=ORANGE):
        animations = []
        squares_in_order = []

        # Collect squares in left-to-right, top-to-bottom order
        for i in range(self.image_height):  # First, go row by row (y-axis)
            for j in range(self.image_width):  # Then, go left to right within each row (x-axis)
                squares_in_order.append((i, j, self.squares[j * self.image_width + i][2]))

        # Apply initial transformation to all squares
        for _, _, square in squares_in_order:
            square.rotate(90 * DEGREES, axis=UP)

        # Create a semi-transparent square the size of the filter
        filter_square = Square(
            side_length=filter_size * original.squares[0][2].get_side_length(),
            stroke_opacity=0,
            fill_opacity=0,
            fill_color=color,
        )

        filter_square_destination = Square(
            side_length=self.squares[0][2].get_side_length(),
            stroke_opacity=0,
            fill_opacity=0,
            fill_color=color,
        )

        # Position the filter in the initial place
        filter_square.move_to(original.squares[0][2].get_corner(UL), aligned_edge=UL)
        filter_square_destination.move_to(self.squares[0][2].get_center())

        # Add the filter square to the original grid
        self.add(filter_square)
        self.add(filter_square_destination)


        for i, j, square in squares_in_order:
            # Create the animation for the square rotation
            square_animation = square.animate.rotate(angle=-90 * DEGREES, axis=UP).set_fill(opacity=1).set_stroke(opacity=1)

            # Move the filter square to follow the current square
            filter_animation = filter_square.animate.move_to(
                original.squares[filter_size * i + (original.image_width * filter_size * j)][2].get_corner(UL), aligned_edge=UL
            ).set_opacity(0.5)

            filter_animation_destination = filter_square_destination.animate.move_to(
                self.squares[i + (self.image_width * j)][2].get_center()
            ).set_opacity(0.5)

            # Combine the square rotation and filter movement into a single animation group
            animations.append(AnimationGroup(square_animation, filter_animation, filter_animation_destination, lag_ratio=0))

        animations.append(filter_square.animate.set_opacity(0))
        animations.append(filter_square_destination.animate.set_opacity(0))

        # Return the animation as a Succession, where each AnimationGroup is run one after the other
        return Succession(*animations, run_time=run_time)


class ConvolutionalLayerScene(Scene):
    def construct(self):
        # Load the image using PIL to get the dimensions
        image_path = "pixel_art.png"
        img = Image.open(image_path)
        img_width, img_height = img.size  # Get image dimensions
        
        # Calculate pixel size dynamically based on the image dimensions
        grid_size = max(img_width, img_height)  # Use the larger dimension for the grid size
        pixel_size = (config.frame_width / 5) / grid_size  # Adjust pixel size to fit the scene
        
        # Create the pixel grid
        image_grid = ImageGrid(img, pixel_size).move_to(LEFT * 4)

        # Create the text for image grid
        image_grid_text = Text("Original Image").next_to(image_grid, UP * 0.5).scale(0.5)

        # Apply convolution with a 5x5 gaussian kernel
        gaussian_img_5x5 = gaussian_convolution(np.array(img, dtype=np.uint8), kernel_size=5)
        gaussian_image_grid_5x5 = ImageGrid(Image.fromarray(gaussian_img_5x5), pixel_size).move_to(ORIGIN + UP * 1.7)

        # Create the text for gaussian image grid
        gaussian_image_grid_5x5_text = Text("Gaussian Kernel").next_to(gaussian_image_grid_5x5, UP * 0.5).scale(0.5)

        # Apply convolution with a Ridge kernel for comparison
        ridge_img = ridge_convolution(np.array(img, dtype=np.uint8))
        ridge_image_grid = ImageGrid(Image.fromarray(ridge_img), pixel_size).move_to(ORIGIN + DOWN * 2.2)

        # Create the text for ridge image grid
        ridge_image_grid_text = Text("Ridge Kernel").next_to(ridge_image_grid, UP * 0.5).scale(0.5)

        # Apply 8x8 average pooling on gaussian image
        pooled_gaussian_img = average_pooling_image(gaussian_img_5x5, pool_size=8)

        # Create the pixel grid for the pooled gaussian image
        pooled_gaussian_grid = ImageGrid(Image.fromarray(pooled_gaussian_img), pixel_size * 8).move_to(RIGHT * 4 + UP * 1.7)

        # Create the text for pooled gaussian image grid
        pooled_gaussian_grid_text = Text("Avg. Pooled Stride 8").next_to(pooled_gaussian_grid, UP * 0.5).scale(0.5)

        # Apply 8x8 average pooling on gaussian image
        pooled_ridge_img = max_pooling_image(ridge_img, pool_size=8)

        # Create the pixel grid for the pooled gaussian image
        pooled_ridge_grid = ImageGrid(Image.fromarray(pooled_ridge_img), pixel_size * 8).move_to(RIGHT * 4 + DOWN * 2.2)

        # Create the text for pooled ridge image grid
        pooled_ridge_grid_text = Text("Max Pooled Stride 8").next_to(pooled_ridge_grid, UP * 0.5).scale(0.5)

        # Create animations for all grids
        animations = [
            image_grid.create(run_time=1.0),
            Create(image_grid_text),
            gaussian_image_grid_5x5.create_from_grid(image_grid, filter_size=1, run_time=5, color=ORANGE),
            Create(gaussian_image_grid_5x5_text),
            ridge_image_grid.create_from_grid(image_grid, filter_size=1, run_time=5, color=PINK),
            Create(ridge_image_grid_text),
            pooled_gaussian_grid.create_from_grid(gaussian_image_grid_5x5, filter_size=8, run_time=5, color=ORANGE),
            Create(pooled_gaussian_grid_text),
            pooled_ridge_grid.create_from_grid(ridge_image_grid, filter_size=8, run_time=5, color=PINK),
            Create(pooled_ridge_grid_text),
        ]

        print("image_grid", image_grid.image_width, image_grid.image_height)
        print("gaussian_image_grid_5x5", gaussian_image_grid_5x5.image_width, gaussian_image_grid_5x5.image_height)

        # Add the grids to the scene and play all animations in parallel
        self.add(image_grid, gaussian_image_grid_5x5, ridge_image_grid, pooled_gaussian_grid, pooled_ridge_grid)
        self.play(AnimationGroup(*animations, lag_ratio=0.5))  # Play in parallel

        self.wait(2)  # Wait for 2 seconds