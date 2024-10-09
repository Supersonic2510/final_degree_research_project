from manim import *
import os

class ImageSequence(Scene):
    def construct(self):
        # Define the number of frames and frame rate
        num_frames = 201
        fps = 10  # 10 frames per second

        # Loop over the frames
        for i in range(num_frames):
            # Load the images for the current frame
            lidar_image = ImageMobject(f"gif generator/lidar/frame_{i}.png")
            front_image = ImageMobject(f"gif generator/img_front/frame_{i}.png")
            rear_image = ImageMobject(f"gif generator/img_rear/frame_{i}.png")
            left_image = ImageMobject(f"gif generator/img_left/frame_{i}.png")
            right_image = ImageMobject(f"gif generator/img_right/frame_{i}.png")

            # Resize images for better fitting
            lidar_image.scale(1.3)
            front_image.scale(0.20)
            rear_image.scale(0.20)
            left_image.scale(0.20)
            right_image.scale(0.20)

            # Create text labels for the images
            lidar_text = Text("Lidar Point Cloud", font_size=24)
            front_text = Text("Camera Front", font_size=18)
            rear_text = Text("Camera Rear", font_size=18)
            left_text = Text("Camera Left", font_size=18)
            right_text = Text("Camera Right", font_size=18)

            # Position the LiDAR image on the left and its text above
            lidar_image.to_edge(LEFT)
            lidar_text.move_to(lidar_image.get_top() + UP * 0.5)

            # Arrange the camera images (2x2 grid manually) and position their texts above each image
            front_image.next_to(lidar_image, RIGHT * 0.60, buff=1).shift(UP * 1)
            front_text.move_to(front_image.get_top() + UP * 0.3)

            rear_image.next_to(front_image, RIGHT * 0.60, buff=0.5).align_to(front_image, UP)
            rear_text.move_to(rear_image.get_top() + UP * 0.3)

            left_image.next_to(front_image, DOWN * 1.5, buff=0.5).align_to(front_image, LEFT)
            left_text.move_to(left_image.get_top() + UP * 0.3)

            right_image.next_to(left_image, RIGHT * 0.60, buff=0.5).align_to(left_image, UP)
            right_text.move_to(right_image.get_top() + UP * 0.3)

            # Add images and texts for the current frame to the scene
            self.add(lidar_image, lidar_text, front_image, front_text, rear_image, rear_text, left_image, left_text, right_image, right_text)

            # Wait for 0.1 seconds to create a 10fps animation
            self.wait(1 / fps)

            # Clear the previous frame's images and text from the scene before adding the next frame
            self.remove(lidar_image, lidar_text, front_image, front_text, rear_image, rear_text, left_image, left_text, right_image, right_text)
