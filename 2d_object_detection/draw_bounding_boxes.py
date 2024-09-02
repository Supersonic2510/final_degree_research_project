import os
import cv2
import yaml

# Base directory and splits
data_dir = 'data/yolo'
splits = ['train', 'val', 'test']
yaml_file_path = os.path.join(data_dir, 'data.yaml')

# Load the data.yaml file to get class names
with open(yaml_file_path, 'r') as file:
    data_yaml = yaml.safe_load(file)

# Function to process and draw bounding boxes on all images in a split
def process_split(split):
    images_dir = os.path.join(data_dir, split, 'images')
    labels_dir = os.path.join(data_dir, split, 'labels')
    output_dir = os.path.join(data_dir, split, 'output_images')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all images in the directory
    for image_file in os.listdir(images_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_id = os.path.splitext(image_file)[0]

            # Load the image
            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape

            # Load the corresponding YOLO annotations
            annotation_path = os.path.join(labels_dir, f'{image_id}.txt')
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as file:
                    annotations = file.readlines()

                # Process each annotation and draw bounding boxes
                for annotation in annotations:
                    class_number, x_center, y_center, width, height = map(float, annotation.strip().split())

                    # Convert YOLO format to bounding box coordinates
                    xmin = int((x_center - width / 2) * image_width)
                    ymin = int((y_center - height / 2) * image_height)
                    xmax = int((x_center + width / 2) * image_width)
                    ymax = int((y_center + height / 2) * image_height)

                    # Get the class name from the data.yaml file
                    class_name = data_yaml['names'][int(class_number)]

                    # Draw the bounding box
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Put the class name text on the image
                    cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Save the image with bounding boxes
                output_image_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_image_path, image)
                print(f"Processed and saved: {output_image_path}")

# Loop through each split (train, val, test) and process all images
for split in splits:
    process_split(split)
