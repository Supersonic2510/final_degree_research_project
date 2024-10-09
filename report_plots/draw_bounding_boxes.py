import cv2
import json

# Load the JSON file
json_file_path = 'images/modified_annotations.json'
with open(json_file_path, 'r') as file:
    json_data = json.load(file)

# Load the original image
original_image_path = 'images/modified_image.jpg'
image = cv2.imread(original_image_path)

# Draw bounding boxes on the image
for obj in json_data['objects']:
    bbox = obj['bbox']
    xmin = int(bbox['xmin'])
    ymin = int(bbox['ymin'])
    xmax = int(bbox['xmax'])
    ymax = int(bbox['ymax'])

    # Draw rectangle
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Save the image with bounding boxes
output_image_path_with_bboxes = 'images/modified_image_drawn.jpg'
cv2.imwrite(output_image_path_with_bboxes, image)

print(f"Image with bounding boxes saved at: {output_image_path_with_bboxes}")
