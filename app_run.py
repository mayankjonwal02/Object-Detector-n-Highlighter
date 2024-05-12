from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO('yolov8n.pt')  # You can specify the YOLO model file you want to use

# Load image using OpenCV (replace 'image_path.jpg' with your image file path)
image_path = 'objects.jpg'
image = cv2.imread(image_path)

# Perform object detection
results = model.track(source = image )  # This will return a results object with detected objects
# print(results[0].boxes)
# Get bounding box information from results[0].boxes
for boxes_info in results[0].boxes.data:


    # Extract box coordinates and class
    x1, y1, x2, y2, ob_id,confidence, class_idx = boxes_info[0], boxes_info[1], boxes_info[2], boxes_info[3], boxes_info[4], boxes_info[5],boxes_info[6]

    # Get class label from the model's class mapping
    class_label = model.names[int(class_idx)]

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2  # Thickness of the bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # # Put text with class label and confidence score on the image
    text = f'{class_label}: {confidence:.2f} : id:{int(ob_id)}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (0, 255, 0)  # Green color for the text
    text_thickness = 2  # Thickness of the text
    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
    cv2.putText(image, text, (int(x1), int(y1) - 10), font, font_scale, text_color, text_thickness)

# # Display the annotated image
cv2.imshow('Object Detection', image)
cv2.waitKey(0)  # Wait for any key to be pressed
cv2.destroyAllWindows()  # Close the OpenCV window
