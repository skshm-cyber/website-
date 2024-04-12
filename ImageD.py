from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('C:\Study\Python\Projects\Yolo-weight\yolov8n.pt')

# Load and resize the image
image_path = "C:\\Study\\Python\\Projects\\T1.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (1069, 720))

# Perform object detection
result = model(image, show=True)

# Wait for any key press
cv2.waitKey(0)
