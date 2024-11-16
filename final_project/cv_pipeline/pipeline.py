# import os
# import ultralytics
# from ultralytics import YOLO
# import cv2

# # Dynamically get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(script_dir, "best.pt")
# model = YOLO(model_path)
# # Initialize the video capture (0 is typically the default camera)
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Process video frames
# try:
#     while True:
#         # Read a frame from the camera
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to grab a frame.")
#             break

#         # Perform inference on the frame
#         results = model.predict(source=frame, save=False, verbose=False)  # Use verbose=False to suppress output

#         # Annotate the frame with predictions
#         annotated_frame = results[0].plot()  # Get the annotated frame

#         # Display the frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("Inference interrupted by user.")

# # Release the camera and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Dynamically get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "best.pt")
model = YOLO(model_path)

# Initialize the video capture (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up matplotlib for live display
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for bounding boxes
HEIGHT_WIDTH_RATIO = 4      # Ratio threshold for classifying as "fallen" or not

# Add a quit flag
quit_program = False  # Initialize quit_program here

# Process video frames
try:
    while not quit_program:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        # Perform inference on the frame
        results = model.predict(source=frame, save=False, verbose=False)  # Use verbose=False to suppress output

        # Filter predictions by confidence
        boxes = results[0].boxes  # Boxes object containing all predictions
        if boxes:
            for box in boxes:
                confidence = box.conf.item()  # Get the confidence score
                if confidence >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers for OpenCV
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0

                    # Annotate the frame
                    label = f"{confidence:.2f}"
                    color = (0, 255, 0)  # Green color for the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Check aspect ratio for classification
                    if (1 / HEIGHT_WIDTH_RATIO) <= aspect_ratio <= HEIGHT_WIDTH_RATIO:
                        print("[RESULT]: Standing up")
                    else:
                        print("[RESULT]: Laying down")
                else:
                    print(f"Filtered out box with confidence {confidence:.2f}")

        # Convert BGR to RGB for matplotlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using matplotlib
        ax.clear()
        ax.imshow(rgb_frame)
        ax.axis("off")
        plt.pause(0.001)  # Short pause for live updates

        # Close the program if the figure window is closed
        if not plt.fignum_exists(fig.number):
            quit_program = True

except KeyboardInterrupt:
    print("Inference interrupted by user.")

# Release the camera and close all windows
cap.release()
plt.close()
