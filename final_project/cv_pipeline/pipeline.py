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

# Add a quit flag
quit_program = False

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

        # Annotate the frame with predictions
        annotated_frame = results[0].plot()  # Get the annotated frame

        # Convert BGR to RGB for matplotlib
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

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

