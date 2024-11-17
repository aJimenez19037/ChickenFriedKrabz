import sys
import os
import importlib

# Define paths to protobuf_old and protobuf_new
protobuf_old_path = os.path.expanduser('~/protobuf_old')
protobuf_new_path = os.path.expanduser('~/protobuf_new')

#Ensure MAVSDK uses protobuf_old
sys.path.insert(0, protobuf_old_path)
import mavsdk
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
from mavsdk.offboard import PositionNedYaw
sys.path.remove(protobuf_old_path)

#Ensure TensorFlow uses protobuf_new
#sys.path.insert(0, protobuf_new_path)
#importlib.reload(sys.modules["google.protobuf"])  # Reload protobuf
#import tensorflow as tf 

from ultralytics import YOLO

import asyncio

import pygame
import cv2

# GStreamer pipelines for forward and downward cameras
CAMERA_PIPELINE_FORWARD = "udpsrc port=5600 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! queue ! appsink sync=false drop=true max-buffers=1"
CAMERA_PIPELINE_DOWNWARD = "udpsrc port=5601 ! application/x-rtp, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! queue! appsink sync=false drop=true max-buffers=1"

class CameraStream:
    def __init__(self, pipeline, cam_name):
        self.cam_name = cam_name
        self.pipeline = pipeline
        self.cap = None

        #Declare Confidence Threshold for YoloV8 Detection
        self.CONFIDENCE_THRESHOLD = 0.25

        #Declare Aspect Ratio Threshold for Fallen Person Detection (Height/Width) and (W/H)
        self.AR_THRESHOLD_HW = 2
        self.AR_THRESHOLD_WH = 2


        # Initialize HOG descriptor with a pre-trained people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        #Initialize Ultralytics YoloV8 Trained Model#1 (Dataset of people from an aerial view w/fallen people NOT classified)
        self.ultra_yolo_model = YOLO("weights/best.pt")

        #Initialize Ultralytics YoloV8 Trained Model#2 (Dataset of people and fallen people already classified)
        self.ultra_yolo_model_2 = YOLO("weights/best2.pt")

        #Specify current detection method: 1 = OpenCV HOG, 2 = YOLO
        self.detection_mode = 1


    async def start(self, keyboard_controller):
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"Failed to open {self.cam_name} stream.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Check detection mode and process frame accordingly
            keyboard_input = keyboard_controller.get_keyboard_input()
            if (keyboard_input[4]):
                if int(keyboard_input[4]) is not self.detection_mode:
                    self.detection_mode = int(keyboard_input[4])
            else:
                self.detection_mode = None

        
            if self.detection_mode == 1:
                frame = self.process_hog(frame)
            elif self.detection_mode == 2:
                frame = self.process_ultra_yolo_1(frame)
            elif self.detection_mode == 3:
                frame = self.process_ultra_yolo_2(frame)
            
            # Display the frame with HOG detections
            cv2.imshow(self.cam_name, frame)
        
            # Press 'q' to exit camera stream processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)  # Add a small delay for non-blocking async

        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stream ended.")


    def process_hog(self, frame):
        # Convert frame to grayscale for HOG processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform HOG detection
        boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), scale=1.05)
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw HOG detections

        detection_count = len(boxes)

        #Overlay the detector label
        cv2.putText(frame, "Detector: OpenCV HOG Descriptor", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        
        return frame
    
    def process_ultra_yolo_1(self, frame):
        #Make prediction
        results = self.ultra_yolo_model.predict(frame, imgsz = 640, conf = 0.25)
        detection_count = 0

        # Annotate frame with YOLO detections
        for result in results:
            detection_count += len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box coordinates to integers
                confidence = box.conf[0]  # Confidence score
                label = self.ultra_yolo_model.names.get(int(box.cls[0])) # Class label (convert class index to name)

                if confidence >= self.CONFIDENCE_THRESHOLD:
                    #Determine Aspect Ratio
                    x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers for OpenCV
                    width = x2 - x1
                    height = y2 - y1
                    AR_HW = height/width if width > 0 else 0
                    AR_WH = width/height if height > 0 else 0

                    #Check Aspect Ratio for Classiciation
                    if (AR_HW < self.AR_THRESHOLD_HW and AR_WH < self.AR_THRESHOLD_WH):
                        label = "STANDING"
                        color = (0,255,0)
                    else:
                        label = "FALLEN"
                        color = (0,0,255)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Overlay the detector label
        cv2.putText(frame, "Detector: YoloV8 Model 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        
        return frame
    
    def process_ultra_yolo_2(self, frame):
        #Make prediction
        results = self.ultra_yolo_model_2.predict(frame, imgsz = 640, conf = 0.25)
        detection_count = 0

        # Annotate frame with YOLO detections
        for result in results:
            detection_count += len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert bounding box coordinates to integers
                confidence = box.conf[0]  # Confidence score
                label = self.ultra_yolo_model_2.names.get(int(box.cls[0])) # Class label (convert class index to name)

                if confidence >= self.CONFIDENCE_THRESHOLD:
                    
                    if label == "Fall-Detected":
                        color = (0,0,255)
                    else:
                        color = (0,255,0)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        
        # Overlay the detector label
        cv2.putText(frame, "Detector: YoloV8 Model 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        
        return frame




class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((400, 400))
        self.detection_mode = None

    def get_key(self, key_name):
        for event in pygame.event.get():
            pass
        key_input = pygame.key.get_pressed()
        key = getattr(pygame, f'K_{key_name}')
        return key_input[key]

    def get_keyboard_input(self):
        forward, right, down, yaw_speed = 0, 0, 0, 0
        speed = 2.5  # meters/second
        yaw_speed_rate = 50  # degrees/second
        detection_mode = None #Default Value

        if self.get_key("a"):
            right = -speed
        elif self.get_key("d"):
            right = speed
        if self.get_key("UP"):
            down = -speed  # Going up decreases the down speed in body frame
        elif self.get_key("DOWN"):
            down = speed
        else:
            down = 0
        if self.get_key("w"):
            forward = speed
        elif self.get_key("s"):
            forward = -speed
        if self.get_key("q"):
            yaw_speed = yaw_speed_rate
        elif self.get_key("e"):
            yaw_speed = -yaw_speed_rate
        
        # Check for mode switching
        if self.get_key("1"):
            detection_mode = 1
            print("Switched to OpenCV HOG detection ...")
        elif self.get_key("2"):
            detection_mode = 2
            print("Switched to Ultralytics YoloV8 Model 1 detection ...")
        elif self.get_key("3"):
            detection_mode = 3
            print("Switched to Ultralytics YOLOV8 Model 2 detection ...")
        elif self.get_key("0"):
            detection_mode = None
            print("Switched OFF Detection")

        self.detection_mode = detection_mode
            

        return [forward, right, down, yaw_speed, self.detection_mode]


class DroneController:
    def __init__(self):
        self.drone = System()

    async def connect(self):
        print("Connecting to drone...")
        await self.drone.connect(system_address="udp://:14540")  # Wait for the connection to be initiated
        
        while True:
            async for state in self.drone.core.connection_state():  # Wait for connection updates
                if state.is_connected:  # Check if connected
                    print("-- Connected to drone!")
                    return  # Exit the function once connected
            print("Still waiting for drone connection...")
            await asyncio.sleep(1)  # Retry after a small delay


    async def arm_and_takeoff(self):
        # Retry arming until successful
        while True:
            try:
                print("-- Arming drone...")
                await self.drone.action.arm()
                print("-- Drone armed successfully!")
                break  # Exit loop once armed
            except Exception as e:
                print(f"Failed to arm: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)  # Retry after delay

        # Command the drone to take off
        while True:
            try:
                print("-- Taking off...")
                await self.drone.action.takeoff()
                print("-- Drone taking off successfully!")
                break  # Exit loop once takeoff succeeds
            except Exception as e:
                print(f"Failed to take off: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)  # Retry after delay

        # Wait for the drone to reach a stable altitude
        await asyncio.sleep(5)

    async def set_offboard_mode(self):
        # Initial setpoint before starting offboard mode
        initial_velocity = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        await self.drone.offboard.set_velocity_body(initial_velocity)

        print("-- Setting offboard mode ...")
        await self.drone.offboard.start()
        print("-- Offboard mode successfully set ...")
    
    # async def hold_position(self):
    #     # Retrieve the current position
    #     async for position_velocity in self.drone.telemetry.position_velocity_ned():
    #         current_position = position_velocity.position #Extract position part
    #         break  # Exit after retrieving the first position

    #     # Retrieve the current yaw
    #     async for attitude in self.drone.telemetry.attitude_euler():
    #         current_yaw = attitude.yaw_deg  # Get yaw in degrees
    #         break  # Exit after retrieving the first attitude
        
    #     await self.drone.offboard.set_position_ned(
    #             PositionNedYaw(
    #                 north_m=current_position.north_m,  # Maintain current north position
    #                 east_m=current_position.east_m,   # Maintain current east position
    #                 down_m=current_position.down_m,  # Hold current altitude
    #                 yaw_deg=current_yaw  # Replace with the desired yaw or maintain current
    #             )
    #         )
            

    async def control_drone(self, keyboard_controller):
        while True:
            vals = keyboard_controller.get_keyboard_input()

            # # Check if all velocity inputs are 0 (hold position)
            # if vals[0] == 0 and vals[1] == 0 and vals[2] == 0 and vals[3] == 0:
            #     await self.hold_position()
            # else:
            velocity = VelocityBodyYawspeed(vals[0], vals[1], vals[2], vals[3])
            await self.drone.offboard.set_velocity_body(velocity)

            # Breaking the loop and landing if 'l' key is pressed
            if keyboard_controller.get_key("l"):
                print("-- Landing")
                await self.drone.action.land()
                break

            await asyncio.sleep(0.1)


async def main():
    # Initialize keyboard controller
    keyboard_controller = KeyboardController()

    # Initialize drone controller (uses MAVSDK)
    drone_controller = DroneController()
    await drone_controller.connect()

    # Start Running the camera stream asynchronously
    print("-- Beginning Camera Stream ...")
    camera_stream_forward = CameraStream(CAMERA_PIPELINE_FORWARD, "Forward Camera Stream")
    camera_stream_downward = CameraStream(CAMERA_PIPELINE_DOWNWARD, "Downward Camera Stream")
    asyncio.create_task(camera_stream_forward.start(keyboard_controller))
    asyncio.create_task(camera_stream_downward.start(keyboard_controller))

    #Begin arming and takeoff sequence
    await drone_controller.arm_and_takeoff()
    await drone_controller.set_offboard_mode()

    # Start controlling the drone using keyboard input
    await drone_controller.control_drone(keyboard_controller)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
