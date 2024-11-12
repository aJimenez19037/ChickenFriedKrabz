import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
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

    async def start(self):
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("Failed to open camera stream.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Process the frame (e.g., display, save, or analyze)
            cv2.imshow(self.cam_name, frame)

            # Press 'q' to exit camera stream processing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)  # Add a small delay for non-blocking async

        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera stream ended.")


class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((400, 400))

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

        if self.get_key("a"):
            right = -speed
        elif self.get_key("d"):
            right = speed
        if self.get_key("UP"):
            down = -speed  # Going up decreases the down speed in body frame
        elif self.get_key("DOWN"):
            down = speed
        if self.get_key("w"):
            forward = speed
        elif self.get_key("s"):
            forward = -speed
        if self.get_key("q"):
            yaw_speed = yaw_speed_rate
        elif self.get_key("e"):
            yaw_speed = -yaw_speed_rate

        return [forward, right, down, yaw_speed]


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

        print("-- Setting offboard mode")
        await self.drone.offboard.start()

    async def control_drone(self, keyboard_controller):
        while True:
            vals = keyboard_controller.get_keyboard_input()
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
    asyncio.create_task(camera_stream_forward.start())
    asyncio.create_task(camera_stream_downward.start())

    #Begin arming and takeoff sequence
    await drone_controller.arm_and_takeoff()
    await drone_controller.set_offboard_mode()

    # Start controlling the drone using keyboard input
    await drone_controller.control_drone(keyboard_controller)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
