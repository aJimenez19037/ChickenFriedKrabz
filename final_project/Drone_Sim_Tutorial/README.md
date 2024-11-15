# Aerial Drone Simulation Tutorial
## Description
As part of this project, we tested our person and fallen person detection model for an aerial application within a drone simulation environment. 
The simulation was conducted using Gazebo and PX4. Gazebo, a widely used robotics simulator, provided the physical environment, 
while PX4, an open-source autopilot software, provides low-level controls of the vehicle as well as pre-defined drone models to use.

Other packages and software used for this simulation are described in the next section.

## Packages and Software Used
#### MAVSDK
Used for drone control and communication, MAVSDK is a library designed for interacting with MAVLink-based systems, such as drones, in a straightforward and high-level manner. 
This library allows us to run commands for arming, takeing off, and landing. Overall, its best quality is abstracting the low-level MAVLink commands into easy-to-use python commands.
Additionally, this can act as an alternative to using MAVROS with ROS to control the PX4 Drone.

Website Resource: https://mavsdk.mavlink.io/main/en/index.html
#### Asyncio
Asyncio is a Python module that facilitates asynchronous program execution, allowing for multiple tasks to run concurrently. With Asyncio, 
we can create non-blocking loops and for our application specifically, we can handle tasks, such as reading camera streams and controlling the drone simultaneously.
However, this is different from multi-threading, as the program is still executed in a single thread.

Website Resource: https://docs.python.org/3/library/asyncio.html
#### pygame
Pygame is a popular library for creating multimedia applications and games. However, for our application, we are using pygame to capture keyboard inputs and control the drone manually during the simulation.

Website Resource: https://www.pygame.org/wiki/about
#### OpenCV
Used for real-time computer-vision tasks, Open-CV allows us to capture, process, and display the video feed taken from the drone's camera sensors within the simulation.

Website Resource: https://opencv.org/links/
#### GStreamer
Gstreamer is a multimedia framework for building audio and video pipelines. Here, we are using it to decode and process the video stream's from the drone's simulated camera sensors. 
It's essentially providing the underlying mechanism to stream video from the simulation, to our separate drone control script. It also integrates well with OpenCV

Website Resource: https://gstreamer.freedesktop.org/

## Environment Setup
Before we can setup our simulations, we need to set up a proper environment to run Gazebo and PX4. If you are on a Windows system, 
my recommended software setup is through Windows Subsystem for Linux (WSL) feature. 
WSL is a compatibility layer that allows linux distributions like Ubuntu to run directly on Windows without needing a virtual machine or separate hardware. 
It's built into Windows and provides seamless integration between the Windows and Linux environments (Sorry MAC Users).

If you have an ubuntu (22.04, 20.04, 18.04) system already, feel free to skip to the PX4 installation section. However, if you are a windows user and would like to use WSL, follow instructions below

### Prerequisites
1. Windows 10 Version 1903 or higher.
2. Administrator access to install software and enable WSL.

### Steps

#### 1. Enable WSL
1. Open PowerShell as Administrator.
2. Run the following command to enable WSL and install the required features:

        wsl --install

#### 2. Set WSL Version to WSL2
By default, WSL 2 will be set up, but if you need to manually configure it:
1. Open PowerShell as Administrator and run:

        wsl --set-default-version 2

This sets WSL 2 as the default version for any new Linux distributions.

#### 3. Install Ubuntu 20.04

  Now that WSL is enabled, you can install Ubuntu 20.04 using the following command:
  
  *NOTE: DO NOT FORGET YOUR USER NAME AND PASSWORD. WRITE IT DOWN
  
      wsl --install -d Ubuntu-20.04

  This command will: Download and install Ubuntu 20.04 on your WSL instance and set it up as your default WSL distribution.

#### 4. Launch WSL Ubuntu 20.04
After the installation is complete, you can launch Ubuntu 20.04 using the following command:

    wsl -d Ubuntu-20.04

*Note: If this is your default or only distribution, you may launch Ubuntu using the following command:

    wsl

#### 5. Update and Upgrade Ubuntu 20.04
Once you're inside the Ubuntu 20.04 environment, it’s a good idea to update the package manager and upgrade the installed packages:

    sudo apt update && sudo apt upgrade -y

#### 6. Verify Installation
You can verify that Ubuntu 20.04 is running by checking the Ubuntu version inside the WSL terminal:

    lsb_release -a

#### 7. Finished!
After step 6, if no problems occured, your WSL and Ubuntu setup should be complete. You can now continue forward with PX4 installation

## PX4 and Gazebo Installation
Note: Cloning the PX4 repository and running the appropriate bash command (in the tutorial) after will also install Gazebo. 

### Installation
#### 1. Navigate to home directory of your wsl environment
    cd ~/
#### 2. Download the PX4 Source Code
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive

#### 3. Run the ubuntu.sh script to install the simulation environment (Gazebo)
    bash ./PX4-Autopilot/Tools/setup/ubuntu.sh

#### 4. Restart computer or WSL environment



   
