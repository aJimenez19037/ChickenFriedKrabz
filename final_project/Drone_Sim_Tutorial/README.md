# Aerial Drone Simulation Tutorial
## Description
As part of this project, we tested our person and fallen person detection model for an aerial application within a drone simulation environment. 
The simulation was conducted using Gazebo and PX4. Gazebo, a widely used robotics simulator, provided the physical environment, 
while PX4, an open-source autopilot software, provides low-level controls of the vehicle as well as pre-defined drone models to use.

Other packages and software used for this simulation are described in the next section below.

<img width="1275" alt="Simulation-Image-OpenCV-HOG-Example" src="https://github.com/user-attachments/assets/37eefe96-a10d-43e2-b2b3-7d1a583f809d">
Figure 1. Screenshot of simulation environment using OpenCV HOG Detector for human and Fallen person detection



<img width="1265" alt="Simulation-Image-YoloV8Model1-Example" src="https://github.com/user-attachments/assets/215d5325-2dd7-4da2-ad2d-b87f841d6334">
Figure 2. Screenshot of simulation environment using YoloV8 Model 1 (Trained on dataset of humans from an aerial view)

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
Used for real-time computer-vision tasks, Open-CV allows us to capture, process, and display the video feed taken from the drone's camera sensors within the simulation as well as our bounding boxes from detection. We will also be using OpenCV for its HOG Descriptor Human Detection model.

Website Resource: https://opencv.org/links/

#### GStreamer
Gstreamer is a multimedia framework for building audio and video pipelines. Here, we are using it to decode and process the video stream's from the drone's simulated camera sensors. 
It's essentially providing the underlying mechanism to stream video from the simulation, to our separate drone control script. It also integrates well with OpenCV

Website Resource: https://gstreamer.freedesktop.org/

#### Ultralytics
Ultralytics is a Python library built to simplify deployment and training of YOLO (You Only Look Once) models for real-time object detection. We use Ultralytics here to load in our YoloV8 models that we trained using Ultralytics and Roboflow Datasets in Google Colab.


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
### What is PX4?
PX4 is an open-source autopilot framework designed for autonomous drones and other unmanned vehicles. It provides a flight control system that can be used to develop   software for multirotors, fixed-wing aircraft, and other autonomous vehicles. PX4 handles critical tasks such as:

- Flight control
- Sensor fusion
- GPS-based navigation
- Handling motor outputs
- Drone configuration and safety features
- PX4 is widely used in research, commercial drones, and hobbyist projects.

Note: Cloning the PX4 repository and running the appropriate bash command (in the tutorial) after will also install Gazebo. 

#### 1. Navigate to home directory of your wsl environment
    cd ~/
#### 2. Download the PX4 Source Code
    git clone https://github.com/PX4/PX4-Autopilot.git --recursive

#### 3. Run the ubuntu.sh script to install the simulation environment (Gazebo)
    bash ./PX4-Autopilot/Tools/setup/ubuntu.sh

#### 4. Restart computer or WSL environment

## Additional Package Installations
After restarting your WSL Environment, restart WSL by opening powershell, and entering "wsl" into the terminal.
Once you are in WSL, travel to your home directory and begin installing the additional packages below.

        cd ~/

#### Install GStreamer

        sudo apt update
        sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad         gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

#### Install mavsdk

        pip install mavsdk

#### Install OpenCV

        sudo apt-get install libopencv-dev python3-opencv

#### Install Ultralytics and its dependencies
This first command install ultralytics without its dependencies since we already installed OpenCV
The second command installs the rest of its dependencies, excluding openCV

        pip install ultralytics --no-deps
        pip install torch torchvision torchaudio numpy scipy pandas matplotlib pyyaml tqdm seaborn tensorboard requests

#### Install Pygame

        pip install pygame

## Setting up the Simulation

#### 1. Clone this Repository 
After navigating to your desired directory, clone this repo using the command below

        git clone https://github.com/aJimenez19037/ChickenFriedKrabz.git

#### 2. Copy tar_uav drone model into PX4 Models directory

        cp -r ChickenFriedKrabz/final_project/px4_mods/tar_uav ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models

#### 3. Copy Drone airframe file into PX4 Airframes directory

        cp -r ChickenFriedKrabz/final_project/px4_mods/4229_gazebo-classic_tar_uav ~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes

#### 4. Add Airframe name to airframes CMakeLists.txt file
        cd ~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes
        code CMakeLists.txt

Add full airframe in appropriate section (4229_gazebo-classic_tar_uav)

#### 5. Copy "people2.world" file into PX4 Worlds directory

        cp -r ChickenFriedKrabz/final_project/world_file/people2.world ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/worlds

#### 6. Add "tar_uav" and "people2" to sitl_targets_gazebo-classic.cmake file
        cd ~/PX4-Autopilot/src/modules/simulation/simulator_mavlink
        code sitl_targets_gazebo-classic.cmake
Add the above words into the cmake file 

#### 7. Copy Human Models into PX4 Models directory
        cp -r ChickenFriedKrabz/final_project/human_models/PatientWheelChair ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/
        cp -r ChickenFriedKrabz/final_project/human_models/casual_female ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/
        cp -r ChickenFriedKrabz/final_project/human_models/male_visitor ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/
        cp -r ChickenFriedKrabz/final_project/human_models/standing_person ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/
        cp -r ChickenFriedKrabz/final_project/human_models/walking_actor ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/
        cp -r ChickenFriedKrabz/final_project/human_models/walking_person ~/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/

#### 8. Build PX4

        cd ~/PX4-Autopilot
        make px4_sitl

## Running Simulation with Drone Control and Human Detection

#### 1. Build drone model and world and launch simulation
        cd ~/PX4-Autopilot
        make px4_sitl gazebo-classic_tar_uav__people2
This will launch your simulation with the drone and world of human models. 

#### 10. Launch Drone Control Script
launch a new wsl tab in your terminal, then navigate to the repository directory

        cd ChickenFriedKrabz/final_project
        python3 drone_control.py

This should launch the drone control script. If everything works fine, a pygame window as well as two camera stream windows should pop up. To control the drone, click on the pygame window and use "wasd" keys for lateral movement, "up" and "down" arrows keys to increase and decrease altitude,  and "q" and "e" keys to yaw the drone.

To perform detection, hold "1" key for OpenCV HOG Detector, and "2" or "3" for our trained YoloV8 Models.

*NOTE: If the simulation launches and the script succesfully connects the simulation, but the camera streams do not pop up, it is likely because the opencv that was installed was not built with Gstreamer functionality on. Our code relies on a Gstreamer pipeline from the simulation to this script to grab the videostream from the simulation.











   

