# Aerial Based Fallen Person Detection
ME369P Final Project
Group Members: Joshua Caratao, Antonio Jimenez, Charles Ramirez

The purpose of our project was to investigate methods for "Aerial-Based Fallen Person detection". The 3 methods we investigated were: 

1) OpenCV Histogram of Oriented Gradients (HOG) descriptor w/SVM model pretrained on pedestrians
2) YoloV8 Detection model custom trained on a dataset of humans from an aerial view + Bounding Box Based Fallen detection
3) YoloV8 Detection model custom trained on a dataset of humans from an aerial view standing up and laying down.

In order to evaluate these detection methods for an aerial application, we used the PX4 Autopilot Stack with a drone model in a Gazebo world that incudes human models to simulate and test performance.

More Information about this project is shared in our presentation link:
https://docs.google.com/presentation/d/1U7FsWdmR6j_znYkj2MU7bBG-hoJp8BWygaLujGOk1p4/edit?usp=sharing

*To learn more about our project code and run this yourself, navigate to the "Drone_Simulation_Tutorial" directory in this repo.



