# Generative Grasping CNN in ROS
This repository consists of code to run Generative Grasping for antipodal grasps in ROS, for the UBC OpenRobotics Software pipeline.

You can read the paper [here](https://arxiv.org/abs/1804.05172)

## Installation

### Depth Camera

GGCNN requires depth images to run its model. As such, ensure that the particular depth camera you have has all it's drivers and is compatible with the ROS infrastructure. 
The following outlines the setup procudures for the Astra Orbecc Sensor. 

First, the `astra_camera` package dependencies are required, 

`sudo apt install ros-noetic-rgbd-launch ros-noetic-libuvc-camera ros-noetic-libuvc-ros`


Next, clone the repository and create the udev rules, 
```cd /ggcnn_ros/src
git clone https://github.com/orbbec/ros_astra_camera

cd /ggcnn_ros
catkin build astra_camera
source devel/setup.bash

roscd astra_camera
./scripts/create_udev_rules

```

To test if the camera is setup correctly, ensure that you have sourced workspace and run, 
`roslaunch astra_camera astra.launch`
Then, in a seperate terminal, run `rviz` and add a camera topic. If you select the world frame to be rgb_optical, you should be able to see the image and depth topics. 


You can  find more information about this package [here](http://wiki.ros.org/astra_camera)

