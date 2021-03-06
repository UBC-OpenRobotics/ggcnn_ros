# Generative Grasping CNN in ROS
This repository consists of code to run Generative Grasping for antipodal grasps in ROS, for the UBC OpenRobotics Software pipeline.

You can read the paper [here](https://arxiv.org/abs/1804.05172)

## TODO

- [x] Implement Basic Subscriber Node
- [x] Allow visualization despite threaded execution
- [ ] Investigate Normalization or Focal point error (unsure which one)
- [ ] Implement Publisher for Best Grasp - investigate optimal message type
- [ ] Implement PID control and interface with Navigation
- [ ] Allow for Training of custom models - with existing datasets or custom

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


### GGCNN
Next, to build ggcnn, 

```
cd /ggcnn_ros
catkin build ggcnn
```

Additionally, the python dependencies can be found in `/ggcnn_ros/src/ggcnn/scripts/requirements.txt` and can be installed with `pip install -r requirements.txt`

## Usage

At the moment, there is only one node `ggcnn_node.py` which subscribes to the depth camera and does a forward pass through the model. The visualization option is enabled by default and will display the RGB, Depth, Q, and Angle images (refer to the paper for more information on what these last two actually mean). Additionally, it will attempt to overlay the grasping rectangle on the RGB and Depth images.

There is one pretrained model currently provided. The node can be launched with, 
```
roslaunch ggcnn ggcnn.launch
```

When it functions correctly, the output should resemble [this demonstration](https://www.youtube.com/watch?v=7nOoxuGEcxA&ab_channel=DougMorrison)
