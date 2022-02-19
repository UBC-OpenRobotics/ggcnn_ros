#!/usr/bin/env python


import rospy

# Message type of depth camera
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# Instantiate
bridge = CvBridge()

import argparse
import torch.utils.data
from torchvision import transforms

from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')



def depth_callback(msg):
    # Depth callback is called every time a new message is published to the depth topic.
    #if args.vis:
    global cv2_rgb

    try:
        cv2_depth = bridge.imgmsg_to_cv2(msg, "passthrough").astype(np.float32)

    except CvBridgeError as e:
        print(e)
    
    else:
        # cv2.imshow('Depth', cv2_depth)
        # cv2.waitKey(1)
        # if cv2_depth:
        generate_grasps(cv2_depth, cv2_rgb.copy())

def rgb_callback(msg):
    # For visualization, add rgb cb that updates global variable
    global cv2_rgb
    try:
        _cv2_rgb = bridge.imgmsg_to_cv2(msg, "bgr8")

    except CvBridgeError as e:
        print(e)
    
    else:
        # cv2.imshow('RGB', _cv2_rgb)
        # cv2.waitKey(1)
        # if cv2_depth:
        cv2_rgb = _cv2_rgb
        


def generate_grasps(depth, rgb):

    # Pre-process
    depth = transforms.ToTensor()(depth)
    depth = torch.unsqueeze(depth, 0) # Add extra dimension at axis 0

    with torch.no_grad():
        xc = depth.to(device)
        pos_output, cos_output, sin_output, width_output = net.forward(xc)  #pos_output, cos_output, sin_output, width_output

        q_img, ang_img, width_img = post_process_output(pos_output, cos_output, sin_output, width_output)

        if args.vis:
            evaluation.plot_output(rgb, np.squeeze(depth), q_img, ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img)
            #plt.colorbar()
            plt.show()



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='GG-CNN Node')
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')

    # Roslaunch appends __name and __log as args, so ignore those.
    args, unknown = parser.parse_known_args()


    # Load model
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    rospy.init_node('ggcnn', anonymous=True) #Don't care about unique name for this node


    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)
    rospy.loginfo(args.vis)
    if args.vis:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        rospy.Subscriber("/camera/rgb/image_rect_color", Image, rgb_callback)


    rospy.spin()