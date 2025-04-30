#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from vs_msgs.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


class ConeDetector(Node):
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        super().__init__("cone_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/cone_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        self.get_logger().info("Cone Detector Initialized")

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################
        conepx = ConeLocationPixel()

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # if self.LineFollower:
        #     cv2.rectangle(image, (0,0), (640, 180), [255, 255, 255], -1)
        #     cv2.rectangle(image, (0, 275), (640, 360), [255,255,255], -1)
            
        bbox = cd_color_segmentation(image, None)
        # bottomv = float(bbox[1][1])
        # bottomu = float(bbox[1][0] + bbox[0][0])//2

        # Assumption input: 
        # bbox for left and right white lines. 
        # The left box gives the bottom left, top right corners of the cone.
        # The right box gives the bottom right, top left corners of the cone.

        #To-do: Haven't taken the middle point. Just take the top and the bottom

        left_bbox = bbox[0]
        right_bbox = bbox[1]

        left_bbox_bottom_left = float(left_bbox[0])
        left_bbox_top_right = float(left_bbox[1])
        right_bbox_bottom_right = float(right_bbox[0])
        right_bbox_top_left = float(right_bbox[1])

        # Find the left line
        left_line_slope = (left_bbox_top_right[1] - left_bbox_bottom_left[1]) / (left_bbox_top_right[0] - left_bbox_bottom_left[0])
        left_line_intercept = left_bbox_bottom_left[1] - left_line_slope * left_bbox_bottom_left[0]

        # Find the right line
        right_line_slope = (right_bbox_top_left[1] - right_bbox_bottom_right[1]) / (right_bbox_top_left[0] - right_bbox_bottom_right[0])
        right_line_intercept = right_bbox_bottom_right[1] - right_line_slope * right_bbox_bottom_right[0]

        # Find the intersection of the two lines
        # y1 = k1 * x1 + b1
        # y2 = k2 * x2 + b2
        # k1 * x + b1 = k2 * x + b2 -> (k1-k2)*x = b2 - b1 -> x = (b2-b1)/(k1 - k2)

        x_intersect = (right_line_intercept - left_line_intercept) / (left_line_slope - right_line_slope)
        y_intersect = left_line_slope * x_intersect + left_line_intercept
        
        #To-do: Bring it closer?
        conepx.u, conepx.v = x_intersect, y_intersect


        self.cone_pub.publish(conepx)
        # img with bounding box
        # bbox_top_left = bbox[0][0], bbox[0][1]
        # bbox_bot_right = bbox[1][0], bbox[1][1]

        #To-do: The boxing dimension may be different
        bbox_top_left = x_intersect - 10, y_intersect - 10
        bbox_bot_right = x_intersect + 10, y_intersect + 10
        cv2.rectangle(image, bbox_top_left, bbox_bot_right, (0,0,255), 2)
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    cone_detector = ConeDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
