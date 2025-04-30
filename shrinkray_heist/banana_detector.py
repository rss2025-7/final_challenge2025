#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
# from vs_msgs.msg import ConeLocationPixel

class BananaDetector(Node):
    """
    # A class for applying your cone detection algorithms to the real robot.
    # Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    # Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self, yolo_dir="/root/yolo", from_tensor_rt=True, threshold=0.5):
        super().__init__("banana_detector")
        # Subscribe to ZED camera RGB frames
        # self.cone_pub = self.create_publisher(ConeLocationPixel, "/relative_cone_px", 10)
        self.debug_pub = self.create_publisher(Image, "/banana_debug_img", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

        from ultralytics import YOLO
        cls = YOLO
        
        self.threshold = threshold
        self.yolo_dir = yolo_dir
        if from_tensor_rt:
            self.model = cls(f"{self.yolo_dir}/yolo11n.engine", task="detect")
        else:
            self.model = cls(f"{self.yolo_dir}/yolo11n.pt", task="detect")

        self.get_logger().info("Banana Detector Initialized")

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
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        results = list(self.model(image, verbose=False))[0]
        boxes = results.boxes

        predictions = []
        # Iterate over the bounding boxes
        for xyxy, conf, cls_idx in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if conf.item() >= self.threshold:
                # Convert bounding box tensor to Python floats
                x1, y1, x2, y2 = xyxy.tolist()
                # Map class index to class label using model/ results
                label = results.names[int(cls_idx.item())]
                predictions.append(((x1, y1, x2, y2), label))

                # TODO check that x1, y1 is actually top left of box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
        
        # bottomv = float(bbox[1][1])
        # bottomu = float(bbox[1][0] + bbox[0][0])//2
        # conepx.u, conepx.v = bottomu, bottomv


        # self.cone_pub.publish(conepx)

         # img with bounding boxes
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    banana_detector = BananaDetector()
    rclpy.spin(banana_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
