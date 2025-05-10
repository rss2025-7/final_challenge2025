#!/usr/bin/env python3
import numpy as np
import rclpy
import time
from rclpy.node import Node
from heist_msgs.msg import HeistState
from std_msgs.msg import Int32
from ackermann_msgs.msg import AckermannDriveStamped
from enum import Enum

class State(Enum):
    FOLLOW = 1
    PARK = 2
    CORRECT = 3
    END = 4

class Obj(Enum):
    BANANA_A = 1
    BANANA_B = 2
    HOME = 3

class BackupController(Node):

    def __init__(self):
        super().__init__("backup_controller")
        # Declare parameters to make them available for use
        self.declare_parameter("drive_topic", "default")
        self.declare_parameter("velocity", -0.5)

        # Fetch constants from the ROS parameter server
        # This is necessary for the tests to be able to test varying parameters!
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.VELOCITY = self.get_parameter('velocity').get_parameter_value().double_value
        self.state_pub = self.create_publisher(Int32, "/change_info", 1)
        self.state_sub = self.create_subscription(HeistState, "/heist_state", self.state_callback, 1)

        self.start_time = 0
        self.started = False
        self.timer = self.create_timer(0.05, self.on_timer)

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            self.DRIVE_TOPIC,
            10)

        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.stamp = self.get_clock().now().to_msg()
        self.drive_msg.header.frame_id = "base_link"

        self.drive_msg.drive.steering_angle = 0.0
        self.drive_msg.drive.steering_angle_velocity = 0.0 # change as quick as possible
        self.drive_msg.drive.speed = -0.5 # m/s
        self.drive_msg.drive.acceleration = 0.0 # change as quick as possible
        self.drive_msg.drive.jerk = 0.0 # change as quick as possible

    def state_callback(self, statemsg):
        if statemsg.state == State.CORRECT.value:
            if not self.started:
                self.started = True
                self.start_time = time.time()

    # def _on_delay(self):
    #     self.timer.cancel()
    #     self.get_logger().info("Backup controller activated")
    #     self.on_timer()

    def on_timer(self):
        # self.get_logger().info(f"{self.started}")
        if self.started:
            self.drive_pub.publish(self.drive_msg)
            if time.time() - self.start_time > 3.5:
                msg = Int32()
                msg.data = State.CORRECT.value
                self.state_pub.publish(msg)
                self.started = False

def main():
    rclpy.init()
    backup_controller = BackupController()
    rclpy.spin(backup_controller)
    backup_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
