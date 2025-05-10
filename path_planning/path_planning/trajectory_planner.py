import rclpy
from rclpy.node import Node

import heapq
assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from heist_msgs.msg import HeistState
from .utils import LineTrajectory
from std_msgs.msg import Int32
import numpy as np
# from skimage.morphology import disk, dilation

from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import Pose

import cv2

from heapq import heappop, heappush
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

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")
        self.declare_parameter('full_run', True)

        self.state_pub = self.create_publisher(Int32, "/change_info", 1)
        self.state_sub = self.create_subscription(HeistState, "/heist_state", self.state_callback, 1)
        self.full_run = self.get_parameter('full_run').get_parameter_value().bool_value

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        # self.visited_points_pub = self.create_publisher(
        #     PoseArray,
        #     "/trajectory/visited_points",
        #     10
        # )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        self.map_data = None
        self.map_info = None
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin = (0.0, 0.0)
        self.rotation_matrix = np.eye(2)
        self.rotation_matrix_inv = np.eye(2)
        self.safety_cost_map = None

        self.robot_radius = 0.75 #0.75
        self.turning_radius = 1.0

        self.current_pose = None
        self.known = [
                      (-4.635417938232422, 25.2034912109375),
                      (-20.21269416809082, 25.44011116027832),
                      (-20.402408599853516, 32.01554870605469),
                      (-27.574138641357422, 33.71652603149414),
                      ]
        self.home = (-19.83066177368164, 1.331664800643921)

        self.goal1 = None

        self.banana1 = None
        self.banana2 = None
        self.path_initialized = False

        self.visualize_search = False
        self.valid_state = False
        self.prev_state = None

    def state_callback(self, statemsg):
        self.get_logger().info(f"Entered state_callback w {statemsg.state}, {statemsg.objective}")
        if not self.path_initialized and statemsg.state == State.FOLLOW.value and self.banana1 is not None and self.banana2 is not None:
            self.get_logger().info(f"entered")
            if statemsg.objective == Obj.BANANA_A.value:
                self.get_logger().info(f"Initialized current pose {0}, px {0}")
                self.plan_path(self.home, self.known[self.banana1], self.map_data)
            elif statemsg.objective == Obj.BANANA_B.value:
                self.get_logger().info(f"Initialized current pose {0}, px {0}")
                self.plan_path(self.known[self.banana1], self.known[self.banana2], self.map_data)
            elif statemsg.objective == Obj.HOME.value:
                self.get_logger().info(f"Initialized current pose {0}, px {0}")
                self.plan_path(self.known[self.banana2], self.home, self.map_data)
            else:
                return
            self.path_initialized = True
        self.prev_state = statemsg.state
        if statemsg.state != State.FOLLOW.value:
            self.path_initialized = False

    def map_cb(self, msg):
        self.map_info = msg.info
        self.map_height = msg.info.height
        self.map_width = msg.info.width
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.get_logger().info("Map received")


        # occupancy grid has values 0 for free space and -1 for unknown
        self.map_data = np.array(msg.data).astype(np.int8).reshape(msg.info.height, msg.info.width)

        # Alter map
        q = (
            msg.info.origin.orientation.x,
            msg.info.origin.orientation.y,
            msg.info.origin.orientation.z,
            msg.info.origin.orientation.w
        )

        _, _, self.yaw = euler_from_quaternion(q) # quaternion to yaw

        self.rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)],
        ])

        self.rotation_matrix_inv = np.array([
            [np.cos(self.yaw), np.sin(self.yaw)],
            [-np.sin(self.yaw), np.cos(self.yaw)],
        ])

        self.dilate_map(self.robot_radius)
        self.calculate_safety_cost_map(maxdist = 500)

        self.get_logger().info("Initialized map")

    def pose_cb(self, pose):
        # pass
        # we don't need to run search on orientation
        self.current_pose = (pose.pose.pose.position.x, pose.pose.pose.position.y) # extract x,y
        self.get_logger().info(f"Initialized current pose {self.current_pose}, px {self.convert_world_to_pixel(self.current_pose)}")
        # self.plan_path(self.current_pose, self.banana1, self.map_data)

        # if self.current_pose is not None and self.banana1 is not None and self.banana2 is not None and self.map_data is not None:

        #     dist_to_banana1 = self.euclidean_distance(self.current_pose, self.banana1)
        #     dist_to_banana2 = self.euclidean_distance(self.current_pose, self.banana2)

        #     if dist_to_banana1 < dist_to_banana2:
        #         self.plan_path(self.current_pose, self.banana1, self.map_data)
        #         self.plan_path(self.banana1, self.banana2, self.map_data)
        #         self.plan_path(self.banana2, self.current_pose, self.map_data)
        #     else:
        #         self.plan_path(self.current_pose, self.banana2, self.map_data)
        #         self.plan_path(self.banana2, self.banana1, self.map_data)
        #         self.plan_path(self.banana1, self.current_pose, self.map_data)

    def get_closest(self, px):
        smallest_dist = 10000000000000000000
        best_i = 0
        for i, pt in enumerate(self.known):
            dist = ((pt[0]-px[0])**2 + (pt[1]-px[1])**2) ** (1/2)
            if dist < smallest_dist:
                smallest_dist = dist
                best_i = i
        return best_i

    def goal_cb(self, msg):
        # we don't need to run search on orientation
        # banana_locations = msg.poses
        if self.goal1 == None:
            self.goal1 = (msg.pose.position.x, msg.pose.position.y)
        else:
            px1 = self.goal1
            px2 = (msg.pose.position.x, msg.pose.position.y)
            banana1 = self.get_closest(px1)
            banana2 = self.get_closest(px2)
            if banana1 < banana2:
                self.banana1 = banana1
                self.banana2 = banana2
            else:
                self.banana1 = banana2
                self.banana2 = banana1
            self.get_logger().info(f"Initialized bananas #{self.banana1} and #{self.banana2}")
            self.goal1 = None

        # self.get_logger().info(f"Initialized goal pose {self.banana1}, px {self.convert_world_to_pixel(self.banana1)}")
        # if not banana_locations:
        #     self.get_logger().info("No banana locations received...")

        # self.banana1 = (banana_locations[0].position.x, banana_locations[0].position.y)
        # self.get_logger().info(f"Initialized banana 1 {self.banana1}, px {self.convert_world_to_pixel(self.banana1)}")

        # self.banana2 = (banana_locations[1].position.x, banana_locations[1].position.y)
        # self.get_logger().info(f"Initialized banana 2 {self.banana2}, px {self.convert_world_to_pixel(self.banana2)}")

    def plan_path(self, start_point, end_point, map):
        self.get_logger().info("Path planning started")
        self.trajectory.clear()

        start_px = self.convert_world_to_pixel(start_point)
        goal_px = self.convert_world_to_pixel(end_point)
        self.get_logger().info(f"Start pixel: {start_px}, Goal pixel: {goal_px}")

        if self.visualize_search:
            visited_points = PoseArray()
            visited_points.header.frame_id = "map"

        # Remember start_px and goal_px are (u,v) format but we index map_data as (v,u)
        def get_neighbors(node):
            neighbors = [
                (node[0] + 1, node[1]),
                (node[0] - 1, node[1]),
                (node[0], node[1] + 1),
                (node[0], node[1] - 1),
                (node[0] + 1, node[1] + 1),
                (node[0] - 1, node[1] - 1),
                (node[0] + 1, node[1] - 1),
                (node[0] - 1, node[1] + 1)
            ]
            return [
                (u, v) for u, v in neighbors
                if 0 <= u <= self.map_width and 0 <= v <= self.map_height
            ]
        def heuristic(a, b):
            # manhattan distance for heuristic
            return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

        open_set = []
        heapq.heapify(open_set)
        came_from = {}
        heapq.heappush(open_set, (0 + heuristic(start_px, goal_px), 0, start_px)) # (f_score, g_score, node)
        g_score = {start_px: 0}
        completed = set()
        path = []
        # self.get_logger().info(f"open set {open_set}")
        while open_set:
            _, current_g, current = heappop(open_set)
            # self.get_logger().info(f"On node {current}")

            if self.visualize_search:
                # Add the current node to visited points
                world_coords = self.convert_pixel_to_world(current)
                pose = Pose()
                pose.position.x = world_coords[0]
                pose.position.y = world_coords[1]
                pose.position.z = 0.0
                visited_points.poses.append(pose)

                # Publish visited points
                self.visited_points_pub.publish(visited_points)

            if current == goal_px:
                self.get_logger().info("Goal reached")
                path = self.reconstruct_path(came_from, goal_px)
                break

            completed.add(current)
            # self.get_logger().info(f"{get_neighbors(current)}")
            for neighbor in get_neighbors(current):
                u, v = neighbor
                # self.get_logger().info(f"safety map value {self.safety_cost_map[v,u]}")
                if neighbor in completed or self.map_data[v,u] != 0:
                    # self.get_logger().info(f"{self.map_data[v,u]}")
                    # self.get_logger().info(f"safety map value {self.safety_cost_map[v,u]}")
                    continue

                tentative_g_score = current_g + self.euclidean_distance(current, neighbor) + self.safety_cost_map[v,u]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    # safety_cost = self.safety_cost_map[v,u]
                    f_score = tentative_g_score + self.euclidean_distance(neighbor, goal_px) # + safety cost# change between euclidean &amp; heuristic
                    # f_score = tentative_g_score + heuristic(neighbor, goal_px)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                    came_from[neighbor] = current

        if path:
            for point in path:
                self.trajectory.addPoint(point)

            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
            self.get_logger().info("Path planned successfully")
            self.get_logger().info(f"Path length: {len(path)}")


    def reconstruct_path(self, came_from, end):
        current = end
        path = [self.convert_pixel_to_world(current)]
        # self.get_logger().info(f"{came_from}")

        while current in came_from:
            current = came_from[current]
            path.insert(0, self.convert_pixel_to_world(current))
        return path

    def dilate_map(self, r):
        r_px = int(r/self.map_resolution)

        footprint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_px, r_px))

        modified_map = np.zeros_like(self.map_data, dtype = np.uint8)
        modified_map[self.map_data != 0] = 1

        dilated_map = cv2.dilate(modified_map, footprint, iterations = 1) # extends unknown boundary

        scaled_map = np.zeros_like(dilated_map, dtype = np.uint8) # scale weights
        scaled_map[dilated_map != 0] = 100

        self.map_data = scaled_map
        self.map_data[:, 1301:] = 100 # dont use right

    def convert_pixel_to_world(self, pixel):
        pixel_coords = np.array([pixel[0], pixel[1]])
        map_coords = np.dot(self.rotation_matrix_inv, pixel_coords) * self.map_resolution
        map_x = map_coords[0] + self.map_origin[0]
        map_y = map_coords[1] + self.map_origin[1]
        return map_x, map_y

    def convert_world_to_pixel(self, point):
        map_x, map_y = point
        map_coords = np.array([map_x - self.map_origin[0], map_y - self.map_origin[1]])
        pixel_coords = np.dot(self.rotation_matrix, map_coords) / self.map_resolution
        pixel_x = int(pixel_coords[0])
        pixel_y = int(pixel_coords[1])
        return pixel_x, pixel_y

    def euclidean_distance(self, a, b):
        # euclidean distance for cost
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def calculate_safety_cost_map(self, maxdist=8):
        """
        Calculate a safety cost map where cost increases as distance to the nearest obstacle decreases.
        """
        safety_cost_map = np.zeros_like(self.map_data, dtype=np.float32)

        # obstacles 1 free space 0
        binary_map = np.zeros_like(self.map_data, dtype=np.uint8)
        binary_map[self.map_data != 0] = 1

        # distance transform
        distance_map = cv2.distanceTransform(1 - binary_map, cv2.DIST_L2, 5)
        # self.get_logger().info(f"{distance_map}")
        # safety costs
        # safety_cost_map = (maxdist - distance_map)
        safety_cost_map = (1*(maxdist - distance_map))**2
        # clip negative values
        # safety_cost_map[safety_cost_map < 0] = 0
        self.safety_cost_map = safety_cost_map
        self.get_logger().info("Safety cost map calculated.")

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
