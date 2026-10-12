#!/usr/bin/env python3
"""Record the Gazebo pose of the robot and one tracked object to CSV.

Purpose
-------
During evaluation the robot drives through a world and the model's trajectory is
compared against a reference path. This node samples ``/model_states`` and
writes one row per message:

    time, robot_x, robot_y, robot_z, object_x, object_y, object_z

Rows are flushed immediately so the CSV stays valid even if the run is killed.
Use ``track_traj2_seed.py`` when several objects must be tracked at once.
"""

import csv
import os

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node


class TrajectoryTracker(Node):
    """Writes robot/object poses from ``/model_states`` to a timestamped CSV."""

    def __init__(self):
        super().__init__('coordinate_tracker')

        # ---- Parameters -------------------------------------------------
        self.declare_parameter("robot_model", "diffdrive_gazebo_bot")
        self.declare_parameter("object_model", "track_object")
        self.declare_parameter("world", "moving")
        self.declare_parameter("model", "NAC")
        self.declare_parameter("seed", 42)

        self.robot_name = self.get_parameter("robot_model").value
        self.object_name = self.get_parameter("object_model").value
        self.world_name = self.get_parameter("world").value
        self.model_name = self.get_parameter("model").value
        self.seed = self.get_parameter("seed").value

        # ---- Output file ------------------------------------------------
        # Name the run by {model}_{world}_{seed} so seeds never overwrite
        # each other and the filename identifies the experiment.
        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "trajectories")
        os.makedirs(csv_dir, exist_ok=True)

        self.csv_path = os.path.join(
            csv_dir, f"{self.model_name}_{self.world_name}_trajectory_{self.seed}.csv")

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time", "robot_x", "robot_y", "robot_z",
            "object_x", "object_y", "object_z",
        ])
        self.csv_file.flush()

        # ---- Input ------------------------------------------------------
        self.model_sub = self.create_subscription(
            ModelStates, "/model_states", self.model_states_callback, 10)

        self.get_logger().info(f"Robot model:  {self.robot_name}")
        self.get_logger().info(f"Object model: {self.object_name}")
        self.get_logger().info(f"CSV file:     {self.csv_path}")

    def model_states_callback(self, msg):
        # Gazebo may not have spawned the models yet; skip until both exist.
        if self.robot_name not in msg.name or self.object_name not in msg.name:
            return

        robot_position = msg.pose[msg.name.index(self.robot_name)].position
        object_position = msg.pose[msg.name.index(self.object_name)].position

        # ModelStates carries no header in this setup, so stamp rows with the
        # node's ROS clock instead.
        time_sec = self.get_clock().now().nanoseconds * 1e-9

        self.csv_writer.writerow([
            f"{time_sec:.4f}",
            f"{robot_position.x:.6f}",
            f"{robot_position.y:.6f}",
            f"{robot_position.z:.6f}",
            f"{object_position.x:.6f}",
            f"{object_position.y:.6f}",
            f"{object_position.z:.6f}",
        ])
        self.csv_file.flush()

    def destroy_node(self):
        if hasattr(self, "csv_file"):
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
