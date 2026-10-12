#!/usr/bin/env python3
"""Record the robot pose together with every tracked Gazebo object.

Unlike ``track_traj.py``, which knows the name of a single object up front, this
node *discovers* the objects at runtime. Gazebo populates the world only after
the launch file finishes, so the tracked set is not known until messages start
arriving; the CSV header is therefore written lazily, on the first callback that
finds something to track, and never again.

Objects are matched either by a name prefix (``track_object``) or, with
``track_all_objects``, every model that is not the robot itself.
"""

import csv
import os

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node


class TrajectoryTracker(Node):
    """Discovers tracked objects from ``/model_states`` and logs their poses."""

    def __init__(self):
        super().__init__('coordinate_tracker')

        # ---- Parameters -------------------------------------------------
        self.declare_parameter("robot_model", "diffdrive_gazebo_bot")
        self.declare_parameter("tracking_prefix", "track_object")
        self.declare_parameter("track_all_objects", False)
        self.declare_parameter("world", "moving")
        self.declare_parameter("model", "NAC")
        self.declare_parameter("seed", 42)

        self.robot_name = self.get_parameter("robot_model").value
        self.tracking_prefix = self.get_parameter("tracking_prefix").value
        self.track_all_objects = self.get_parameter("track_all_objects").value
        self.world_name = self.get_parameter("world").value
        self.model_name = self.get_parameter("model").value
        self.seed = self.get_parameter("seed").value

        # ---- Output file ------------------------------------------------
        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "trajectories")
        os.makedirs(csv_dir, exist_ok=True)

        self.csv_path = os.path.join(
            csv_dir, f"{self.model_name}_{self.world_name}_trajectory_{self.seed}.csv")

        self.csv_file = open(self.csv_path, "w", newline="")
        # Left as None until the first tracked object appears: the header needs
        # one coordinate triple per object, which is unknown until then.
        self.csv_writer = None

        self.tracking_objects = []

        # ---- Input ------------------------------------------------------
        self.model_sub = self.create_subscription(
            ModelStates, "/model_states", self.model_states_callback, 10)

        self.get_logger().info(f"Robot model: {self.robot_name}")
        self.get_logger().info(f"Tracking prefix: {self.tracking_prefix}")
        self.get_logger().info(f"Track all objects: {self.track_all_objects}")
        self.get_logger().info("Waiting for Gazebo models...")

    # ---- Object discovery ----------------------------------------------
    def find_tracking_objects(self, model_names):
        """Return the subset of ``model_names`` that should be logged."""
        detected = []
        for name in model_names:
            if name == self.robot_name:          # never track the robot itself
                continue
            if self.track_all_objects or name.startswith(self.tracking_prefix):
                detected.append(name)
        return detected

    def update_tracking_objects(self, model_names):
        """Merge newly seen objects into the tracked set; write the header once."""
        new_objects = [n for n in self.find_tracking_objects(model_names)
                       if n not in self.tracking_objects]
        if not new_objects:
            return

        self.get_logger().info("Detected tracking objects: " + ", ".join(new_objects))
        self.tracking_objects.extend(new_objects)
        self.tracking_objects.sort()

        # The header is written exactly once, on the callback that discovers the
        # first objects. Objects appearing later are appended as columns only if
        # they sort before existing ones -- so keep the recorded world static.
        if self.csv_writer is None:
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["time", "robot_x", "robot_y", "robot_z"] +
                                     [f"{obj}_{axis}"
                                      for obj in self.tracking_objects
                                      for axis in "xyz"])
            self.csv_file.flush()
            self.get_logger().info("CSV header created.")
            self.get_logger().info(f"Tracking objects: {self.tracking_objects}")

    # ---- Callbacks -----------------------------------------------------
    def model_states_callback(self, msg):
        if self.robot_name not in msg.name:
            self.get_logger().warn(f"Robot '{self.robot_name}' not found in Gazebo.")
            return

        robot_position = msg.pose[msg.name.index(self.robot_name)].position
        self.update_tracking_objects(msg.name)

        # Nothing to record until at least one object has been discovered.
        if not self.tracking_objects:
            return

        time_sec = self.get_clock().now().nanoseconds * 1e-9
        row = [f"{time_sec:.4f}",
               f"{robot_position.x:.6f}",
               f"{robot_position.y:.6f}",
               f"{robot_position.z:.6f}"]

        for object_name in self.tracking_objects:
            if object_name in msg.name:
                position = msg.pose[msg.name.index(object_name)].position
                row.extend([f"{position.x:.6f}",
                            f"{position.y:.6f}",
                            f"{position.z:.6f}"])
            else:
                # Object has vanished this cycle: pad with blanks to keep the
                # row width aligned with the header.
                row.extend(["", "", ""])

        self.csv_writer.writerow(row)
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
