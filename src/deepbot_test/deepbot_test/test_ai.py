#!/usr/bin/env python3
"""Drive the simulated robot with a trained behavioural-cloning policy.

Each incoming camera frame is resized and pushed into a fixed-length window; once
the window is full the model turns the last ``SEQUENCE_LENGTH`` frames into a
``(linear.x, angular.z)`` pair, which is un-scaled, clipped to the vehicle's
limits and published on ``/cmd_vel``.

Select the checkpoint with the ``model`` parameter, e.g.
``ros2 run deepbot_test test_ai_node --ros-args -p model:=LAN``.
"""

import os

# The network is small enough that CPU inference keeps up with the camera, and
# the node is often launched next to Gazebo on a machine with no spare VRAM.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
from collections import deque

import cv2
import keras
import numpy as np
import rclpy
import tensorflow as tf
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH

# Checkpoints are stored by pickling the model itself; loading one therefore
# executes pickle data, hence the explicit opt-in below.
keras.config.enable_unsafe_deserialization()

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
SHOW_IMAGE = True

# Steering/speed limits of the platform, applied as a last defence against a
# policy that predicts outside the range it was trained on.
MAX_LINEAR_X = 0.8
MAX_ANGULAR_Z = 1.5


class AiInference(Node):
    def __init__(self):
        super().__init__("ai_inference")

        self.declare_parameter("model", "NAC")
        self.model_name = self.get_parameter("model").get_parameter_value().string_value

        # Some architectures (NAC) use stochastic layer initialisation that is
        # not fully pinned by the saved weights; fixing the seed keeps repeated
        # evaluation runs comparable.
        self.declare_parameter("seed", 42)
        self.seed = self.get_parameter("seed").value
        tf.random.set_seed(int(self.seed))
        np.random.seed(int(self.seed))

        self.model = keras.models.load_model(
            os.path.join(WEIGHTS_DIR, f"deepbot_{self.model_name}.keras"))

        with open(os.path.join(WEIGHTS_DIR, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        self.frames = deque(maxlen=SEQUENCE_LENGTH)
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera_sensor/image_raw/compressed",
            self.image_callback, 10)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info(f"Loaded model: {self.model_name}")

    def image_callback(self, msg):
        try:
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warning("Failed to decode image")
                return

            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
            self.frames.append(image)

            # Wait until the window holds a full sequence before predicting.
            if len(self.frames) < SEQUENCE_LENGTH:
                return

            x = np.expand_dims(np.asarray(self.frames, dtype=np.float32), axis=0)

            # Model output lives in MinMax-scaled space; map it back to m/s and
            # rad/s before publishing.
            prediction = self.scaler.inverse_transform(
                self.model.predict(x, verbose=0))[0]
            prediction[0] = np.clip(prediction[0], -MAX_LINEAR_X, MAX_LINEAR_X)
            prediction[1] = np.clip(prediction[1], -MAX_ANGULAR_Z, MAX_ANGULAR_Z)

            twist = Twist()
            twist.linear.x = float(prediction[0])
            twist.angular.z = float(prediction[1])
            self.twist_pub.publish(twist)

            if SHOW_IMAGE:
                display = cv2.resize(image.astype(np.uint8), (320, 240))
                cv2.putText(display, f"linear.x: {prediction[0]:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"angular.z: {prediction[1]:.3f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"{self.model_name} Predictions", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = AiInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
