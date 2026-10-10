#!/home/waleed/anaconda3/envs/tf/bin/python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
from collections import deque
import cv2
import keras
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from .utils import SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, resize
    
keras.config.enable_unsafe_deserialization()

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model_weights",)
SHOW_IMAGE = True

class AiInference(Node):
    def __init__(self):
        super().__init__("ai_inference")

        #loading model
        self.declare_parameter("model", "NAC")
        self.model_name = self.get_parameter("model").get_parameter_value().string_value
        self.model = keras.models.load_model(os.path.join(WEIGHTS_DIR,f"deepbot_{self.model_name}.keras"))

        with open(f"{WEIGHTS_DIR}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.frames = deque(maxlen=SEQUENCE_LENGTH)
        self.image_sub = self.create_subscription(CompressedImage, '/camera_sensor/image_raw/compressed', self.image_callback, 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info(f"Loaded model: {self.model_name}")

    def image_callback(self, msg):
        try:
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warning("Failed to decode image")
                return
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
            self.frames.append(image)
            if len(self.frames) < SEQUENCE_LENGTH:
                return
            x = np.expand_dims(np.asarray(self.frames, dtype=np.float32), axis=0)
            prediction = self.scaler.inverse_transform(self.model.predict(x, verbose=0))[0]
            prediction[0] = np.clip(prediction[0], -0.8, 0.8)
            prediction[1] = np.clip(prediction[1], -1.5, 1.5)
            twist = Twist()
            twist.linear.x = float(prediction[0])
            twist.angular.z = float(prediction[1])
            self.twist_pub.publish(twist)

            if SHOW_IMAGE:
                display = cv2.resize(image.astype(np.uint8), (320, 240))
                cv2.putText(display, f"linear.x: {prediction[0]:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"angular.z: {prediction[1]:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"{self.model_name} Predictions", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    rclpy.shutdown()
                    
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AiInference()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
