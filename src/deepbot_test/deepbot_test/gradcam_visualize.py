#!/usr/bin/env python3
"""Visualise which image regions drive the policy, using Grad-CAM on each conv block.

The robot is driven exactly as in ``test_ai.py``, but the same forward pass also
yields the five Conv2D activation maps. The gradient of one output (steering or
throttle) w.r.t. each map produces a heatmap, shown in a 3x2 grid: the original
frame plus one panel per conv block. The maps are also written to ``.mp4`` files
so a full episode can be reviewed later.

Keys: ``1`` select linear.x, ``2`` select angular.z, ``r`` start/stop recording,
``q``/``Esc`` quit.
"""

import os
import pickle
from collections import deque
from datetime import datetime

import cv2
import keras
import numpy as np
import rclpy
import tensorflow as tf
from geometry_msgs.msg import Twist
from keras.layers import (LSTM, Activation, Conv2D, Dense, Flatten,
                          GlobalAveragePooling2D, Input, Lambda,
                          MultiHeadAttention, TimeDistributed)
from keras.models import Model
from keras.regularizers import l2
from keras_nac.layers import LAN, NAC
from ncps.keras import CfC, LTC
from ncps.wirings import AutoNCP
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from .utils import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH

# Loading a pickled checkpoint executes pickle data, hence the explicit opt-in.
keras.config.enable_unsafe_deserialization()

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradcam_recordings")

SHOW_IMAGE = True
SHOW_GRADCAM = True
GRADCAM_TIMESTEP = -1          # which frame of the window to explain (-1 = newest)
GRADCAM_ALPHA = 0.75           # heatmap blend weight over the original frame
RECORD_FPS = 10.0

GRADCAM_TARGETS = {ord("1"): 0, ord("2"): 1}   # key -> output index
GRADCAM_TARGET_NAMES = {0: "linear.x", 1: "angular.z"}

# Steering/speed limits, matching test_ai.py.
MAX_LINEAR_X = 0.8
MAX_ANGULAR_Z = 1.5


class AiInference(Node):
    def __init__(self):
        super().__init__("ai_inference")

        self.declare_parameter("model", "NAC")
        self.model_name = self.get_parameter("model").get_parameter_value().string_value

        # ``model`` is the plain predictor; ``gradcam_model`` is the same graph
        # with every Conv2D output exposed alongside the prediction. Both share
        # weights, so only one needs to be exercised.
        self.model, self.gradcam_model = self.build_models(self.model_name)

        with open(os.path.join(WEIGHTS_DIR, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        self.gradcam_target = 0
        self.frames = deque(maxlen=SEQUENCE_LENGTH)

        self.recording = False
        self.video_writers = []
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        self.subscription = self.create_subscription(
            CompressedImage, "/camera_sensor/image_raw/compressed",
            self.image_callback, 10)
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.get_logger().info(f"Loaded model: {self.model_name}")
        self.get_logger().info(
            "Press r to start/stop recording, 1 for linear.x, 2 for angular.z, "
            "q or Esc to quit")

    # ---- Model ---------------------------------------------------------
    def build_models(self, rnn="NAC"):
        """Build the predictor and the activation-tapped twin for ``rnn``.

        The conv trunk mirrors ``train.py:build_model`` exactly, because the
        weights are loaded positionally from the checkpoint; only the recurrent
        cell and the extra output taps differ.
        """
        inp = Input(shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
        x = TimeDistributed(Lambda(lambda x: x / 127.5 - 1.0))(inp)

        conv_outputs = []
        x = TimeDistributed(Conv2D(64, (7, 7), strides=(2, 2), activation="relu",
                                   padding="same", kernel_regularizer=l2(1e-5)))(x)
        conv_outputs.append(x)
        x = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation="relu",
                                   padding="same", kernel_regularizer=l2(1e-5)))(x)
        conv_outputs.append(x)
        x = TimeDistributed(Conv2D(64, (3, 3), strides=(3, 3), activation="relu",
                                   padding="same", kernel_regularizer=l2(3.02e-5)))(x)
        conv_outputs.append(x)
        x = TimeDistributed(Conv2D(48, (7, 7), strides=(3, 3), activation="relu",
                                   padding="same", kernel_regularizer=l2(1e-5)))(x)
        conv_outputs.append(x)
        x = TimeDistributed(Conv2D(24, (7, 7), strides=(2, 2), activation="relu",
                                   padding="same", kernel_regularizer=l2(1.4e-5)))(x)
        conv_outputs.append(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)

        if rnn == "NAC":
            x = NAC(d_model=64, num_heads=16, sparsity=0.8, topk=16, activation="tanh",
                    return_sequences=False, use_riemann_sum=False)(x)
        elif rnn == "LSTM":
            x = LSTM(64, activation="tanh", return_sequences=False)(x)
        elif rnn == "LAN":
            x = LAN(d_model=64, num_heads=16, topk=16, activation="tanh",
                    return_sequences=False)(x)
        elif rnn == "MHA":
            x = MultiHeadAttention(key_dim=64, num_heads=16)(x, x)
            x = Activation("tanh")(x)
            x = Flatten()(x)
        elif rnn == "LTC":
            x = Dense(32, activation="relu")(x)
            x = LTC(AutoNCP(units=32, output_size=2), return_sequences=False)(x)
        elif rnn == "CfC":
            x = Dense(32, activation="relu")(x)
            x = CfC(AutoNCP(units=32, output_size=2), return_sequences=False)(x)
        else:
            raise ValueError(f"Unknown rnn policy: {rnn!r}")

        if rnn not in ("LTC", "CfC"):
            # The ncps cells emit the 2-vector directly; the rest need a head.
            x = Dense(32, activation="relu")(x)
            x = Dense(2, activation="linear")(x)

        model = Model(inputs=inp, outputs=x)
        gradcam_model = Model(inputs=inp, outputs=conv_outputs + [x])

        weights_path = os.path.join(WEIGHTS_DIR, f"deepbot_{rnn}.keras")
        model.load_weights(weights_path)
        return model, gradcam_model

    # ---- Grad-CAM ------------------------------------------------------
    def compute_gradcams(self, x, target_index=0, timestep=-1):
        """Return the prediction and one normalised heatmap per conv block.

        Standard Grad-CAM: average the incoming gradients over the spatial axes
        to get per-channel weights, take the weighted sum of activations, ReLU,
        then min-max normalise so every panel renders at a usable contrast.
        """
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.gradcam_model(x_tensor, training=False)
            target = outputs[-1][:, target_index]

        heatmaps = []
        for activation in outputs[:-1]:
            gradients = tape.gradient(target, activation)
            activation_t = activation[:, timestep, :, :, :]
            gradients_t = gradients[:, timestep, :, :, :]

            weights = tf.reduce_mean(gradients_t, axis=(1, 2), keepdims=True)
            cam = tf.nn.relu(tf.reduce_sum(weights * activation_t, axis=-1))
            cam = (cam - tf.reduce_min(cam, axis=(1, 2), keepdims=True)) / (
                tf.reduce_max(cam, axis=(1, 2), keepdims=True)
                - tf.reduce_min(cam, axis=(1, 2), keepdims=True) + 1e-8)
            heatmaps.append(cam[0].numpy())

        del tape
        return outputs[-1].numpy(), heatmaps

    def make_gradcam_overlay(self, image, heatmap):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]),
                             interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 1 - GRADCAM_ALPHA, heatmap, GRADCAM_ALPHA, 0)

    def add_label(self, image, text):
        image = image.copy()
        cv2.rectangle(image, (0, 0), (image.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def build_gradcam_grid(self, image, heatmaps):
        """Original frame plus the five conv panels, tiled 3x2."""
        panels = [self.add_label(image, "Original")]
        for i, heatmap in enumerate(heatmaps):
            panels.append(self.add_label(self.make_gradcam_overlay(image, heatmap),
                                         f"Conv2D #{i + 1}"))
        top = np.hstack((panels[0], panels[1]))
        middle = np.hstack((panels[2], panels[3]))
        bottom = np.hstack((panels[4], panels[5]))
        return np.vstack((top, middle, bottom))

    # ---- Recording -----------------------------------------------------
    def start_recording(self):
        if self.recording:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writers = []
        for name in ["original", "conv1", "conv2", "conv3", "conv4", "conv5"]:
            path = os.path.join(RECORDINGS_DIR,
                                f"{self.model_name}_{name}_{timestamp}.mp4")
            self.video_writers.append(cv2.VideoWriter(path, fourcc, RECORD_FPS, (320, 240)))
        self.recording = True
        self.get_logger().info(f"Recording started: {RECORDINGS_DIR}")

    def stop_recording(self):
        if not self.recording:
            return
        for writer in self.video_writers:
            writer.release()
        self.video_writers = []
        self.recording = False
        self.get_logger().info(f"Recording saved: {RECORDINGS_DIR}")

    def record_frames(self, original, heatmaps):
        if not self.recording:
            return
        self.video_writers[0].write(original)
        for i, heatmap in enumerate(heatmaps):
            self.video_writers[i + 1].write(self.make_gradcam_overlay(original, heatmap))

    # ---- ROS callback --------------------------------------------------
    def image_callback(self, msg):
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return

            display_frame = frame.copy()
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # The network was trained on RGB; OpenCV hands us BGR.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)

            self.frames.append(frame_rgb)
            if len(self.frames) < SEQUENCE_LENGTH:
                return
            x = np.array(self.frames, dtype=np.float32)[None, ...]

            if SHOW_GRADCAM:
                prediction, heatmaps = self.compute_gradcams(
                    x, self.gradcam_target, GRADCAM_TIMESTEP)
            else:
                prediction = self.model(x, training=False).numpy()
                heatmaps = []

            prediction = self.scaler.inverse_transform(prediction)[0]
            linear_x = float(np.clip(prediction[0], -MAX_LINEAR_X, MAX_LINEAR_X))
            angular_z = float(np.clip(prediction[1], -MAX_ANGULAR_Z, MAX_ANGULAR_Z))

            cmd = Twist()
            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self.publisher.publish(cmd)

            if SHOW_GRADCAM:
                self.record_frames(display_frame, heatmaps)
                grid = self.build_gradcam_grid(display_frame, heatmaps)
                cv2.putText(grid, f"linear.x: {linear_x:.3f}",
                            (10, grid.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(grid, f"angular.z: {angular_z:.3f}",
                            (10, grid.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
                if self.recording:
                    cv2.putText(grid, "RECORDING", (grid.shape[1] - 180, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                                cv2.LINE_AA)
                cv2.imshow("Grad-CAM", grid)
            elif SHOW_IMAGE:
                cv2.putText(display_frame, f"linear.x: {linear_x:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"angular.z: {angular_z:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("AI Inference", display_frame)

            self.handle_key(cv2.waitKey(1) & 0xFF)

        except Exception as e:
            self.get_logger().error(str(e))

    def handle_key(self, key):
        if key in GRADCAM_TARGETS:
            self.gradcam_target = GRADCAM_TARGETS[key]
            self.get_logger().info(
                f"Grad-CAM target: {GRADCAM_TARGET_NAMES[self.gradcam_target]}")
        elif key == ord("r"):
            self.stop_recording() if self.recording else self.start_recording()
        elif key in (ord("q"), 27):
            self.stop_recording()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AiInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_recording()
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
