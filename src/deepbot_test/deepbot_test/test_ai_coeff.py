#!/usr/bin/env python3
"""Drive the robot and log the LAN's learned dynamics alongside each command.

This is ``test_ai.py`` with a probe attached. The control path is identical; on
every ``coeff_every``-th frame the node also computes the local DCM coefficients
of the trained LAN's gate ODE and appends them to a CSV, so the driving run and
the dynamics analysis refer to exactly the same states.

The coefficients
----------------
Per (head, retrieved pair) the logit obeys ``da/dt = -gamma * a + phi``. Two ways
of describing how an intervention reaches it are logged:

    partial = B_int * a0 + C_ext     only the direct pair -> gate path
    total   = <grad_u G, A_out>      the derivative of the whole field G

They differ by ``omitted_term``, i.e. the part of the perturbation that acts on
``gamma`` in its role as the *decay* multiplying ``a`` rather than as a drive.
``dG_emp`` is the same total field change measured by finite differences, so a
match between ``total`` and ``dG_emp`` is a check on the analytic gradient.

Cost
----
One batched Jacobian per logged frame, dominated by the LSTM backward pass
(~2 s on CPU for 16 heads x 8 pairs). Raise ``coeff_every`` on slower machines.
"""

import csv
import os
import pickle
import time
from collections import deque

import cv2
import keras
import numpy as np
import rclpy
import tensorflow as tf
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# Both layers must be importable so Keras can resolve the serialized names
# ("keras_nac>LAN", "LanGateExtractor") when the checkpoint is deserialised.
from .dcm_coefficients import EPS, find_lan_layer
from .liquid_attention import LAN, LanGateExtractor  # noqa: F401
from .utils import IMAGE_HEIGHT, IMAGE_WIDTH, SEQUENCE_LENGTH

# Loading a pickled checkpoint executes pickle data, hence the explicit opt-in.
keras.config.enable_unsafe_deserialization()

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights")
DEFAULT_WEIGHTS_DIR = WEIGHTS_DIR

MAX_LINEAR_X = 0.8
MAX_ANGULAR_Z = 1.5

# Columns produced by the batched estimator, plus the trailing A_out vector.
BASE_COLUMNS = [
    "t_wall", "t_ros", "step", "linear_x", "angular_z",
    "A", "B_int", "C_ext", "a0", "phi", "gamma",
    "dG_pred_partial", "dG_emp", "omitted_term", "K", "H",
]


class AiInferenceCoeff(Node):
    def __init__(self):
        super().__init__("ai_inference_coeff")

        # ---- Parameters ------------------------------------------------
        self.declare_parameter("model", "LAN")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("coeff_every", 5)
        self.declare_parameter("A_out_seed", 0)
        self.declare_parameter("iota", 1e-3)
        self.declare_parameter("weights_dir", DEFAULT_WEIGHTS_DIR)
        self.declare_parameter("show_image", True)

        self.model_name = self.get_parameter("model").get_parameter_value().string_value
        self.coeff_every = int(self.get_parameter("coeff_every").value)
        self.A_out_seed = int(self.get_parameter("A_out_seed").value)
        self.iota = float(self.get_parameter("iota").value)
        weights_dir = self.get_parameter("weights_dir").get_parameter_value().string_value
        self.show_image = bool(self.get_parameter("show_image").value)

        csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories")
        self.csv_path = os.path.join(csv_dir, "coefficient_trajectory.csv")

        # ---- Model and scaler ------------------------------------------
        model_file = os.path.join(weights_dir, f"deepbot_{self.model_name}.keras")
        self.model = keras.models.load_model(
            model_file, custom_objects={"LAN": LAN,
                                        "LanGateExtractor": LanGateExtractor})
        with open(os.path.join(weights_dir, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        self.get_logger().info(f"Loaded model: {self.model_name} from {model_file}")

        # ---- Probe -----------------------------------------------------
        # Only LAN has gates to probe. Non-LAN checkpoints are still driven, they
        # just produce no coefficient columns.
        self.lan = None
        self.trunk = None
        self.gate_model = None
        self.A_out = None
        self.has_coeffs = False
        try:
            self.lan = find_lan_layer(self.model)
            self._build_coeff_graphs()

            # A fixed random unit direction stands in for the (unavailable)
            # physical intervention; seeding it makes runs comparable.
            rng = np.random.default_rng(self.A_out_seed)
            v = rng.standard_normal(self.lan.depth * 2).astype("float32")
            self.A_out = v / np.linalg.norm(v)
            self.has_coeffs = True
            self.get_logger().info(
                f"Coefficient probe ENABLED (LAN d_model={self.lan.d_model}, "
                f"heads={self.lan.num_heads}, topk={self.lan.topk})")
        except Exception as e:
            self.get_logger().warn(
                f"Coefficient probe DISABLED for model '{self.model_name}': {e}")

        # ---- CSV -------------------------------------------------------
        os.makedirs(csv_dir, exist_ok=True)
        n_aout = len(self.A_out) if self.A_out is not None else 0
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(BASE_COLUMNS + [f"A_out_{i}" for i in range(n_aout)])
        self._csv_file.flush()
        self.get_logger().info(f"Writing coefficients to {self.csv_path}")
        self._step = 0

        # ---- ROS plumbing ----------------------------------------------
        self.frames = deque(maxlen=SEQUENCE_LENGTH)
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera_sensor/image_raw/compressed",
            self.image_callback, 10)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    # ---- Coefficient graphs --------------------------------------------
    def _build_coeff_graphs(self):
        """Build the trunk->features model and the per-head gate model."""
        lan = self.lan
        self.trunk = keras.Model(inputs=self.model.inputs[0], outputs=lan.input)

        F = lan.input.shape[-1]
        inp = keras.Input(shape=(None, F), dtype="float32", name="lan_in")
        u_h, a_h, phi_h, gam_h, idx = LanGateExtractor(lan, name="gate_extract")(inp)
        self.gate_model = keras.Model(inp, {
            "u_heads": u_h, "a_heads": a_h, "phi_heads": phi_h,
            "gam_heads": gam_h, "idx": idx,
            "u": keras.ops.mean(u_h, axis=1),
            "gate_raw": keras.ops.mean(a_h, axis=1),
        })

    def _gate_logits(self, us):
        """All per-head logits for a driver sequence [H, K, 2D] -> [H, K].

        The H heads act as the batch dimension, so each runs its own LSTM over
        the retrieved pairs -- matching how the layer runs at inference.
        """
        h = self.lan.gate_in(us, training=False)                 # [H, K, d_model]
        return tf.reshape(self.lan.gate_out(h, training=False),  # [H, K, 1]
                          (tf.shape(us)[0], tf.shape(us)[1]))

    def _all_coefficients(self, u_heads):
        """Per-(head, pair) coefficients from a single backward pass.

        All H*K logits come from one LSTM pass over the [H, K, 2D] driver. The
        four fields (a, gamma, phi, G) are then stacked into one [4M]-valued
        function so a single ``batch_jacobian`` yields every derivative, rather
        than four separate tapes. For each pair m only its own [2D] block of the
        Jacobian is kept -- the cross-pair entries are the LSTM's memory and are
        deliberately excluded from these local coefficients.
        """
        us = tf.convert_to_tensor(u_heads, dtype="float32")      # [H, K, 2D]
        H, K, D2 = int(us.shape[0]), int(us.shape[1]), int(us.shape[2])
        M = H * K
        a_out = np.asarray(self.A_out, dtype="float32")          # [2D]

        pairs = [(m // K, m % K) for m in range(M)]

        def jac_of_stacked(fns):
            """One batched Jacobian for several [M]-valued fields at once."""
            us_flat = tf.reshape(us, (1, M * D2))
            P = len(fns)

            def f(x):
                usr = tf.reshape(x, (H, K, D2))
                return tf.reshape(tf.concat([fn(usr) for fn in fns], axis=0), (1, P * M))

            with tf.GradientTape() as jt:
                jt.watch(us_flat)
                out = f(us_flat)
            J = jt.batch_jacobian(out, us_flat)                  # [1, P*M, M*D2]
            J = tf.reshape(J, (P, M, H, K, D2)).numpy()
            return [np.stack([J[p, m][pairs[m]] for m in range(M)]) for p in range(P)]

        def f_a(u):
            return tf.reshape(self._gate_logits(u), (M,))

        def f_g(u):
            return tf.reshape(tf.nn.softplus(self._gate_logits(u)) + EPS, (M,))

        def f_p(u):
            return tf.reshape(tf.nn.relu(self._gate_logits(u)), (M,))

        def f_G(u):
            # G = -gamma * a + phi, the ODE's right-hand side.
            a = self._gate_logits(u)
            return tf.reshape(-(tf.nn.softplus(a) + EPS) * a + tf.nn.relu(a), (M,))

        ga, gg, gp, gf = jac_of_stacked([f_a, f_g, f_p, f_G])

        # Operating-point values of the gates at this driver.
        a = self._gate_logits(us)
        gam = tf.nn.softplus(a) + EPS
        a_v, gam_v = a.numpy().reshape(M), gam.numpy().reshape(M)
        phi_v = tf.nn.relu(a).numpy().reshape(M)

        # Project every gradient onto the intervention direction.
        C_ext = gp.dot(a_out)
        B_int = -(gg.dot(a_out))
        partial = B_int * a_v + C_ext
        total = gf.dot(a_out)

        return dict(A=-gam_v, B_int=B_int, C_ext=C_ext,
                    a0=a_v, phi=phi_v, gamma=gam_v,
                    partial=partial, total=total)

    def _empirical_dG(self, u_heads):
        """Finite-difference dG/diota per (head, pair).

        Every retrieved pair is perturbed along A_out at once and each pair's own
        field change is read from the same forward pass.
        """
        us = tf.convert_to_tensor(u_heads, dtype="float32")      # [H, K, 2D]
        H, K, D2 = int(us.shape[0]), int(us.shape[1]), int(us.shape[2])
        a_out = tf.convert_to_tensor(self.A_out, dtype="float32")
        delta = tf.broadcast_to(a_out[None, None, :], (H, K, D2))

        def field(usq):
            a = self._gate_logits(usq)
            return -(tf.nn.softplus(a) + EPS) * a + tf.nn.relu(a)

        G0 = field(us)
        G1 = field(us + self.iota * delta)
        return ((G1 - G0) / self.iota).numpy()                   # [H, K]

    def _coefficients_for(self, feats):
        """Coefficients for one frame, averaged over all heads and pairs."""
        u_heads = self.gate_model.predict(feats, verbose=0)["u_heads"][0]  # [H, K, 2D]
        H, K = int(u_heads.shape[0]), int(u_heads.shape[1])

        d = self._all_coefficients(u_heads)
        agg = {key: float(np.mean(d[key]))
               for key in ("A", "B_int", "C_ext", "a0", "phi", "gamma",
                           "partial", "total")}
        agg["dG_emp"] = float(np.mean(self._empirical_dG(u_heads)))
        agg["omitted_term"] = agg["total"] - agg["partial"]
        agg["K"], agg["H"] = K, H
        return agg

    # ---- ROS callback --------------------------------------------------
    def image_callback(self, msg):
        try:
            image = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warning("Failed to decode image")
                return

            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float32)
            self.frames.append(image)
            if len(self.frames) < SEQUENCE_LENGTH:
                return

            x = np.expand_dims(np.asarray(self.frames, dtype=np.float32), axis=0)

            # ---- control, identical to test_ai.py ----------------------
            prediction = self.scaler.inverse_transform(
                self.model.predict(x, verbose=0))[0]
            prediction[0] = np.clip(prediction[0], -MAX_LINEAR_X, MAX_LINEAR_X)
            prediction[1] = np.clip(prediction[1], -MAX_ANGULAR_Z, MAX_ANGULAR_Z)

            twist = Twist()
            twist.linear.x = float(prediction[0])
            twist.angular.z = float(prediction[1])
            self.twist_pub.publish(twist)

            # ---- coefficients ------------------------------------------
            if self.has_coeffs and self._step % self.coeff_every == 0:
                feats = self.trunk.predict(x, verbose=0)
                c = self._coefficients_for(feats)
                row = [
                    f"{time.time():.6f}",
                    f"{self.get_clock().now().nanoseconds / 1e9:.6f}",
                    self._step,
                    f"{twist.linear.x:.6f}", f"{twist.angular.z:.6f}",
                    f"{c['A']:.8g}", f"{c['B_int']:.8g}", f"{c['C_ext']:.8g}",
                    f"{c['a0']:.8g}", f"{c['phi']:.8g}", f"{c['gamma']:.8g}",
                    f"{c['partial']:.8g}", f"{c['dG_emp']:.8g}",
                    f"{c['omitted_term']:.8g}", c["K"], c["H"],
                ]
                row.extend(f"{float(w):.8g}" for w in self.A_out)
                self.log_coefficients(row)

            self._step += 1

            if self.show_image:
                display = cv2.resize(image.astype(np.uint8), (320, 240))
                cv2.putText(display, f"linear.x: {twist.linear.x:.3f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"angular.z: {twist.angular.z:.3f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"{self.model_name} Predictions", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")

    def log_coefficients(self, row):
        try:
            self._csv.writerow(row)
            self._csv_file.flush()   # keep the CSV complete on a hard stop
        except Exception as e:
            self.get_logger().error(f"CSV write failed: {e}")

    def destroy_node(self):
        try:
            self._csv_file.flush()
            self._csv_file.close()
            self.get_logger().info(f"Closed CSV: {self.csv_path}")
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AiInferenceCoeff()
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
