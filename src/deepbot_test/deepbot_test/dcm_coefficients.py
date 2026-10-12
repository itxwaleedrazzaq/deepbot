"""Recover the coefficients of a trained LAN's learned dynamics.

The layer's per-pair logit obeys ``da/dt = -gamma(u) * a + phi(u)``, where
``gamma`` and ``phi`` are the two heads of the gate network. Around the operating
point this is a first-order (bilinear) system, and its Jacobians give three
quantities that are comparable across models:

    A       = -gamma(u0)                state decay (how fast a logit forgets)
    B_int   = -<grad_u gamma, A_out>    self-stabilising response
    C_ext   =  <grad_u phi,   A_out>    external drive from an intervention

``A_out`` here is an arbitrary unit direction in driver space: the encoder that
maps a real physical intervention to pixel changes is not available offline, so
callers pass a direction and the code reports the projection onto it.

The subtlety this module exists to document: the gate LSTM runs over the *whole*
sequence of retrieved pairs for a given (head, query). A pair's logit therefore
depends on the pairs retrieved before it. ``gate_raw_single_pair`` ignores that
and is kept only to demonstrate the error; ``gate_raw_heads`` is the faithful
version and is what any gradient should be taken through.
"""

import os

# Force CPU: the probe is a handful of small Jacobians, and this keeps it usable
# on a headless evaluation box with no GPU or with one already in use.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import keras
import numpy as np
import tensorflow as tf

from .liquid_attention import LAN, LanGateExtractor, TAU_EPSILON

keras.config.enable_unsafe_deserialization()

TF_DTYPE = "float32"
EPS = TAU_EPSILON


# ---------------------------------------------------------------------------
# Model surgery
# ---------------------------------------------------------------------------
def load_lan_model(weights_path):
    """Load a checkpoint and freeze it (the probe only reads gradients)."""
    model = keras.models.load_model(weights_path, custom_objects={"LAN": LAN})
    model.trainable = False
    return model


def find_lan_layer(model):
    """Locate the LAN layer, looking one level down for wrapped layers."""
    for layer in model.layers:
        if layer.__class__.__name__ == "LAN":
            return layer
    for layer in model.layers:
        for sub in getattr(layer, "layers", []):
            if sub.__class__.__name__ == "LAN":
                return sub
    raise RuntimeError("No LAN layer found in the loaded model.")


def build_trunk(model, lan_layer):
    """Model mapping the image sequence to the LAN layer's input [B, T, F].

    This is the encoder side of A_out: everything the conv trunk does to the
    pixels before the attention layer ever sees them.
    """
    return keras.Model(inputs=model.inputs[0], outputs=lan_layer.input)


def build_lan_gates(lan_layer):
    """Model mapping a LAN input [B, T, F] to a dict of gate quantities.

    Head-resolved keys (``*_heads``) are the faithful ones; the plain keys are
    head means included for convenience.
    """
    F = lan_layer.input.shape[-1]
    inp = keras.Input(shape=(None, F), dtype=TF_DTYPE, name="lan_in")
    u_h, a_h, phi_h, gam_h, idx = LanGateExtractor(lan_layer, name="gate_extract")(inp)
    return keras.Model(inp, {
        "u_heads": u_h, "a_heads": a_h, "phi_heads": phi_h, "gam_heads": gam_h,
        "idx": idx,
        "u": keras.ops.mean(u_h, axis=1),
        "phi": keras.ops.mean(phi_h, axis=1),
        "gamma": keras.ops.mean(gam_h, axis=1),
        "gate_raw": keras.ops.mean(a_h, axis=1),
    })


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def gate_raw_single_pair(lan_layer, u_pair):
    """DEPRECATED -- presents one pair as a length-1 LSTM sequence.

    WRONG by construction: it discards the LSTM state accumulated over the pairs
    retrieved before this one, so the returned logit is not the one the trained
    model uses. Kept as the counter-example; use ``gate_raw_heads`` instead.
    """
    x = tf.reshape(u_pair, (1, 1, u_pair.shape[-1]))
    h = lan_layer.gate_in(x, training=False)
    return tf.squeeze(lan_layer.gate_out(h, training=False))


def gate_raw_heads(lan_layer, u_heads_seq, h_index, k_index):
    """FAITHFUL pre-activation ``a`` for one (head, pair) of a driver sequence.

    ``u_heads_seq`` is [H, K, 2D] in the exact order the LSTM consumes it. The
    LSTM genuinely runs per (head, query) over this sequence, so differentiating
    this scalar gives the true ``grad_u phi`` / ``grad_u gamma`` of the layer.
    """
    us = tf.convert_to_tensor(u_heads_seq, dtype=TF_DTYPE)
    # Treat the H leading heads as a batch: each runs its own independent LSTM.
    h = lan_layer.gate_in(us, training=False)
    a = lan_layer.gate_out(h, training=False)
    return a[h_index, k_index, 0]


def gate_raw_sequence_batch(lan_layer, u_seq):
    """All K logits for a single un-batched driver sequence."""
    x = tf.reshape(u_seq, (1, u_seq.shape[0], u_seq.shape[-1]))
    a = lan_layer.gate_out(lan_layer.gate_in(x, training=False), training=False)
    return tf.reshape(a, [-1])


def gates_from_logit(a, tau_epsilon=EPS):
    """The two gate heads of a logit: phi = relu(a), gamma = softplus(a) + eps."""
    return tf.nn.relu(a), tf.nn.softplus(a) + tau_epsilon


# ---------------------------------------------------------------------------
# Autodiff helpers
# ---------------------------------------------------------------------------
def jacobian_rows(fn, x, out_dim):
    """Dense Jacobian of ``fn: x[N] -> [out_dim]``; returns J with shape
    [out_dim, N]."""
    x = tf.convert_to_tensor(x, dtype=TF_DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.reshape(fn(x), [-1])
    return tf.stack([tf.reshape(tape.gradient(y[i], x), [-1])
                     for i in range(y.shape[0])])


def scalar_grad(fn, x):
    """Gradient of a scalar ``fn`` at ``x``, preserving ``x``'s shape."""
    x = tf.convert_to_tensor(x, dtype=TF_DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = fn(x)
    g = tape.gradient(y, x)
    if g is None:
        raise RuntimeError("Gradient is None: fn does not depend on x.")
    return g


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
class LanDCMProbe:
    """Estimates A, B_int and C_ext for a trained LAN checkpoint.

    Two modes, which differ only in what ``u`` means:

    ``std``  u is the LAN input feature vector produced by the conv trunk.
    ``wp``   u is a [q ; k] pair, the object the gate actually consumes.
    """

    def __init__(self, weights_path, mode="std"):
        assert mode in ("std", "wp")
        self.mode = mode

        self.model = load_lan_model(weights_path)
        self.lan = find_lan_layer(self.model)
        self.d_model = self.lan.d_model
        self.num_heads = self.lan.num_heads
        self.depth = self.lan.depth
        self.topk = self.lan.topk

        self.trunk = build_trunk(self.model, self.lan)
        self.gate_model = build_lan_gates(self.lan)

    # ---- driver extraction ---------------------------------------------
    def features_at(self, x_images):
        """Images [B, 8, 75, 75, 3] -> LAN input features [B, 8, F]."""
        return self.trunk.predict(x_images, verbose=0)

    def driver_at(self, x_images):
        """Images -> gate dict, with drivers for the last position's pairs."""
        return self.driver_from_features(self.features_at(x_images))

    def driver_from_features(self, feats):
        return self.gate_model.predict(feats, verbose=0)

    # ---- collapsed-level scalars ---------------------------------------
    def scalars_at_driver(self, u_sample):
        """Gate scalars at a driver sample. Not implemented: the collapsed level
        is model-specific and subclasses/tests supply their own."""
        raise NotImplementedError

    # ---- A_out: encoder Jacobian ---------------------------------------
    def encoder_jacobian(self, x_images, driver_index=0, n_probe_dirs=1, seed=0):
        """J = d u / d x for one retrieved pair, via per-output VJP.

        The image is 75*75*3*8 = 135000 inputs, so a dense Jacobian is not
        tractable; each row is obtained with one backward pass instead.
        """
        inp = tf.convert_to_tensor(x_images, dtype=TF_DTYPE)
        with tf.GradientTape() as tape:
            tape.watch(inp)
            u = self.gate_model(inp, training=False)["u"]      # [1, K, 2D]

        u_vec = tf.reshape(u[:, driver_index, :], [-1])
        return tf.stack([tf.reshape(tape.gradient(u_vec[i], inp), [-1])
                         for i in range(u_vec.shape[0])])      # [2D, N_pixels]

    # ---- gate gradients ------------------------------------------------
    def gate_gradients(self, u_pair, a_out):
        """(B_int, C_ext, grad_gamma, grad_phi) at the operating point ``u_pair``.

        Uses ``gate_raw_single_pair``, so the gradients are those of the
        *unrolled single-pair* gate rather than the trained layer's. Kept for
        comparison against the batched estimator in ``test_ai_coeff``.
        """
        u0 = tf.convert_to_tensor(u_pair, dtype=TF_DTYPE)

        def phi_of_u(u):
            return tf.nn.relu(gate_raw_single_pair(self.lan, u))

        def gamma_of_u(u):
            return tf.nn.softplus(gate_raw_single_pair(self.lan, u)) + EPS

        gphi = scalar_grad(phi_of_u, u0)
        ggamma = scalar_grad(gamma_of_u, u0)

        a_out_t = tf.convert_to_tensor(a_out, dtype=TF_DTYPE)
        C_ext = tf.reduce_sum(gphi * a_out_t)
        B_int = -tf.reduce_sum(ggamma * a_out_t)
        return float(B_int), float(C_ext), ggamma.numpy(), gphi.numpy()

    def A_state_coupling(self, u_pair):
        """A = -gamma(u0), the decay felt by a logit at this operating point."""
        a = gate_raw_single_pair(self.lan, tf.convert_to_tensor(u_pair, dtype=TF_DTYPE))
        return float(tf.nn.softplus(a) + EPS)

    # ---- empirical finite-difference check -----------------------------
    def empirical_delta_logit(self, x_images, perturb_dir_img, iota):
        """Field G(a, u) = phi - gamma * a before and after an image perturbation.

        A one-step Euler residual of the ODE, used as the ground truth the
        analytic coefficients above should reproduce.
        """
        x0 = tf.convert_to_tensor(x_images, dtype=TF_DTYPE)
        x1 = x0 + iota * tf.convert_to_tensor(perturb_dir_img, dtype=TF_DTYPE)

        g0 = self.gate_model.predict(np.asarray(x0), verbose=0)
        g1 = self.gate_model.predict(np.asarray(x1), verbose=0)

        def field(g):
            return g["phi"] - g["gamma"] * g["gate_raw"]

        return field(g0), field(g1)
