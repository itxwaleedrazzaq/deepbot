"""LAN: a continuous-time attention layer with per-pair liquid gates.

Why this exists
---------------
Plain softmax attention scores a pair ``(query, key)`` with a single dot product
and turns the scores into weights in one shot. LAN instead treats each retrieved
pair as driving a scalar first-order ODE,

        da/dt = -tau(u) * a + phi(u) ,        u = [q ; k]

whose fixed point is ``phi / tau`` -- the "liquid" time constant. The ODE is
integrated with a few explicit Euler steps, and the resulting ``a`` are used as
the softmax logits. Because ``phi`` and ``tau`` are produced by a small LSTM over
the retrieved pairs, a pair's logit depends not only on its own ``(q, k)`` but on
the other pairs retrieved alongside it: the gate has memory, which a per-pair MLP
would not.

Two knobs matter in practice
----------------------------
``use_pairwise``
    ``False`` (default) keeps only the top-``k`` pairs by dot product, so the
    softmax runs over ``k`` entries per query. Set it to ``True`` to score every
    pair, at O(T^2) memory.
``return_attention`` / ``return_sequences``
    Return the attention map and/or the full sequence instead of the last step.

The companion ``LanGateExtractor`` below re-exposes the internals of a *trained*
layer on a symbolic tensor, which is how ``dcm_coefficients.py`` probes the
learned gates after training.
"""

import tensorflow as tf

# Added to softplus(gate) so tau is never zero; the ODE would otherwise lose its
# restoring term and the Euler update would be unbounded.
TAU_EPSILON = 1e-6


@tf.keras.utils.register_keras_serializable(package="keras_nac", name="LAN")
class LAN(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        topk: int = 8,
        euler_steps: int = 2,
        use_sink_gate: bool = True,
        activation=None,
        tau_epsilon: float = TAU_EPSILON,
        delta_t: float = 0.01,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_pairwise: bool = False,
        return_attention: bool = False,
        return_sequences: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.topk = int(topk)
        self.euler_steps = int(euler_steps)
        self.delta_t = float(delta_t)
        self.use_sink_gate = use_sink_gate
        self.tau_epsilon = tau_epsilon
        self.dropout = dropout
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_pairwise = use_pairwise
        self.return_attention = return_attention
        self.return_sequences = return_sequences

        self.q_dense = tf.keras.layers.Dense(d_model, use_bias=use_bias, name="q_proj")
        self.k_dense = tf.keras.layers.Dense(d_model, use_bias=use_bias, name="k_proj")
        self.v_dense = tf.keras.layers.Dense(d_model, use_bias=use_bias, name="v_proj")
        self.out_dense = tf.keras.layers.Dense(d_model, use_bias=use_bias, name="out_proj")

        # Two heads over each pair u = [q ; k]: the LSTM supplies the history
        # across retrieved pairs, the Dense collapses its state to one scalar.
        self.gate_in = tf.keras.layers.LSTM(d_model, use_bias=use_bias,
                                            return_sequences=True, name="gate_in")
        self.gate_out = tf.keras.layers.Dense(1, use_bias=use_bias, name="gate_out")

        self.sink_gate = tf.keras.layers.Dense(d_model, activation="sigmoid",
                                               use_bias=True, name="gate")
        self.attn_dropout = tf.keras.layers.Dropout(dropout)

    def build(self, input_shape):
        super().build(input_shape)

    # ---- head plumbing -------------------------------------------------
    def split_heads(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (B, T, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])          # [B, H, T, D]

    def combine_heads(self, x):
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        return tf.reshape(x, (B, T, self.d_model))

    def pairwise_concat(self, q, k):
        """All [q ; k] pairs -> [B, H, Tq, Tk, 2D]. O(T^2), used when
        ``use_pairwise`` is set."""
        q_exp = tf.expand_dims(q, axis=3)
        k_exp = tf.expand_dims(k, axis=2)
        q_tile = tf.tile(q_exp, [1, 1, 1, tf.shape(k)[2], 1])
        k_tile = tf.tile(k_exp, [1, 1, tf.shape(q)[2], 1, 1])
        return tf.concat([q_tile, k_tile], axis=-1)

    def sparse_topk_pairwise(self, q, k, K=None):
        """Only the top-``K`` keys per query -> [B, H, Tq, K, 2D]."""
        if K is None:
            K = self.topk
        scores = tf.einsum("bhqd,bhkd->bhqk", q, k)
        K_eff = tf.minimum(K, tf.shape(k)[2])

        _, topk_idx = tf.math.top_k(scores, k=K_eff)
        k_gathered = tf.gather(k, topk_idx, batch_dims=2, axis=2)

        q_tiled = tf.tile(tf.expand_dims(q, axis=3), [1, 1, 1, K_eff, 1])
        topk_pairs = tf.concat([q_tiled, k_gathered], axis=-1)
        return topk_pairs, topk_idx

    def _retrieve_pairs(self, q, k):
        """Route to the dense or the sparse pair construction."""
        if self.use_pairwise:
            return self.pairwise_concat(q, k), None
        return self.sparse_topk_pairwise(q, k)

    def _gate_heads(self, pairs):
        """Run the gate LSTM over retrieved pairs -> (phi, tau, gate_raw).

        ``pairs`` is [..., Tk, F]; the batch is flattened so the LSTM sees one
        sequence of retrieved pairs per (batch, head, query).
        """
        shape = tf.shape(pairs)
        B, H, Tq, Tk, F = shape[0], shape[1], shape[2], shape[3], shape[4]
        flat = tf.reshape(pairs, (B * H * Tq, Tk, F))
        gate_raw = tf.reshape(self.gate_out(self.gate_in(flat)), (-1, H, Tq, Tk))
        phi = tf.nn.relu(gate_raw)
        tau = tf.nn.softplus(gate_raw) + self.tau_epsilon
        return phi, tau, gate_raw

    def compute_phi_tau(self, q, k):
        """(phi, tau, retrieved indices) for every query, before the ODE."""
        pairs, idx = self._retrieve_pairs(q, k)
        phi, tau, _ = self._gate_heads(pairs)
        return phi, tau, idx

    def gates_and_pairs(self, q, k):
        """Same quantities as ``compute_phi_tau`` plus the driver pairs.

        Returns
            phi, tau : [B, H, Tq, Tk_ret]
            idx      : [B, H, Tq, Tk_ret] retrieved indices (None if dense)
            pairs    : [B, H, Tq, Tk_ret, 2D] the [q ; k] drivers u
        """
        pairs, idx = self._retrieve_pairs(q, k)
        phi, tau, _ = self._gate_heads(pairs)
        return phi, tau, idx, pairs

    # ---- forward pass --------------------------------------------------
    def call(self, inputs, mask=None, training=None):
        q_in = k_in = v_in = inputs
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            q_in, k_in, v_in = inputs
        elif isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            (q_in, v_in) = inputs
            k_in = q_in

        if isinstance(mask, (list, tuple)):
            mask = mask[0] if len(mask) > 0 else None
            if mask is None or (hasattr(mask, "shape") and mask.shape == ()):
                mask = None

        q = self.q_dense(q_in)
        k = self.k_dense(k_in)
        v = self.v_dense(v_in)

        qh, kh, vh = self.split_heads(q), self.split_heads(k), self.split_heads(v)

        phi, tau, topk_idx = self.compute_phi_tau(qh, kh)

        # Explicit Euler on da/dt = -tau * a + phi, starting from a = 0. The step
        # is capped so the update stays stable for the largest tau in the batch.
        dt = tf.cast(self.delta_t, phi.dtype)
        dt = tf.minimum(dt, 1.0 / (tf.reduce_max(tau) + 1e-12))

        a = tf.zeros_like(phi)
        for _ in range(self.euler_steps):
            a = a + dt * (-tau * a + phi)

        attn_logits = a
        if mask is not None:
            mask = tf.cast(mask, attn_logits.dtype)
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
            attn_logits = attn_logits + (1.0 - mask) * tf.constant(
                -1e9, dtype=attn_logits.dtype)

        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        if self.use_pairwise:
            output = tf.matmul(attn_weights, vh)
        else:
            vh_topk = tf.gather(vh, topk_idx, batch_dims=2, axis=2)
            output = tf.reduce_sum(attn_weights[..., tf.newaxis] * vh_topk, axis=3)

        combined = self.combine_heads(output)
        if self.use_sink_gate:
            # Per-channel gate on the query, letting the layer suppress the whole
            # attention output when no retrieved pair is informative.
            combined = combined * self.sink_gate(q_in)
        out = self.activation(self.out_dense(combined))

        result = out if self.return_sequences else out[:, -1, :]
        if self.return_attention:
            return result, attn_weights
        return result

    # ---- inspection ----------------------------------------------------
    def retrieve_gates(self, x):
        """Head-averaged gates at the LAST query position.

        The model runs with ``return_sequences=False``, so the last position is
        the one that produced the prediction. Averaging over heads matches the
        scalar-per-pair logit of the theory.

        Returns
            u_last        [B, K, 2D]  driver of each retrieved pair
            phi_last      [B, K]
            tau_last      [B, K]      (== gamma)
            gate_raw_last [B, K]      pre-activation a, recovered from tau
            idx           [B, H, Tq, K]
        """
        q = self.q_dense(x)
        k = self.k_dense(x)
        phi, tau, idx, pair = self.gates_and_pairs(self.split_heads(q),
                                                   self.split_heads(k))

        u_last = tf.reduce_mean(pair[:, :, -1, :, :], axis=1)
        phi_last = tf.reduce_mean(phi[:, :, -1, :], axis=1)
        tau_last = tf.reduce_mean(tau[:, :, -1, :], axis=1)

        # Inverse of softplus, clipped away from the epsilon floor where the
        # mapping would be numerically unstable.
        gate_raw_last = tf.math.log(
            tf.math.expm1(tf.maximum(tau_last - self.tau_epsilon, 1e-12)))
        return u_last, phi_last, tau_last, gate_raw_last, idx

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "tau_epsilon": self.tau_epsilon,
            "topk": self.topk,
            "euler_steps": self.euler_steps,
            "delta_t": self.delta_t,
            "use_sink_gate": self.use_sink_gate,
            "dropout": self.dropout,
            "activation": tf.keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "use_pairwise": self.use_pairwise,
            "return_attention": self.return_attention,
            "return_sequences": self.return_sequences,
        })
        return cfg


class LanGateExtractor(tf.keras.layers.Layer):
    """Re-expose a trained LAN's retrieved pairs and gate logits as a graph op.

    The layer above keeps its gates internal; the DCM probe needs them on a
    KerasTensor so it can be differentiated end to end. This wrapper is that
    window, and is deliberately *per-head* and *last-query-only*: the gate LSTM
    runs once per (head, query), so a per-head quantity is the faithful one, and
    the model's prediction comes from the last query position alone.

    Outputs
        u_heads   [B, H, K, 2D]  exact [q ; k] driver the gate LSTM consumes
        a_heads   [B, H, K]      gate logits (pre-activation)
        phi_heads [B, H, K]      relu(a)
        gam_heads [B, H, K]      softplus(a) + eps   (== tau)
        idx       [B, H, Tq, K]  retrieved key indices
    """

    def __init__(self, lan, **kwargs):
        super().__init__(**kwargs)
        self.lan = lan
        self.tau_epsilon = lan.tau_epsilon

    def call(self, x, training=None):
        qh = self.lan.split_heads(self.lan.q_dense(x))
        kh = self.lan.split_heads(self.lan.k_dense(x))

        pair, idx = self.lan.sparse_topk_pairwise(qh, kh)   # [B, H, Tq, K, 2D]
        phi, tau, gate_raw = self.lan._gate_heads(pair)

        last = -1
        return (pair[:, :, last, :, :],      # u_heads
                gate_raw[:, :, last, :],     # a_heads
                phi[:, :, last, :],          # phi_heads
                tau[:, :, last, :],          # gam_heads
                idx)

    def get_config(self):
        return super().get_config()
