# from __future__ import annotations

from typing import Any, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .constants import (
    HISTORY_LEN,
    JOINT_POS_DIM,
    NUM_MOTION_TAGS,
    OUTPUT_DIM,
    ROOT_ANG_VEL_DIM,
    ROOT_VEL_DIM,
    WAYPOINT_DIM,
)
from .mamba2 import Mamba2Block, Mamba2Cache, Mamba2Config, RMSNorm


class FeatureNormalizer(eqx.Module):
    """Feature normalizer."""

    mean: Array
    std: Array

    def __init__(self, mean: Array, std: Array):
        self.mean = mean
        self.std = jnp.where(std < 1e-8, 1.0, std)

    def normalize(self, x: Array) -> Array:
        return (x - self.mean) / self.std

    def denormalize(self, x: Array) -> Array:
        return x * self.std + self.mean


def normalize_inputs_from_model(
    input_normalizer: FeatureNormalizer,
    positions: jax.Array,
    root_vel: jax.Array,
    root_ang_vel: jax.Array,
    waypoints: jax.Array,
) -> Tuple[Array, Array, Array, Array]:
    """Normalize input features using model's input normalizer."""
    # Reconstruct full feature vector [..., 76]
    full_features = jnp.concatenate(
        [positions, root_vel, root_ang_vel, waypoints], axis=-1
    )
    # Normalize
    norm_features = input_normalizer.normalize(full_features)
    # Extract components
    norm_positions = norm_features[..., :66]
    norm_root_vel = norm_features[..., 66:69]
    norm_root_ang_vel = norm_features[..., 69:70]
    norm_waypoints = norm_features[..., 70:76]
    return norm_positions, norm_root_vel, norm_root_ang_vel, norm_waypoints


def create_empty_cache(
    cfg: Mamba2Config, batch_size: int = 1, dtype=jnp.float32
) -> Mamba2Cache:
    """Create an empty cache for Mamba2 model."""
    conv_dim = cfg.intermediate_size + 2 * cfg.d_state
    cache_len = cfg.conv_kernel - 1

    conv_states = tuple(
        jnp.zeros((batch_size, conv_dim, cache_len), dtype=dtype)
        for _ in range(cfg.num_hidden_layers)
    )
    ssm_states = tuple(
        jnp.zeros((batch_size, cfg.num_heads, cfg.head_dim, cfg.d_state), dtype=dtype)
        for _ in range(cfg.num_hidden_layers)
    )

    return Mamba2Cache(ssm_states=ssm_states, conv_states=conv_states)


class MambaMotionModel(eqx.Module):
    config: Mamba2Config = eqx.field(static=True)
    input_proj: eqx.nn.Linear
    layers: Tuple[Mamba2Block, ...]
    final_norm: Any
    output_proj: eqx.nn.Linear
    input_normalizer: FeatureNormalizer
    output_normalizer: FeatureNormalizer

    def __init__(
        self,
        input_normalizer: FeatureNormalizer,
        output_normalizer: FeatureNormalizer,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        d_state: int = 64,
        expand_factor: int = 2,
        d_conv: int = 4,
        chunk_size: int = 256,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 3 + num_layers)

        # Input projection: positions(66) + velocities(66) + root_vel(3) +
        # root_ang_vel(1) + waypoints(6) + tags(15) = 157
        input_dim = (
            JOINT_POS_DIM
            + JOINT_POS_DIM
            + ROOT_VEL_DIM
            + ROOT_ANG_VEL_DIM
            + WAYPOINT_DIM
            + NUM_MOTION_TAGS
        )
        self.input_proj = eqx.nn.Linear(input_dim, hidden_dim, key=keys[0])

        self.config = Mamba2Config(
            d_model=hidden_dim,
            d_state=d_state,
            conv_kernel=d_conv,
            expand=expand_factor,
            chunk_size=chunk_size,
            num_hidden_layers=num_layers,
            hidden_act="silu",
        )  # type: ignore

        layers = []
        for i in range(num_layers):
            layers.append(Mamba2Block(self.config, key=keys[1 + i]))
        self.layers = tuple(layers)

        self.final_norm = RMSNorm(
            self.config.d_model, eps=self.config.layer_norm_epsilon, key=keys[-2]
        )
        self.output_proj = eqx.nn.Linear(hidden_dim, output_dim, key=keys[-1])

        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

    def _compute_velocities(
        self, positions: jax.Array, dt: float = 1.0 / 30.0
    ) -> jax.Array:
        """Compute velocities for a single sequence [seq_len, 66]."""
        v = (positions[1:] - positions[:-1]) / dt
        zero_padding = jnp.zeros((1, 66), dtype=v.dtype)
        return jnp.concatenate([zero_padding, v], axis=0)

    def __call__(
        self,
        positions: Float[Array, "seq_len 66"],
        root_vel: Float[Array, "seq_len 3"],
        root_ang_vel: Float[Array, "seq_len 1"],
        waypoints: Float[Array, "seq_len 6"],
        tags: Float[Array, "seq_len"],
    ) -> Float[Array, "seq_len output_dim"]:
        """This predicts next frame for all positions in seq"""
        velocities = self._compute_velocities(positions)

        # Convert tag indices to one-hot encoding [seq_len, 15]
        one_hot_tags = jax.nn.one_hot(tags, NUM_MOTION_TAGS, axis=-1)

        # Concatenate all features: positions + velocities + root_vel + root_ang_vel + waypoints + tags
        # [seq_len, 66 + 66 + 3 + 1 + 6 + 15] = [seq_len, 157]
        x_in = jnp.concatenate(
            [positions, velocities, root_vel, root_ang_vel, waypoints, one_hot_tags],
            axis=-1,
        )

        x_in = jax.vmap(self.input_proj)(x_in)

        x_in = jnp.expand_dims(x_in, axis=0)

        for layer in self.layers:
            x_in = layer(x_in, conv_state=None, ssm_state=None)[0]

        x_in = jnp.squeeze(x_in, axis=0)

        def apply_norm(h: Array) -> Array:
            return self.final_norm(h)

        x_in = jax.vmap(apply_norm)(x_in)

        output = jax.vmap(self.output_proj)(x_in)

        return output

    def step(
        self,
        positions: Float[Array, "66"],
        root_vel: Float[Array, "3"],
        root_ang_vel: Float[Array, "1"],
        waypoints: Float[Array, "6"],
        tag: int,
        prev_positions: Float[Array, "66"],
        cache: Mamba2Cache | None = None,
    ) -> tuple[Float[Array, "70"], Mamba2Cache]:
        """Single-step causal inference with state caching, for inference"""
        # Create cache if not provided
        if cache is None:
            cache = create_empty_cache(self.config, batch_size=1)

        dt = 1.0 / 30.0  # 30 FPS
        velocity = (positions - prev_positions) / dt

        one_hot_tag = jax.nn.one_hot(jnp.array([tag]), NUM_MOTION_TAGS, axis=-1)[0]

        x_in = jnp.concatenate(
            [positions, velocity, root_vel, root_ang_vel, waypoints, one_hot_tag]
        )

        x_in = self.input_proj(x_in)

        x_in = x_in[None, None, :]

        new_conv_states = []
        new_ssm_states = []
        for i, layer in enumerate(self.layers):
            x_in, new_conv_state, new_ssm_state = layer(
                x_in, conv_state=cache.conv_states[i], ssm_state=cache.ssm_states[i]
            )
            new_conv_states.append(new_conv_state)
            new_ssm_states.append(new_ssm_state)

        x_in = x_in[0, 0, :]

        x_in = self.final_norm(x_in)

        output = self.output_proj(x_in)

        new_cache = Mamba2Cache(
            ssm_states=tuple(new_ssm_states), conv_states=tuple(new_conv_states)
        )

        return output, new_cache

    def normalize_inputs(
        self,
        positions: Float[Array, "... 66"],
        root_vel: Float[Array, "... 3"],
        root_ang_vel: Float[Array, "... 1"],
        waypoints: Float[Array, "... 6"],
    ) -> Tuple[Array, Array, Array, Array]:
        return normalize_inputs_from_model(
            self.input_normalizer, positions, root_vel, root_ang_vel, waypoints
        )
