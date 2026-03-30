"""Mamba-2 implementation in Equinox.

Based on "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality" (Dao & Gu, 2024).

Adapted from CosmoNaught/mamba2-jax (Flax NNX) to Equinox.
"""
# from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, List, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


@jax.tree_util.register_pytree_node_class
@dataclass
class Mamba2Cache:
    """Cache for Mamba2 SSM and convolution states."""
    ssm_states: Tuple[Array, ...]  # (batch, heads, head_dim, state_size) per layer
    conv_states: Tuple[Array, ...]  # (batch, conv_dim, kernel_size - 1) per layer
    
    def tree_flatten(self):
        children = (self.ssm_states, self.conv_states)
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ssm_states, conv_states = children
        return cls(ssm_states=ssm_states, conv_states=conv_states)


@dataclass(frozen=True)
class Mamba2Config:
    """Configuration for Mamba2 models."""

    d_model: int = 512
    d_state: int = 64
    head_dim: int = 64
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    num_hidden_layers: int = 8
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: Literal["silu", "gelu", "relu", "tanh"] = "silu"
    A_initializer_range: Tuple[float, float] = (1.0, 16.0)
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    residual_in_fp32: bool = True

    @property
    def intermediate_size(self) -> int:
        return int(self.expand * self.d_model)

    @property
    def num_heads(self) -> int:
        return self.intermediate_size // self.head_dim


ACT2FN = {
    "silu": jax.nn.silu,
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
}


def segsum(x: Array) -> Array:
    """Stable segment sum calculation. Input: (..., T) -> Output: (..., T, T)."""
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def _pad_seq_dim(x: Array, pad_size: int) -> Array:
    """Pad zeros at the end of the sequence dimension (axis=1)."""
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def ssd_forward(
    x: Float[Array, "batch seq heads head_dim"],
    dt: Float[Array, "batch seq heads"],
    A: Float[Array, "heads"],
    B_mat: Float[Array, "batch seq heads state"],
    C_mat: Float[Array, "batch seq heads state"],
    chunk_size: int,
    D: Float[Array, "heads"],
    dt_bias: Float[Array, "heads"],
    dt_min: float,
    dt_max: float,
    initial_states: Array | None = None,
    return_final_states: bool = False,
) -> tuple[Float[Array, "batch seq heads head_dim"], Array | None]:
    """SSD (State Space Duality) forward pass with chunked computation.

    Args:
        x: Input tensor (batch_size, seq_len, num_heads, head_dim)
        dt: Time deltas (batch_size, seq_len, num_heads)
        A: State transition scalar per head (num_heads,)
        B_mat: Input-to-state matrix (batch_size, seq_len, num_heads, state_size)
        C_mat: State-to-output matrix (batch_size, seq_len, num_heads, state_size)
        chunk_size: Size of chunks for efficient computation
        D: Skip connection weights (num_heads,)
        dt_bias: Bias for time deltas (num_heads,)
        dt_min: Minimum time delta after clamping
        dt_max: Maximum time delta after clamping
        initial_states: Optional initial SSM states (batch, 1, heads, head_dim, state_size)
        return_final_states: Whether to return final SSM states

    Returns:
        y: Output tensor (batch_size, seq_len, num_heads, head_dim)
        final_state: Optional final states (batch_size, heads, head_dim, state_size)
    """
    B_size, seq_len, num_heads, head_dim = x.shape
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # Apply dt bias with softplus and clamp
    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    # Pad tensors along sequence dimension
    x_padded = _pad_seq_dim(x, pad_size)
    dt_padded = _pad_seq_dim(dt, pad_size)
    B_padded = _pad_seq_dim(B_mat, pad_size)
    C_padded = _pad_seq_dim(C_mat, pad_size)

    # D residual connection
    D_residual = D.reshape(1, 1, num_heads, 1) * x_padded

    # Discretize x and A
    x_disc = x_padded * dt_padded[..., None]
    A_disc = A.astype(x_disc.dtype) * dt_padded

    # Chunk everything
    def chunk_tensor(t):
        b, cl, *remaining = t.shape
        return t.reshape(b, cl // chunk_size, chunk_size, *remaining)

    x_blk = chunk_tensor(x_disc)
    A_blk = chunk_tensor(A_disc)
    B_blk = chunk_tensor(B_padded)
    C_blk = chunk_tensor(C_padded)

    # A cumsum over intra-chunk time dimension
    A_blk2 = jnp.transpose(A_blk, (0, 3, 1, 2))
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)

    # 1. Intra-chunk (diagonal blocks)
    L_mat = jnp.exp(segsum(A_blk2))
    Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blk, B_blk, L_mat, x_blk)

    # 2. States within each chunk
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B_blk, decay_states, x_blk)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])
    states = jnp.concatenate([initial_states, states], axis=1)

    A_end = A_cumsum[..., -1]
    A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))
    decay_chunk = jnp.exp(segsum(A_end_padded))
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]

    # 4. Convert states -> outputs
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", C_blk, states, state_decay_out)

    y = Y_diag + Y_off
    b, c, l, h, p = y.shape
    y = y.reshape(b, c * l, h, p)
    y = y + D_residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    return (y, final_state) if return_final_states else (y, None)


class RMSNorm(eqx.Module):
    """RMSNorm with optional residual gating."""

    weight: Float[Array, "hidden_size"]
    eps: float = eqx.field(static=True)
    gate_residual: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        gate_residual: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.weight = jnp.ones((hidden_size,))
        self.eps = eps
        self.gate_residual = gate_residual

    def __call__(self, hidden_states: Array, residual: Array | None = None) -> Array:
        x = hidden_states.astype(jnp.float32)
        if residual is not None and self.gate_residual:
            x = x * jax.nn.silu(residual.astype(jnp.float32))
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps) * self.weight
        return x.astype(hidden_states.dtype)


class DepthwiseConv1d(eqx.Module):
    """Depthwise causal 1D convolution with state caching."""

    conv: eqx.nn.Conv1d
    kernel_size: int = eqx.field(static=True)

    def __init__(
        self,
        features: int,
        kernel_size: int,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.kernel_size = kernel_size
        # Depthwise conv: groups = features
        self.conv = eqx.nn.Conv1d(
            in_channels=features,
            out_channels=features,
            kernel_size=kernel_size,
            stride=1,
            padding=0,  # Manual causal padding
            groups=features,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self, 
        x: Float[Array, "batch seq features"],
        conv_state: Array | None = None
    ) -> tuple[Array, Array]:
        """Forward pass with optional state caching.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            conv_state: Optional cached state (batch, features, kernel_size-1)
            
        Returns:
            output: (batch, seq_len, features)
            new_conv_state: (batch, features, kernel_size-1)
        """
        batch_size, seq_len, features = x.shape
        cache_len = self.kernel_size - 1
        
        # Combine with cached state if provided
        if conv_state is None:
            x_padded = jnp.pad(x, ((0, 0), (cache_len, 0), (0, 0)), mode="constant")
        else:
            # Prepend cached state: (batch, features, cache_len) -> (batch, cache_len, features)
            conv_state_t = jnp.transpose(conv_state, (0, 2, 1))
            x_padded = jnp.concatenate([conv_state_t, x], axis=1)
        
        # Conv1d expects (batch, features, seq_len)
        x_transposed = jnp.transpose(x_padded, (0, 2, 1))
        out = jax.vmap(self.conv)(x_transposed)
        # Back to (batch, seq_len, features)
        output = jnp.transpose(out, (0, 2, 1))
        
        # Extract new state from the end
        new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))
        
        return output, new_conv_state


class Mamba2Mixer(eqx.Module):
    """Mamba2 mixer block using the SSD algorithm."""

    cfg: Mamba2Config = eqx.field(static=True)
    in_proj: eqx.nn.Linear
    conv1d: DepthwiseConv1d
    dt_bias: Float[Array, "num_heads"]
    A_log: Float[Array, "num_heads"]
    D: Float[Array, "num_heads"]
    norm: RMSNorm
    out_proj: eqx.nn.Linear
    act: Any = eqx.field(static=True)

    def __init__(
        self,
        cfg: Mamba2Config,
        *,
        key: PRNGKeyArray,
    ):
        self.cfg = cfg
        self.act = ACT2FN[cfg.hidden_act]

        # Input projection
        proj_size = 2 * (cfg.intermediate_size + cfg.d_state) + cfg.num_heads
        self.in_proj = eqx.nn.Linear(
            cfg.d_model, proj_size, use_bias=cfg.use_bias, key=key
        )

        # Depthwise conv
        conv1d_dim = cfg.intermediate_size + 2 * cfg.d_state
        conv_key, key = jax.random.split(key)
        self.conv1d = DepthwiseConv1d(
            conv1d_dim, cfg.conv_kernel, use_bias=cfg.use_conv_bias, key=conv_key
        )

        # SSM parameters
        dt_key, A_key, D_key = jax.random.split(key, 3)
        low, high = cfg.time_step_min, cfg.time_step_max
        floor = cfg.time_step_floor
        dt_init = jnp.exp(
            jax.random.uniform(dt_key, (cfg.num_heads,))
            * (jnp.log(high) - jnp.log(low))
            + jnp.log(low)
        )
        dt_init = jnp.maximum(dt_init, floor)
        # Inverse softplus
        self.dt_bias = dt_init + jnp.log(-jnp.expm1(-dt_init))

        A_low, A_high = cfg.A_initializer_range
        A_init = jax.random.uniform(
            A_key, (cfg.num_heads,), minval=A_low, maxval=A_high
        )
        self.A_log = jnp.log(A_init)

        self.D = jnp.ones((cfg.num_heads,))

        # Internal norm and output projection
        norm_key, out_key = jax.random.split(D_key)
        self.norm = RMSNorm(
            cfg.intermediate_size, eps=1e-5, gate_residual=True, key=norm_key
        )
        self.out_proj = eqx.nn.Linear(
            cfg.intermediate_size, cfg.d_model, use_bias=cfg.use_bias, key=out_key
        )

    def __call__(
        self, 
        hidden_states: Float[Array, "batch seq d_model"],
        conv_state: Array | None = None,
        ssm_state: Array | None = None
    ) -> tuple[Array, Array, Array]:
        """Forward pass with optional state caching for causal inference.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, d_model)
            conv_state: Optional cached conv state (batch_size, conv_dim, kernel_size-1)
            ssm_state: Optional cached SSM state (batch_size, heads, head_dim, d_state)
            
        Returns:
            output: (batch_size, seq_len, d_model)
            new_conv_state: (batch_size, conv_dim, kernel_size-1)
            new_ssm_state: (batch_size, heads, head_dim, d_state)
        """
        B_size, L, _ = hidden_states.shape

        # 1) Parallel projection
        zxbcdt = jax.vmap(jax.vmap(self.in_proj))(hidden_states)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.cfg.intermediate_size
            - 2 * self.cfg.d_state
            - self.cfg.num_heads
        ) // 2

        # Split: z0, x0, z, xBC, dt
        z0 = zxbcdt[..., :d_mlp]
        x0 = zxbcdt[..., d_mlp : 2 * d_mlp]
        z = zxbcdt[
            ...,
            2 * d_mlp : 2 * d_mlp + self.cfg.intermediate_size,
        ]
        xBC = zxbcdt[
            ...,
            2 * d_mlp + self.cfg.intermediate_size : 2 * d_mlp
            + self.cfg.intermediate_size
            + self.cfg.intermediate_size
            + 2 * self.cfg.d_state,
        ]
        dt = zxbcdt[..., -self.cfg.num_heads :]

        # 2) Depthwise causal convolution with state caching
        xBC, new_conv_state = self.conv1d(xBC, conv_state=conv_state)
        xBC = self.act(xBC)
        # Split xBC into x, B, C using indices (not sizes)
        # xBC shape: [..., intermediate_size + d_state + d_state]
        split_indices = [
            self.cfg.intermediate_size,
            self.cfg.intermediate_size + self.cfg.d_state,
        ]
        x, B_t, C_t = jnp.split(xBC, split_indices, axis=-1)

        # 3) SSD forward with state caching
        A = -jnp.exp(self.A_log.astype(jnp.float32))

        B_exp = jnp.broadcast_to(
            jnp.expand_dims(B_t, 2), (B_size, L, self.cfg.num_heads, self.cfg.d_state)
        )
        C_exp = jnp.broadcast_to(
            jnp.expand_dims(C_t, 2), (B_size, L, self.cfg.num_heads, self.cfg.d_state)
        )

        # Prepare initial state for SSM if provided
        init_state = ssm_state[:, None, ...] if ssm_state is not None else None
        
        y, new_ssm_state = ssd_forward(
            x=x.reshape(B_size, L, self.cfg.num_heads, self.cfg.head_dim),
            dt=dt,
            A=A,
            B_mat=B_exp,
            C_mat=C_exp,
            chunk_size=self.cfg.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_min=self.cfg.time_step_min,
            dt_max=self.cfg.time_step_max,
            initial_states=init_state,
            return_final_states=True,
        )
        y = y.reshape(B_size, L, -1)

        # 4) Residual gate normalization
        y = self.norm(y, residual=z)
        if d_mlp > 0:
            y = jnp.concatenate([self.act(z0) * x0, y], axis=-1)

        # 5) Output projection
        output = jax.vmap(jax.vmap(self.out_proj))(y)
        return output, new_conv_state, new_ssm_state


class Mamba2Block(eqx.Module):
    """Single Mamba2 block with pre-norm and residual connection with state caching."""

    norm: RMSNorm
    mixer: Mamba2Mixer
    residual_in_fp32: bool = eqx.field(static=True)

    def __init__(
        self,
        cfg: Mamba2Config,
        *,
        key: PRNGKeyArray,
    ):
        self.residual_in_fp32 = cfg.residual_in_fp32
        norm_key, mixer_key = jax.random.split(key)
        self.norm = RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon, key=norm_key)
        self.mixer = Mamba2Mixer(cfg, key=mixer_key)

    def __call__(
        self, 
        hidden_states: Array,
        conv_state: Array | None = None,
        ssm_state: Array | None = None
    ) -> tuple[Array, Array, Array]:
        """Forward pass with optional state caching for causal inference.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, d_model)
            conv_state: Optional cached conv state
            ssm_state: Optional cached SSM state
            
        Returns:
            output: (batch_size, seq_len, d_model)
            new_conv_state: Updated conv state
            new_ssm_state: Updated SSM state
        """
        residual = hidden_states
        hs = jax.vmap(self.norm)(hidden_states)
        comp_dtype = self.mixer.in_proj.weight.dtype
        hs = hs.astype(comp_dtype)
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hs_out, new_conv_state, new_ssm_state = self.mixer(hs, conv_state=conv_state, ssm_state=ssm_state)
        return residual + hs_out.astype(residual.dtype), new_conv_state, new_ssm_state


class Mamba2Model(eqx.Module):
    """Mamba2 model for sequence modeling."""

    config: Mamba2Config = eqx.field(static=True)
    input_proj: eqx.nn.Linear
    layers: Tuple[Mamba2Block, ...]
    final_norm: RMSNorm

    def __init__(
        self,
        config: Mamba2Config,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        keys = jax.random.split(key, 2 + config.num_hidden_layers)

        # Project input to d_model
        self.input_proj = eqx.nn.Linear(config.d_model, config.d_model, key=keys[0])

        # Mamba2 blocks
        layers = []
        for i in range(config.num_hidden_layers):
            layers.append(Mamba2Block(config, key=keys[1 + i]))
        self.layers = tuple(layers)

        # Final norm
        self.final_norm = RMSNorm(
            config.d_model, eps=config.layer_norm_epsilon, key=keys[-1]
        )

    def __call__(
        self, inputs: Float[Array, "batch seq d_model"]
    ) -> Float[Array, "batch seq d_model"]:
        """Forward pass.

        Args:
            inputs: Input tensor (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Project input
        hidden_states = jax.vmap(jax.vmap(self.input_proj))(inputs)

        # Pass through layers (no cache in this simple model)
        for layer in self.layers:
            hidden_states, _, _ = layer(hidden_states, conv_state=None, ssm_state=None)

        # Final normalization
        hidden_states = jax.vmap(self.final_norm)(hidden_states)

        return hidden_states
