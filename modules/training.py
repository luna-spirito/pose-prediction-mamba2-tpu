# from __future__ import annotations

import time  # type: ignore
from functools import partial
from typing import Any, Tuple  # type: ignore

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray

from .constants import (  # type: ignore
    HISTORY_LEN,
    NUM_MOTION_TAGS,
    OUTPUT_DIM,
    RAW_FEATURES_DIM,
)
from .model import FeatureNormalizer, MambaMotionModel

OUTPUT_POS_SLICE = slice(0, 66)  # joint_pos_rel
OUTPUT_VEL_SLICE = slice(66, 69)  # root_vel_global
OUTPUT_ANG_SLICE = slice(69, 70)  # root_ang_vel


def cast_to_bf16(x: jax.Array) -> jax.Array:
    # Only cast floating-point arrays (ignores ints/booleans)
    if eqx.is_inexact_array(x):
        return x.astype(jnp.bfloat16)
    return x


def make_fixed_chunks(ribbon, seq_len: int):
    """Slices ribbon into independent, non-overlapping sequence chunks."""
    chunk_size = seq_len + 1  # Need seq_len + 1 for inputs and targets
    num_chunks = len(ribbon) // chunk_size
    truncated = ribbon[: num_chunks * chunk_size]
    # Shape: [num_chunks, seq_len + 1, features]
    return truncated.reshape(num_chunks, chunk_size, -1)


def get_batch(
    chunked_features: jax.Array, chunked_tags: jax.Array, chunked_indices: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fetch pre-sliced sequences directly by index using basic array slicing."""

    def get_device_batch(device_features, device_tags, b_idx):
        # Direct indexing. No dynamic_slice needed!
        features_window = device_features[b_idx]  # [local_batch, seq_len + 1, dims]
        tags_window = device_tags[b_idx]

        input_features = features_window[:, :-1]
        input_tags = tags_window[:, :-1]
        target_seq = features_window[:, 1:, :RAW_FEATURES_DIM]

        positions = input_features[..., :66]
        root_vel = input_features[..., 66:69]
        root_ang_vel = input_features[..., 69:70]
        waypoints = input_features[..., 70:76]

        return positions, root_vel, root_ang_vel, waypoints, input_tags, target_seq

    return jax.vmap(get_device_batch)(chunked_features, chunked_tags, chunked_indices)


def create_cosine_schedule(
    peak_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=min_lr / peak_lr,
    )
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps],
    )


def loss_fn(
    model: MambaMotionModel,
    positions: jax.Array,
    root_vel: jax.Array,
    root_ang_vel: jax.Array,
    waypoints: jax.Array,
    input_tags: jax.Array,
    targets: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Compute MSE loss for single-step motion prediction."""
    # Normalize inputs (model expects normalized inputs) - use model's method for consistency
    positions, root_vel, root_ang_vel, waypoints = model.normalize_inputs(
        positions, root_vel, root_ang_vel, waypoints
    )

    targets_normalized = model.output_normalizer.normalize(targets).astype(jnp.float32)

    predictions = jax.vmap(model)(
        positions, root_vel, root_ang_vel, waypoints, input_tags
    )

    # Compute MSE loss in normalized space
    loss = jnp.mean(jnp.square(predictions - targets_normalized))

    return loss, predictions


def train(
    dataset_file: str = "processed_mamba.npz",
    out_model: str = "mamba_motion_model.eqx",
    epochs: int = 1000,
    batch_size: int = 256,
    peak_lr: float = 1e-4,
    min_lr: float = 1e-7,
    warmup_epochs: int = 100,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
) -> None:
    print("Loading data...")
    data = np.load(dataset_file)

    # Base features ribbon (76 dims: 70 base + 6 waypoints)
    train_features_ribbon = jnp.array(data["train_ribbon"])
    val_features_ribbon = jnp.array(data["val_ribbon"])
    # Tags ribbon (15 dims)
    train_tags_ribbon = jnp.array(data["train_tags_ribbon"])
    val_tags_ribbon = jnp.array(data["val_tags_ribbon"])

    mean_input = jnp.array(data["mean_input"])  # [76]
    std_input = jnp.array(data["std_input"])  # [76]
    mean_output = jnp.array(data["mean_output"])  # [70]
    std_output = jnp.array(data["std_output"])  # [70]

    print(f"    Train: {len(train_features_ribbon)} frames")
    print(f"    Val: {len(val_features_ribbon)} frames")

    hist_len_int = int(HISTORY_LEN)
    num_train_samples = len(train_features_ribbon) - hist_len_int
    num_val_samples = len(val_features_ribbon) - hist_len_int

    hist_len_int = int(HISTORY_LEN)

    print(f"Slicing datasets into fixed, non-overlapping chunks...")
    train_chunks_all = make_fixed_chunks(train_features_ribbon, hist_len_int)
    train_tags_all = make_fixed_chunks(train_tags_ribbon, hist_len_int)

    val_chunks_all = make_fixed_chunks(val_features_ribbon, hist_len_int)
    val_tags_all = make_fixed_chunks(val_tags_ribbon, hist_len_int)

    num_devices = len(jax.devices())

    if len(train_chunks_all) < batch_size:
        raise ValueError(
            f"Not enough training data for batch_size={batch_size}. "
            f"Only {len(train_chunks_all)} chunks available, need at least {batch_size}. "
            f"Try reducing batch_size or using more training data."
        )
    if len(val_chunks_all) < batch_size:
        raise ValueError(
            f"Not enough validation data for batch_size={batch_size}. "
            f"Only {len(val_chunks_all)} chunks available, need at least {batch_size}. "
            f"Try reducing batch_size."
        )

    valid_train_chunks = (len(train_chunks_all) // batch_size) * batch_size
    valid_val_chunks = (len(val_chunks_all) // batch_size) * batch_size

    train_chunks_all = train_chunks_all[:valid_train_chunks]
    train_tags_all = train_tags_all[:valid_train_chunks]
    val_chunks_all = val_chunks_all[:valid_val_chunks]
    val_tags_all = val_tags_all[:valid_val_chunks]

    local_train_chunks = valid_train_chunks // num_devices
    local_val_chunks = valid_val_chunks // num_devices
    local_batch_size = batch_size // num_devices

    print(f"Total train chunks: {valid_train_chunks} ({local_train_chunks} per device)")
    print(f"Total val chunks:   {valid_val_chunks} ({local_val_chunks} per device)")

    steps_per_epoch = valid_train_chunks // batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    print("Creating model...")
    input_normalizer = FeatureNormalizer(mean_input, std_input)
    output_normalizer = FeatureNormalizer(mean_output, std_output)

    key = jax.random.PRNGKey(42)
    model_key, train_key = jax.random.split(key)

    model = MambaMotionModel(
        input_normalizer=input_normalizer,
        output_normalizer=output_normalizer,
        output_dim=OUTPUT_DIM,
        hidden_dim=256,
        num_layers=8,
        d_state=64,
        expand_factor=2,
        d_conv=4,
        chunk_size=128,
        key=model_key,
    )

    params_count = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    print(f"Parameters: {params_count:,}")

    print(f"Start training with:")
    print(f"Epochs: {epochs}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Peak LR: {peak_lr:.2e}")
    print(f"Min LR: {min_lr:.2e}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient clip: {grad_clip}")

    lr_schedule = create_cosine_schedule(
        peak_lr=peak_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    print(f"Using {num_devices} devices...")
    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(device_mesh, axis_names=("devices",))

    device_sharding = NamedSharding(mesh, P("devices"))
    replicated_sharding = NamedSharding(mesh, P())

    def replicate_model_state(x: jax.Array) -> jax.Array:
        return jax.device_put(x, replicated_sharding) if eqx.is_array(x) else x

    model = jax.tree_util.tree_map(replicate_model_state, model, is_leaf=eqx.is_array)
    opt_state = jax.tree_util.tree_map(
        replicate_model_state, opt_state, is_leaf=eqx.is_array
    )

    chunked_train_features = train_chunks_all.reshape(
        num_devices, local_train_chunks, hist_len_int + 1, -1
    )
    chunked_train_tags = train_tags_all.reshape(
        num_devices, local_train_chunks, hist_len_int + 1
    )

    chunked_val_features = val_chunks_all.reshape(
        num_devices, local_val_chunks, hist_len_int + 1, -1
    )
    chunked_val_tags = val_tags_all.reshape(
        num_devices, local_val_chunks, hist_len_int + 1
    )

    train_features_device = jax.device_put(chunked_train_features, device_sharding)
    train_tags_device = jax.device_put(chunked_train_tags, device_sharding)
    val_features_device = jax.device_put(chunked_val_features, device_sharding)
    val_tags_device = jax.device_put(chunked_val_tags, device_sharding)

    pred_features_device = jax.device_put(chunked_train_features, device_sharding)

    @eqx.filter_jit
    def generate_local_batch_indices(
        keys: jax.Array,
        local_samples: int,
        local_batch_size: int,
    ) -> jax.Array:
        def local_perm(key):
            perm = jax.random.permutation(key, local_samples)
            num_batches = local_samples // local_batch_size
            return perm[: num_batches * local_batch_size].reshape(
                num_batches, local_batch_size
            )

        indices = jax.vmap(local_perm)(keys)
        # Transpose to [num_batches, num_devices, local_batch_size] for jax.lax.scan
        return jnp.swapaxes(indices, 0, 1)

    @eqx.filter_jit
    def train_epoch(
        model: MambaMotionModel,
        opt_state: optax.OptState,
        batch_indices: jax.Array,
        train_chunks: jax.Array,
        p_chunks: jax.Array,
        alpha: jax.Array,
    ):
        print("(Starting compilation...)")

        def scan_fn(
            carry: Tuple[Any, ...], b_idx: jax.Array
        ) -> Tuple[Tuple[Any, ...], Tuple[jax.Array, jax.Array, jax.Array]]:
            m, o_s = carry

            # 1. Fetch chunks directly
            pos_t, vel_t, ang_t, wp_t, inp_tags, tgts_t = get_batch(
                train_chunks, train_tags_device, b_idx
            )
            pos_p, vel_p, ang_p, wp_p, _, tgts_p = get_batch(
                p_chunks, train_tags_device, b_idx
            )

            # 2. Flatten for the model
            def flatten_device_batch(x):
                return x.reshape(-1, *x.shape[2:])

            positions = flatten_device_batch(pos_p)
            root_vel = flatten_device_batch(vel_p)
            root_ang_vel = flatten_device_batch(ang_p)
            waypoints = flatten_device_batch(wp_p)
            inp_tags_flat = flatten_device_batch(inp_tags)
            tgts = flatten_device_batch(tgts_t)  # pure goal targets

            # Compute loss
            (loss, preds), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                m, positions, root_vel, root_ang_vel, waypoints, inp_tags_flat, tgts
            )

            filtered_m = eqx.filter(m, eqx.is_array)
            updates, new_o_s = optimizer.update(grads, o_s, filtered_m)
            new_m = eqx.apply_updates(m, updates)

            preds_denorm = new_m.output_normalizer.denormalize(preds)
            preds_per_device = preds_denorm.reshape(
                num_devices, local_batch_size, hist_len_int, OUTPUT_DIM
            )

            return (new_m, new_o_s), (loss, b_idx, preds_per_device)

        (model, opt_state), (losses, all_b_idx, all_preds) = jax.lax.scan(
            scan_fn, (model, opt_state), batch_indices
        )

        def bulk_update_local(local_p_chunks, local_all_b_idx, local_all_preds):
            flat_indices = local_all_b_idx.flatten()
            flat_preds = local_all_preds.reshape(-1, hist_len_int, OUTPUT_DIM)
            return local_p_chunks.at[flat_indices, 1:, :RAW_FEATURES_DIM].set(
                flat_preds
            )

        all_b_idx_swapped = jnp.swapaxes(all_b_idx, 0, 1)
        all_preds_swapped = jnp.swapaxes(all_preds, 0, 1)

        new_pred_chunks = jax.vmap(bulk_update_local)(
            p_chunks, all_b_idx_swapped, all_preds_swapped
        )

        return model, opt_state, new_pred_chunks, jnp.mean(losses)

    @eqx.filter_jit
    def val_step(model: MambaMotionModel, indices: jax.Array) -> jax.Array:
        pos_t, vel_t, ang_t, wp_t, inp_tags, tgts_t = get_batch(
            val_features_device, val_tags_device, indices
        )

        def flatten_device_batch(x):
            return x.reshape(-1, *x.shape[2:])

        pos_flat = flatten_device_batch(pos_t)
        vel_flat = flatten_device_batch(vel_t)
        ang_flat = flatten_device_batch(ang_t)
        wp_flat = flatten_device_batch(wp_t)
        tags_flat = flatten_device_batch(inp_tags)

        # Normalize inputs before passing to model - use model's method for consistency
        pos_norm, vel_norm, ang_norm, wp_norm = model.normalize_inputs(
            pos_flat, vel_flat, ang_flat, wp_flat
        )

        # Forward pass (flattened)
        predictions_normalized = jax.vmap(model)(
            pos_norm, vel_norm, ang_norm, wp_norm, tags_flat
        )

        targets_normalized = model.output_normalizer.normalize(
            flatten_device_batch(tgts_t)
        ).astype(jnp.float32)
        return jnp.mean(jnp.square(predictions_normalized - targets_normalized))

    print(f"Starting training...")
    best_val_loss = float("inf")
    best_epoch = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        train_key, epoch_key, perm_key = jax.random.split(train_key, 3)

        current_step = epoch * steps_per_epoch
        alpha = jnp.array(1.0 - 1.0 * min(epoch / (epochs / 2.0), 1.0))

        device_keys = jax.random.split(perm_key, num_devices)
        device_keys = jax.device_put(device_keys, device_sharding)

        batch_indices = generate_local_batch_indices(
            device_keys, local_train_chunks, local_batch_size
        )
        batch_indices = jax.device_put(
            batch_indices, NamedSharding(mesh, P(None, "devices"))
        )

        model, opt_state, pred_features_device, epoch_loss = train_epoch(
            model,
            opt_state,
            batch_indices,
            train_features_device,
            pred_features_device,
            alpha,
        )
        epoch_loss = float(epoch_loss)
        epoch_time = time.time() - epoch_start

        # Validation phase - Also mapped to local shards
        val_start = time.time()
        # Validation uses local_val_chunks
        local_val_subset_size = min(1024 // num_devices, local_val_chunks)

        def get_local_val_indices(key):
            return jax.random.randint(
                key, (local_val_subset_size,), 0, local_val_chunks
            )

        val_keys = jax.random.split(epoch_key, num_devices)
        val_keys = jax.device_put(val_keys, device_sharding)

        val_indices = jax.vmap(get_local_val_indices)(val_keys)
        val_indices = jax.device_put(val_indices, device_sharding)

        val_loss = float(val_step(model, val_indices))
        val_time = time.time() - val_start

        current_lr = float(lr_schedule(current_step))

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            eqx.tree_serialise_leaves(out_model, model)
            improved = " [BEST]"

        total_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1:04d}/{epochs} | "
            f"Alpha: {float(alpha):.3f} | "
            f"LR: {current_lr:.2e} | "
            f"Train: {epoch_loss:.6f} | "
            f"Val: {val_loss:.6f}{improved} | "
            f"Time: {epoch_time:.1f}s (val: {val_time:.1f}s) | "
            f"Total: {total_time / 60:.1f}m"
        )
    eqx.tree_serialise_leaves(f"final_{out_model}", model)

    print(f"Training complete")
    print(f"Best model at epoch {best_epoch} with val_loss: {best_val_loss:.6f}")
    print(f"Model saved to {out_model}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
