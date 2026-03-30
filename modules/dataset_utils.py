import glob
import os
from typing import List, Optional, Tuple

import jax  # type: ignore
import jax.numpy as jnp
import numpy as np

from .constants import (
    JOINT_POS_DIM,
    LAFAN_PATH,
    MOTION_TAGS,
    NUM_JOINTS,
    NUM_MOTION_TAGS,
    RAW_FEATURES_DIM,
    ROOT_ANG_VEL_DIM,
    ROOT_VEL_DIM,
    WAYPOINT_DIM,
    WAYPOINT_DIR_VEL_DIM,
    WAYPOINT_TARGET_VEL_DIM,
    extract,
)


def length(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))


def extract_motion_tag(filename: str) -> int:
    """Extract motion tag index from filename.

    Returns the index of the matching motion tag, or -1 if no match found.
    """
    import re

    # Remove numbers from the start (e.g., "aiming1" -> "aiming")
    name = re.sub(r"^\d+", "", filename)
    # Remove _subjectN.bvh suffix
    name = re.sub(r"_.*\.bvh$", "", name)
    # Remove trailing numbers
    name = re.sub(r"\d+$", "", name)

    for i, tag in enumerate(MOTION_TAGS):
        if name.lower().startswith(tag.lower()):
            return i
    return -1  # Unknown tag


def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / (length(x, axis=axis) + eps)


def quat_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]
    return np.concatenate(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )


def quat_mul_vec(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)


def quat_fk(
    lrot: np.ndarray, lpos: np.ndarray, parents: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray]:
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            quat_mul_vec(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(quat_mul(gr[parents[i]], lrot[..., i : i + 1, :]))
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def compute_waypoint_features(
    root_pos: np.ndarray,
    root_vel: np.ndarray,
    offset: int = 5,
) -> np.ndarray:
    """Compute waypoint features for each frame.

    For frame i, computes:
    - waypoint_dir_vel: (root_pos[i+offset] - root_pos[i]) / offset [3]
    - waypoint_target_vel: root_vel[i+offset] [3]

    Args:
        root_pos: [B, T, 3] global root positions
        root_vel: [B, T, 3] global root velocities
        offset: number of frames to look ahead (default 5)

    Returns:
        [B, T, 6] waypoint features
    """
    B, T, _ = root_pos.shape

    # waypoint_dir_vel: velocity needed to reach waypoint in offset frames
    waypoint_dir_vel = np.zeros_like(root_pos)
    # Can compute for frames where i+offset < T
    if T > offset:
        waypoint_dir_vel[:, :-offset] = (
            root_pos[:, offset:] - root_pos[:, :-offset]
        ) / offset

    # waypoint_target_vel: velocity at the waypoint (frame i+offset)
    waypoint_target_vel = np.zeros_like(root_vel)
    if T > offset:
        waypoint_target_vel[:, :-offset] = root_vel[:, offset:]

    return np.concatenate([waypoint_dir_vel, waypoint_target_vel], axis=-1)


def compute_raw_features(
    local_q: np.ndarray,
    local_x: np.ndarray,
    parents: Tuple[int, ...],
    motion_tag: int = -1,
    dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """Compute raw features (76 dims with waypoints + tags) for dataset.

    Features (76 dims without tags, 91 with tags):
    - joint_pos_rel[66]: positions relative to root
    - root_vel_global[3]: root velocity
    - root_ang_vel[1]: root angular velocity
    - waypoint_dir_vel[3]: velocity to reach waypoint (frame i+5)
    - waypoint_target_vel[3]: velocity at waypoint (frame i+5)
    - motion_tag[15]: one-hot encoded motion category (optional, added later)

    Note: root_pos_global and root_angle removed (non-stationary, computed externally).
          Joint velocities are computed on-the-fly in the model.
    """
    B, T, J, _ = local_x.shape
    _, global_x = quat_fk(local_q, local_x, parents)
    root_pos_global = global_x[:, :, 0, :]
    forward = global_x[:, :, 1, :] - global_x[:, :, 0, :]
    forward_xz = forward.copy()
    forward_xz[:, :, 1] = 0
    forward_xz = forward_xz / (length(forward_xz, axis=-1) + 1e-8)
    root_angles = -np.arctan2(forward_xz[:, :, 2], forward_xz[:, :, 0])
    root_vel_global = np.zeros_like(root_pos_global)
    root_vel_global[:, 1:] = (root_pos_global[:, 1:] - root_pos_global[:, :-1]) / dt
    root_vel_global[:, 0] = root_vel_global[:, 1]
    root_ang_vel = np.zeros_like(root_angles[..., np.newaxis])
    root_ang_vel[:, 1:] = (
        root_angles[:, 1:, np.newaxis] - root_angles[:, :-1, np.newaxis]
    ) / dt
    root_ang_vel[:, 0] = root_ang_vel[:, 1]
    joint_pos_rel = global_x - root_pos_global[:, :, np.newaxis, :]

    # Compute waypoint features
    waypoint_features = compute_waypoint_features(
        root_pos_global, root_vel_global, offset=5
    )

    # Concatenate: 66 + 3 + 1 + 3 + 3 = 76 dims
    features = np.concatenate(
        [
            joint_pos_rel.reshape(B, T, -1),  # [66] positions
            root_vel_global,  # [3] root velocity
            root_ang_vel,  # [1] root angular velocity
            waypoint_features,  # [6] waypoint features
        ],
        axis=-1,
    )
    return features


def prepare_dataset(
    bvh_path: Optional[str] = None,
    out_file: str = "processed_mamba.npz",
    train_actors: Optional[List[str]] = None,
    val_actors: Optional[List[str]] = None,
    min_clip_length: int = 300,
) -> None:
    if bvh_path is None:
        bvh_path = f"{LAFAN_PATH}/lafan1"
    if train_actors is None:
        train_actors = ["subject1", "subject2", "subject3", "subject4"]
    if val_actors is None:
        val_actors = ["subject5"]

    if extract is None:
        raise RuntimeError(
            "LaFAN1 `extract` module not found. Is the dataset connected?"
        )

    extract_module = extract  # type: ignore

    print(f"Preparing dataset...")
    print(f"Train actors: {train_actors}")
    print(f"Val actors: {val_actors}")
    print(f"Min clip length: {min_clip_length}")

    def process_actor(
        actor: str,
    ) -> Tuple[List[np.ndarray], List[int], Tuple[int, ...], np.ndarray]:
        print(f"[{actor}] Loading...")

        pattern = os.path.join(bvh_path, f"*_{actor}.bvh")
        bvh_files = glob.glob(pattern)

        if not bvh_files:
            print(f"[{actor}] no bvh")
            return [], [], (0,), np.zeros((NUM_JOINTS, 3))

        clips: List[np.ndarray] = []
        clip_tags: List[int] = []
        parents_res: Optional[Tuple[int, ...]] = None
        offsets_res: Optional[np.ndarray] = None

        for bvh_file in sorted(bvh_files):
            filename = os.path.basename(bvh_file)
            print(f"{filename}...", end=" ")

            try:
                anim = extract_module.read_bvh(bvh_file)

                if anim.pos.shape[0] < min_clip_length:
                    print(f"skip (too short: {anim.pos.shape[0]})")
                    continue

                if parents_res is None:
                    parents_res = tuple(anim.parents.tolist())
                    _, global_pos = quat_fk(anim.quats[0:1], anim.pos[0:1], parents_res)
                    offsets_res = global_pos[0].copy()

                # Extract motion tag from filename
                motion_tag = extract_motion_tag(filename)
                if motion_tag < 0:
                    print(f"warn: unknown tag", end=" ")

                Q = anim.quats[np.newaxis, ...]
                X = anim.pos[np.newaxis, ...]
                features = compute_raw_features(Q, X, parents_res)
                clip = features[0]
                clips.append(clip)
                clip_tags.append(motion_tag)
                print(f"OK ({anim.pos.shape[0]})")

            except Exception as e:
                print(f"ERR: {e}")
                continue

        print(f"[{actor}] Loaded {len(clips)} clips")

        if len(clips) == 0 or parents_res is None or offsets_res is None:
            return [], [], (0,), np.zeros((NUM_JOINTS, 3))

        return clips, clip_tags, parents_res, offsets_res

    # train
    all_train_clips: List[np.ndarray] = []
    all_train_tags: List[int] = []
    parents: Tuple[int, ...] = (0,)
    offsets_raw: np.ndarray = np.zeros((NUM_JOINTS, 3))

    for actor in train_actors:
        clips, tags, p, o = process_actor(actor)
        if len(clips) > 0:
            parents, offsets_raw = p, o
            all_train_clips.extend(clips)
            all_train_tags.extend(tags)

    # val
    all_val_clips: List[np.ndarray] = []
    all_val_tags: List[int] = []
    for actor in val_actors:
        clips, tags, _, _ = process_actor(actor)
        all_val_clips.extend(clips)
        all_val_tags.extend(tags)

    def pack_to_ribbon(
        clips: List[np.ndarray],
        tags: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ribbon = np.concatenate(clips, axis=0)
        lengths = np.array([len(c) for c in clips])
        starts = np.zeros_like(lengths)
        starts[1:] = np.cumsum(lengths)[:-1]
        # Create tag ribbon - each frame gets its clip's tag (as int, -1 = unknown)
        tag_ribbon = np.zeros(len(ribbon), dtype=np.int32)
        pos = 0
        for clip, tag in zip(clips, tags):
            if tag >= 0:  # Valid tag
                tag_ribbon[pos : pos + len(clip)] = tag
            else:
                tag_ribbon[pos : pos + len(clip)] = -1  # Unknown tag
            pos += len(clip)
        return ribbon, starts, lengths, tag_ribbon

    train_ribbon, train_starts, train_lengths, train_tags_ribbon = pack_to_ribbon(
        all_train_clips, all_train_tags
    )
    val_ribbon, val_starts, val_lengths, val_tags_ribbon = pack_to_ribbon(
        all_val_clips, all_val_tags
    )

    print(f"Normalization paramters...")

    train_ribbon_jax = jnp.array(train_ribbon)

    pos_data = train_ribbon_jax[:, :JOINT_POS_DIM]
    mean_pos = jnp.mean(pos_data)
    std_pos = jnp.std(pos_data)

    root_vel_data = train_ribbon_jax[:, JOINT_POS_DIM : JOINT_POS_DIM + ROOT_VEL_DIM]
    mean_root_vel = jnp.mean(root_vel_data)
    std_root_vel = jnp.std(root_vel_data)

    ang_vel_data = train_ribbon_jax[
        :,
        JOINT_POS_DIM + ROOT_VEL_DIM : JOINT_POS_DIM + ROOT_VEL_DIM + ROOT_ANG_VEL_DIM,
    ]
    mean_ang_vel = jnp.mean(ang_vel_data)
    std_ang_vel = jnp.std(ang_vel_data)

    wp_start = JOINT_POS_DIM + ROOT_VEL_DIM + ROOT_ANG_VEL_DIM
    wp_dir_vel_data = train_ribbon_jax[:, wp_start : wp_start + WAYPOINT_DIR_VEL_DIM]
    mean_wp_dir_vel = jnp.mean(wp_dir_vel_data)
    std_wp_dir_vel = jnp.std(wp_dir_vel_data)

    wp_target_vel_data = train_ribbon_jax[
        :,
        wp_start + WAYPOINT_DIR_VEL_DIM : wp_start
        + WAYPOINT_DIR_VEL_DIM
        + WAYPOINT_TARGET_VEL_DIM,
    ]
    mean_wp_target_vel = jnp.mean(wp_target_vel_data)
    std_wp_target_vel = jnp.std(wp_target_vel_data)

    std_pos = jnp.where(std_pos < 1e-8, 1.0, std_pos)
    std_root_vel = jnp.where(std_root_vel < 1e-8, 1.0, std_root_vel)
    std_ang_vel = jnp.where(std_ang_vel < 1e-8, 1.0, std_ang_vel)
    std_wp_dir_vel = jnp.where(std_wp_dir_vel < 1e-8, 1.0, std_wp_dir_vel)
    std_wp_target_vel = jnp.where(std_wp_target_vel < 1e-8, 1.0, std_wp_target_vel)

    mean_base = np.concatenate(
        [
            np.full(JOINT_POS_DIM, float(mean_pos)),
            np.full(ROOT_VEL_DIM, float(mean_root_vel)),
            np.full(ROOT_ANG_VEL_DIM, float(mean_ang_vel)),
        ]
    )
    std_base = np.concatenate(
        [
            np.full(JOINT_POS_DIM, float(std_pos)),
            np.full(ROOT_VEL_DIM, float(std_root_vel)),
            np.full(ROOT_ANG_VEL_DIM, float(std_ang_vel)),
        ]
    )

    mean_input = np.concatenate(
        [
            mean_base,
            np.full(WAYPOINT_DIR_VEL_DIM, float(mean_wp_dir_vel)),
            np.full(WAYPOINT_TARGET_VEL_DIM, float(mean_wp_target_vel)),
        ]
    )
    std_input = np.concatenate(
        [
            std_base,
            np.full(WAYPOINT_DIR_VEL_DIM, float(std_wp_dir_vel)),
            np.full(WAYPOINT_TARGET_VEL_DIM, float(std_wp_target_vel)),
        ]
    )

    print(f"Statistics computed for frames {len(train_ribbon)}")
    print(f"Positions: μ={mean_pos:.4f}, σ={std_pos:.4f} (all {JOINT_POS_DIM} dims)")
    print(
        f"Root velocity: μ={mean_root_vel:.4f}, σ={std_root_vel:.4f} (all {ROOT_VEL_DIM} dims)"
    )
    print(f"Root angular velocity: μ={mean_ang_vel:.4f}, σ={std_ang_vel:.4f}")
    print(f"Waypoint dir velocity: μ={mean_wp_dir_vel:.4f}, σ={std_wp_dir_vel:.4f}")
    print(
        f"Waypoint target velocity: μ={mean_wp_target_vel:.4f}, σ={std_wp_target_vel:.4f}"
    )

    offsets = offsets_raw - offsets_raw[0]

    np.savez_compressed(
        out_file,
        train_ribbon=train_ribbon,
        train_tags_ribbon=train_tags_ribbon,
        train_starts=train_starts,
        train_lengths=train_lengths,
        val_ribbon=val_ribbon,
        val_tags_ribbon=val_tags_ribbon,
        val_starts=val_starts,
        val_lengths=val_lengths,
        parents=np.array(parents),
        offsets=offsets,
        mean_input=mean_input,
        std_input=std_input,
        mean_output=mean_base,
        std_output=std_base,
    )

    print(f"Dataset saved to `{out_file}`")
