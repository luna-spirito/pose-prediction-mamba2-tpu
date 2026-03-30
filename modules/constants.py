# Constants for motion model
import sys

# LAFAN dataset path
LAFAN_PATH = "/kaggle/input/datasets/star2578/lafan1"
if LAFAN_PATH not in sys.path:
    sys.path.append(LAFAN_PATH)

try:
    import extract
except ImportError:
    extract = None

# Core constants
NUM_JOINTS = 22
HISTORY_LEN = 100

# Feature dimensions
JOINT_POS_DIM = NUM_JOINTS * 3  # 66
ROOT_VEL_DIM = 3
ROOT_ANG_VEL_DIM = 1
WAYPOINT_DIR_VEL_DIM = 3  # (root_pos[i+5] - root_pos[i]) / 5 - velocity needed to reach waypoint
WAYPOINT_TARGET_VEL_DIM = 3  # root_vel[i+5] - velocity at waypoint

# Motion tags (extracted from filenames)
MOTION_TAGS = [
    "aiming",
    "dance", 
    "fallAndGetUp",
    "fight",
    "fightAndSports",
    "ground",
    "jumps",
    "multipleActions",
    "obstacles",
    "push",
    "pushAndFall",
    "pushAndStumble",
    "run",
    "sprint",
    "walk",
]
NUM_MOTION_TAGS = len(MOTION_TAGS)

# Total feature dimensions
RAW_FEATURES_DIM = JOINT_POS_DIM + ROOT_VEL_DIM + ROOT_ANG_VEL_DIM  # 70 (base features)
WAYPOINT_DIM = WAYPOINT_DIR_VEL_DIM + WAYPOINT_TARGET_VEL_DIM  # 6 (waypoint features)
FEATURE_DIM = RAW_FEATURES_DIM + WAYPOINT_DIM + NUM_MOTION_TAGS  # 70 + 6 + 15 = 91
OUTPUT_DIM = RAW_FEATURES_DIM  # 70 (we only predict base features)
