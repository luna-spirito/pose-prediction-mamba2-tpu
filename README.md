# WIP Mamba2 motion prediction with closed-loop training

A JAX/Equinox mamba2 model for human motion prediction, featuring autoregressive training without teacher forcing and optimized for Google Cloud TPU acceleration, training on Kaggle.



https://github.com/user-attachments/assets/6a758bf7-8efe-48f0-a24f-1f1cb08ef357



## Overview

This repository implements a motion prediction architecture based on Mamba2 (Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality) to autoregressively predict human poses from the LAFAN1 motion capture dataset.

Conventionally, sequence-to-sequence approaches rely on teacher forcing during training and predict discrete tokens. This repository predicts continuous human pose, and naive teacher forcing produces models, catastrophically susceptible to exposre bias.

To fix this issue, we experiment with a "closed-loop" training, where the model's own predictions propagate to future epochs as input, ensuring:
- Near-zero performance penalty for "Scheduling Sampling"-esque approach.
- Resistance to accumulated prediction errors during inference.

## Limitations

This is a work-in-progress project, and while the trained model maintained a great stability, quality is found to degrade over long sequences, producing questionable "foot sliding" movements.

## Architecture

### Mamba2 Backbone

**Configuration**:
- Hidden dimension: 256
- State dimension: 64
- Expansion factor: 2
- Convolution kernel: 4
- Number of layers: 8

### Input Representation

The model processes a 157-dimensional input vector per timestep:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Joint positions | 66 | 22 joints × 3 coordinates (root-relative) |
| Joint velocities | 66 | Finite differences computed on-the-fly |
| Root velocity | 3 | Global translation velocity |
| Root angular velocity | 1 | Y-axis rotation rate |
| Waypoint features | 6 | Future trajectory hints (t+5) |
| Motion tags | 15 | One-hot encoded action categories|

### Output Space

The model predicts 70 dimensions comprising:
- Joint positions (66D)
- Root velocity (3D)  
- Root angular velocity (1D)

## Dataset

**LAFAN1** (Laboratory of Animation Fidelity and Naturalness):
- 77 motion sequences across 15 action categories
- 22-joint skeleton at 30 Hz
- 496,672 total frames
- 80/20 train/validation split

Preprocessing includes:
- Root-centering and Y-up alignment
- Quaternion-to-rotation-matrix conversion
- Per-feature mean/std normalization
- Non-overlapping sequence chunking (101 frames per sample)
