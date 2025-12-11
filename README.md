# Robo-Barrow: Force-Based Legged Locomotion

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Based on](https://img.shields.io/badge/based%20on-unitree__rl__lab-blue)](https://github.com/unitreerobotics/unitree_rl_lab)

## Overview

Force-based locomotion controller for Unitree Go2. The robot follows external forces applied by users instead of velocity commands, enabling intuitive physical guidance for construction and material transport applications.

**Key Features:**
- End-to-end RL policy using proprioceptive force estimation (no explicit force sensors)
- Force compliance instead of disturbance rejection
- Physics-based force computation from IMU and joint encoders
- Training on flat terrain and slopes with stair generalization testing

Built on [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) and [IsaacLab](https://github.com/isaac-sim/IsaacLab).

## Installation

Follow the standard [unitree_rl_lab installation](https://github.com/unitreerobotics/unitree_rl_lab#installation), then checkout this branch:
```bash
git clone https://github.com/yourusername/unitree_rl_lab.git
cd unitree_rl_lab
git checkout go2-rsl
conda activate env_isaaclab
./unitree_rl_lab.sh -i
```

## Technical Approach

**Standard velocity-tracking:**
```
velocity_command → policy → joint_actions (reject disturbances)
```

**Force-following (this work):**
```
user pushes robot → F_external computed from sensors → policy → joint_actions (minimize force)
```

The robot detects external forces using momentum conservation:
```
F_external = m*a_measured - F_expected
```

where acceleration is measured from IMU and expected forces from motor commands. No explicit force input required.

**Observations:** `[F_external, ω, gravity, q, q̇, last_action]` (no velocity commands)

**Rewards:** Minimize external forces + maintain stability

## Training
```bash
# Train force-following policy
./unitree_rl_lab.sh -t --task Unitree-Go2-Force-Following

# Play trained policy
./unitree_rl_lab.sh -p --task Unitree-Go2-Force-Following
```

## Deployment

Same deployment pipeline as unitree_rl_lab: sim2sim (Mujoco) → sim2real (physical robot).

See [Deploy](https://github.com/unitreerobotics/unitree_rl_lab#deploy) section in base repository.
