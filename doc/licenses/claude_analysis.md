# Claude Analysis: Implementing Hartmann's Multi-Stage Compliance Curriculum in IsaacLab

## Executive Summary

This document analyzes how to implement the Hartmann et al. (2024) "Deep Compliant Control" multi-stage episodic training approach in IsaacLab. The key challenge is that IsaacLab's curriculum system is designed for gradual parameter changes, not the time-bounded stage transitions required by Hartmann's method.

**Key Finding:** IsaacLab curricula cannot directly implement time-based stage transitions within episodes. A custom solution using environment-level state tracking is required.

---

## 1. Understanding the Hartmann Approach

### 1.1 The Three-Stage Episode Structure

Each 4-second episode is divided into:

| Stage | Duration | Purpose | Reward Modification |
|-------|----------|---------|---------------------|
| **Walking** | 2.0s | Normal locomotion | Full tracking rewards active |
| **Recovery** | 1.0s | Compliant response to push | Tracking rewards frozen to constants |
| **Post-Recovery** | 1.0s | Return to normal walking | Full tracking rewards restored |

### 1.2 The Critical Innovation: Frozen Tracking Rewards

During the **Recovery** stage:
```python
# WALKING STAGE: Normal rewards
r_lin = exp(-8 * [(v_x - v*_x)^2 + (v_z - v*_z)^2])  # Penalizes velocity error

# RECOVERY STAGE: Frozen to constant (~0.85 of max)
r_lin_recovery = 0.85  # No penalty for not tracking velocity!

# Energy terms REMAIN ACTIVE in both stages:
r_e = |tau * q_dot|  # Mechanical power
r_tau = ||tau||^2    # Actuator losses
```

This allows the robot to **not fight the disturbance** while still being penalized for aggressive/inefficient motions.

### 1.3 Entry Condition: The Adaptive Push Curriculum

Pushes are only applied when the robot demonstrates good walking:
```
IF average_r_lin > 0.85 * max_r_lin:
    THEN trigger push and enter recovery stage
ELSE:
    REMAIN in walking stage (no push applied)
```

---

## 2. Current IsaacLab Architecture Analysis

### 2.1 CurriculumManager Behavior

From `/home/katari/IsaacLab/source/isaaclab/isaaclab/managers/curriculum_manager.py`:

```python
def compute(self, env_ids: Sequence[int] | None = None):
    """Called every step, iterates over all curriculum terms."""
    for name, term_cfg in zip(self._term_names, self._term_cfgs):
        state = term_cfg.func(self._env, env_ids, **term_cfg.params)
        self._curriculum_state[name] = state
```

**Key observation:** Curricula are computed **every step** and receive `env_ids` of environments that just reset.

### 2.2 EventManager Modes

From `/home/katari/IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py`:

| Mode | Trigger | Use Case |
|------|---------|----------|
| `prestartup` | Before sim starts | USD-level randomization |
| `startup` | Once at training start | One-time initialization |
| `reset` | Every environment reset | State randomization |
| `interval` | Time-based intervals | Periodic disturbances |
| **Custom** | User-defined | **Our solution** |

**Important:** The `interval` mode samples random intervals per environment and applies events when the timer expires. It does NOT support conditional triggering based on reward performance.

### 2.3 Available Curriculum Classes

From `/home/katari/IsaacLab/source/isaaclab/isaaclab/envs/mdp/curriculums.py`:

1. **`modify_reward_weight`**: Changes reward weight after N steps
2. **`modify_env_param`**: Modifies any attribute via dotted path
3. **`modify_term_cfg`**: Convenience wrapper for manager term configs

**Limitation:** These modify parameters globally or based on step count, not based on within-episode time or per-environment conditions.

---

## 3. Problems with Current Implementation

### 3.1 Syntax Errors in `multi_stage_compliance_curriculum`

Location: `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py:64-109`

```python
def multi_stage_compliance_curriculum(
    ...
    event_term_name: str = "push_robot"  # Missing comma!
    reward_threshold: float = 0.85,
) -> torch.Tensor:
"""Curriculum disabling track rewards during push"""  # Docstring not indented!
    ...
    push_env_term.params[velocity_range[x]] = (-0.5, 0.5)  # Invalid syntax!
    # Should be: push_env_term.params["velocity_range"]["x"] = (-0.5, 0.5)
```

### 3.2 Fundamental Design Flaw: Episode Boundary Only

The current implementation only acts at episode boundaries:
```python
if env.common_step_counter % env.max_episode_length == 0:
    # This only triggers at episode resets, not within episodes!
```

Hartmann requires stage transitions **within** the 4-second episode.

### 3.3 No Time Tracking for Stage Duration

The code lacks:
- Per-environment stage tracking
- Timer for stage duration
- Mechanism to restore rewards after recovery period

### 3.4 Synchronous Reward Modification

The code attempts to modify rewards synchronously:
```python
reward_term_vel.weight = reward_term_vel.weight / 50
# ... do push ...
reward_term_vel.weight = reward_term_vel.weight * 50  # Immediate!
```

This doesn't create a 1-second recovery window; it modifies and restores in the same step.

---

## 4. Proposed Architecture

### 4.1 Core Concept: Per-Environment Stage State Machine

Create a state machine that tracks each environment's stage:

```
     ┌─────────────────────────────────────────────────────────────┐
     │                                                             │
     │    WALKING (2s)                                             │
     │    ┌────────────┐                                           │
     │    │ Track full │──── avg_r_lin > 0.85 ────►┌────────────┐ │
     │    │ rewards    │                           │ RECOVERY   │ │
     │    │            │◄─── timer > 1s ───────────│ (1s)       │ │
     │    │            │                           │ Frozen     │ │
     │    └────────────┘                           │ tracking   │ │
     │         ▲                                   │ rewards    │ │
     │         │                                   └─────┬──────┘ │
     │         │                                         │        │
     │         │                                  timer > 1s      │
     │         │                                         │        │
     │         │       ┌────────────┐                    │        │
     │         └───────│POST-RECOV  │◄───────────────────┘        │
     │                 │ (1s)       │                              │
     │                 │ Full       │                              │
     │                 │ rewards    │                              │
     │                 └────────────┘                              │
     │                                                             │
     └─────────────────────────────────────────────────────────────┘
```

### 4.2 Required State Tensors

Add to environment initialization:
```python
# Per-environment state machine
self._compliance_stage = torch.zeros(num_envs, dtype=torch.int, device=device)
# 0 = WALKING, 1 = RECOVERY, 2 = POST_RECOVERY

# Per-environment timer (seconds since stage entry)
self._stage_timer = torch.zeros(num_envs, dtype=torch.float, device=device)

# Per-environment walking reward accumulator
self._walking_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
self._walking_steps = torch.zeros(num_envs, dtype=torch.int, device=device)

# Cached original reward weights (saved once at init)
self._original_track_lin_weight = None
self._original_track_ang_weight = None
```

### 4.3 Implementation Strategy: Custom ManagerTermBase Class

Create a stateful curriculum class that:
1. Inherits from `ManagerTermBase`
2. Maintains per-environment state
3. Modifies rewards per-step based on stage

```python
class HartmannComplianceCurriculum(ManagerTermBase):
    """Multi-stage compliance curriculum following Hartmann et al. (2024)."""

    # Stage constants
    WALKING = 0
    RECOVERY = 1
    POST_RECOVERY = 2

    # Timing (seconds)
    WALKING_DURATION = 2.0
    RECOVERY_DURATION = 1.0
    POST_RECOVERY_DURATION = 1.0

    # Entry threshold
    REWARD_THRESHOLD = 0.85

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Initialize state tensors
        self._stage = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)
        self._stage_timer = torch.zeros(env.num_envs, device=env.device)
        self._walking_reward_accum = torch.zeros(env.num_envs, device=env.device)
        self._walking_step_count = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)

        # Cache original reward weights
        self._track_lin_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy")
        self._track_ang_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z")
        self._original_lin_weight = self._track_lin_cfg.weight
        self._original_ang_weight = self._track_ang_cfg.weight

        # Push event configuration
        self._push_cfg = env.event_manager.get_term_cfg("push_robot")

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset state for terminated environments."""
        if env_ids is None:
            env_ids = slice(None)
        self._stage[env_ids] = self.WALKING
        self._stage_timer[env_ids] = 0.0
        self._walking_reward_accum[env_ids] = 0.0
        self._walking_step_count[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], **kwargs):
        """Called every step to manage stage transitions."""
        dt = env.step_dt

        # Update timers for all environments
        self._stage_timer += dt

        # Get current tracking reward (before any modification)
        current_lin_reward = env.reward_manager._episode_sums["track_lin_vel_xy"]

        # ========== STAGE TRANSITIONS ==========

        # WALKING -> RECOVERY transition
        walking_mask = self._stage == self.WALKING
        walking_envs = walking_mask.nonzero(as_tuple=True)[0]

        if len(walking_envs) > 0:
            # Accumulate walking reward
            self._walking_reward_accum[walking_envs] += current_lin_reward[walking_envs]
            self._walking_step_count[walking_envs] += 1

            # Check if walking duration reached AND reward threshold met
            ready_for_push = (
                (self._stage_timer[walking_envs] >= self.WALKING_DURATION) &
                (self._walking_reward_accum[walking_envs] / self._walking_step_count[walking_envs].float()
                 > self._original_lin_weight * self.REWARD_THRESHOLD)
            )

            transition_envs = walking_envs[ready_for_push]
            if len(transition_envs) > 0:
                self._enter_recovery(env, transition_envs)

        # RECOVERY -> POST_RECOVERY transition
        recovery_mask = self._stage == self.RECOVERY
        recovery_envs = recovery_mask.nonzero(as_tuple=True)[0]

        if len(recovery_envs) > 0:
            recovery_done = self._stage_timer[recovery_envs] >= self.RECOVERY_DURATION
            transition_envs = recovery_envs[recovery_done]
            if len(transition_envs) > 0:
                self._enter_post_recovery(env, transition_envs)

        # POST_RECOVERY -> WALKING transition
        post_recovery_mask = self._stage == self.POST_RECOVERY
        post_recovery_envs = post_recovery_mask.nonzero(as_tuple=True)[0]

        if len(post_recovery_envs) > 0:
            post_done = self._stage_timer[post_recovery_envs] >= self.POST_RECOVERY_DURATION
            transition_envs = post_recovery_envs[post_done]
            if len(transition_envs) > 0:
                self._enter_walking(env, transition_envs)

        # ========== APPLY STAGE-SPECIFIC REWARDS ==========
        self._apply_reward_modifications(env)

        # Return stage distribution for logging
        return {
            "walking_fraction": (self._stage == self.WALKING).float().mean().item(),
            "recovery_fraction": (self._stage == self.RECOVERY).float().mean().item(),
            "post_recovery_fraction": (self._stage == self.POST_RECOVERY).float().mean().item(),
        }

    def _enter_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """Transition to recovery stage: apply push and freeze tracking rewards."""
        self._stage[env_ids] = self.RECOVERY
        self._stage_timer[env_ids] = 0.0

        # Apply velocity push (Hartmann: up to 1.0 m/s horizontal)
        # This directly calls the push function for these environments
        from isaaclab.envs.mdp import push_by_setting_velocity
        push_by_setting_velocity(
            env,
            env_ids,
            velocity_range={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            asset_cfg=SceneEntityCfg("robot")
        )

    def _enter_post_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """Transition to post-recovery stage: restore tracking rewards."""
        self._stage[env_ids] = self.POST_RECOVERY
        self._stage_timer[env_ids] = 0.0

    def _enter_walking(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """Transition back to walking stage: reset accumulators."""
        self._stage[env_ids] = self.WALKING
        self._stage_timer[env_ids] = 0.0
        self._walking_reward_accum[env_ids] = 0.0
        self._walking_step_count[env_ids] = 0

    def _apply_reward_modifications(self, env: ManagerBasedRLEnv):
        """Apply per-step reward modifications based on stage."""
        # This is the tricky part - we need to modify rewards per-environment
        # Option 1: Multiply reward output by mask
        # Option 2: Store modification factor in env.extras for reward function to read

        recovery_mask = (self._stage == self.RECOVERY).float()

        # Store the mask in env for reward functions to access
        env._compliance_recovery_mask = recovery_mask
        env._compliance_frozen_reward_value = 0.85 * self._original_lin_weight
```

### 4.4 Modified Reward Functions

The reward functions need to check for the recovery mask:

```python
def track_lin_vel_xy_exp_compliant(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Tracking reward with compliance support."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Normal tracking reward computation
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] -
                     asset.data.root_lin_vel_b[:, :2]),
        dim=1
    )
    normal_reward = torch.exp(-lin_vel_error / std)

    # Check if compliance curriculum is active
    if hasattr(env, '_compliance_recovery_mask'):
        recovery_mask = env._compliance_recovery_mask
        frozen_value = env._compliance_frozen_reward_value

        # Blend: recovery envs get frozen value, others get normal
        return (1 - recovery_mask) * normal_reward + recovery_mask * frozen_value

    return normal_reward
```

---

## 5. Alternative Implementation: Custom Environment Wrapper

If modifying the reward functions is not desirable, create a wrapper environment:

```python
class ComplianceEnvWrapper(ManagerBasedRLEnv):
    """Environment wrapper that implements Hartmann's multi-stage curriculum."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Initialize compliance state
        self._init_compliance_state()

    def _init_compliance_state(self):
        self._compliance_stage = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self._stage_timer = torch.zeros(self.num_envs, device=self.device)
        # ... (same as above)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Pre-step: update stage and modify reward weights if needed
        self._update_compliance_stages()

        # Normal step
        obs, reward, terminated, truncated, info = super().step(action)

        # Post-step: Apply reward masking for recovery stage
        reward = self._apply_compliance_reward_mask(reward)

        return obs, reward, terminated, truncated, info

    def _update_compliance_stages(self):
        """Update stage state machine before each step."""
        # ... (stage transition logic)

    def _apply_compliance_reward_mask(self, reward: torch.Tensor) -> torch.Tensor:
        """Modify rewards based on compliance stage."""
        # During recovery: replace tracking component with constant
        # ... (reward modification logic)
        return reward
```

---

## 6. Detailed Implementation Steps

### Step 1: Fix Syntax Errors

In `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`:

```python
# Remove or fix the broken multi_stage_compliance_curriculum function
# Replace with the HartmannComplianceCurriculum class
```

### Step 2: Create the Stateful Curriculum Class

Create a new file: `hartmann_curriculum.py`

```python
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.events import push_by_setting_velocity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class HartmannComplianceCurriculum(ManagerTermBase):
    """Implementation of Hartmann et al. (2024) multi-stage compliance training."""

    # ... (full implementation as shown in section 4.3)
```

### Step 3: Modify Reward Functions

Update `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`:

Add compliance-aware tracking rewards that check for the recovery mask.

### Step 4: Update Environment Configuration

In `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/force_env_cfg.py`:

```python
from unitree_rl_lab.tasks.locomotion.mdp.hartmann_curriculum import HartmannComplianceCurriculum

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Existing curricula
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)

    # Hartmann compliance curriculum
    compliance_stages = CurrTerm(
        func=HartmannComplianceCurriculum,
        params={
            "walking_duration": 2.0,
            "recovery_duration": 1.0,
            "post_recovery_duration": 1.0,
            "reward_threshold": 0.85,
            "push_velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        }
    )
```

### Step 5: Update Episode Length

The episode length should be 4 seconds to match Hartmann:

```python
def __post_init__(self):
    # ...
    self.episode_length_s = 4.0  # Changed from 20.0 to match Hartmann
```

### Step 6: Disable Existing Push Event

Comment out or modify the existing `push_robot` event since the curriculum will handle pushes:

```python
@configclass
class EventCfg:
    # ... other events ...

    # Disable: curriculum handles pushes
    # push_robot = EventTerm(...)
```

---

## 7. Key Implementation Considerations

### 7.1 Per-Environment vs Global State

Hartmann's approach requires **per-environment** state tracking:
- Each environment can be in a different stage
- Each environment has its own timer
- Push timing is independent per environment

IsaacLab's default curriculum infrastructure doesn't support this well, hence the need for a custom `ManagerTermBase` class.

### 7.2 Reward Modification Granularity

Two approaches for modifying rewards during recovery:

**Option A: Modify reward weights globally (NOT recommended)**
- Changes affect all environments
- Doesn't support per-environment stages

**Option B: Modify reward OUTPUT per-environment (Recommended)**
- Use masks/tensors to selectively modify rewards
- Store mask in `env.extras` or custom attribute
- Reward functions read and apply the mask

### 7.3 Push Application

Hartmann applies push as a **velocity offset** (impulse), not a continuous force:

```python
# Correct: Set velocity directly (impulse-like)
asset.write_root_velocity_to_sim(vel_w + push_vel, env_ids=env_ids)

# Incorrect: Apply continuous force
asset.set_external_force_and_torque(forces, torques, env_ids)  # This is continuous
```

### 7.4 Timing Synchronization

Ensure stage timing uses simulation time:
```python
dt = env.step_dt  # Time per step in seconds
self._stage_timer += dt
```

Not step counts:
```python
# Don't do this - doesn't account for varying step_dt
self._stage_timer += 1
```

---

## 8. Testing and Validation

### 8.1 Unit Tests

Test the state machine transitions:
```python
def test_walking_to_recovery_transition():
    # Setup environment with high tracking reward
    # Verify transition after 2.0 seconds

def test_recovery_to_post_recovery_transition():
    # Start in recovery stage
    # Verify transition after 1.0 seconds

def test_reward_freezing_during_recovery():
    # Verify tracking rewards return constant during recovery
    # Verify energy rewards still computed normally
```

### 8.2 Logging Metrics

Add these metrics to monitor curriculum behavior:
- Stage distribution (% envs in each stage)
- Average time in each stage
- Number of successful recovery transitions
- Reward values during each stage

---

## 9. References to Source Code

| Component | File Path |
|-----------|-----------|
| Current curriculum implementation | `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py` |
| Current reward functions | `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py` |
| Environment configuration | `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/force_env_cfg.py` |
| IsaacLab curriculum classes | `/home/katari/IsaacLab/source/isaaclab/isaaclab/envs/mdp/curriculums.py` |
| IsaacLab curriculum manager | `/home/katari/IsaacLab/source/isaaclab/isaaclab/managers/curriculum_manager.py` |
| IsaacLab event manager | `/home/katari/IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py` |
| IsaacLab events (push_by_setting_velocity) | `/home/katari/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` |
| IsaacLab MDP reference examples | `/home/katari/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/` |

---

## 10. Summary of Required Changes

1. **Delete/replace** the broken `multi_stage_compliance_curriculum` function
2. **Create** new `HartmannComplianceCurriculum` class inheriting from `ManagerTermBase`
3. **Modify** tracking reward functions to support per-environment masking
4. **Update** environment configuration to use new curriculum
5. **Adjust** episode length to 4.0 seconds
6. **Disable** existing `push_robot` interval event
7. **Add** logging for stage distribution metrics

The key insight is that IsaacLab's curriculum system is designed for gradual parameter changes across training, not for within-episode stage transitions. Implementing Hartmann's approach requires custom state tracking at the environment level.

---
---

# PART 2: Detailed Learning Guide - Understanding IsaacLab Curricula From First Principles

This section is written as a teaching guide. We will build understanding step-by-step, starting from what you already have working in your codebase.

---

## 11. Memory Concerns: Why This Approach is Actually Lightweight

Before diving in, let me address your concern about memory and "silent failures."

### 11.1 How Much Memory Are We Actually Using?

Let's calculate the memory for 4096 environments (your default):

```python
# State tensors we need:
self._stage = torch.zeros(4096, dtype=torch.int, device=device)        # 4096 * 4 bytes = 16 KB
self._stage_timer = torch.zeros(4096, dtype=torch.float, device=device) # 4096 * 4 bytes = 16 KB
self._reward_accum = torch.zeros(4096, dtype=torch.float, device=device) # 4096 * 4 bytes = 16 KB
self._step_count = torch.zeros(4096, dtype=torch.int, device=device)    # 4096 * 4 bytes = 16 KB

# TOTAL: ~64 KB
```

For comparison:
- Your robot has 12 joints × 4096 envs × 4 bytes = 196 KB just for joint positions
- The observation buffer with history=2 uses megabytes
- A single neural network layer uses more memory than all our state tensors

**Conclusion:** Memory is not a concern. These are tiny tensors.

### 11.2 Silent Failures: How to Prevent Them

Silent failures happen when:
1. Tensors end up on wrong device (CPU vs GPU)
2. Tensor shapes mismatch
3. NaN/Inf values accumulate

**Prevention strategies we will use:**
```python
# Always use env.device - this matches where all other tensors live
self._stage = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)

# Always reset on episode termination - prevents accumulation
def reset(self, env_ids):
    self._stage[env_ids] = 0
    self._stage_timer[env_ids] = 0.0

# Add assertions during development
assert self._stage.device == env.device, "Device mismatch!"
assert not torch.isnan(self._stage_timer).any(), "NaN in timer!"
```

---

## 12. Understanding What You Already Have: Your Current Curriculum

Let's look at your working `lin_vel_cmd_levels` function in detail:

**File:** `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/curriculums.py`

```python
def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,          # The environment object - contains everything
    env_ids: Sequence[int],           # Which environments just reset (not used here)
    reward_term_name: str = "track_lin_vel_xy",  # Name of reward to check
) -> torch.Tensor:                    # Must return something for logging
```

**Line-by-line explanation:**

```python
    # Step 1: Get the command term that controls velocity commands
    command_term = env.command_manager.get_term("base_velocity")
```
- `env.command_manager` is the manager that handles all commands (velocity, heading, etc.)
- `get_term("base_velocity")` returns the specific command term you defined in `CommandsCfg`
- This is the same object as your `UniformLevelVelocityCommandCfg`

```python
    # Step 2: Get the current and limit ranges from the command term
    ranges = command_term.cfg.ranges           # Current allowed range, e.g., lin_vel_x=(-0.1, 0.1)
    limit_ranges = command_term.cfg.limit_ranges  # Maximum allowed range, e.g., lin_vel_x=(-1.0, 1.0)
```
- `ranges` is what the robot is currently allowed to do
- `limit_ranges` is the maximum it can ever do (the "final" curriculum level)

```python
    # Step 3: Get the reward term configuration
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
```
- `env.reward_manager` manages all rewards
- `get_term_cfg("track_lin_vel_xy")` returns the `RewTerm` object you defined
- This has `.weight` (e.g., 0.8) and `.func` (the reward function)

```python
    # Step 4: Calculate average reward per second
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
```

Let me break this down piece by piece:
- `env.reward_manager._episode_sums` is a dictionary: `{"reward_name": tensor of shape [num_envs]}`
- `_episode_sums["track_lin_vel_xy"]` gives the accumulated reward for each environment this episode
- `[env_ids]` selects only the environments that just reset
- `torch.mean(...)` averages across those environments
- `/ env.max_episode_length_s` normalizes by episode duration

```python
    # Step 5: Check if we should increase difficulty (only at episode boundaries)
    if env.common_step_counter % env.max_episode_length == 0:
```
- `env.common_step_counter` is the total number of steps across all training
- `env.max_episode_length` is steps per episode (e.g., 1000 steps for 20s episode at 50Hz)
- This `if` only triggers when an episode ends

```python
        # Step 6: If performance is good enough, expand the velocity range
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
```

Breaking this down:
- `reward_term.weight * 0.8` = 0.8 * 0.8 = 0.64 threshold
- If average reward > 0.64, the robot is tracking well
- `delta_command = [-0.1, 0.1]` means: expand range by 0.1 on each side
- `torch.clamp(..., min, max)` ensures we don't exceed `limit_ranges`
- `.tolist()` converts tensor back to Python list (required for config)

```python
    # Step 7: Return something for logging
    return torch.tensor(ranges.lin_vel_x[1], device=env.device)
```
- The curriculum manager logs whatever you return
- This shows up in TensorBoard as "Curriculum/lin_vel_cmd_levels"

### 12.1 The Pattern: Simple Curriculum Functions

Your current curriculum is a **simple function** that:
1. Takes `env` and `env_ids` as arguments
2. Reads some state from the environment
3. Maybe modifies some configuration
4. Returns a value for logging

This pattern works when:
- ✅ Changes happen at episode boundaries
- ✅ Changes are global (same for all environments)
- ✅ No state needs to be remembered between calls

This pattern does NOT work when:
- ❌ Changes need to happen mid-episode
- ❌ Different environments need different treatment
- ❌ State must persist across multiple steps

---

## 13. Understanding IsaacLab's Two Curriculum Patterns

IsaacLab has two ways to define curricula:

### Pattern 1: Simple Functions (What You Currently Use)

```python
def my_curriculum(env, env_ids, param1="default") -> torch.Tensor:
    # Do something
    return some_value
```

**Used as:**
```python
my_curriculum_term = CurrTerm(func=my_curriculum, params={"param1": "custom"})
```

### Pattern 2: Class-Based (What We Need for Temporal Stages)

```python
class MyCurriculum(ManagerTermBase):
    def __init__(self, cfg, env):
        # Called ONCE when training starts
        # Initialize any state here

    def __call__(self, env, env_ids, **kwargs):
        # Called EVERY STEP
        # Do your logic here
        return some_value

    def reset(self, env_ids):
        # Called when environments reset
        # Clean up state for those environments
```

**Used as:**
```python
my_curriculum_term = CurrTerm(func=MyCurriculum, params={"param1": "custom"})
```

The key difference: **class-based curricula can store state between calls**.

---

## 14. Building the Temporal Stage Curriculum Step-by-Step

Let's build a `TemporalStageCurriculum` from scratch, explaining every decision.

### 14.1 The File Structure

Create a new file: `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/temporal_stage_curriculum.py`

### 14.2 The Imports (Line by Line)

```python
from __future__ import annotations
```
**Why:** This allows us to use `ManagerBasedRLEnv` in type hints before it's imported. Python 3.7+ feature for cleaner type annotations.

```python
import torch
```
**Why:** We need PyTorch for tensor operations. All IsaacLab computations use tensors.

```python
from collections.abc import Sequence
```
**Why:** `env_ids` parameter is typed as `Sequence[int]` - this is an abstract type that includes lists, tuples, and other iterables.

```python
from typing import TYPE_CHECKING
```
**Why:** We only need some imports for type checking, not at runtime. This avoids circular imports.

```python
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
```
**Why:** `ManagerBasedRLEnv` is only imported when a type checker (like mypy) runs, not when the code executes. This prevents import errors while still getting type hints.

```python
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
```
**What each one is:**
- `CurriculumTermCfg`: The configuration class for curriculum terms (passed to `__init__`)
- `ManagerTermBase`: The base class we inherit from to create stateful curricula
- `SceneEntityCfg`: Used to specify which asset (robot) to apply pushes to

### 14.3 The Class Definition

```python
class TemporalStageCurriculum(ManagerTermBase):
    """
    A curriculum that divides each episode into temporal stages.

    This enables different reward behaviors at different times within an episode.
    Used for compliance training where we want to:
    1. Walk normally (tracking rewards active)
    2. Respond to push (tracking rewards frozen)
    3. Recover (tracking rewards restored)
    """
```

**Why inherit from `ManagerTermBase`?**

Looking at IsaacLab's source (`/home/katari/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py`):

```python
class ManagerTermBase:
    """Base class for manager terms that are classes."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self._cfg = cfg
        self._env = env

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
```

This base class:
- Stores `cfg` and `env` for us
- Defines `reset()` which gets called when environments terminate
- Requires us to implement `__call__()` for the per-step logic

### 14.4 Class Constants

```python
    # Stage identifiers - using integers for efficiency
    STAGE_WALKING = 0
    STAGE_RECOVERY = 1
    STAGE_POST_RECOVERY = 2
```

**Why integers instead of strings?**
- Faster comparison: `x == 0` is faster than `x == "walking"`
- Less memory: int32 vs string object
- Tensor-friendly: can store in integer tensors

### 14.5 The `__init__` Method (Runs Once at Training Start)

```python
    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        """
        Initialize the temporal stage curriculum.

        Args:
            cfg: The curriculum term configuration from CurrTerm(...)
            env: The environment instance
        """
        # ALWAYS call parent's __init__ first!
        super().__init__(cfg, env)
```

**Why call `super().__init__()`?**
- The parent class stores `self._cfg = cfg` and `self._env = env`
- If you skip this, you won't have access to these later
- It's a Python best practice for inheritance

```python
        # Extract parameters from cfg.params dictionary
        # These come from: CurrTerm(func=..., params={"walking_duration": 2.0, ...})
        self.walking_duration = cfg.params.get("walking_duration", 2.0)
        self.recovery_duration = cfg.params.get("recovery_duration", 1.0)
        self.post_recovery_duration = cfg.params.get("post_recovery_duration", 1.0)
        self.reward_threshold = cfg.params.get("reward_threshold", 0.85)
```

**What is `cfg.params.get()`?**
- `cfg.params` is the dictionary you pass to `CurrTerm(..., params={...})`
- `.get("key", default)` returns the value if key exists, otherwise returns default
- This makes parameters optional - you can override or use defaults

```python
        # Get the velocity range for pushes
        push_range = cfg.params.get("push_velocity_range", {"x": (-1.0, 1.0), "y": (-1.0, 1.0)})
        self.push_vel_x = push_range.get("x", (-1.0, 1.0))
        self.push_vel_y = push_range.get("y", (-1.0, 1.0))
```

**Why nested dictionaries?**
- Matches the format expected by IsaacLab's `push_by_setting_velocity` function
- Makes configuration more readable in the environment config

```python
        # ============================================================
        # STATE TENSORS - These track per-environment state
        # ============================================================

        # Which stage each environment is in (0, 1, or 2)
        # Shape: [num_envs] = [4096] for your config
        # Memory: 4096 * 4 bytes = 16 KB
        self._stage = torch.zeros(
            env.num_envs,           # One value per environment
            dtype=torch.int32,      # Integer type (0, 1, or 2)
            device=env.device       # Same device as environment (GPU)
        )
```

**Critical: `device=env.device`**
- IsaacLab runs on GPU when available
- All tensors MUST be on the same device
- If you use `device="cpu"` here, you'll get errors when comparing with GPU tensors

```python
        # How long (in seconds) each environment has been in its current stage
        # Shape: [num_envs]
        # Reset to 0.0 when entering a new stage
        self._stage_timer = torch.zeros(
            env.num_envs,
            dtype=torch.float32,    # Float for sub-second precision
            device=env.device
        )
```

**Why float32?**
- We add `env.step_dt` each step (e.g., 0.02 seconds)
- Need precision for accurate timing
- float32 is enough (float64 would be overkill)

```python
        # Accumulated reward during walking stage
        # Used to check if robot is walking well before applying push
        self._walking_reward_sum = torch.zeros(env.num_envs, device=env.device)

        # Number of steps in walking stage (for averaging)
        self._walking_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
```

**Why track reward sum AND step count separately?**
- Average = sum / count
- Can't compute running average without both
- Resetting just one would give wrong results

```python
        # ============================================================
        # CACHE ORIGINAL REWARD WEIGHTS
        # ============================================================

        # Get the reward term configurations
        # These are the RewTerm objects from your RewardsCfg
        self._track_lin_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy")
        self._track_ang_cfg = env.reward_manager.get_term_cfg("track_ang_vel_z")

        # Store the original weights (we'll need these to compute frozen values)
        self._original_lin_weight = self._track_lin_cfg.weight  # e.g., 0.8
        self._original_ang_weight = self._track_ang_cfg.weight  # e.g., 0.5
```

**Why cache original weights?**
- We compute frozen reward as: `0.85 * original_weight`
- If we modified the weight and then tried to read it, we'd get wrong value
- Cache once at init, use forever

### 14.6 The `reset` Method (Runs When Environments Terminate)

```python
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """
        Reset state for environments that just terminated.

        This is called by the CurriculumManager when environments reset.

        Args:
            env_ids: Indices of environments that reset.
                     None means all environments.
        """
        # Handle the "all environments" case
        if env_ids is None:
            env_ids = slice(None)  # slice(None) is equivalent to [:] - selects all
```

**What is `slice(None)`?**
- It's the Python object that represents `[:]` (select all)
- `tensor[slice(None)]` = `tensor[:]` = all elements
- More efficient than creating a list of all indices

```python
        # Reset all state tensors for the terminated environments
        self._stage[env_ids] = self.STAGE_WALKING
        self._stage_timer[env_ids] = 0.0
        self._walking_reward_sum[env_ids] = 0.0
        self._walking_step_count[env_ids] = 0
```

**Why is reset necessary?**
- When an episode ends (robot falls, timeout, etc.), the environment resets
- If we don't reset our state, the new episode would start mid-stage
- Example: robot falls during recovery → new episode should start in walking, not recovery

### 14.7 The `__call__` Method (Runs Every Step)

```python
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        **kwargs  # Catches any extra params from CurrTerm(params={...})
    ) -> dict:
        """
        Main logic - called every simulation step.

        Args:
            env: The environment (same as self._env, but passed for consistency)
            env_ids: Environments that just reset (we mostly ignore this)
            **kwargs: Additional parameters from cfg.params

        Returns:
            Dictionary of metrics for logging
        """
```

**Why does `__call__` receive `env` when we already have `self._env`?**
- IsaacLab convention: all term functions receive `env` as first arg
- Consistency with simple function-based terms
- They should be the same object

```python
        # ============================================================
        # STEP 1: UPDATE TIMERS
        # ============================================================

        # env.step_dt is the time per simulation step in seconds
        # e.g., if sim runs at 50 Hz, step_dt = 0.02 seconds
        dt = env.step_dt

        # Add dt to ALL timers (every environment)
        # This is a simple tensor addition: [4096] + scalar
        self._stage_timer += dt
```

**How often does this run?**
- Every simulation step (e.g., 50 times per second)
- For 4096 environments simultaneously
- That's 4096 × 50 = 204,800 updates per second

```python
        # ============================================================
        # STEP 2: GET CURRENT REWARDS
        # ============================================================

        # _episode_sums is a dictionary maintained by reward_manager
        # Key: reward term name
        # Value: tensor of shape [num_envs] with accumulated reward this episode
        current_reward = env.reward_manager._episode_sums["track_lin_vel_xy"]
```

**Is accessing `_episode_sums` safe?**
- The `_` prefix means "internal" by convention
- But it's commonly used in IsaacLab curricula (see their examples)
- It won't be removed without major version change

```python
        # ============================================================
        # STEP 3: HANDLE WALKING STAGE
        # ============================================================

        # Create a boolean mask: True for envs in walking stage
        walking_mask = (self._stage == self.STAGE_WALKING)

        # Get indices of walking environments
        # nonzero() returns indices where condition is True
        # as_tuple=True returns a tuple of (indices,) instead of 2D tensor
        walking_envs = walking_mask.nonzero(as_tuple=True)[0]
```

**What does `.nonzero(as_tuple=True)[0]` do?**

```python
# Example:
stage = torch.tensor([0, 1, 0, 2, 0])  # 5 environments
mask = (stage == 0)                     # [True, False, True, False, True]
indices = mask.nonzero(as_tuple=True)[0]  # tensor([0, 2, 4])
```

```python
        # Only process if there are walking environments
        if len(walking_envs) > 0:
            # Accumulate rewards for walking environments
            self._walking_reward_sum[walking_envs] += current_reward[walking_envs]
            self._walking_step_count[walking_envs] += 1
```

**Why check `len(walking_envs) > 0`?**
- Avoid division by zero later
- Skip unnecessary computation
- Tensor operations on empty tensors can cause issues

```python
            # Calculate average reward for walking environments
            avg_reward = (
                self._walking_reward_sum[walking_envs] /
                self._walking_step_count[walking_envs].float()  # Convert int to float for division
            )

            # Check two conditions for transitioning to recovery:
            # 1. Has been walking long enough (>= walking_duration)
            # 2. Is walking well (avg_reward > threshold)
            time_condition = self._stage_timer[walking_envs] >= self.walking_duration
            performance_condition = avg_reward > (self._original_lin_weight * self.reward_threshold)

            # Both conditions must be true
            ready_for_push = time_condition & performance_condition
```

**Understanding the `&` operator:**
- `&` is bitwise AND for tensors
- Both conditions must be True for result to be True
- Different from `and` which is for Python booleans

```python
            # Get indices of environments ready to transition
            transition_envs = walking_envs[ready_for_push]

            if len(transition_envs) > 0:
                self._enter_recovery(env, transition_envs)
```

**What is `walking_envs[ready_for_push]`?**
```python
# Example:
walking_envs = torch.tensor([0, 2, 5, 7])  # Envs in walking stage
ready_for_push = torch.tensor([False, True, True, False])  # Which are ready
transition_envs = walking_envs[ready_for_push]  # tensor([2, 5])
```

```python
        # ============================================================
        # STEP 4: HANDLE RECOVERY STAGE (similar pattern)
        # ============================================================

        recovery_mask = (self._stage == self.STAGE_RECOVERY)
        recovery_envs = recovery_mask.nonzero(as_tuple=True)[0]

        if len(recovery_envs) > 0:
            # Simple time-based transition (no performance check)
            time_up = self._stage_timer[recovery_envs] >= self.recovery_duration
            transition_envs = recovery_envs[time_up]

            if len(transition_envs) > 0:
                self._enter_post_recovery(env, transition_envs)
```

```python
        # ============================================================
        # STEP 5: HANDLE POST-RECOVERY STAGE
        # ============================================================

        post_recovery_mask = (self._stage == self.STAGE_POST_RECOVERY)
        post_recovery_envs = post_recovery_mask.nonzero(as_tuple=True)[0]

        if len(post_recovery_envs) > 0:
            time_up = self._stage_timer[post_recovery_envs] >= self.post_recovery_duration
            transition_envs = post_recovery_envs[time_up]

            if len(transition_envs) > 0:
                self._enter_walking(env, transition_envs)
```

```python
        # ============================================================
        # STEP 6: STORE STAGE INFO FOR REWARD FUNCTIONS
        # ============================================================

        # Create a mask that reward functions can read
        # 1.0 for recovery stage, 0.0 for others
        recovery_mask_float = (self._stage == self.STAGE_RECOVERY).float()

        # Store on the environment object
        # Reward functions will check: if hasattr(env, '_temporal_stage_recovery_mask')
        env._temporal_stage_recovery_mask = recovery_mask_float
        env._temporal_stage_frozen_value = self._original_lin_weight * self.reward_threshold
```

**Why store on `env` object?**
- Reward functions receive `env` as argument
- They can't access our curriculum object directly
- `env` is the shared communication channel

```python
        # ============================================================
        # STEP 7: RETURN METRICS FOR LOGGING
        # ============================================================

        # Calculate what fraction of environments are in each stage
        num_envs = float(env.num_envs)

        return {
            "walking_frac": (self._stage == self.STAGE_WALKING).sum().item() / num_envs,
            "recovery_frac": (self._stage == self.STAGE_RECOVERY).sum().item() / num_envs,
            "post_recovery_frac": (self._stage == self.STAGE_POST_RECOVERY).sum().item() / num_envs,
        }
```

**What does `.item()` do?**
- Converts a single-element tensor to a Python number
- Required because we're returning a dictionary (not tensor)
- Example: `torch.tensor(5).item()` → `5` (Python int)

### 14.8 The Transition Helper Methods

```python
    def _enter_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """
        Transition specified environments to recovery stage.
        Applies a velocity push to simulate external disturbance.
        """
        # Update stage
        self._stage[env_ids] = self.STAGE_RECOVERY

        # Reset timer (recovery duration starts now)
        self._stage_timer[env_ids] = 0.0

        # ============================================================
        # APPLY THE PUSH
        # ============================================================

        # Import the push function from IsaacLab
        from isaaclab.envs.mdp.events import push_by_setting_velocity

        # Apply push to the specified environments
        push_by_setting_velocity(
            env=env,
            env_ids=env_ids,
            velocity_range={
                "x": self.push_vel_x,  # e.g., (-1.0, 1.0)
                "y": self.push_vel_y,  # e.g., (-1.0, 1.0)
            },
            asset_cfg=SceneEntityCfg("robot")  # Which asset to push
        )
```

**What does `push_by_setting_velocity` do?**

Looking at IsaacLab source (`events.py:1046-1071`):
```python
def push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg):
    asset = env.scene[asset_cfg.name]  # Get the robot

    # Get current velocity
    vel_w = asset.data.root_vel_w[env_ids]

    # Sample random velocity offsets
    ranges = torch.tensor([velocity_range["x"], velocity_range["y"], ...])
    push_vel = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape)

    # Add push to current velocity
    vel_w += push_vel

    # Write back to simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
```

**This is an impulse, not a force:**
- Force would need to be applied every step
- This directly changes velocity (instantaneous impulse)
- More like being pushed by a person

```python
    def _enter_post_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """Transition to post-recovery stage."""
        self._stage[env_ids] = self.STAGE_POST_RECOVERY
        self._stage_timer[env_ids] = 0.0
        # No special action needed - just tracking rewards restored
```

```python
    def _enter_walking(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        """Transition back to walking stage."""
        self._stage[env_ids] = self.STAGE_WALKING
        self._stage_timer[env_ids] = 0.0

        # Reset reward accumulators for the new walking phase
        self._walking_reward_sum[env_ids] = 0.0
        self._walking_step_count[env_ids] = 0
```

---

## 15. Modifying Your Reward Function

Your existing reward function needs a small modification to check for the recovery mask.

### 15.1 Current Reward Function (for reference)

From IsaacLab's `rewards.py`:
```python
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] -
            asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1
    )
    return torch.exp(-lin_vel_error / std)
```

### 15.2 Modified Version for Temporal Stages

Add this to your `rewards.py`:

```python
def track_lin_vel_xy_exp_staged(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Velocity tracking reward with temporal stage support.

    During recovery stage, returns a frozen constant value.
    This allows the robot to not fight disturbances.
    """
    asset = env.scene[asset_cfg.name]

    # Compute normal tracking reward
    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] -
            asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1
    )
    normal_reward = torch.exp(-lin_vel_error / std)

    # ============================================================
    # CHECK FOR TEMPORAL STAGE CURRICULUM
    # ============================================================

    # Check if the curriculum has stored a recovery mask
    if hasattr(env, '_temporal_stage_recovery_mask'):
        # Get the mask (1.0 for recovery, 0.0 for other stages)
        recovery_mask = env._temporal_stage_recovery_mask

        # Get the frozen value to use during recovery
        frozen_value = env._temporal_stage_frozen_value

        # Blend based on mask:
        # - recovery_mask=0: (1-0)*normal + 0*frozen = normal
        # - recovery_mask=1: (1-1)*normal + 1*frozen = frozen
        return (1.0 - recovery_mask) * normal_reward + recovery_mask * frozen_value

    # If no curriculum active, return normal reward
    return normal_reward
```

**How does the blending work?**

```python
# Example for 4 environments:
normal_reward = torch.tensor([0.9, 0.7, 0.8, 0.6])
recovery_mask = torch.tensor([0.0, 1.0, 0.0, 1.0])  # Envs 1,3 in recovery
frozen_value = 0.68  # 0.85 * 0.8

result = (1 - recovery_mask) * normal_reward + recovery_mask * frozen_value
# = [1.0, 0.0, 1.0, 0.0] * [0.9, 0.7, 0.8, 0.6] + [0.0, 1.0, 0.0, 1.0] * 0.68
# = [0.9, 0.0, 0.8, 0.0] + [0.0, 0.68, 0.0, 0.68]
# = [0.9, 0.68, 0.8, 0.68]
```

---

## 16. Updating Your Environment Configuration

### 16.1 Add the Import

In `force_env_cfg.py`, add:

```python
from unitree_rl_lab.tasks.locomotion.mdp.temporal_stage_curriculum import TemporalStageCurriculum
```

### 16.2 Update the Curriculum Configuration

```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Keep your existing curricula
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)

    # Add the temporal stage curriculum
    temporal_stages = CurrTerm(
        func=TemporalStageCurriculum,  # The class, not an instance!
        params={
            "walking_duration": 2.0,      # Seconds in walking stage
            "recovery_duration": 1.0,     # Seconds in recovery stage
            "post_recovery_duration": 1.0, # Seconds in post-recovery stage
            "reward_threshold": 0.85,     # Must achieve 85% of max reward to trigger push
            "push_velocity_range": {
                "x": (-1.0, 1.0),  # Push velocity range in x (m/s)
                "y": (-1.0, 1.0),  # Push velocity range in y (m/s)
            },
        }
    )
```

**Note:** `func=TemporalStageCurriculum` passes the **class**, not an instance. IsaacLab will instantiate it.

### 16.3 Update the Reward Configuration

Change your tracking reward to use the staged version:

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Changed from track_lin_vel_xy_exp to track_lin_vel_xy_exp_staged
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp_staged,  # Use the staged version
        weight=0.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # ... rest of your rewards unchanged ...
```

### 16.4 Disable the Existing Push Event

Since the curriculum handles pushes, disable the interval-based push:

```python
@configclass
class EventCfg:
    """Configuration for events."""

    # ... other events ...

    # COMMENTED OUT - curriculum handles pushes now
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(5.0, 10.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )
```

### 16.5 Adjust Episode Length (Optional)

For Hartmann's exact setup:

```python
def __post_init__(self):
    # ... other settings ...
    self.episode_length_s = 4.0  # 2s walking + 1s recovery + 1s post-recovery
```

But you can keep 20s episodes and have multiple push-recovery cycles per episode.

---

## 17. Complete Code: Ready to Copy

### 17.1 The Curriculum File

Create `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/temporal_stage_curriculum.py`:

```python
"""
Temporal Stage Curriculum for Compliance Training.

This curriculum divides each episode into stages with different reward behaviors.
Based on the approach from Hartmann et al. (2024) "Deep Compliant Control".
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TemporalStageCurriculum(ManagerTermBase):
    """
    Curriculum that divides episodes into temporal stages.

    Stages:
        0 (WALKING): Normal tracking rewards, accumulate performance
        1 (RECOVERY): Frozen tracking rewards, energy rewards active
        2 (POST_RECOVERY): Tracking rewards restored

    Transitions:
        WALKING -> RECOVERY: After walking_duration AND if performance > threshold
        RECOVERY -> POST_RECOVERY: After recovery_duration
        POST_RECOVERY -> WALKING: After post_recovery_duration
    """

    # Stage constants
    STAGE_WALKING = 0
    STAGE_RECOVERY = 1
    STAGE_POST_RECOVERY = 2

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Extract parameters with defaults
        self.walking_duration = cfg.params.get("walking_duration", 2.0)
        self.recovery_duration = cfg.params.get("recovery_duration", 1.0)
        self.post_recovery_duration = cfg.params.get("post_recovery_duration", 1.0)
        self.reward_threshold = cfg.params.get("reward_threshold", 0.85)

        push_range = cfg.params.get("push_velocity_range", {"x": (-1.0, 1.0), "y": (-1.0, 1.0)})
        self.push_vel_x = push_range.get("x", (-1.0, 1.0))
        self.push_vel_y = push_range.get("y", (-1.0, 1.0))

        # State tensors
        self._stage = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self._stage_timer = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._walking_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._walking_step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

        # Cache original reward weights
        self._track_lin_cfg = env.reward_manager.get_term_cfg("track_lin_vel_xy")
        self._original_lin_weight = self._track_lin_cfg.weight

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage[env_ids] = self.STAGE_WALKING
        self._stage_timer[env_ids] = 0.0
        self._walking_reward_sum[env_ids] = 0.0
        self._walking_step_count[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], **kwargs) -> dict:
        dt = env.step_dt
        self._stage_timer += dt

        current_reward = env.reward_manager._episode_sums["track_lin_vel_xy"]

        # Handle WALKING stage
        walking_envs = (self._stage == self.STAGE_WALKING).nonzero(as_tuple=True)[0]
        if len(walking_envs) > 0:
            self._walking_reward_sum[walking_envs] += current_reward[walking_envs]
            self._walking_step_count[walking_envs] += 1

            avg_reward = self._walking_reward_sum[walking_envs] / self._walking_step_count[walking_envs].float()
            ready = (
                (self._stage_timer[walking_envs] >= self.walking_duration) &
                (avg_reward > self._original_lin_weight * self.reward_threshold)
            )
            transition_envs = walking_envs[ready]
            if len(transition_envs) > 0:
                self._enter_recovery(env, transition_envs)

        # Handle RECOVERY stage
        recovery_envs = (self._stage == self.STAGE_RECOVERY).nonzero(as_tuple=True)[0]
        if len(recovery_envs) > 0:
            done = self._stage_timer[recovery_envs] >= self.recovery_duration
            transition_envs = recovery_envs[done]
            if len(transition_envs) > 0:
                self._enter_post_recovery(env, transition_envs)

        # Handle POST_RECOVERY stage
        post_envs = (self._stage == self.STAGE_POST_RECOVERY).nonzero(as_tuple=True)[0]
        if len(post_envs) > 0:
            done = self._stage_timer[post_envs] >= self.post_recovery_duration
            transition_envs = post_envs[done]
            if len(transition_envs) > 0:
                self._enter_walking(env, transition_envs)

        # Store mask for reward functions
        env._temporal_stage_recovery_mask = (self._stage == self.STAGE_RECOVERY).float()
        env._temporal_stage_frozen_value = self._original_lin_weight * self.reward_threshold

        # Return metrics
        n = float(env.num_envs)
        return {
            "walking_frac": (self._stage == self.STAGE_WALKING).sum().item() / n,
            "recovery_frac": (self._stage == self.STAGE_RECOVERY).sum().item() / n,
            "post_recovery_frac": (self._stage == self.STAGE_POST_RECOVERY).sum().item() / n,
        }

    def _enter_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        self._stage[env_ids] = self.STAGE_RECOVERY
        self._stage_timer[env_ids] = 0.0

        from isaaclab.envs.mdp.events import push_by_setting_velocity
        push_by_setting_velocity(
            env, env_ids,
            velocity_range={"x": self.push_vel_x, "y": self.push_vel_y},
            asset_cfg=SceneEntityCfg("robot")
        )

    def _enter_post_recovery(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        self._stage[env_ids] = self.STAGE_POST_RECOVERY
        self._stage_timer[env_ids] = 0.0

    def _enter_walking(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        self._stage[env_ids] = self.STAGE_WALKING
        self._stage_timer[env_ids] = 0.0
        self._walking_reward_sum[env_ids] = 0.0
        self._walking_step_count[env_ids] = 0
```

### 17.2 The Modified Reward Function

Add to `/home/katari/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`:

```python
def track_lin_vel_xy_exp_staged(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Velocity tracking with temporal stage support."""
    asset: Articulation = env.scene[asset_cfg.name]

    lin_vel_error = torch.sum(
        torch.square(
            env.command_manager.get_command(command_name)[:, :2] -
            asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1
    )
    normal_reward = torch.exp(-lin_vel_error / std)

    if hasattr(env, '_temporal_stage_recovery_mask'):
        mask = env._temporal_stage_recovery_mask
        frozen = env._temporal_stage_frozen_value
        return (1.0 - mask) * normal_reward + mask * frozen

    return normal_reward
```

---

## 18. Debugging Tips

### 18.1 Verify Tensors Are on Correct Device

```python
# Add this to __init__ for debugging:
print(f"Stage tensor device: {self._stage.device}")
print(f"Environment device: {env.device}")
assert self._stage.device == env.device
```

### 18.2 Monitor Stage Transitions in TensorBoard

The returned dictionary from `__call__` appears in TensorBoard under "Curriculum/temporal_stages/...":
- `walking_frac`: Should start at 1.0, decrease as pushes happen
- `recovery_frac`: Should spike after pushes
- `post_recovery_frac`: Should follow recovery with 1s delay

### 18.3 Check for NaN Values

```python
# Add this check periodically:
if torch.isnan(self._stage_timer).any():
    print("WARNING: NaN in stage timer!")
    self._stage_timer = torch.nan_to_num(self._stage_timer, 0.0)
```

---

## 19. Summary: What You Learned

1. **Curriculum patterns**: Simple functions vs class-based with `ManagerTermBase`
2. **State management**: Using tensors to track per-environment state
3. **Memory efficiency**: A few float32 tensors for 4096 envs = ~64KB
4. **Device consistency**: Always use `device=env.device`
5. **Tensor operations**: Masks, nonzero, indexing for conditional logic
6. **Inter-component communication**: Storing data on `env` for rewards to read
7. **IsaacLab patterns**: How managers, terms, and configurations work together