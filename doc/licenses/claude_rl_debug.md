# Claude RL Debug Analysis: Training Collapse in Force Environment

**Date:** 2026-01-31
**Training Log:** `logs/rsl_rl/unitree_go2_force/2026-01-30_11-58-29`
**Environment:** `force_env_cfg.py`
**Baseline Comparison:** `velocity_env_cfg.py` (known working)

---

## Executive Summary

The training run experienced a **catastrophic collapse** where episode lengths dropped from ~240 steps (4.8 seconds) to just **7 steps (0.14 seconds)** by iteration 50. The robot is terminating almost immediately after spawning due to **bad orientation** (tilting over). The primary cause is the `TemporalStageCurriculum` applying strong velocity pushes (±1.0 m/s) too early in training, before the robot learned stable locomotion.

---

## 1. Quantitative Analysis from TensorBoard Logs

### 1.1 Episode Length Progression

| Iteration | Episode Length (steps) | Notes |
|-----------|------------------------|-------|
| 0 | 14.12 | Initial learning |
| 4 | 117.24 | Robot learning to stand |
| 13 | **240.65** | Peak performance |
| 17 | 137.05 | Beginning of decline |
| 19 | 103.85 | Rapid deterioration |
| 50 | 9.29 | Collapse |
| 100+ | **7.00** | Complete failure (0.14s) |

**Observation:** Training peaked at iteration 13, then rapidly collapsed. By iteration 50, episodes last only 7 steps out of a maximum 1000 (20 seconds × 50 Hz).

### 1.2 Termination Cause Analysis

| Termination | First 5 Iterations (avg) | Last 5 Iterations (avg) | Trend |
|-------------|--------------------------|-------------------------|-------|
| `bad_orientation` | 3.3 | **585.0** | +17,600% |
| `base_contact` | 0.0 | 0.0 | No change |
| `time_out` | 4.4 | 0.0 | -100% |

**Critical Finding:** `bad_orientation` terminations exploded from ~3 to ~585 per iteration. The robot is consistently falling over (tilting beyond the 0.8 radian limit). No robots reach the 20-second timeout anymore.

### 1.3 Curriculum Stage Distribution

| Iteration | Walking Frac | Recovery Frac | Post-Recovery Frac |
|-----------|--------------|---------------|-------------------|
| 0-3 | 1.000 | 0.000 | 0.000 |
| 4 | **0.380** | 0.620 | 0.000 |
| 5 | 0.257 | **0.743** | 0.000 |
| 10 | 0.753 | 0.247 | 0.000 |
| 20+ | 1.000 | 0.000 | 0.000 |

**Critical Observation:**
- At iteration 4-5, 62-74% of environments transitioned to RECOVERY stage and received velocity pushes
- By iteration 20+, all environments are stuck in WALKING stage
- **Why?** Episodes terminate so quickly (~7 steps = 0.14s) that they never reach the 2-second `walking_duration` threshold to trigger a push

### 1.4 Reward Signal Analysis

| Reward Term | Min | Max | Mean |
|-------------|-----|-----|------|
| `alive` | 0.001 | 0.036 | 0.001 |
| `track_lin_vel_xy` | 0.005 | 0.152 | 0.007 |
| `track_ang_vel_z` | 0.002 | 0.041 | 0.003 |
| `base_pose` | -0.052 | -0.003 | -0.004 |
| `undesired_contacts` | -0.021 | 0.000 | -0.000 |

**Total Mean Reward Progression:**
- First 5 iterations: [-1.6, -3.8, -6.0, -8.2, -10.1]
- Last 5 iterations: [-0.055, -0.055, -0.053, -0.055, -0.055]

**Issue:** The `alive` reward averages only 0.001 (weight=0.15). For a 20s episode at 50Hz, this should be ~150 (1000 steps × 0.15). Getting 0.001 confirms episodes last only ~7 steps.

---

## 2. Root Cause Analysis

### 2.1 Revised Understanding: Pushes Did NOT Immediately Cause Collapse

**Important Correction:** The initial analysis incorrectly stated pushes caused immediate collapse. The data shows:

| Phase | Iterations | Recovery Frac | Episode Length | What Happened |
|-------|------------|---------------|----------------|---------------|
| Pre-push | 0-3 | 0% | 14→90 | Learning to stand |
| **Handling pushes** | 4-13 | 6-74% | 117→240 | **Robot successfully handled pushes!** |
| Decline starts | 14-16 | 11-33% | 212→178 | Something starts going wrong |
| Collapse | 17-24 | 3-12% | 137→28 | Rapid failure |

The robot handled pushes successfully for **9 iterations** (4-13) while episode length continued to increase. The collapse started around iteration 14-17, not immediately after pushes began.

### 2.2 What Actually Triggered Collapse?

The velocity command curriculum **never advanced** (stayed at 0.1 m/s throughout), so increasing difficulty wasn't the cause. Possible factors:

1. **Cumulative policy degradation**: Repeated push experiences may have gradually destabilized the policy
2. **Reward signal issues**: Mean reward peaked at -17.0 at iteration 13 (most negative = longest episode with most penalties)
3. **Training dynamics**: Once episode length started dropping, less practice time accelerated the collapse

### 2.3 The TemporalStageCurriculum Configuration

```python
# From force_env_cfg.py
temporal_stages = CurrTerm(
    func=TemporalStageCurriculum,
    params={
        "walking_duration": 2.0,        # Only 2 seconds before push
        "reward_threshold": 0.5,        # 50% of max reward triggers push
        "push_velocity_range": {
            "x": (-1.0, 1.0),           # Strong push: ±1.0 m/s
            "y": (-1.0, 1.0),
        },
    },
)
```

**Problem Breakdown:**

1. **Threshold Too Low:** With `reward_threshold: 0.5` and `track_lin_vel_xy` weight of 0.8, the push triggers when average reward reaches just 0.4 (0.5 × 0.8). This happens very early in training.

2. **Push Too Strong:** The push velocity of ±1.0 m/s is double what the working `velocity_env_cfg.py` uses (±0.5 m/s).

3. **No Progressive Curriculum:** The push goes from 0% to 100% intensity instantly. There's no gradual increase based on robot capability.

### 2.2 Comparison with Working velocity_env_cfg.py

| Parameter | force_env_cfg (broken) | velocity_env_cfg (works) |
|-----------|------------------------|--------------------------|
| Push mechanism | TemporalStageCurriculum | Interval-based EventTerm |
| Push timing | After 2s walking | Random 5-10s intervals |
| Push velocity | ±1.0 m/s | ±0.5 m/s |
| Push trigger | 50% reward threshold | Time-based (no performance check) |
| Progressive curriculum | No | Implicitly (later in episode = more stable) |

### 2.3 The Cascade Failure

1. **Iteration 4-5:** Robot achieves 50% tracking reward, triggering pushes
2. **Push applied:** ±1.0 m/s velocity impulse destabilizes unstable robot
3. **Robot falls:** Triggers `bad_orientation` termination
4. **Short episodes:** Only 7 steps = 0.14 seconds
5. **No learning:** Robot never experiences walking, only falling
6. **Policy degradation:** Learns to "accept" falling as inevitable
7. **Self-reinforcing:** Worse policy → faster falls → less learning

---

## 3. Secondary Issues Identified

### 3.1 Reward Signal During Collapse

The `alive` reward (weight=0.15) is too weak to provide meaningful survival incentive:

```
With 7-step episodes: 7 × 0.15 = 1.05 total alive reward
With 1000-step episodes: 1000 × 0.15 = 150 total alive reward
```

The difference (150x) in potential reward is lost, removing any gradient signal for staying upright longer.

### 3.2 Missing Stability Prerequisite

According to Hartmann et al. (2024), pushes should only be applied when:
> "average r_lin > 85% of maximum value" (Section 1.2.5)

The current implementation uses 50%, which is too permissive.

### 3.3 Initial Robot State

From `env.yaml`:
```yaml
init_state:
  pos: [0.0, 0.0, 0.4]  # 0.4m spawn height
```

The robot spawns at 0.4m height, but `base_pose_penalty` targets 0.31m (`desired_height: 0.31`). This 9cm discrepancy creates immediate instability as the robot "falls" to reach target height.

---

## 4. Suggested Fixes (with API Documentation Citations)

All suggestions reference the [IsaacLab MDP API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html).

### 4.1 Fix TemporalStageCurriculum Threshold

**Current:**
```python
"reward_threshold": 0.5,  # Too low - triggers at 40% performance
```

**Suggested:**
```python
"reward_threshold": 0.85,  # Matches Hartmann paper
```

**API Reference:** The threshold should match the reward weight scaling. Per IsaacLab's reward function design, `track_lin_vel_xy_exp` returns values in [0, 1], so with weight 0.8, the threshold should be at least 0.68 (0.85 × 0.8) to ensure stable locomotion.

### 4.2 Reduce Push Velocity (Progressive Curriculum)

**Current:**
```python
"push_velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
```

**Suggested:** Implement a progressive push curriculum:
```python
# Start with gentle pushes
"push_velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
# Gradually increase based on training progress
```

**API Reference:** `push_by_setting_velocity` from `isaaclab.envs.mdp.events` directly sets root velocity. The velocity magnitude directly impacts stability - larger values require more robust policies to handle.

### 4.3 Increase Alive Reward Weight

**Current:**
```python
alive = RewTerm(func=mdp.is_alive, weight=0.15)
```

**Suggested:**
```python
alive = RewTerm(func=mdp.is_alive, weight=0.5)  # Or higher
```

**API Reference:** `is_alive` returns 1.0 for each step the robot hasn't terminated. Per IsaacLab documentation, this "provides a constant positive reward for each timestep the robot remains alive," which should be weighted appropriately against task rewards.

### 4.4 Increase Walking Duration Before Push

**Current:**
```python
"walking_duration": 2.0,
```

**Suggested:**
```python
"walking_duration": 5.0,  # More time to stabilize
```

This ensures the robot has demonstrated sustained stability before being challenged with pushes.

### 4.5 Match Spawn Height to Target Height

**Current:**
```python
init_state.pos: [0.0, 0.0, 0.4]  # 0.4m
desired_height: 0.31              # 0.31m
```

**Suggested:** Either:
- Lower spawn height to 0.31m
- Or increase `desired_height` to 0.35m

This prevents the initial "drop" that destabilizes the robot.

### 4.6 Add Termination Condition Relaxation During Early Training

Consider temporarily relaxing `bad_orientation` limit during initial training:

**Current:**
```python
bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
```

**API Reference:** `bad_orientation` terminates when "the robot's orientation deviates excessively from upright." The limit_angle parameter (in radians) controls sensitivity. Consider starting with 1.0 and decreasing via curriculum.

### 4.7 Disable TemporalStageCurriculum Initially

The most conservative fix: disable the curriculum entirely until the robot can reliably walk:

```python
# Comment out or remove:
# temporal_stages = CurrTerm(func=TemporalStageCurriculum, ...)
```

Train a stable walking policy first (using the working `velocity_env_cfg.py` approach), then add compliance training as a second phase.

---

## 5. Recommended Training Strategy

Based on Hartmann et al. (2024) and the analysis above:

### Phase 1: Stable Locomotion (disable TemporalStageCurriculum)
1. Train until robot achieves consistent 85%+ tracking reward
2. Episode lengths should regularly reach timeout (20s)
3. `bad_orientation` terminations should be < 10% of episodes

### Phase 2: Gentle Push Introduction
1. Enable TemporalStageCurriculum with:
   - `reward_threshold: 0.85`
   - `walking_duration: 5.0`
   - `push_velocity_range: ±0.3 m/s`

### Phase 3: Progressive Push Strengthening
1. Gradually increase push velocity to ±1.0 m/s
2. Only increase when robot maintains 80%+ success rate
3. Reduce walking duration to 2.0s as robot stabilizes

---

## 6. Key Metrics to Monitor

When retraining, monitor these TensorBoard metrics:

| Metric | Healthy Value | Current Value |
|--------|---------------|---------------|
| `Train/mean_episode_length` | > 500 steps | 7 steps |
| `Episode_Termination/bad_orientation` | < 50/iter | 585/iter |
| `Episode_Termination/time_out` | > 3/iter | 0/iter |
| `Episode_Reward/alive` | > 0.5 | 0.001 |
| `Curriculum/temporal_stages/walking_frac` | > 0.7 | 1.0 (stuck) |

---

## 7. Conclusion

The training failure is a classic case of **curriculum mismatch** - the challenge (strong velocity pushes) was introduced before the prerequisite skill (stable locomotion) was learned. The robot never had a chance to learn walking because it was being pushed over immediately.

The fix is straightforward: delay and gradually introduce the push challenges, matching the approach that made `velocity_env_cfg.py` successful. The Hartmann paper's 85% threshold exists precisely to prevent this failure mode.

---

## 8. Testing Without TemporalStageCurriculum

To isolate whether the curriculum is the issue, you can disable it and test basic locomotion learning.

### Option A: Comment out in force_env_cfg.py

```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)

    # DISABLED FOR TESTING - uncomment to re-enable
    # temporal_stages = CurrTerm(
    #     func=TemporalStageCurriculum,
    #     params={...},
    # )
```

### Option B: Also remove the staged reward function

If you disable the curriculum, you should also use the standard tracking reward:

```python
# In RewardsCfg:
track_lin_vel_xy = RewTerm(
    func=mdp.track_lin_vel_xy_exp,  # Use standard version, not _staged
    weight=0.8,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
)
```

### Expected Outcome

Without the curriculum:
- Episode lengths should consistently reach 500+ steps
- `bad_orientation` terminations should stay low (<50/iteration)
- `time_out` terminations should be common (robots reaching 20s)
- Velocity command curriculum should eventually advance beyond 0.1 m/s

If training still collapses without the curriculum, the issue lies elsewhere (reward weights, termination thresholds, or initial conditions).

---

## 9. Second Training Run Analysis (2026-01-31_00-59-22)

After fixing the reward weights to match `velocity_env_cfg.py`, the robot successfully learned to walk. However, the **TemporalStageCurriculum never triggered** - the robot stayed in WALKING stage for all 28,144 iterations.

### 9.1 Curriculum Never Triggered

| Metric | Value |
|--------|-------|
| Total iterations | 28,144 |
| Max recovery_frac | **0.0000** |
| Walking_frac (entire run) | 1.0000 |

The robot learned to walk successfully (episode length: 14 → 965 steps, mean_reward: -0.05 → 32.5), but **no pushes were ever applied**.

### 9.2 Root Cause: Threshold Calculation Bug

The curriculum checks:
```python
avg_reward > self._original_lin_weight * self.reward_threshold
```

Where:
- `self._original_lin_weight = 1.5` (track_lin_vel_xy weight)
- `self.reward_threshold = 0.5`
- **Threshold = 1.5 × 0.5 = 0.75**

But looking at the actual per-step rewards:

| Iter | Episode Length | track_lin per step | Threshold |
|------|----------------|-------------------|-----------|
| 100 | 990.7 | 0.00145 | 0.75 |
| 500 | 954.7 | 0.00145 | 0.75 |
| 1000 | 961.7 | 0.00145 | 0.75 |

**The per-step reward is 0.00145, but the threshold expects 0.75!**

This is a ~500x mismatch. The curriculum will **never** trigger because the condition can never be satisfied.

### 9.3 Why the Mismatch?

The `_episode_sums` from IsaacLab appears to be normalized or scaled. Checking the `alive` reward:
- Weight = 0.15
- `is_alive()` returns 1.0
- Expected per-step = 0.15
- Actual per-step = 0.000151 (1000x smaller!)

The logged Episode_Reward values are divided by ~1000 (possibly number of environments or some other normalization).

### 9.4 Suggested Fix (DO NOT APPLY YET - For Review)

The threshold comparison needs to account for this scaling. Options:

**Option A: Use a much lower threshold**
```python
# In force_env_cfg.py
"reward_threshold": 0.0005,  # Instead of 0.5
```

**Option B: Fix the calculation in temporal_stage_curriculum.py**
```python
# Instead of comparing weighted values, compare raw reward function output
# The exp() function returns [0, 1], so threshold should be in that range
avg_reward = episode_sum[walking_envs] / self._walking_step_count[walking_envs].float().clamp(min=1)
# Divide by weight to get raw reward value
raw_avg_reward = avg_reward / self._original_lin_weight
ready = (
    (self._stage_timer[walking_envs] >= self.walking_duration) &
    (raw_avg_reward > self.reward_threshold)  # Compare raw value to threshold
)
```

**Option C: Check IsaacLab's actual _episode_sums format**

The `env.reward_manager._episode_sums` might not be what we think. Need to verify whether it's:
- Per-environment cumulative sum
- Averaged across environments
- Normalized by some factor

---

## 10. Understanding RL Training Metrics (Educational Guide)

This section explains the TensorBoard metrics from a reinforcement learning perspective.

### 10.1 The Big Picture: What is RL Doing?

Imagine teaching a dog to sit:
1. Dog tries something random (exploring)
2. You give treat or no treat (reward)
3. Dog learns: "when I sit, I get treats" (policy update)
4. Dog sits more often (exploitation)

RL does the same thing, but with math:
1. **Policy (π)**: The "brain" - a neural network that maps observations → actions
2. **Value function (V)**: Predicts "how good is this situation?" (expected future rewards)
3. **Reward (r)**: Immediate feedback signal
4. **Return (G)**: Total discounted future rewards: G = r₀ + γr₁ + γ²r₂ + ...

### 10.2 PPO Algorithm (What RSL-RL Uses)

PPO (Proximal Policy Optimization) is the algorithm training your robot. Here's how it works:

#### Step 1: Collect Experience
```
For each environment (4096 of them):
    1. Observe state s
    2. Policy outputs action a ~ π(a|s)
    3. Execute action, get reward r, new state s'
    4. Store (s, a, r, s') in buffer
```

#### Step 2: Compute Advantages
The **advantage** A(s,a) answers: "Was this action better or worse than average?"

```
A(s,a) = Q(s,a) - V(s)
       = (actual return) - (expected return)
```

- A > 0: Action was better than expected → increase probability
- A < 0: Action was worse than expected → decrease probability

#### Step 3: Update Policy (The Surrogate Loss)
PPO uses a "clipped" objective to prevent too-large updates:

```
L_surrogate = E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)]

where:
    ratio = π_new(a|s) / π_old(a|s)  # How much did we change?
    ε = 0.2 (clip range)
```

**Intuition**: We want to increase probability of good actions, but not too much at once (stability).

### 10.3 Interpreting Each Metric

#### `Loss/surrogate` (The Policy Loss)

**What it is**: Measures how much we're improving the policy.

**Values observed**:
| Iter | Surrogate Loss |
|------|----------------|
| 0 | -0.0048 |
| 100 | -0.0056 |
| 500 | -0.0073 |

**Interpretation**:
- **Negative is good!** We maximize this (minimize negative = maximize positive)
- Typical range: -0.01 to -0.001
- If it goes to 0: Policy stopped improving (converged or stuck)
- If very negative (< -0.1): Updates might be too aggressive

**The math**:
```
L_clip = E_t[min(r_t(θ) × A_t, clip(r_t(θ), 1-ε, 1+ε) × A_t)]

where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

#### `Loss/value_function` (The Critic Loss)

**What it is**: How wrong is our value prediction?

**Values observed**:
| Iter | Value Loss |
|------|------------|
| 0 | 0.0130 |
| 100 | 0.0009 |
| 500 | 0.0132 |

**Interpretation**:
- Lower is better (we want accurate predictions)
- Typical range: 0.001 to 0.1
- If very high: Rewards are unpredictable or environment is chaotic
- If zero: Suspicious - might indicate a bug

**The math**:
```
L_value = E[(V_θ(s) - V_target)²]

where V_target = returns (actual discounted rewards received)
```

**Physical meaning**: The critic is learning to predict "if the robot is in this pose, how much reward will it get in the future?" High loss means predictions are inaccurate.

#### `Loss/entropy`

**What it is**: How "random" is the policy?

**Values observed**:
| Iter | Entropy |
|------|---------|
| 0 | 16.98 |
| 100 | 6.18 |
| 500 | 12.03 |
| 28143 | 21.74 |

**Interpretation**:
- **High entropy**: Policy is uncertain, outputs are random (exploring)
- **Low entropy**: Policy is confident, outputs are deterministic (exploiting)
- Healthy training: Entropy decreases over time as policy becomes confident

**The math**:
```
H(π) = -E[log π(a|s)]

For Gaussian policy: H = 0.5 × log(2πe × σ²) × action_dim
```

**Physical meaning**: When entropy is high, the robot tries many different joint movements. When low, it has "decided" on a specific motion pattern.

**Your data shows**: Entropy dropped from 17 to 6 (learning), then rose back to 22. This might indicate the policy became uncertain again - possibly because the task (standing still vs walking) keeps changing.

#### `Policy/mean_noise_std`

**What it is**: The standard deviation of action noise (exploration).

**Values observed**:
| Iter | Noise Std |
|------|-----------|
| 0 | 0.994 |
| 100 | 0.410 |
| 500 | 0.664 |
| 28143 | 8.857 |

**Interpretation**:
- PPO uses a Gaussian policy: a = μ(s) + σ × ε, where ε ~ N(0,1)
- High σ: Actions are noisy (more exploration)
- Low σ: Actions are precise (more exploitation)
- σ typically decreases during training

**Physical meaning**: σ = 0.5 means the robot's joint commands vary by ±0.5 from the "intended" action. This is how it explores different movements.

**Your data shows**: Noise started at 1.0, dropped to 0.4 (good), but then exploded to 8.9! This is abnormal - the policy lost confidence and started exploring wildly.

#### `Loss/learning_rate`

**What it is**: How big are the update steps?

**Values observed**:
| Iter | Learning Rate |
|------|---------------|
| 0 | 0.000296 |
| 10 | 0.010000 |
| 100 | 0.000585 |
| 28143 | 0.000010 |

**Interpretation**:
- RSL-RL uses adaptive learning rate based on KL divergence
- If policy changes too much: LR decreases (be more careful)
- If policy changes too little: LR increases (be more aggressive)
- Range: typically 1e-5 to 1e-2

**The math**:
```
if KL > 2 × KL_target:
    lr = lr / 1.5  # Policy changed too much, slow down
if KL < 0.5 × KL_target:
    lr = lr × 1.5  # Policy barely changed, speed up
```

### 10.4 What Your Training Run Tells Us

Looking at iteration 28143 (end of training):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| mean_reward | 32.5 | Good! Robot is getting positive rewards |
| episode_length | 965 | Good! Nearly full episodes (1000 max) |
| noise_std | 8.86 | **BAD!** Policy lost confidence |
| entropy | 21.7 | **BAD!** Very uncertain |
| learning_rate | 1e-5 | Very low - stopped learning |

**Diagnosis**: The robot learned to walk (high reward, long episodes), but the policy became increasingly uncertain over time. The high noise/entropy at the end suggests:
1. The velocity commands keep changing (resampled every 10s)
2. 10% of environments have zero command (standing still)
3. The policy is confused about "should I walk or stand?"

This is actually normal for velocity-tracking - the policy needs to handle many different commands.

### 10.5 The Training Loop Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│                     PPO TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │ Observe  │───►│  Policy  │───►│ Execute  │                   │
│  │  State   │    │ π(a|s)   │    │  Action  │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│       ▲                               │                          │
│       │                               ▼                          │
│       │                         ┌──────────┐                    │
│       │                         │  Get     │                    │
│       │                         │ Reward r │                    │
│       │                         └──────────┘                    │
│       │                               │                          │
│       └───────────────────────────────┘                          │
│                                                                  │
│  After N steps:                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Compute advantages A = Q(s,a) - V(s)                  │   │
│  │ 2. Update policy: maximize L_surrogate                   │   │
│  │ 3. Update critic: minimize L_value                       │   │
│  │ 4. Adjust learning rate based on KL divergence           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.6 Key Equations Summary

| Concept | Equation | Meaning |
|---------|----------|---------|
| Return | G_t = Σ γ^k × r_{t+k} | Total discounted future reward |
| Value | V(s) = E[G_t \| s_t = s] | Expected return from state s |
| Advantage | A(s,a) = Q(s,a) - V(s) | How much better than average |
| Policy gradient | ∇J = E[∇log π(a\|s) × A] | Direction to improve policy |
| PPO objective | L = E[min(ratio×A, clip(ratio)×A)] | Stable policy update |
| Entropy | H = -E[log π(a\|s)] | Policy randomness |

### 10.7 What to Look for When Debugging

**Healthy training**:
- mean_reward: Increasing, then plateaus
- episode_length: Increasing toward max
- noise_std: Decreasing from ~1.0 to ~0.3
- entropy: Gradually decreasing
- surrogate_loss: Negative, stable around -0.005
- value_loss: Decreasing, then small and stable

**Signs of problems**:
- noise_std exploding (>2.0): Policy lost confidence
- entropy collapsing to 0: Policy became deterministic too fast
- surrogate_loss = 0: Policy stopped updating
- value_loss very high: Environment is unpredictable
- episode_length collapsing: Robot keeps failing

---

## 11. Successful Training Run (2026-01-31_21-45-26)

After fixing the reward threshold calculation to match `curriculums.py` pattern, the temporal stage curriculum is now **working correctly**.

### 11.1 Training Progress

| Iteration | Mean Reward | Episode Length | Status |
|-----------|-------------|----------------|--------|
| 0 | -0.06 | 11.4 | Starting |
| 24 | 11.81 | 552.6 | Learning |
| 49 | 25.47 | 983.5 | Good |
| 74 | 29.51 | 989.6 | Excellent |
| 99 | 31.81 | 988.5 | Converged |

**Key Success Indicators:**
- Episode length reached ~988/1000 steps (19.8 seconds out of 20)
- Reward increased from negative to 31.8
- No terminations due to falling (`bad_orientation = 0`)

### 11.2 Curriculum Stage Distribution

| Iteration | Walking | Recovery | Post-Recovery |
|-----------|---------|----------|---------------|
| 0 | 100% | 0% | 0% |
| 24 | 61% | 38% | 1% |
| 49 | 69% | 15% | 16% |
| 74 | 66% | 17% | 18% |
| 99 | 72% | 14% | 14% |

**Analysis:**
- The curriculum **now triggers correctly** (recovery_frac > 0)
- ~14% of envs are in RECOVERY stage (receiving pushes)
- ~14% are in POST_RECOVERY (recovering from pushes)
- ~72% are in WALKING (either haven't triggered or cycled back)

This distribution makes sense:
- Walking duration: 2.0s → 10% of episode
- Recovery duration: 1.0s → 5% of episode
- Post-recovery: 1.0s → 5% of episode
- But many envs cycle through multiple times per episode

### 11.3 Reward Breakdown

| Reward Term | First | Last | Interpretation |
|-------------|-------|------|----------------|
| track_lin_vel_xy | 0.018 | 1.373 | Good velocity tracking |
| track_ang_vel_z | 0.004 | 0.499 | Good angular tracking |
| energy | -0.0001 | -0.0017 | Low energy penalty |
| alive | 0.002 | 0.150 | Staying alive consistently |

### 11.4 Termination Analysis

| Termination Type | First | Last | Interpretation |
|------------------|-------|------|----------------|
| time_out | 1.6 | 1.7 | Robots reaching full episodes |
| bad_orientation | 0.0 | 0.0 | No falls - stable |
| base_contact | 0.0 | 0.0 | No body contact with ground |

**This is excellent!** The robot is not falling over, meaning it's successfully handling the pushes during recovery stage.

### 11.5 Loss Metrics

| Metric | First | Last | Interpretation |
|--------|-------|------|----------------|
| surrogate | -0.0043 | -0.0104 | Policy improving |
| value_function | 0.0134 | 0.0023 | Value predictions improving |
| entropy | 17.01 | 10.34 | Policy becoming more confident |

### 11.6 What Made This Work

The key fix was using the **correct threshold calculation** matching `curriculums.py`:

```python
# Pattern from curriculums.py:
reward_rate = episode_sum / max_episode_length_s
if reward_rate > weight * threshold:
    transition_to_recovery()
```

This gives a consistent scale where:
- `reward_rate` accumulates over time
- `weight * threshold` provides a comparable threshold
- Curriculum triggers when robot demonstrates good tracking

### 11.7 Next Steps

1. **Add staged angular velocity reward** - Currently only `rlin` is frozen during recovery, but paper says both `rlin` and `rang` should be frozen
2. **Add push visualization** - Show red arrows indicating push direction during recovery
3. **Tune durations** - Experiment with different walking/recovery/post-recovery durations
4. **Increase push strength** - Gradually increase from ±1.0 m/s to test robustness

---

## Appendix: Files Analyzed

| File | Purpose |
|------|---------|
| `force_env_cfg.py` | Environment configuration (now working) |
| `velocity_env_cfg.py` | Baseline environment (reference) |
| `temporal_stage_curriculum.py` | Push curriculum implementation |
| `rewards.py` | Custom reward functions including staged versions |
| `events.py` | Custom event functions (push with return) |
| `events.out.tfevents.*` | TensorBoard training logs |
| `env.yaml` | Saved environment parameters |