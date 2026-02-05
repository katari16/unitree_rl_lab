# Thesis Project Context: Force-Following Locomotion for Quadruped Robots

## Project Overview

This thesis will build upon the "Deep Compliant Control" approach by Hartmann et al. (ICRA 2024) to develop a force-following locomotion controller for quadruped robots. The goal is to create a controller that enables quadrupeds to exhibit compliant behaviors in response to external forces, particularly for applications like human-robot interaction (e.g., leash-guided navigation, collaborative transport).

**Primary Inspiration:** Hartmann et al., "Deep Compliant Control for Legged Robots" (ICRA 2024)

**Robot Platform:** Quadruped robots (initially Unitree Go2)

**Development Framework:** Isaac Lab (successor to Isaac Gym), ROS2

---

## 1. Hartmann et al. (2024) - Deep Compliant Control - DETAILED METHODOLOGY

### 1.1 Core Problem & Insight

**Problem:** Deep RL policies for legged locomotion typically generate stiff, high-frequency motions when responding to unexpected disturbances. This is unnatural, energy-inefficient, and potentially unsafe.

**Key Insight:** Stiff responses occur because agents are incentivized to maximize task rewards at ALL times, even during perturbations. The policy tries to perfectly track commanded velocities even when being pushed, leading to aggressive corrective actions.

**Solution:** Introduce an explicit **recovery stage** in training where tracking rewards are temporarily relaxed, allowing the robot to recover naturally from disturbances before resuming task execution.

### 1.2 Implementation Details

#### 1.2.1 Observation Space (83-dimensional)

The observation vector encodes both current and previous states to enable disturbance detection without force sensors (POMDP formulation):

**Current State (ô_t):**
- Base height: y [m]
- Local gravity vector: g_x, g_y, g_z [m/s²]
- Joint angles: q_0...11 [rad] (12 joints)
- Local linear velocity: v_x, v_y, v_z [m/s]
- Local angular velocity: ω_x, ω_y, ω_z [rad/s]
- Joint rates: q̇_0...11 [rad/s]

**Previous State:**
- Previous observation: ô_{t-1}
- Previous action: a_{t-1} (PD controller targets)

**Task Commands:**
- Desired forward velocity: v*_z [m/s]
- Desired sideways velocity: v*_x [m/s]
- Desired yaw rate: ω*_y [rad/s]

**Full observation:** o_t = {ô_{t-1}, a_{t-1}, ô_t, v*_x, v*_z, ω*_y} ∈ ℝ^83

**Rationale for Historical States:**
Inspired by momentum-based external force observers (Morlando et al.), the policy learns to detect external disturbances by comparing current and previous joint states and PD controller inputs. Without ground reaction force sensors, this temporal information is crucial for identifying when external forces are applied.

#### 1.2.2 Action Space

**Actions:** 12-dimensional joint target angles for PD controller
**PD Gains:** K_p = 40, K_d = 1 (moderate gains to allow compliance)
**Control Frequency:** 60 Hz policy updates, 240 Hz simulation

#### 1.2.3 Reward Structure

**Positive Rewards (r_pos):**
```
r_pos = max(0, w_lin·r_lin + w_ang·r_ang + w_h·r_h)

r_lin = exp(-8·[(v_x - v*_x)² + (v_z - v*_z)²])        [w_lin = 0.8]
r_ang = exp(-8·(ω_z - ω*_z)²)                           [w_ang = 0.5]
r_h = Σ(p^peak_{fi,y} / p^des_{fi,y} - 1)²              [w_h = -0.7]
```

**Negative Penalties (r_neg):**
```
r_neg = w_e·r_e + w_τ·r_τ + w_pose·r_pose + w_cl·r_cl

r_e = |τ·q̇|                                             [w_e = -0.015]  (mechanical power)
r_τ = ||τ||²_2                                          [w_τ = -0.0015] (actuator losses)
r_pose = φ² + ψ² + 10·(y - y_des)²                      [w_pose = -2.0] (pitch, roll, height)
r_cl = Σ||v_{i,xz}||²_2·(p_{fi,y} - p^des_{fi,y})²     [w_cl = -0.1]   (foot clearance)
```

**Total Reward:**
```
r = r_pos · exp(0.45 · r_neg)
```

**Key Design Choices:**
- Energy efficiency terms (r_e, r_τ) encourage natural, compliant movements
- Exponential structure ensures positive rewards, preventing early termination
- Foot height reward (r_h) encourages larger strides for better terrain adaptation
- Dense foot clearance reward (r_cl) during swing phase

#### 1.2.4 Multi-Stage Episodic Training (THE KEY INNOVATION)

Each 4-second episode is divided into THREE stages:

**STAGE 1: Walking Stage (2.0s)**
- Robot walks undisturbed
- Full reward function applied
- Desired velocity randomly sampled:
  - Forward: [-1.0, 1.0] m/s
  - Sideways: [-0.5, 0.5] m/s
  - Yaw rate: [-0.5, 0.5] rad/s

**STAGE 2: Recovery Stage (1.0s) - THE COMPLIANCE MECHANISM**

**Perturbation Application:**
- Impulse applied to robot base as velocity offset
- Horizontal disturbance: up to 1.0 m/s (forward/sideways, uniformly sampled)
- Rotational disturbance: up to 1.0 rad/s (all three axes)

**Modified Reward Function:**
```python
# Replace tracking rewards with constants
r_lin_recovery = <average r_lin from walking stage>  # ~0.85 of max
r_ang_recovery = <average r_ang from walking stage>

# Energy terms remain active!
r_pos = max(0, w_lin·r_lin_recovery + w_ang·r_ang_recovery + w_h·r_h)
r_neg = w_e·r_e + w_τ·r_τ + w_pose·r_pose + w_cl·r_cl
```

**Critical Insight:** By giving constant tracking rewards regardless of actual velocity tracking performance, the policy is NOT penalized for deviating from commanded velocities during recovery. The energy penalties (r_e, r_τ) become the dominant terms, encouraging smooth, energy-efficient recovery rather than aggressive corrective actions.

**STAGE 3: Post-Recovery Stage (1.0s)**
- Full reward function restored
- Ensures robot returns to normal walking after recovery
- Prevents performance degradation after disturbances

#### 1.2.5 Training Curriculum

**Adaptive Push Curriculum:**
1. **Early Phase:** Only trigger recovery/post-recovery if walking performance is good
   - Threshold: average r_lin > 85% of maximum value
   - Prevents learning overly cautious behaviors due to early failures
   - Avoids local optima from premature collapse penalties

2. **Progressive Velocity Scaling:**
   - Gradually increase max desired velocity from low speeds to 1.0 m/s
   - Ensures smooth difficulty progression

3. **Environment Randomization:**
   - Gravity vector adjusted to simulate slopes up to 10% incline
   - Resampled at episode start
   - Enables terrain adaptation without explicit terrain modeling

4. **Adaptive Learning Rate:**
   - Based on KL-divergence monitoring (Rudin et al.)
   - Prevents training instability

#### 1.2.6 Simulation Details

**Physics Engine:** Open Dynamics Engine (ODE)
**Simulation Frequency:** 240 Hz
**Actuator Latency:** 30 ms (critical for sim-to-real transfer)
**Early Termination Conditions:**
- Base height < 0.2 m (collapse)
- Non-foot contact with ground

**Training Hyperparameters (PPO):**
```
Batch Size: 8192
Mini-batch Size: 512
Epochs: 30
Discount Factor: 0.99
GAE Lambda: 0.95
Clip Range: 0.2
Entropy Coefficient: 0.01
Learning Rate: Adaptive
Initial Std Dev: exp(-1)
Network Architecture: [256, 128, 64] (ELU activation)
```

**Training Resources:**
- GPU: GTX 1070 or better (≥10GB VRAM recommended)
- CPU: Ryzen 7 3700X
- Training Time: ~15 hours for 200M samples
- Multiple policies trained with different random seeds for reproducibility

#### 1.2.7 Hardware Deployment

**Sim-to-Real Transfer:**

1. **State Estimation:**
   - Kalman filter-based estimator (Bledt et al.)
   - Contact estimator (supervised learning, 240 Hz)
   - Inputs: Last 4 timesteps of joint states and actions
   - Outputs: Boolean contact states per foot

2. **Dynamics Randomization:**
   - Friction coefficient: [0.5, 1.2]
   - Joint calibration offset: N(0, 0.01 rad) per joint

3. **Robot Platform:** Unitree Go1
   - PD gains: K_p = 40, K_d = 1
   - 12 actuated joints (hip, thigh, calf per leg)

### 1.3 Key Results & Observations

**Quantitative Improvements:**
- **Push Recovery:** 78% of test cases passed at 80% success threshold (vs 73% for baseline)
- **Energy Efficiency:** 15% less mechanical power during recovery
- **Torque Reduction:** 6% lower motor torques after perturbation
- **Recovery Time:** 85% longer decay time (0.617s vs 0.333s) - SMOOTHER recovery

**Behavioral Characteristics:**
- **Compliant policies:** Gentle reactions, gradual recovery, lower forces on obstacles
- **Baseline policies:** Stiff responses, aggressive corrections, high forces
- **Leash behavior:** Compliant policy follows force direction; baseline resists
- **Obstacle collision:** Compliant deflects smoothly; baseline pushes through

**Terrain Performance:**
- Successfully navigated ball pits (0.5kg spheres, 5cm radius)
- Adapted to uneven terrain (Perlin noise, 5cm magnitude)
- Note: Terrains NOT seen during training - generalization!

---

## 2. Related Work - Brief Summaries

### 2.1 Gu et al. (2025) - Hierarchical Cooperative Locomotion

**Paper:** "Hierarchical Cooperative Locomotion Control of Human and Quadruped Robot Based on Interactive Force Guidance" (IEEE/ASME Transactions on Mechatronics, 2025)

**Key Contribution:** Model-based hierarchical control for human-quadruped cooperation (like walking a dog or camel transport)

**Approach:**
- **Hybrid dynamic model** considering interaction forces
- **Two-level hierarchical control:**
  - High level: Model Predictive Control (MPC) for optimal trajectory
  - Low level: Nonlinear controller for trajectory tracking
- Enables quadruped to follow human guidance through leash/handle

**Technical Details:**
- Explicit modeling of human-robot interaction forces
- MPC-based planning accounts for force disturbances
- Foothold adjustment based on interactive forces
- Focus on stability during cooperative transport

**Relevance to Thesis:**
- Alternative (model-based) approach to force-following
- Could provide comparison: Learning-based (Hartmann) vs Model-based (Gu)
- Explicitly models interaction forces (vs implicit learning)
- Hierarchical structure might inspire multi-level policy design

**Key Difference:** Model-based with explicit force modeling vs learning-based with implicit force detection

---

## 3. Proposed Thesis Direction: "Robo-Barrow" Force-Following Locomotion

### 3.1 Research Questions

1. **Can Hartmann's compliance mechanism enable direct force-following locomotion?**
   - Move in direction of applied forces rather than just recovering from them
   - Applications: Leash guidance, collaborative transport, human-robot interaction

2. **How should the reward structure be modified for force-following vs disturbance recovery?**
   - Should force direction be explicitly observed or implicitly learned?
   - How to balance compliance (following forces) with task objectives?

3. **What are the limits of proprioception-based force sensing?**
   - Force magnitude estimation accuracy
   - Force direction estimation accuracy
   - Comparison with force/torque sensor ground truth

4. **How does force-following scale across different terrains and payloads?**
   - Flat vs uneven terrain
   - Different payload weights
   - Dynamic vs static forces

### 3.2 Proposed Methodology

**Phase 1: Reproduce Hartmann's Approach**
- Implement multi-stage episodic training in Isaac Lab
- Verify compliant behavior matches reported results
- Establish baseline for comparison

**Phase 2: Extend to Force-Following**
- Modify reward structure to encourage movement in force direction
- Experiment with different force input modalities:
  - Implicit: Only proprioception (like Hartmann)
  - Explicit: Add force/torque sensor observations
  - Hybrid: Both proprioception + sensor confirmation

**Phase 3: Application-Specific Training**
- Wheelbarrow scenario: Following leash while carrying payload
- Object transport: Cooperative pushing/pulling
- Terrain adaptation: Force-following on slopes/uneven ground

**Phase 4: Hardware Validation**
- Deploy on quadruped hardware (Unitree Go1 or ANYmal)
- Real-world force-following tests
- Comparison with model-based approaches (MPC baseline)

### 3.3 Technical Implementation Plan

**Simulation Framework:**
- Isaac Lab (GPU-parallelized RL training)
- Robot models: Unitree Go1, ANYmal C (if available)
- Physics: realistic actuator dynamics, latency modeling

**Training Infrastructure:**
- PPO algorithm (proven for locomotion)
- Multi-stage curriculum learning
- Domain randomization for sim-to-real transfer

**ROS2 Integration:**
- Policy deployment as ROS2 node
- State estimation pipeline
- Visualization and debugging tools

**Evaluation Metrics:**
- Force-following accuracy (direction and magnitude)
- Energy efficiency
- Terrain adaptability
- Robustness to varying payloads

---

## 4. Key Technical Challenges

### 4.1 Reward Engineering

**Challenge:** Balancing multiple objectives
- Track commanded velocities
- Follow applied forces
- Maintain stability
- Minimize energy consumption

**Approach:** Stage-based reward modification (inspired by Hartmann)
- Normal walking: Velocity tracking rewards
- Force application: Directional alignment rewards + energy minimization
- Recovery: Smooth transition back to normal walking

### 4.2 Force Estimation Without Sensors

**Challenge:** Accurately estimate force magnitude and direction from proprioception

**Possible Solutions:**
1. **Temporal patterns:** Include multiple timesteps (current approach)
2. **Recurrent networks:** LSTM/GRU for better temporal modeling
3. **Privileged learning:** Train with force sensors in sim, deploy without

### 4.3 Generalization Across Scenarios

**Challenge:** Policy should work for:
- Different force magnitudes
- Various force application points
- Multiple terrains
- Varying payloads

**Approach:** 
- Extensive domain randomization
- Curriculum from simple to complex scenarios
- Multi-task learning framework

### 4.4 Safety & Stability

**Challenge:** Don't compromise stability while being compliant

**Approach:**
- Conservative early termination conditions
- Gradual compliance increase during training
- Stability penalties in reward function

---

## 5. Expected Contributions

### 5.1 Scientific Contributions

1. **Novel application of compliance training for force-following**
   - Extension of Hartmann's recovery-focused approach
   - Direct force-following as intentional behavior

2. **Comparison of learning-based vs model-based force control**
   - Benchmark against MPC/hierarchical approaches
   - Analysis of trade-offs

3. **Understanding of proprioception-based force sensing limits**
   - Quantification of estimation accuracy
   - Guidelines for sensor-free vs sensor-based approaches

### 5.2 Practical Contributions

1. **Open-source implementation in Isaac Lab**
   - Reproducible baseline
   - Community resource for compliant locomotion research

2. **ROS2-deployable controller**
   - Ready for real robot deployment
   - Integration with existing quadruped platforms

3. **Application demonstrations**
   - Leash-guided navigation
   - Cooperative transport
   - Human-robot collaboration scenarios

---

## 6. References & Resources

### Primary Papers
1. Hartmann et al., "Deep Compliant Control for Legged Robots," ICRA 2024
   - PDF: https://crl.ethz.ch/papers/hartmann2024deep.pdf
   - Lab: Computational Robotics Lab, ETH Zurich

2. Portela et al., "Learning Force Control for Legged Manipulation," ICRA 2024
   - arXiv: 2405.01402
   - Code: https://github.com/Improbable-AI/learning-compliance

3. Gu et al., "Hierarchical Cooperative Locomotion Control," IEEE/ASME ToM 2025
   - DOI: 10.1109/TMECH.2025.3535721

### Useful Prior Work
- Hwangbo et al., "Learning Agile and Dynamic Motor Skills," Science Robotics 2019
- Rudin et al., "Learning to Walk in Minutes," CoRL 2021
- Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain," Science Robotics 2020
- Margolis & Agrawal, "Walk These Ways," CoRL 2022

### Tools & Frameworks
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Isaac Gym (predecessor): https://developer.nvidia.com/isaac-gym
- RSL RL: https://github.com/leggedrobotics/rsl_rl
- ROS2: https://docs.ros.org/

---

## 7. Timeline Suggestion (6-month thesis)

**Month 1-2: Foundation**
- Literature review completion
- Isaac Lab setup and familiarization
- Reproduce Hartmann baseline

**Month 3-4: Core Development**
- Implement force-following modifications
- Iterative reward tuning
- Simulation experiments

**Month 5: Validation**
- Hardware deployment
- Real-world experiments
- Data collection and analysis

**Month 6: Writing & Defense**
- Thesis writing
- Final experiments
- Defense preparation

---

## 8. Open Questions for Discussion

1. **Force input modality:** Fully implicit (proprioception) vs explicit force sensing vs hybrid?

2. **Baseline comparisons:** Which controllers to compare against?
   - Hartmann's compliant policy
   - Standard RL without compliance
   - Model-based MPC (like Gu et al.)
   - Classical impedance control

3. **Application focus:** Which scenario to prioritize?
   - Leash-guided navigation (dog-walking)
   - Wheelbarrow transport
   - Cooperative object pushing
   - Other?

4. **Robot platform:** Unitree Go1 (available at RSL) vs ANYmal (more capable)?

5. **Success metrics:** What defines "good" force-following?
   - Directional accuracy threshold?
   - Acceptable latency?
   - Energy efficiency targets?

---

## 9. Key Takeaways from Hartmann Paper

### What Makes It Work

1. **Recovery stage with relaxed tracking rewards** - The key innovation
   - Allows natural compliance without fighting the disturbance
   - Energy terms dominate during recovery

2. **Multi-stage training curriculum**
   - Ensures good walking before introducing disturbances
   - Prevents degenerate solutions

3. **Temporal observations**
   - Current + previous states enable implicit force detection
   - No force sensors needed

4. **Moderate PD gains** (K_p=40, K_d=1)
   - Not too stiff (K_p=80 would be too aggressive)
   - Not too soft (K_p=20 might be unstable)

5. **Energy-based rewards**
   - Encourages natural, efficient movements
   - Aligns with biological compliance

### What Could Be Improved

1. **Explicit force following not demonstrated**
   - Only shows recovery from perturbations
   - Doesn't actively track force directions

2. **Limited to flat/slightly uneven terrain**
   - No stairs, gaps, or highly irregular terrain

3. **Single robot platform**
   - Only tested on Unitree Go2
   - Generalization to other robots unclear

4. **No payload experiments**
   - Doesn't test with carried weights
   - Relevant for transport applications

These gaps are opportunities for the thesis!

---