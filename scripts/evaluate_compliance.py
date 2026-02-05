"""
Compliance Evaluation Script.

Measures how far a standing robot displaces when pushed.
A compliant robot will "go with" the push (larger displacement), while a stiff
robot will fight the push (smaller displacement).

Based on Hartmann et al. (2024) "Deep Compliant Control" evaluation methodology.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate compliance of a trained policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--settle_time", type=float, default=2.0, help="Time (s) to let robot settle before push")
parser.add_argument("--observe_time", type=float, default=5.0, help="Time (s) to observe after push")
parser.add_argument("--push_magnitude", type=float, default=1.0, help="Push velocity magnitude (m/s) or force magnitude (N) if --use_force")
parser.add_argument("--push_direction", type=str, default="y", choices=["x", "y", "-x", "-y"], help="Push direction")
parser.add_argument("--use_force", action="store_true", help="Use external force instead of velocity impulse")
parser.add_argument("--force_duration", type=float, default=0.1, help="Duration to apply force (s), only used with --use_force")
parser.add_argument("--loop", action="store_true", help="Loop continuously until Ctrl+C")
parser.add_argument("--output_dir", type=str, default="logs/compliance_eval", help="Output directory for logs")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now we can import Isaac-specific modules
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401 - registers environments

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz


def apply_velocity_push(env, env_ids: torch.Tensor, velocity: torch.Tensor):
    """Apply a velocity push to the robot (instantaneous impulse)."""
    asset = env.unwrapped.scene["robot"]
    vel_w = asset.data.root_vel_w[env_ids].clone()
    vel_w[:, :3] += velocity
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_force(env, env_ids: torch.Tensor, force: torch.Tensor, body_name: str = "base"):
    """Apply an external force to the robot's body."""
    asset = env.unwrapped.scene["robot"]
    num_envs = len(env_ids)

    # Get body index
    body_ids = asset.find_bodies(body_name)[0]
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1

    # Create force tensor: (num_envs, num_bodies, 3)
    forces = torch.zeros(num_envs, num_bodies, 3, device=asset.device)
    forces[:, 0, :] = force  # Apply to first body (base)

    # Zero torques
    torques = torch.zeros_like(forces)

    # Set forces
    asset.set_external_force_and_torque(forces, torques, body_ids=body_ids, env_ids=env_ids)


def clear_force(env, env_ids: torch.Tensor, body_name: str = "base"):
    """Clear external forces on the robot."""
    asset = env.unwrapped.scene["robot"]
    num_envs = len(env_ids)

    body_ids = asset.find_bodies(body_name)[0]
    num_bodies = len(body_ids) if isinstance(body_ids, list) else 1

    forces = torch.zeros(num_envs, num_bodies, 3, device=asset.device)
    torques = torch.zeros_like(forces)

    asset.set_external_force_and_torque(forces, torques, body_ids=body_ids, env_ids=env_ids)


def create_force_visualizer():
    """Create red arrow visualization marker for force."""
    marker_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Compliance/force_arrow")
    marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    return VisualizationMarkers(marker_cfg)


def update_force_visualization(visualizer, robot, push_dir: torch.Tensor, magnitude: float, device):
    """Update force arrow position and orientation."""
    num_envs = robot.data.root_pos_w.shape[0]

    # Position: above robot base
    positions = robot.data.root_pos_w.clone()
    positions[:, 2] += 0.5  # Offset above robot

    # Scale arrow by force magnitude (normalized for visibility)
    scale_factor = min(magnitude / 100.0, 2.0)  # Cap at 2x for very large forces
    scales = torch.tensor([[0.3 + scale_factor, 0.3, 0.3]], device=device).repeat(num_envs, 1)

    # Orientation: point in force direction
    # Compute yaw angle from push direction
    yaw = torch.atan2(push_dir[1], push_dir[0])
    zeros = torch.zeros(num_envs, device=device)
    yaws = torch.full((num_envs,), yaw.item(), device=device)
    orientations = quat_from_euler_xyz(zeros, zeros, yaws)

    visualizer.visualize(positions, orientations, scales)


def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args_cli.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Parse environment config
    env_cfg = parse_env_cfg(
        "Unitree-Go2-Force",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Set zero velocity commands (robot should stand still)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0  # All envs standing
    env_cfg.commands.base_velocity.resampling_time_range = (100.0, 100.0)

    # Disable any curriculum pushes
    if hasattr(env_cfg.events, 'periodic_push'):
        env_cfg.events.periodic_push = None

    # Create environment
    env = gym.make("Unitree-Go2-Force", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry("Unitree-Go2-Force", "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    # Get device
    device = env.device

    # Push direction vector
    push_dir = torch.zeros(3, device=device)
    if args_cli.push_direction == "x":
        push_dir[0] = 1.0
    elif args_cli.push_direction == "-x":
        push_dir[0] = -1.0
    elif args_cli.push_direction == "y":
        push_dir[1] = 1.0
    elif args_cli.push_direction == "-y":
        push_dir[1] = -1.0

    # Push vector (velocity or force depending on mode)
    push_vector = push_dir * args_cli.push_magnitude

    # Simulation parameters
    dt = env.unwrapped.step_dt
    settle_steps = int(args_cli.settle_time / dt)
    observe_steps = int(args_cli.observe_time / dt)
    force_steps = int(args_cli.force_duration / dt) if args_cli.use_force else 0

    # Robot reference
    robot = env.unwrapped.scene["robot"]

    print(f"\n{'='*60}")
    print("Compliance Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Settle time: {args_cli.settle_time} s")
    print(f"Observe time: {args_cli.observe_time} s")
    if args_cli.use_force:
        print(f"Push mode: FORCE")
        print(f"Force: {args_cli.push_magnitude} N in {args_cli.push_direction} direction")
        print(f"Force duration: {args_cli.force_duration} s")
    else:
        print(f"Push mode: VELOCITY")
        print(f"Velocity: {args_cli.push_magnitude} m/s in {args_cli.push_direction} direction")
    if args_cli.loop:
        print(f"Loop mode: ON (Ctrl+C to stop)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Data storage (only from first cycle)
    time_data = []
    displacement_data = []
    first_cycle = True
    cycle_count = 0
    env_ids = torch.arange(args_cli.num_envs, device=device)
    force_visualizer = None

    try:
        while True:
            cycle_count += 1
            print(f"\n--- Cycle {cycle_count} ---")

            # Reset and settle
            obs, _ = env.reset()
            print("Letting robot settle...")
            for step in range(settle_steps):
                with torch.no_grad():
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                if dones.any():
                    print("Robot fell during settling!")
                    break

            # Record pre-push position
            pre_push_pos = robot.data.root_pos_w.clone()
            print(f"Pre-push position: x={pre_push_pos[0, 0].item():.3f}, y={pre_push_pos[0, 1].item():.3f}")

            # Apply push
            if args_cli.use_force:
                print(f"Applying force: {push_vector.cpu().numpy()} N")
                apply_force(env, env_ids, push_vector)
                if force_visualizer is None:
                    force_visualizer = create_force_visualizer()
                force_visualizer.set_visibility(True)
                update_force_visualization(force_visualizer, robot, push_dir, args_cli.push_magnitude, device)
            else:
                print(f"Applying velocity impulse: {push_vector.cpu().numpy()} m/s")
                apply_velocity_push(env, env_ids, push_vector)

            # Observe
            force_active = args_cli.use_force
            print("Observing response...")
            for step in range(observe_steps):
                current_time = step * dt

                # Update force visualization while active
                if args_cli.use_force and force_active and force_visualizer is not None:
                    update_force_visualization(force_visualizer, robot, push_dir, args_cli.push_magnitude, device)

                # Clear force after duration
                if args_cli.use_force and force_active and step >= force_steps:
                    print(f"  t={current_time:.2f}s: Force cleared")
                    clear_force(env, env_ids)
                    force_active = False
                    if force_visualizer is not None:
                        force_visualizer.set_visibility(False)

                with torch.no_grad():
                    actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                # Compute displacement in push direction
                current_pos = robot.data.root_pos_w
                displacement_vec = current_pos[:, :3] - pre_push_pos[:, :3]
                displacement_in_push_dir = torch.sum(displacement_vec * push_dir, dim=1)

                # Log data only on first cycle
                if first_cycle:
                    time_data.append(current_time)
                    displacement_data.append(displacement_in_push_dir.cpu().numpy())

                if step % 50 == 0:
                    print(f"  t={current_time:.2f}s: displacement={displacement_in_push_dir.mean().item():.3f} m")

                if dones.any():
                    print(f"Robot fell at t={current_time:.2f}s")
                    break

            first_cycle = False

            # Exit if not looping
            if not args_cli.loop:
                break

    except KeyboardInterrupt:
        print(f"\n\nStopped after {cycle_count} cycles.")

    # Convert to arrays
    time_data = np.array(time_data)
    displacement_data = np.array(displacement_data)  # (steps, num_envs)

    # Compute statistics
    disp_mean = displacement_data.mean(axis=1)
    disp_std = displacement_data.std(axis=1) if args_cli.num_envs > 1 else np.zeros_like(disp_mean)

    max_idx = np.argmax(np.abs(disp_mean))
    max_disp = disp_mean[max_idx]
    max_disp_std = disp_std[max_idx]
    max_time = time_data[max_idx]

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Max displacement: {max_disp:.3f} +/- {max_disp_std:.3f} m")
    print(f"Time to max: {max_time:.2f} s")
    print(f"Final displacement: {disp_mean[-1]:.3f} m")
    print(f"{'='*60}\n")

    # ==================== PLOTTING ====================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot displacement over time
    ax.plot(time_data, disp_mean, color='#2c3e50', linewidth=1.5)

    # Error band if multiple envs
    if args_cli.num_envs > 1:
        ax.fill_between(time_data, disp_mean - disp_std, disp_mean + disp_std,
                       alpha=0.3, color='#2c3e50')

    # Mark max displacement
    ax.scatter(max_time, max_disp, s=80, c='#e74c3c', marker='o',
               edgecolors='white', linewidths=1, zorder=3)
    ax.annotate(f'$d_{{max}}$ = {abs(max_disp):.2f} m',
                xy=(max_time, max_disp),
                xytext=(max_time + 0.3, max_disp + 0.02 * np.sign(max_disp)),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))

    # Zero line
    ax.axhline(y=0, color='#95a5a6', linestyle='--', linewidth=1)

    push_type = "force" if args_cli.use_force else "velocity"
    ax.set_xlabel('Time after push (s)')
    ax.set_ylabel(f'Displacement in {args_cli.push_direction} (m)')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    plot_file = os.path.join(output_dir, "displacement.pdf")
    plt.savefig(plot_file, bbox_inches='tight')
    plot_file_png = os.path.join(output_dir, "displacement.png")
    plt.savefig(plot_file_png, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")

    # Save data
    np.savez(os.path.join(output_dir, "data.npz"),
             time=time_data, displacement=displacement_data,
             push_direction=args_cli.push_direction,
             push_magnitude=args_cli.push_magnitude,
             use_force=args_cli.use_force,
             force_duration=args_cli.force_duration if args_cli.use_force else 0.0)

    plt.show()

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()