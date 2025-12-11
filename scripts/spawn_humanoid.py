"""Spawn robot and monitor base velocity to debug physics."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug robot physics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# Import your robot config
from unitree_rl_lab.assets.robots.unitree import HUMANOIDV2_CFG as ROBOT_CFG


@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    """Minimal scene with just ground and robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Run simulation and print debug info."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0
            # Reset robot state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("\n[INFO]: Reset robot state")

        # Apply default joint positions (just hold pose)
        robot.set_joint_position_target(robot.data.default_joint_pos)

        # Write and step
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        # Print debug info every 50 steps
        if count % 50 == 0:
            root_pos = robot.data.root_pos_w[0]
            root_lin_vel = robot.data.root_lin_vel_w[0]
            root_ang_vel = robot.data.root_ang_vel_w[0]
            joint_vel = robot.data.joint_vel[0]

            print(f"\n--- Step {count} ---")
            print(f"Root pos:     {root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}")
            print(f"Root lin vel: {root_lin_vel[0]:.3f}, {root_lin_vel[1]:.3f}, {root_lin_vel[2]:.3f}")
            print(f"Root ang vel: {root_ang_vel[0]:.3f}, {root_ang_vel[1]:.3f}, {root_ang_vel[2]:.3f}")
            print(f"Joint vel max: {joint_vel.abs().max():.3f}")
            
            # Check for explosion
            if root_lin_vel.abs().max() > 100 or joint_vel.abs().max() > 100:
                print("*** EXPLOSION DETECTED ***")
            if torch.isnan(root_pos).any() or torch.isnan(root_lin_vel).any():
                print("*** NaN DETECTED ***")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 0.0, 2.0], [0.0, 0.0, 1.0])

    scene_cfg = DebugSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete. Watching robot...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()