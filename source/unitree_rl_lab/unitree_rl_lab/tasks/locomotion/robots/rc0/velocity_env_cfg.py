import math
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import HUMANOIDV0_CFG as ROBOT_CFG #this is the maxon motor cfg, import other cfgs as needed
from unitree_rl_lab.tasks.locomotion import mdp



@configclass
class SimpleSceneCfg(InteractiveSceneCfg):
    """Minimal scene: flat ground + robot + contact sensor."""


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", 
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )



@configclass
class SimpleEventCfg:
    """Minimal events: Add randomization later."""

    # Reset robot pose
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset joints to default
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0), 
            "velocity_range": (0.0, 0.0), 
        },
    )

@configclass
class SimpleCommandsCfg:
    """Fixed velocity range - no curriculum initially."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,      
        rel_heading_envs=0.0,        
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.5, 0.5),
        ),
    )



# ACTIONS - Only control legs and waist
@configclass
class SimpleActionsCfg:
    """Control only what's needed for walking."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_leg_flexor", "left_leg_abductor", "left_leg_rotor",
            "left_knee", "left_ankle_roll", "left_ankle_pitch",
            "right_leg_flexor", "right_leg_abductor", "right_leg_rotor",
            "right_knee", "right_ankle_roll", "right_ankle_pitch",
            # Waist (2 DOF) 
            "hip_rotor", "hip_abductor0",
        ],
        scale=0.25,  # Actions scaled by 0.25 rad
        use_default_offset=True,  
    )



@configclass
class SimpleObservationsCfg:
    """What the policy needs to know."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive observations only - no vision/heightmap."""


        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))


        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))


        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()



@configclass
class SimpleRewardsCfg:
    """
    POSITIVE rewards for TASK goals:
      - Track commanded velocity
      - Stay alive
    
    NEGATIVE rewards for REGULARIZATION (small weights):
      - Smooth actions
      - Energy efficiency
      - Joint limits
    
    NEGATIVE rewards for SAFETY (larger weights):
      - Bad contacts
      - Falling
    
    """

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.5)

# regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    energy = RewTerm(func=mdp.energy, weight=-1e-4)

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)

    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link", "femur_left_1", "femur_right_1"],
            ),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)



@configclass
class SimpleTerminationsCfg:
    """Simple termination: timeout or fall."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.7},  # Generous threshold
    )

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0}, 
    )

@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """
    Simple humanoid locomotion environment.

    """

    # Scene
    scene: SimpleSceneCfg = SimpleSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: SimpleObservationsCfg = SimpleObservationsCfg()
    actions: SimpleActionsCfg = SimpleActionsCfg()
    commands: SimpleCommandsCfg = SimpleCommandsCfg()
    rewards: SimpleRewardsCfg = SimpleRewardsCfg()
    terminations: SimpleTerminationsCfg = SimpleTerminationsCfg()
    events: SimpleEventCfg = SimpleEventCfg()

    curriculum = None

    def __post_init__(self):
        """Simulation settings."""
        self.decimation = 4      
        self.episode_length_s = 20.0 
        
        self.sim.dt = 0.005   
        self.sim.render_interval = self.decimation
        
        self.scene.contact_forces.update_period = self.sim.dt



@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32

