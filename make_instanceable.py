from isaaclab.app import AppLauncher
AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverterCfg, UrdfConverter

converter_cfg = UrdfConverterCfg(
    asset_path="/home/ubuntu/ethr_rc_robot_assets/humanoid-description/humanoid_final_copy_description/urdf/humanoid_final_copy.urdf",
    usd_dir="/home/ubuntu/ethr_rc_robot_assets/humanoidv0/",
    usd_file_name="humanoidv0_instanceable.usd",
    make_instanceable=True,
    fix_base=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=10.0,
        ),
    ),
)
converter = UrdfConverter(converter_cfg)
print(f"Saved to: {converter.usd_path}")
