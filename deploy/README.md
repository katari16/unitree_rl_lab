# Deployment: MuJoCo Simulation + Go2 Controller

## Terminal 1: MuJoCo Simulator

```bash
unset ROS_DISTRO AMENT_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH CYCLONEDDS_URI
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu
export CYCLONEDDS_URI='<CycloneDDS><Domain><Iceoryx><Enable>false</Enable></Iceoryx></Domain></CycloneDDS>'
cd ~/unitree_mujoco/simulate/build && ./unitree_mujoco
```

## Terminal 2: Go2 RL Controller

```bash
unset ROS_DISTRO AMENT_PREFIX_PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH CYCLONEDDS_URI
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:/home/katari/unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib
cd ~/unitree_rl_lab/deploy/robots/go2/build && ./go2_ctrl -n lo
```

## Notes

- Start Terminal 1 first, wait for the MuJoCo window to appear, then start Terminal 2.
- Both must use the loopback interface (`lo`) to communicate via DDS.
- The `unset` commands clear ROS 2 / Isaac Lab environment variables that conflict with CycloneDDS.
- The Iceoryx disable flag is only needed for `unitree_mujoco` (not for `go2_ctrl`).

---

# Deployment: Real Go2 Robot

## Important: Releasing the Default Controller

Before deploying a policy on the real Go2 robot, you **must** release the default sport mode controller. Failure to do so will result in conflicting commands being sent to the motors, causing unpredictable behavior.

Unlike the G1 (which has a hardware button combination `L2+R2 → L2+A → L2+B` to enter debug mode), the **Go2 requires using the SDK programmatically**.

### Option 1: Python Script (Before Running C++ Controller)

Run this script to safely release the default controller:

```python
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

ChannelFactoryInitialize(0, "eth0")  # Use your network interface

# Initialize clients
sc = SportClient()
sc.SetTimeout(5.0)
sc.Init()

msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()

# Release default controller
status, result = msc.CheckMode()
while result['name']:
    sc.StandDown()       # Safely lower the robot first
    msc.ReleaseMode()    # Release sport mode
    status, result = msc.CheckMode()
    time.sleep(1)

print("Default controller released. Ready for low-level control.")
```

### Option 2: Fix in C++ Code (Recommended)

The C++ deployment code should call `unitree::robot::go2::shutdown()` at startup. This function is available in the SDK header `unitree/dds_wrapper/robots/go2/go2.h` and does the same as the Python script above.

In `main.cpp`, modify `init_fsm_state()`:

```cpp
void init_fsm_state()
{
    // Release default controller FIRST
    unitree::robot::go2::shutdown();

    // ... rest of initialization
}
```

## Running the Controller

```bash
cd ~/unitree_rl_lab/deploy/robots/go2/build
./go2_ctrl --network eth0  # Use your network interface (e.g., eth0, enp0s31f6)
```

## Controller Button Mapping

| Action | Button Combination |
|--------|-------------------|
| Enter FixStand | L2 (hold) + A |
| Start RL Policy | Start |
| Emergency Stop (Passive) | L2 (hold) + B |

## Go2 vs G1 Differences

| Feature | G1 | Go2 |
|---------|----|----|
| Hardware debug mode | L2+R2 → L2+A → L2+B | None (use SDK) |
| Release controller | `MotionSwitcherClient.ReleaseMode()` | Same |
| Exit debug mode | Reboot required | SDK can re-enable |
