from icecream import ic
import mujoco_py as mpy
import numpy as np


def set_state_from_array(sim, arr):
    state = sim.get_state()
    joints = [f'robot0_joint{i}' for i in range(1, 8)]
    joint_addrs = [sim.model.get_joint_qpos_addr(joint) for joint in joints]
    for addr, val in zip(joint_addrs, arr):
        state.qpos[addr] = val

    ic("Updated state to", arr)
    sim.set_state(state)


def test_forward_kinematics(sim):
    sim.forward()

    eef_site = 'gripper0_grip_site'
    eef_site_id = sim.model.site_name2id(eef_site)
    eef_pos = sim.data.site_xpos[eef_site_id]
    eef_quat = sim.data.site_xmat[eef_site_id]
    ic(eef_pos, eef_quat)


def test_joint_limits(sim):
    model = sim.model
    __import__('ipdb').set_trace()



def main():
    model = mpy.load_model_from_path('./robot_with_gripper.xml')
    sim = mpy.MjSim(model)
    # viewer = mpy.MjViewer(sim)

    xinit = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi/4])
    set_state_from_array(sim, xinit)

    # test_forward_kinematics(sim)

    for _ in range(100):
        sim.forward()
        # viewer.render()

    test_joint_limits(sim)


if __name__ == "__main__":
    main()
