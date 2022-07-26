"""
Running operational space control with the PyGame display, using an exponential
additive signal when to push away from joints.
The target location can be moved by clicking on the background.
"""
import numpy as np

from abr_control.arms import threejoint as arm
from abr_control.controllers import OSC, AvoidJointLimits, Damping

# from abr_control.arms import twojoint as arm
from abr_control.interfaces.no_display import NoDisplay

print("\nClick to move the target.\n")

# initialize our robot config
robot_config = arm.Config()
# create our arm simulation
arm_sim = arm.ArmSim(robot_config)

avoid = AvoidJointLimits(
    robot_config,
    min_joint_angles=[np.pi / 5.0] * robot_config.N_JOINTS,
    max_joint_angles=[np.pi / 2.0] * robot_config.N_JOINTS,
    max_torque=[100.0] * robot_config.N_JOINTS,
)
# damp the movements of the arm
damping = Damping(robot_config, kv=10)
# create an operational space controller
ctrlr = OSC(
    robot_config,
    kp=100,
    null_controllers=[avoid, damping],
    # control (x, y) out of [x, y, z, alpha, beta, gamma]
    ctrlr_dof=[True, True, False, False, False, False],
)


def on_click(self, mouse_x, mouse_y):
    self.target[0] = self.mouse_x
    self.target[1] = self.mouse_y


# create our interface
interface = NoDisplay(
    arm_sim,
    dt=0.001
)
interface.connect()

# create a target [x, y, z]]
target_xyz = [0, 2, 0]
# create a target orientation [alpha, beta, gamma]
target_angles = [0, 0, 0]

try:
    print("\nSimulation starting...\n")

    count = 0
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx("EE", feedback["q"])

        target = np.hstack([target_xyz, target_angles])
        # generate an operational space control signal
        u = ctrlr.generate(
            q=feedback["q"],
            dq=feedback["dq"],
            target=target,
        )
        print(u)
        # apply the control signal, step the sim forward
        interface.send_forces(u)

        # change target location once hand is within
        # 5mm of the target
        if np.sqrt(np.sum((target_xyz - hand_xyz) ** 2)) < 0.005:
            target_xyz = np.array(
                [np.random.random() * 2 - 1, np.random.random() * 2 + 1, 0]
            )

        count += 1

finally:
    # stop and reset the simulation
    interface.disconnect()

    print("Simulation terminated...")
