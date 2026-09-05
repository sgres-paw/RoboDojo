"""Reusable manipulation skills a scripted expert composes into a task."""

from typing import Any

import numpy as np
import transforms3d as t3d

from src.datagen.arrays import to_numpy
from src.datagen.grasp_bank import load_scene_grasps

GRIPPER_OPEN = 1.0  # Jaws fully apart
GRIPPER_CLOSED = 0.3  # Grip that holds; grasp and place must match or the carry drops it

# --- Scene queries ---


def _arm_robot(env: Any, arm: str) -> Any:
    """Return the robot object for the "left" or "right" arm."""
    return env.robot_manager.get_robot_by_arm_name(f"{arm}_arm")


def current_ee_pose(env: Any, arm: str) -> np.ndarray:
    """Return one arm's current world ee pose (7,) as xyz + wxyz."""
    return env.robot_manager.get_real_endpose(_arm_robot(env, arm), env_idx_list=[0])[0]  # (7,)


def current_gripper_opening(env: Any, arm: str) -> float:
    """Return one arm's current gripper opening, normalised to the 0..1 action range."""
    # == Note ==
    # An action carries a 0..1 opening; the joint moves between gripper_scale [low, high].
    # take_action maps command to joint, so reading it back takes the inverse:
    #   command -> joint   raw = command * (high - low) + low
    #   joint -> command   command = (raw - low) / (high - low)
    # sign == -1 closes the joint as the command grows, so the command is 1 - command
    # ==========
    robot = _arm_robot(env, arm)
    raw_joint_value = float(env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0][0])
    joint_low, joint_high = robot.gripper_scale
    opening = (raw_joint_value - joint_low) / (joint_high - joint_low)
    return opening if robot.gripper_move["sign"] == 1 else 1.0 - opening


def object_position(env: Any, object_label: str) -> np.ndarray:
    """Return a scene object's world position (3,)."""
    position, _ = env.scene_manager.layout_manager.get_instance_pose(env_idx=0, label=object_label, relative=False)
    return to_numpy(position)[:3]  # (3,)


def object_orientation(env: Any, object_label: str) -> np.ndarray:
    """Return a scene object's world orientation (4,) as wxyz."""
    _, quaternion = env.scene_manager.layout_manager.get_instance_pose(env_idx=0, label=object_label, relative=False)
    return to_numpy(quaternion)[:4]  # (4,)


def object_bbox(env: Any, object_label: str) -> np.ndarray:
    """Return a scene object's oriented bounding box vertices (8, 3) in its own frame."""
    layout_manager = env.scene_manager.layout_manager
    instance_name = layout_manager.get_instance_name(0, object_label)
    return layout_manager.get_instance_bbox_vertices(inst_name=instance_name, env_idx=0)  # (8, 3)


# --- Motion ---


def move(
    env: Any,
    arm: str,  # "left" or "right"
    ee_pose: np.ndarray,  # (7,) xyz + wxyz, world frame
    gripper_opening: float,  # 0..1, closed to open
    position_tolerance: float = 0.005,  # Metres from the waypoint that count as arrived
    max_steps_per_segment: int = 10,  # One step advances about 24 mm, so 10 covers a 5 cm segment
    segment_length: float = 0.05,  # Metres of travel per Cartesian leg
) -> float:
    """Step one arm along a near-straight path to a world ee pose (7,); return metres still to go."""
    # take_action interpolates in joint space and advances ~24 mm, so one long command bows
    # off the straight line and sweeps the gripper sideways. Short Cartesian legs stay near it.
    start_position = current_ee_pose(env, arm)[:3].copy()
    distance = np.linalg.norm(start_position - ee_pose[:3])
    segment_count = max(1, int(np.ceil(distance / segment_length)))

    # Read the idle arm once: feeding its measured pose back makes its tracking error the next
    # target, so it creeps down onto whatever is under it.
    idle_arm = "right" if arm == "left" else "left"
    idle_ee_pose = list(current_ee_pose(env, idle_arm))
    idle_gripper_opening = [current_gripper_opening(env, idle_arm)]

    for segment_index in range(1, segment_count + 1):
        waypoint = ee_pose.copy()  # Orientation is the target's throughout; only position steps
        waypoint[:3] = start_position + (ee_pose[:3] - start_position) * (segment_index / segment_count)
        for _ in range(max_steps_per_segment):
            env.take_action(
                {
                    f"{arm}_ee_pose": list(waypoint),
                    f"{arm}_ee_joint_state": [gripper_opening],
                    f"{idle_arm}_ee_pose": idle_ee_pose,
                    f"{idle_arm}_ee_joint_state": idle_gripper_opening,
                }
            )
            # Check after acting: move also drives the gripper at a pose already reached.
            # take_action is a no-op once end_flag is set, so stop rather than spin.
            distance = np.linalg.norm(current_ee_pose(env, arm)[:3] - waypoint[:3])
            if distance < position_tolerance or env.end_flag[0]:
                break
        if env.end_flag[0]:
            break
    return distance


# --- Skills ---


def grasp(
    env: Any,
    arm: str,  # "left" or "right"
    object_label: str,
    standoff_distance: float = 0.10,  # Metres held back before sliding in along the approach axis
    max_tilt_degrees: float = 25.0,  # How far the approach may tilt from straight down
    gripper_open: float = GRIPPER_OPEN,
    gripper_closed: float = GRIPPER_CLOSED,
    position_tolerance: float = 0.005,  # Metres from the target that count as arrived
    max_steps_per_segment: int = 10,  # One step advances about 24 mm, so 10 covers a 5 cm segment
    approach_height: float = 0.15,  # Metres above the standoff to enter from
) -> float:
    """Approach an object from its standoff and close on it; return metres short at the grasp."""

    def approach_axis(ee_pose: np.ndarray) -> np.ndarray:
        """Return the unit approach direction (3,) of an ee_link pose (7,)."""
        # == Note ==
        # R = quat2mat(q) holds the gripper's own axes as columns, in the pose's own frame:
        #   R[:, 0]  approach     direction the gripper travels toward the object
        #   R[:, 1]  sideways     R[:, 2] x R[:, 0]
        #   R[:, 2]  jaw opening  direction the fingers separate along
        # Only the approach is used: to reject side grasps, and to back off to a standoff.
        # ==========
        return t3d.quaternions.quat2mat(ee_pose[3:])[:, 0]  # (3,) unit vector

    def pregrasp_pose(ee_pose: np.ndarray) -> np.ndarray:
        """Return the pose (7,) standoff_distance metres back from a grasp pose along its approach axis."""
        # == Note ==
        # R and t as in grasp_bank; a = R_grasp[:, 0] is the approach axis.
        #   t_pre = t_grasp - standoff * a
        #   R_pre = R_grasp
        # Orientation is held, so the approach is a straight slide along a. Offsetting along
        # world z instead drags the jaws sideways: most bank grasps are far from vertical.
        # ==========
        return ee_pose - np.r_[approach_axis(ee_pose) * standoff_distance, np.zeros(4)]  # (7,)

    def is_reachable(ee_pose: np.ndarray) -> bool:
        """Report whether inverse kinematics can put this arm at a world ee pose (7,)."""
        result = env.robot_manager.solve_ik(target_pose=list(ee_pose), env_idx=0, robot=_arm_robot(env, arm))
        return result["status"] == "Success"

    def select_grasp() -> np.ndarray:
        """Return the best-scoring reachable bank grasp (7,) for this arm."""
        world_down = np.array([0.0, 0.0, -1.0])
        min_downward_cosine = np.cos(np.radians(max_tilt_degrees))
        for ee_pose in load_scene_grasps(env, object_label):
            # The bank ignores the table, so its top candidate is often a side grasp. Both are
            # unit vectors, so their dot product is cos(tilt from straight down).
            if approach_axis(ee_pose) @ world_down < min_downward_cosine:
                continue
            # The standoff must be reachable too, or the approach cannot be executed.
            if is_reachable(ee_pose) and is_reachable(pregrasp_pose(ee_pose)):
                return ee_pose  # (7,)
        raise RuntimeError(f"No reachable grasp for {object_label!r} with the {arm} arm")

    ee_pose = select_grasp()

    # A straight run from wherever the arm is still cuts across the object when it comes from
    # the side. Enter from directly above the standoff instead, then descend.
    standoff_pose = pregrasp_pose(ee_pose)
    above_pose = standoff_pose.copy()
    above_pose[2] += approach_height

    move(env, arm, above_pose, gripper_open, position_tolerance, max_steps_per_segment)  # Up clear of the object
    move(env, arm, standoff_pose, gripper_open, position_tolerance, max_steps_per_segment)  # Down to the standoff
    distance = move(env, arm, ee_pose, gripper_open, position_tolerance, max_steps_per_segment)  # Slide in, jaws open
    # Same pose, gripper only, so distance never shrinks here.
    move(env, arm, ee_pose, gripper_closed, position_tolerance, max_steps_per_segment)  # Close on the object
    return distance


def place(
    env: Any,
    arm: str,  # "left" or "right"
    object_label: str,
    destination_position: np.ndarray,  # (3,) world position the object should end up at
    destination_orientation: np.ndarray | None = None,  # (4,) wxyz; None leaves it as the jaws hold it
    approach_height: float = 0.10,  # Metres above the release pose to travel at
    gripper_open: float = GRIPPER_OPEN,
    gripper_closed: float = GRIPPER_CLOSED,
    position_tolerance: float = 0.005,  # Metres from the target that count as arrived
    max_steps_per_segment: int = 10,  # One step advances about 24 mm, so 10 covers a 5 cm segment
) -> float:
    """Carry a held object to a world position (3,) and release it; return metres short of the release pose."""

    def release_pose() -> np.ndarray:
        """Return the ee pose (7,) that lands the held object on its destination, from where it sits now."""
        # == Note ==
        # A held object keeps a fixed offset from the gripper, so aim the gripper by the
        # offset the object still has to travel:
        #   t_ee' = t_ee + (destination - t_object)      orientation unchanged
        # Asking for an orientation turns that offset too, by R_fix = R_target @ R_object^-1:
        #   q_ee' = q_fix * q_ee
        #   t_ee' = destination - R_fix @ (t_object - t_ee)
        # ==========
        ee_pose = current_ee_pose(env, arm)
        current_object_position = object_position(env, object_label)
        if destination_orientation is None:
            return np.r_[ee_pose[:3] + destination_position - current_object_position, ee_pose[3:]]  # (7,)

        correction = t3d.quaternions.qmult(
            destination_orientation, t3d.quaternions.qconjugate(object_orientation(env, object_label))
        )
        held_offset = t3d.quaternions.quat2mat(correction) @ (current_object_position - ee_pose[:3])
        return np.r_[destination_position - held_offset, t3d.quaternions.qmult(correction, ee_pose[3:])]  # (7,)

    # Three poses: lift_pose straight above the pick-up, approach_pose high over the
    # destination, descent_pose where the jaws open.
    approach_pose = release_pose()
    approach_pose[2] += approach_height

    # Lift straight up first: a diagonal start drags whatever the object still sits over.
    # Command the grip every leg; under load the jaws read wider than commanded.
    lift_pose = current_ee_pose(env, arm).copy()
    lift_pose[2] = max(lift_pose[2], approach_pose[2])
    move(env, arm, lift_pose, gripper_closed, position_tolerance, max_steps_per_segment)  # Straight up, still held
    move(env, arm, approach_pose, gripper_closed, position_tolerance, max_steps_per_segment)  # Across, still high

    # Re-measure: the object shifts and turns in the jaws mid-carry, so the earlier pose no
    # longer lands it right.
    approach_pose = current_ee_pose(env, arm).copy()
    descent_pose = release_pose()

    distance = move(env, arm, descent_pose, gripper_closed, position_tolerance, max_steps_per_segment)  # Down onto it
    move(env, arm, descent_pose, gripper_open, position_tolerance, max_steps_per_segment)  # Open the jaws
    move(env, arm, approach_pose, gripper_open, position_tolerance, max_steps_per_segment)  # Back up, empty
    return distance
