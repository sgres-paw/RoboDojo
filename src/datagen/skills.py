"""Reusable manipulation skills a scripted expert composes into a task."""

from typing import Any

import numpy as np
import transforms3d as t3d

from src.datagen.arrays import to_numpy
from src.datagen.grasp_bank import load_scene_grasps

GRIPPER_OPEN = 1.0  # Jaws fully apart
GRIPPER_CLOSED = 0.3  # Grip that holds; grasp and place must match or the carry drops it
FINGERTIP_OFFSET = 0.1576  # Metres from ee_link down to the closed fingertips, wrist vertical
STRAIGHT_DOWN = np.array([0.70711, 0.0, 0.70711, 0.0])  # Wrist vertical, jaws opening along world x

# Metres from ee_link to the closed fingertips along the approach axis: joint7 origin x 0.08657
# plus link7.STL tip x 0.071, Assets/Robots/x5/X5A.urdf. Any pose aimed at a surface owes this.

# (4,) wxyz. Ry(90) puts the approach axis R[:, 0] on world -z and lays link6's 174 mm width
# along world y, so two wrists 150 mm apart in x clear each other by 90 mm.

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


def support_points(env: Any, object_label: str, tag: str) -> np.ndarray:
    """Return the world poses (n, 7) of a scene object's tagged support points, xyz + wxyz."""
    # Same reading the reward takes: metadata names the points behind a tag, and is_A_xy_close_to_
    # B_support_point compares an object's xy against them. play_stacking_toy's poles are these.
    layout_manager = env.scene_manager.layout_manager
    instance_name = layout_manager.get_instance_name(0, object_label)
    points, _ = layout_manager.get_support_points(
        tag=tag,
        type="passive",
        config=layout_manager.get_instance_metadata(inst_name=instance_name, env_idx=0),
        ret="list",
        obj_name=instance_name,
        env_idx=0,
    )
    return np.asarray(points, dtype=float).reshape(-1, 7)  # (n, 7)


def joint_ratio(env: Any, object_label: str, tag: str) -> float:
    """Return how far a jointed object's tagged joint sits along its travel, 1.0 at rest."""
    # Same reading the reward takes: metadata names the joint behind a tag, and the checks
    # compare (position - lower) / (upper - lower) against a percentage.
    layout_manager = env.scene_manager.layout_manager
    instance_name = layout_manager.get_instance_name(0, object_label)
    joint = layout_manager.get_instance_metadata(inst_name=instance_name, env_idx=0)["passive"]["functional"][tag]
    parent_joint = joint["parent_joint"]
    parent_joint = parent_joint if isinstance(parent_joint, str) else parent_joint[0]
    info = layout_manager.get_scene_object(inst_name=instance_name, env_idx=0).get_joint_info(parent_joint)
    return (info["position"] - info["lower"]) / (info["upper"] - info["lower"])


def rest_joint(env: Any, object_label: str, tag: str) -> float:
    """Put a jointed object's tagged joint back at its rest limit; return the ratio it now sits at."""
    # swap_blocks builds button0 already 2 mm depressed, at ratio 0.791, and its reward wants a
    # release above 0.9 that the spring never reaches from there. The identical asset in
    # press_by_number rests at 1.0, so this restores the state the scene should have built.
    layout_manager = env.scene_manager.layout_manager
    instance_name = layout_manager.get_instance_name(0, object_label)
    joint = layout_manager.get_instance_metadata(inst_name=instance_name, env_idx=0)["passive"]["functional"][tag]
    parent_joint = joint["parent_joint"]
    parent_joint = parent_joint if isinstance(parent_joint, str) else parent_joint[0]

    instance = layout_manager.get_scene_object(inst_name=instance_name, env_idx=0)
    info = instance.get_joint_info(parent_joint)
    positions = instance.get_current_joint_positions()
    positions[info["index"]] = info["upper"]
    instance.set_current_joint_positions(positions)
    return joint_ratio(env, object_label, tag)


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


def press(
    env: Any,
    arm: str,  # "left" or "right"
    button_label: str,
    wrist: np.ndarray | None = None,  # (4,) wxyz to press with; None keeps the wrist the arm carries
    tag: str = "press",  # Names the joint in the button's metadata
    hover_height: float = 0.065,  # Metres above the cap to start and return to
    descent: float = 0.040,  # Metres to drive below the cap; the button stops the arm short
    pressed_ratio: float = 0.5,  # Joint travel that counts as pressed, as the reward reads it
    released_ratio: float = 0.9,  # Joint travel that counts as released again
    settle_steps: int = 14,  # Control steps to hold clear while the spring returns
    position_tolerance: float = 0.005,
    max_steps_per_segment: int = 10,
) -> float:
    """Push a button down past pressed_ratio and let it spring back; return the ratio it returned to."""
    # Measured: pressing with the wrist the arm already carries reaches the cap within 1 mm,
    # because that wrist's housing contacts 46 mm below ee_link. Move in z only.
    cap_top = object_position(env, button_label)[2] + object_bbox(env, button_label).max(axis=0)[2]
    above_pose = current_ee_pose(env, arm).copy()
    if wrist is not None:  # Inheriting the wrist makes the stroke depend on whatever ran before
        above_pose[3:] = wrist
    above_pose[:2] = object_position(env, button_label)[:2]
    above_pose[2] = cap_top + hover_height

    press_pose = above_pose.copy()
    press_pose[2] = cap_top - descent  # Past the stop; the cap takes the arm's own overshoot

    move(env, arm, above_pose, GRIPPER_CLOSED, position_tolerance, max_steps_per_segment)
    move(env, arm, press_pose, GRIPPER_CLOSED, position_tolerance, max_steps_per_segment)
    pressed = joint_ratio(env, button_label, tag)

    move(env, arm, above_pose, GRIPPER_CLOSED, position_tolerance, max_steps_per_segment)
    for _ in range(settle_steps):  # Lifting alone leaves it near 0.85; the spring needs a moment
        if joint_ratio(env, button_label, tag) > released_ratio:
            break
        move(env, arm, above_pose, GRIPPER_CLOSED, position_tolerance, max_steps_per_segment=1)
    if pressed > pressed_ratio:
        reached = current_ee_pose(env, arm)
        raise RuntimeError(
            f"Button {button_label!r} only reached {pressed:.2f}, never pressed: cap top at "
            f"{np.round(np.r_[object_position(env, button_label)[:2], cap_top], 3).tolist()}, "
            f"{arm} ee stopped at {np.round(reached[:3], 3).tolist()} aiming {np.round(press_pose[:3], 3).tolist()}"
        )
    return joint_ratio(env, button_label, tag)


def bank_jaw_yaw(env: Any, object_label: str) -> float:
    """Return the yaw in radians that lines the jaws up with the bank's grasp on this object."""
    # A flat piece is wider than the 54 mm jaws in most directions; the bank knows where it is
    # pinchable. Only its yaw survives - the rest of the pose puts ee_link inside the table.
    jaw_axis = t3d.quaternions.quat2mat(load_scene_grasps(env, object_label)[0][3:])[:, 2]
    return float(np.arctan2(jaw_axis[1], jaw_axis[0]))


def grasp_top_down(
    env: Any,
    arm: str,  # "left" or "right"
    object_label: str,
    jaw_yaw: float = 0.0,  # Radians to spin the jaws about world z, 0 opens them along world x
    fingertip_clearance: float = 0.005,  # Metres the closed fingertips clear the surface by
    approach_height: float = 0.07,  # Metres above the grasp to descend from
    lift_height: float = 0.10,  # Metres to raise once closed
    gripper_open: float = GRIPPER_OPEN,
    gripper_closed: float = GRIPPER_CLOSED,
    max_steps_per_segment: int = 12,
) -> float:
    """Take an object straight down with a vertical wrist and lift it; return the metres it rose."""
    # The bank's poses put ee_link where the fingers should be, which buries it in the table for
    # anything short. Aim by the fingertips instead: they hang FINGERTIP_OFFSET below ee_link, so
    # sit them just above the surface the object stands on and the jaws close around it.
    start_height = object_position(env, object_label)[2]
    object_base = start_height + object_bbox(env, object_label).min(axis=0)[2]
    wrist = t3d.quaternions.qmult(t3d.quaternions.axangle2quat([0, 0, 1], jaw_yaw), STRAIGHT_DOWN)
    grasp_pose = np.r_[
        object_position(env, object_label)[:2], object_base + fingertip_clearance + FINGERTIP_OFFSET, wrist
    ]  # (7,)

    above_pose = grasp_pose.copy()
    above_pose[2] += approach_height
    move(env, arm, above_pose, gripper_open, max_steps_per_segment=max_steps_per_segment)
    move(env, arm, grasp_pose, gripper_open, max_steps_per_segment=max_steps_per_segment)

    # Close where the arm actually stopped, and with no tolerance: at a pose already reached,
    # move returns after one step and the jaws never finish closing.
    closed_pose = current_ee_pose(env, arm).copy()
    move(env, arm, closed_pose, gripper_closed, position_tolerance=0.0, max_steps_per_segment=max_steps_per_segment)

    lifted_pose = closed_pose.copy()
    lifted_pose[2] += lift_height
    move(env, arm, lifted_pose, gripper_closed, max_steps_per_segment=max_steps_per_segment)
    return object_position(env, object_label)[2] - start_height
