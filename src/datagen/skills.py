"""Reusable manipulation skills a scripted expert composes into a task."""

from copy import deepcopy
from typing import Any

import numpy as np
import transforms3d as t3d

from env.global_configs import BATCH_NUM
from src.datagen.arrays import to_numpy
from src.datagen.grasp_bank import load_scene_grasps
from utils.transformer import cal_quat_dis

GRIPPER_OPEN = 1.0  # Jaws fully apart
GRIPPER_CLOSED = 0.3  # Grip that holds; grasp and place must match or the carry drops it
FINGERTIP_OFFSET = 0.1576  # Metres from ee_link down to the closed fingertips, wrist vertical
WAYPOINTS_PER_LEG = 6  # Planner rows kept per leg. It interpolates at the 4 ms sim step, so following
# every row costs hundreds of actions and a task runs out of its step budget; take_action lerps
# between whatever rows it is given, so a handful per leg tracks the same path far more cheaply.
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


def current_joint_positions(env: Any, arm: str) -> np.ndarray:
    """Return one arm's current joint positions (n,)."""
    return np.asarray(env.robot_manager.get_joint(_arm_robot(env, arm), env_idx_list=[0])[0], dtype=float)  # (n,)


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


def _sample_path(path: np.ndarray) -> list[np.ndarray]:
    """Return at most WAYPOINTS_PER_LEG evenly spaced rows of a planned joint path (T, n), last included."""
    rows = np.asarray(path)
    stride = max(1, int(np.ceil(len(rows) / WAYPOINTS_PER_LEG)))
    sampled = list(rows[stride - 1 :: stride])
    if not sampled or not np.array_equal(sampled[-1], rows[-1]):
        sampled.append(rows[-1])
    return sampled


def move(
    env: Any,
    arm: str,  # "left" or "right"
    ee_pose: np.ndarray,  # (7,) xyz + wxyz, world frame
    gripper_opening: float,  # 0..1, closed to open
    position_tolerance: float = 0.005,  # Metres from the target that count as arrived
    rotation_tolerance_degrees: float = 5.0,  # Degrees from the target wrist that count as arrived
    settle_steps: int = 8,  # Control steps held at the last waypoint while the arm converges
) -> float:
    """Plan a collision-free path to a world ee pose (7,) and follow it; return metres still to go."""
    # curobo solves the whole path once, in joint space, so the gripper no longer bows off the
    # straight line and the wrist arrives with the position instead of lagging behind it. An
    # unreachable goal comes back as status "Fail" rather than an arm that quietly does not move:
    # the ee action branch drops the joint key when IK fails (eval_env.py:415), which is why a bad
    # pose used to look like a no-op. Both arms must appear in a joint action, so the idle one is
    # commanded at its measured joints - it cannot creep, because nothing re-derives IK for it.
    robot = _arm_robot(env, arm)
    idle_arm = "right" if arm == "left" else "left"
    idle_robot = _arm_robot(env, idle_arm)
    robot_manager = env.robot_manager

    plan = robot_manager.planner[robot.robot_name].plan_path(
        curr_joint_pos=robot_manager.get_joint(robot, env_idx_list=[0])[0],
        target_ee_pose=list(ee_pose),
        real_robot_pose=deepcopy(robot.entity_origin_pose),
    )
    if plan["status"] != "Success" or plan.get("position") is None:
        raise RuntimeError(f"No path for the {arm} arm to {np.round(np.asarray(ee_pose)[:3], 3).tolist()}")

    waypoints = _sample_path(plan["position"])

    arm_key = robot_manager.process_name(robot.arm_name)
    gripper_key = robot_manager.process_name(robot.gripper_name)
    idle_arm_key = robot_manager.process_name(idle_robot.arm_name)
    idle_gripper_key = robot_manager.process_name(idle_robot.gripper_name)
    idle_joints = list(np.asarray(robot_manager.get_joint(idle_robot, env_idx_list=[0])[0], dtype=float))
    idle_gripper = [current_gripper_opening(env, idle_arm)]

    def command(joint_positions: np.ndarray) -> None:
        """Drive one control step with both arms in joint space."""
        env.take_action(
            {
                arm_key: list(np.asarray(joint_positions, dtype=float)),
                gripper_key: [gripper_opening],
                idle_arm_key: idle_joints,
                idle_gripper_key: idle_gripper,
            }
        )

    for joint_positions in waypoints:
        command(joint_positions)
        if env.end_flag[0]:
            break

    # Hold the last waypoint until both halves of the pose land. The jaws also need the wait: the
    # gripper command is rate-limited to a fifth of its range per step (control_manager.py:22).
    distance = float(np.linalg.norm(current_ee_pose(env, arm)[:3] - np.asarray(ee_pose)[:3]))
    for _ in range(settle_steps):
        reached = current_ee_pose(env, arm)
        distance = float(np.linalg.norm(reached[:3] - np.asarray(ee_pose)[:3]))
        turn = float(np.degrees(cal_quat_dis(reached[3:], np.asarray(ee_pose)[3:])))
        if (distance < position_tolerance and turn < rotation_tolerance_degrees) or env.end_flag[0]:
            break
        command(waypoints[-1])
    return distance


def rest(
    env: Any,
    arm: str,
    joint_positions: np.ndarray,  # (n,) the configuration to return to
    gripper_opening: float = GRIPPER_OPEN,
    joint_tolerance: float = 0.02,  # Radians on the worst joint that count as arrived
    settle_steps: int = 12,  # Control steps held at the goal while the arm converges
) -> float:
    """Plan back to a joint configuration (n,) and follow it; return radians still to go."""
    # Homing is a joint goal, not a pose goal. Asked for the home *pose*, the planner refuses it:
    # home hangs the closed fingertips at z 0.764 against a table top of 0.765, so the goal itself
    # is in collision. The joint vector the arm started from cannot be, and going joint-to-joint
    # also drops the hand-built lift-and-cross legs, whose retreat height was often out of reach.
    robot = _arm_robot(env, arm)
    idle_arm = "right" if arm == "left" else "left"
    idle_robot = _arm_robot(env, idle_arm)
    robot_manager = env.robot_manager

    plan = robot_manager.planner[robot.robot_name].plan_joint(
        start_joint_pos=current_joint_positions(env, arm),
        goal_joint_pos=np.asarray(joint_positions, dtype=float),
    )
    if plan["status"] != "Success" or plan.get("position") is None:
        raise RuntimeError(f"No path for the {arm} arm back to its start configuration")

    waypoints = _sample_path(plan["position"])

    idle_joints = list(current_joint_positions(env, idle_arm))
    idle_gripper = [current_gripper_opening(env, idle_arm)]

    def command(target_joints: np.ndarray) -> None:
        """Drive one control step with both arms in joint space."""
        env.take_action(
            {
                robot_manager.process_name(robot.arm_name): list(np.asarray(target_joints, dtype=float)),
                robot_manager.process_name(robot.gripper_name): [gripper_opening],
                robot_manager.process_name(idle_robot.arm_name): idle_joints,
                robot_manager.process_name(idle_robot.gripper_name): idle_gripper,
            }
        )

    for joints in waypoints:
        command(joints)
        if env.end_flag[0]:
            break

    # Hold the last waypoint until the joints actually land. Without it the arm stops wherever the
    # final command left it: measured 17 mm from home in position but the wrist a long way round,
    # which fails all_robot_back_to_origin on its 20 degree half.
    goal = np.asarray(joint_positions, dtype=float)
    error = float(np.max(np.abs(current_joint_positions(env, arm) - goal)))
    for _ in range(settle_steps):
        error = float(np.max(np.abs(current_joint_positions(env, arm) - goal)))
        if error < joint_tolerance or env.end_flag[0]:
            break
        command(goal)
    return error


def nudge(
    env: Any,
    arm: str,
    ee_pose: np.ndarray,  # (7,) xyz + wxyz, world frame
    gripper_opening: float,  # 0..1, closed to open
    steps: int = 8,  # Control steps to press for; contact stops the arm before the target
) -> float:
    """Servo one arm toward a world ee pose (7,) without planning; return metres still to go."""
    # For contact strokes only. A press deliberately aims through the cap and lets the button stop
    # the arm, so its target sits below the table surface and the planner refuses it - correctly,
    # since it will not plan into collision. Driving the ee target directly restores the overshoot
    # the press depends on, at the cost of no collision checking, so keep the travel short.
    idle_arm = "right" if arm == "left" else "left"
    idle_ee_pose = list(current_ee_pose(env, idle_arm))
    idle_gripper = [current_gripper_opening(env, idle_arm)]

    # Step toward the target rather than commanding it outright. The ee branch solves IK per action
    # and simply omits the joint key when it fails, so a pose past the surface moves the arm not at
    # all: measured, it stopped 65 mm above the cap. Each intermediate pose is still solvable, so
    # the arm creeps down until the button stops it.
    start = current_ee_pose(env, arm)[:3].copy()
    target = np.asarray(ee_pose, dtype=float)
    for step in range(1, steps + 1):
        waypoint = target.copy()
        waypoint[:3] = start + (target[:3] - start) * (step / steps)
        env.take_action(
            {
                f"{arm}_ee_pose": list(waypoint),
                f"{arm}_ee_joint_state": [gripper_opening],
                f"{idle_arm}_ee_pose": idle_ee_pose,
                f"{idle_arm}_ee_joint_state": idle_gripper,
            }
        )
        if env.end_flag[0]:
            break
    return float(np.linalg.norm(current_ee_pose(env, arm)[:3] - target[:3]))


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
        """Return the first bank grasp (7,) the planner can actually reach with this arm."""
        # Scored by whether a path exists, not whether IK has a solution: solve_ik answers for the
        # goal alone and accepts poses the arm cannot be driven to, which is how unreachable grasps
        # used to be chosen. plan_batch asks the same question for a whole batch in one call.
        world_down = np.array([0.0, 0.0, -1.0])
        min_downward_cosine = np.cos(np.radians(max_tilt_degrees))
        # The bank ignores the table, so its top candidate is often a side grasp. Both are unit
        # vectors, so their dot product is cos(tilt from straight down).
        upright = [
            pose
            for pose in load_scene_grasps(env, object_label)
            if approach_axis(pose) @ world_down >= min_downward_cosine
        ]

        robot = _arm_robot(env, arm)
        for start in range(0, len(upright), BATCH_NUM):
            candidates = upright[start : start + BATCH_NUM]  # plan_batch raises above the configured cap
            result = env.robot_manager.planner[robot.robot_name].plan_batch(
                env.robot_manager.get_joint(robot, env_idx_list=[0])[0],
                [list(pregrasp_pose(pose)) for pose in candidates],
                deepcopy(robot.entity_origin_pose),
            )
            for index, status in enumerate(result["status"]):
                if status == "Success":
                    return candidates[index]  # (7,)
        raise RuntimeError(f"No reachable grasp for {object_label!r} with the {arm} arm")

    ee_pose = select_grasp()

    # A straight run from wherever the arm is still cuts across the object when it comes from
    # the side. Enter from directly above the standoff instead, then descend.
    standoff_pose = pregrasp_pose(ee_pose)
    above_pose = standoff_pose.copy()
    above_pose[2] += approach_height

    # Entering from overhead keeps the jaws off the object on the way in, but the entry pose is
    # often out of reach - for cover_blocks it is out of reach for all 100 bank grasps. That used
    # to pass unnoticed, because a move to an unreachable pose quietly did nothing and the run
    # carried on from the standoff. Now the planner says so, and the standoff, already offset back
    # along the approach axis, is a safe enough entry on its own when the overhead pose is refused.
    try:
        move(env, arm, above_pose, gripper_open, position_tolerance)  # Up clear of the object
    except RuntimeError:  # Optional leg: solve_ik accepts poses the planner will not plan a path to
        pass
    move(env, arm, standoff_pose, gripper_open, position_tolerance)  # Down to the standoff
    distance = move(env, arm, ee_pose, gripper_open, position_tolerance)  # Slide in, jaws open
    # Same pose, gripper only, so distance never shrinks here.
    move(env, arm, ee_pose, gripper_closed, position_tolerance)  # Close on the object
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
) -> float:
    """Carry a held object to a world position (3,) and release it; return metres short of the release pose."""

    HELD_OFFSET_XY = 0.12  # Metres the object may sit from under the gripper and still be held

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

    # Nothing below means anything if the object is not in the jaws: release_pose aims the gripper
    # by the offset the object still has to travel, so a dropped object throws the target metres
    # away and the planner refuses a pose that was never sensible. Measured: targets at
    # (-0.217, -0.177, 1.071) and y=+0.167, off the table. Say so instead.
    def check_still_held() -> None:
        """Raise unless the object is still under the gripper."""
        # Measured horizontally, not in 3D: a held object hangs nearly under the gripper, while a
        # dropped one keeps a similar 3D distance yet sits well to the side. Layout 11 put a cube at
        # y=-0.49, behind the table edge, and a 3D test still called it held.
        if np.linalg.norm(object_position(env, object_label)[:2] - current_ee_pose(env, arm)[:2]) > HELD_OFFSET_XY:
            at = np.round(object_position(env, object_label), 3).tolist()
            raise RuntimeError(f"{object_label!r} is not in the {arm} jaws, it is at {at}")

    check_still_held()

    # Three poses: lift_pose straight above the pick-up, approach_pose high over the
    # destination, descent_pose where the jaws open.
    approach_pose = release_pose()
    approach_pose[2] += approach_height

    # Lift straight up first: a diagonal start drags whatever the object still sits over.
    # Command the grip every leg; under load the jaws read wider than commanded.
    # Lifting first keeps the held object off whatever it still sits over, but the pose is derived
    # from wherever the arm happens to be, and raising z at a forward xy leaves the arm's envelope:
    # measured refusals at y=-0.06 and y=-0.09, well in front of the mats. It is an optimisation,
    # not a requirement - the planner already routes over the table - so a refusal just skips it.
    # Rising before the crossing matters more than it looks: the planner models the arm and the
    # table, never the object in the jaws, so a direct path happily drags a held cube across a mat
    # and knocks it out. Take the highest lift that plans rather than skipping the leg outright.
    held_height = current_ee_pose(env, arm)[2]
    for lift_height in (max(held_height, approach_pose[2]), max(held_height, approach_pose[2]) - 0.05, held_height):
        lift_pose = current_ee_pose(env, arm).copy()
        lift_pose[2] = lift_height
        try:
            move(env, arm, lift_pose, gripper_closed, position_tolerance)  # Straight up, still held
            break
        except RuntimeError:
            continue
    # A fixed lift above the release can leave the arm's envelope - the handover puts the ee at
    # (0.005, -0.093, 1.038), far in y and high in z, which the left arm cannot reach. The height
    # only has to clear whatever sits beside the target, so take the highest one that plans.
    for height in (approach_height, approach_height / 2, 0.0):
        approach_pose = release_pose()
        approach_pose[2] += height
        try:
            move(env, arm, approach_pose, gripper_closed, position_tolerance)  # Across, still high
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"No reachable approach above {object_label!r} for the {arm} arm")

    # Re-measure: the object shifts and turns in the jaws mid-carry, so the earlier pose no
    # longer lands it right.
    # Re-check here, not only on entry: the object can slip during the lift or the crossing, and
    # release_pose() would then aim the gripper metres away from anything sensible.
    check_still_held()
    approach_pose = current_ee_pose(env, arm).copy()
    descent_pose = release_pose()

    distance = move(env, arm, descent_pose, gripper_closed, position_tolerance)  # Down onto it
    move(env, arm, descent_pose, gripper_open, position_tolerance)  # Open the jaws
    # Retracting after the release only clears the cube for whatever comes next; the next skill
    # moves this arm anyway. A refusal here is not a failed placement, so it does not fail the carry.
    try:
        move(env, arm, approach_pose, gripper_open, position_tolerance)  # Back up, empty
    except RuntimeError:
        pass
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

    # Planned where a plan exists, servoed where it does not. The cap sits between the two bases,
    # and from some carries the planner finds no path to the hover pose even though the arm can
    # creep there: measured on layout 36, x=0.0 with the left arm.
    try:
        move(env, arm, above_pose, GRIPPER_CLOSED, position_tolerance)  # Planned, in free space
    except RuntimeError:
        nudge(env, arm, above_pose, GRIPPER_CLOSED, steps=12)
    nudge(env, arm, press_pose, GRIPPER_CLOSED)  # Servoed, through the cap
    pressed = joint_ratio(env, button_label, tag)

    move(env, arm, above_pose, GRIPPER_CLOSED, position_tolerance)  # Planned again, back clear
    for _ in range(settle_steps):  # Lifting alone leaves it near 0.85; the spring needs a moment
        if joint_ratio(env, button_label, tag) > released_ratio:
            break
        move(env, arm, above_pose, GRIPPER_CLOSED, position_tolerance)
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
    move(env, arm, above_pose, gripper_open)
    move(env, arm, grasp_pose, gripper_open)

    # Close where the arm actually stopped, and with no tolerance: at a pose already reached,
    # move returns after one step and the jaws never finish closing.
    closed_pose = current_ee_pose(env, arm).copy()
    move(env, arm, closed_pose, gripper_closed, position_tolerance=0.0)

    lifted_pose = closed_pose.copy()
    lifted_pose[2] += lift_height
    move(env, arm, lifted_pose, gripper_closed)
    return object_position(env, object_label)[2] - start_height
