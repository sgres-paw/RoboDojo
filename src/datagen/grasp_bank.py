"""Read a mesh's `grasps/proposal.json` and place its grasps where the object now sits."""

import json
from typing import Any

import numpy as np
import transforms3d as t3d

from env.global_configs import ASSETS_PATH, BENCHMARK
from src.datagen.arrays import to_numpy


def load_grasp_proposals(category_name: str, category_index: int, object_type: str) -> np.ndarray:
    """Return one mesh's grasp proposals as object-frame poses (N, 7) xyz + wxyz, best score first."""
    proposal_path = (
        f"{ASSETS_PATH}/Object/{BENCHMARK}/{object_type.capitalize()}"
        f"/{category_name}/{category_index:05d}/grasps/proposal.json"
    )

    # == Note ==
    # File holds:
    #   {"uuid": ..., "grasp": [[[x, y, z, qw, qx, qy, qz], score], ...]}
    # 100 proposals, best score first.
    #
    # Each entry claims: put the gripper there, close the jaws, and the object is held.
    #   - x, y, z          position, in metres
    #   - qw, qx, qy, qz   orientation, unit quaternion in wxyz order (not scipy's xyzw)
    #
    # Poses are in the object's frame, so they travel with it; to_world_ee_poses puts
    # them where the object actually is.
    #
    # All 552 shipped meshes have banks. For a mesh you author, write the same file:
    #   - antipodal sampling with trimesh: surface point pairs, opposing normals, gap
    #     under the 54 mm jaw width in Assets/Robots/x5/robot_config.yml
    #   - a pretrained grasp network predicts scored poses from the mesh or a point
    #     cloud, and handles shapes sampling cannot: GraspGen, Contact-GraspNet, AnyGrasp
    # ==========

    with open(proposal_path) as proposal_file:
        entries = json.load(proposal_file)["grasp"]
    return np.array([pose for pose, _ in entries], dtype=float)  # (N, 7)


def to_world_ee_poses(
    grasp_poses: np.ndarray,  # (N, 7) xyz + wxyz, object frame
    object_position: np.ndarray,  # (3,)
    object_orientation: np.ndarray,  # (4,) wxyz
) -> np.ndarray:
    """Transform object-frame grasp proposals into world end-effector poses (N, 7)."""
    # == Note ==
    # Object frame keeps a grasp reusable at every spawn. The arm takes world coordinates,
    # the same frame get_instance_pose reports object poses in.
    #
    # T_AB is the 4x4 transform placing frame B in frame A: rotation R, translation t.
    # W world, O object, G grasp, E end effector.
    #
    #   T_WE = T_WO @ T_OG @ T_GE
    #     T_WO   the object's pose now
    #     T_OG   one bank proposal
    #     T_GE   bank_to_ee_link, rotation only
    #
    #   R_WE = R_WO @ R_OG @ R_GE
    #   t_WE = R_WO @ t_OG + t_WO
    # ==========

    # Approach runs along the gripper frame's own z axis in the bank, x in ee_link.
    # Right-multiplying a grasp rotation by this swaps them. Franka ships the same
    # matrix as `delta_matrix` in Assets/Robots/franka/robot_config.yml.
    bank_to_ee_link = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)

    object_rotation = t3d.quaternions.quat2mat(object_orientation)
    world_poses = []
    for grasp_pose in grasp_poses:
        grasp_rotation = t3d.quaternions.quat2mat(grasp_pose[3:])
        world_rotation = object_rotation @ grasp_rotation @ bank_to_ee_link
        world_poses.append(
            np.concatenate(
                [
                    object_rotation @ grasp_pose[:3] + object_position,
                    t3d.quaternions.mat2quat(world_rotation),
                ]
            )
        )
    return np.array(world_poses)  # (N, 7) xyz + wxyz, world frame


def load_scene_grasps(env: Any, object_label: str, env_idx: int = 0) -> np.ndarray:
    """Return a scene object's grasp proposals as world end-effector poses (N, 7), best score first."""
    # Approach directions vary per mesh. Callers must filter in world frame.
    layout_manager = env.scene_manager.layout_manager
    instance_name = layout_manager.get_instance_name(env_idx, object_label)
    object_type = layout_manager.instance_type_by_env[env_idx][instance_name]
    object_records = layout_manager.object_records_by_type[object_type.capitalize()]
    mesh_metadata = object_records.metadata_by_env[env_idx][instance_name]

    grasp_poses = load_grasp_proposals(mesh_metadata["model_name"], mesh_metadata["model_id"], object_type)
    # relative=False adds the env origin, so the pose is world even with several envs.
    object_position, object_orientation = layout_manager.get_instance_pose(
        env_idx=env_idx, inst_name=instance_name, relative=False
    )
    return to_world_ee_poses(grasp_poses, to_numpy(object_position)[:3], to_numpy(object_orientation)[:4])  # (N, 7)
