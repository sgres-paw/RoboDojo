"""Capture a scripted expert's rollout and write it as an xspark v1.0 episode."""

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np

from src.datagen.arrays import to_numpy
from XPolicyLab.utils.process_data import images_encoding


def record_episode(env: Any, run_task: Callable[[Any], None]) -> list[dict]:
    """Run a task's skill sequence and return one observation frame per control step."""
    # Every skill steps through take_action, so wrapping it records them all, skills untouched.
    frames = []
    original_take_action = env.take_action

    def take_action_and_record(action: dict) -> None:
        """Capture the observation the action starts from, then take it."""
        if not env.end_flag[0]:  # take_action is a no-op past the end; it would duplicate frames
            frames.append(env.get_obs())
        original_take_action(action)

    env.take_action = take_action_and_record
    try:
        run_task(env)
    finally:
        env.take_action = original_take_action
    frames.append(env.get_obs())  # The state the last action reached
    return frames


def write_episode(frames: list[dict], hdf5_path: Path) -> None:
    """Write captured frames to one xspark v1.0 HDF5 episode."""
    # action[t] is the state its step reached, so it equals state[t+1]; exact in the official demos.
    states, actions = frames[:-1], frames[1:]

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as episode:
        for group_name, group_frames in (("state", states), ("action", actions)):
            for key in group_frames[0][group_name]:  # get_obs names one frame, the file a series
                episode[f"{group_name}/{key}s"] = np.array([to_numpy(frame[group_name][key]) for frame in group_frames])
        # The official files store a byte-identical copy under this name.
        for arm in ("left", "right"):
            episode[f"state/{arm}_delta_ee_poses"] = episode[f"state/{arm}_ee_poses"][()]

        for camera_name, camera_data in states[0]["vision"].items():
            images = [to_numpy(frame["vision"][camera_name]["color"]) for frame in states]
            jpeg_frames, longest_jpeg_bytes = images_encoding(images)
            # One fixed-width byte column; numpy pads the shorter frames, as the encoder expects.
            episode[f"vision/{camera_name}/colors"] = np.array(jpeg_frames, dtype=f"S{longest_jpeg_bytes}")
            episode[f"vision/{camera_name}/shape"] = np.array(images[0].shape)
            episode[f"vision/{camera_name}/intrinsic_matrix"] = to_numpy(camera_data["intrinsic_matrix"])
            episode[f"vision/{camera_name}/extrinsic_matrix"] = np.array(
                [to_numpy(frame["vision"][camera_name]["extrinsic_matrix"]) for frame in states]
            )

        episode["additional_info/frequency"] = states[0]["additional_info"]["frequency"]
        episode["data_format_version"] = states[0]["data_format_version"]
        episode["instruction"] = str(states[0]["instruction"])  # np.str_ is UTF-32; h5py takes only str
