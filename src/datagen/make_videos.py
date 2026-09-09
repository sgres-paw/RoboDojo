"""Render one camera of episode HDF5 files to mp4 at native resolution."""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FPS = 25  # The benchmark records at 25 Hz


def decode(frame_bytes: np.void) -> np.ndarray:
    """Decode one stored JPEG frame into a BGR image (H, W, 3)."""
    return cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)


def render(episode_path: Path, out_path: Path, camera: str) -> int:
    """Write one camera of one episode to mp4 at its native resolution; return frames written."""
    with h5py.File(episode_path, "r") as episode:
        stream = episode["vision"][camera]["colors"]
        writer = None
        for index in range(len(stream)):
            image = decode(stream[index])
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (image.shape[1], image.shape[0])
                )
            writer.write(image)
        if writer is not None:
            writer.release()
        return len(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--camera", type=str, default="cam_head", help="Which camera stream to render.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.episodes:
        target = args.out_dir / f"{path.parent.name}_{path.stem}_{args.camera}.mp4"
        frames = render(path, target, args.camera)
        print(f"[video] {target.name}  {frames} frames  {target.stat().st_size / 1e6:.1f} MB", flush=True)
