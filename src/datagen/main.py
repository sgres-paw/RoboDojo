"""Generate demonstration episodes for one task."""

import argparse
import importlib
import os
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from src.datagen.env_builder import create_env
from src.datagen.recorder import record_episode, write_episode

SUCCESS_REWARD = 1.0 - 1e-3  # get_reward is 1.0 only if every stage passed; epsilon as in reward_manager.py:605


def main(args: argparse.Namespace, simulation_app: Any) -> None:
    """Attempt one episode per layout and keep the ones the task's own scorer passes."""
    run_task = importlib.import_module(f"src.datagen.tasks.{args.task_name}").run
    episode_dir = args.output_dir / args.task_name
    env = create_env(args.task_name, simulation_app)

    # Layouts are pre-baked files, so a run cannot outlast the shipped set.
    attempts = min(args.num_episodes, len(env.seed_manager.seed_list))
    if attempts < args.num_episodes:
        print(f"[datagen] only {attempts} layouts exist; attempting those", flush=True)

    kept_count = 0
    for layout_id in range(attempts):  # reset indexes pre-baked layout files, not an RNG seed
        env.reset(seed=[layout_id])
        env.run_reward()  # Registers the staged checks; step() consumes them as they pass

        try:
            frames = record_episode(env, run_task)
        except RuntimeError as error:  # No reachable grasp: an unusable layout, not a failure to fix
            print(f"[datagen] layout {layout_id} skipped: {error}", flush=True)
            continue

        reward = env.reward_manager.get_reward(final_check=True)[0]
        if reward < SUCCESS_REWARD:
            stages_left = len(env.reward_manager.check_list[0])  # Reward is 0.0 either way; this says how far it got
            print(f"[datagen] layout {layout_id} failed with {stages_left} stages left", flush=True)
            continue

        write_episode(frames, episode_dir / f"episode_{kept_count:07d}.hdf5")
        kept_count += 1
        print(f"[datagen] layout {layout_id} kept {len(frames)} frames", flush=True)

    print(f"[datagen] {kept_count}/{attempts} episodes written to {episode_dir}", flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)  # Kit's crash reporter otherwise keeps the process alive polling a display


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", type=str, default="cover_blocks", help="Task to generate episodes for.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Layouts to attempt, one episode each.")
    parser.add_argument("--output_dir", type=Path, default=Path("datagen_result"), help="Where episodes are written.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = AppLauncher(args).app  # create_env needs this before it builds anything
    main(args, simulation_app)
