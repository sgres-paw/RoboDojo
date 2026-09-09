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
    episode_dir = args.output_dir / args.task_name / f"seed_{args.seed}"
    env = create_env(args.task_name, simulation_app, eval_seed=args.seed)

    # Layouts are pre-baked files, so a run cannot outlast the shipped set. Articulated
    # scenes cannot be re-reset either: every reset mints new instance names and only rigid
    # objects are re-registered under them, so run those one layout per process.
    last_layout = min(args.first_layout + args.num_episodes, len(env.seed_manager.seed_list))
    if last_layout - args.first_layout < args.num_episodes:
        print(f"[datagen] only {last_layout} layouts exist; attempting up to that", flush=True)

    kept_count = 0
    for layout_id in range(args.first_layout, last_layout):  # reset indexes pre-baked layout files, not an RNG seed
        env.reset(seed=[layout_id])
        env.run_reward()  # Registers the staged checks; step() consumes them as they pass

        try:
            # Capturing frames costs 70 ms of the 156 ms a control step takes, so a run that only
            # needs to know which layouts the expert solves skips it and goes about twice as fast.
            frames = run_task(env) if args.no_record else record_episode(env, run_task)
        except RuntimeError as error:  # No reachable grasp: an unusable layout, not a failure to fix
            print(f"[datagen] layout {layout_id} skipped: {error}", flush=True)
            continue

        # The env decides success itself, in is_episode_end (eval_client/eval_env.py): as soon as
        # get_reward clears the threshold it sets end_flag and success, and stops accepting actions.
        # That flag is what the benchmark records for a policy, so it is what counts here too. Our
        # own final_check can disagree - layout 52 of swap_blocks is marked a success at step 538
        # and then scores 0 on a second evaluation, because the staged checks were already consumed.
        if env.success[0]:
            reward = 1.0
        else:
            reward = env.reward_manager.get_reward(final_check=True)[0]
        if reward < SUCCESS_REWARD:
            stages_left = len(env.reward_manager.check_list[0])  # Reward is 0.0 either way; this says how far it got
            print(f"[datagen] layout {layout_id} failed with {stages_left} stages left", flush=True)
            continue

        kept_count += 1
        if args.no_record:
            print(f"[datagen] layout {layout_id} solved", flush=True)
            continue
        write_episode(frames, episode_dir / f"episode_{layout_id:07d}.hdf5")  # By layout, so runs never collide
        print(f"[datagen] layout {layout_id} kept {len(frames)} frames", flush=True)

    attempted = last_layout - args.first_layout
    verb = "solved" if args.no_record else f"written to {episode_dir}"
    print(f"[datagen] {kept_count}/{attempted} {verb}", flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)  # Kit's crash reporter otherwise keeps the process alive polling a display


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_name", type=str, default="cover_blocks", help="Task to generate episodes for.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Layouts to attempt, one episode each.")
    parser.add_argument("--first_layout", type=int, default=0, help="Layout to start from.")
    parser.add_argument("--seed", type=int, default=0, help="Layout set to draw from; 0 to 2 ship with the assets.")
    parser.add_argument("--output_dir", type=Path, default=Path("datagen_result"), help="Where episodes are written.")
    parser.add_argument(
        "--no_record", action="store_true", help="Report which layouts the expert solves without writing episodes."
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = AppLauncher(args).app  # create_env needs this before it builds anything
    main(args, simulation_app)
