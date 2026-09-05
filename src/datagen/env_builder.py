"""Create a RoboDojo eval environment that runs without a policy server."""

from pathlib import Path
import types
from typing import Any

from omegaconf import OmegaConf

from env.global_configs import BENCHMARK, ROOT_DIR
import src.eval_client.eval_env as eval_env
from task.RoboDojo import task_registry
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization


def create_env(task_name: str, simulation_app: Any) -> Any:
    """Return an environment for one task, with a single env and a stubbed policy client."""
    # == Note ==
    # Config assembly copied from src/eval_client/main.py:253-338, minus resume handling.
    # Datagen differs from stock eval in four ways:
    #   - one env, since the skills drive env_idx 0 only
    #   - no policy server: WsModelClient is stubbed, deploy_cfg.port never dialled
    #   - policy_name only names the eval_result subdirectory (eval_env.py:84)
    #   - camera matrices collected: eval skips them, recording needs them
    # ==========
    eval_config_path = Path(ROOT_DIR) / "env_cfg"
    task_config_path = Path(ROOT_DIR) / "task" / BENCHMARK / "config"

    eval_cfg = load_yaml(eval_config_path / "arx_x5.yml")
    # num_envs feeds SeedManager, which has no default; policy_name names the output dir.
    eval_cfg.update(task_name=task_name, num_envs=1, policy_name="datagen")
    eval_cfg["observation"]["vision"].update(intrinsic_matrix=True, extrinsic_matrix=True)

    # arx_x5.yml names one YAML per section; load each by that name.
    config_sections = {
        section: load_yaml(eval_config_path / section / f"{eval_cfg['config'][section]}.yml")
        for section in ("sim", "scene", "camera", "robot")
    }
    config_sections["sim"]["scene"]["num_envs"] = 1  # sim_config.yml ships 10
    config_sections["sim"]["seed"] = [0]  # Absent from the YAML, read by base_env

    # One tree for the whole env: the section YAMLs, the task, and the eval and deploy blocks.
    env_cfg = OmegaConf.create(
        {
            **config_sections,
            "task_env": load_yaml(task_registry.task_config_path(task_config_path, task_name)),
            "eval_cfg": eval_cfg,
            "deploy_cfg": {"port": 0},  # eval_env.py:180 raises without one
        }
    )
    # Upstream passes, in main.py's order: randomization, then the task's own overrides.
    env_cfg = process_randomization(env_cfg)
    env_cfg, _ = process_config(env_cfg, task_name=task_name)

    # camera_manager UnboundLocalErrors without this. Must follow process_config, which
    # replaces env_cfg.camera outright for tasks naming their own camera config.
    OmegaConf.update(env_cfg, "camera.default_frequency", eval_cfg["observation"]["collect_freq"], force_add=True)

    # EvalEnv constructs this client and calls it on reset; no server answers either.
    # Patches the module for the whole process, which is what we want here.
    eval_env.WsModelClient = lambda *a, **k: types.SimpleNamespace(call=lambda **k: None, close=lambda: None)
    return eval_env.create_eval_env(env_cfg, simulation_app)
