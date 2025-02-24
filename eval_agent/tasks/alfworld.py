import os
import json
import yaml
import logging
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment as envs

from eval_agent.tasks.base import Task


logger = logging.getLogger("agent_frame")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        env: envs.AlfredTWEnv,
        obs: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs

        self.env = env

    @classmethod
    def load_task(cls, env, game_file, batch_size: int = 1):
        
        # env = env.init_env(batch_size=batch_size)
        env.game_files = [game_file]
        obs, infos = env.reset()
        obs = "\n".join(obs[0].split("\n\n")[1:])
        return cls(
            task_id=0,
            game_file=game_file,
            env=env,
            obs=obs,
        )
        
        
    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1, batch_size: int = 1) -> Tuple[Iterable[Task], int]:
        os.environ["ALFWORLD_DATA"] = "eval_agent/data/alfworld"
        alfworld_data_path = os.environ.get("ALFWORLD_DATA")

        with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)
        
        if split == 'train':
            split = "train"
            N_TASKS = 3321
        elif split == 'dev':
            split = "eval_in_distribution"
            N_TASKS = 140
        elif split == 'test':
            split = "eval_out_of_distribution"
            N_TASKS = 134

        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.AlfredTWEnv)
        env = env.init_env(batch_size=batch_size)

        if part_num > 1:
            assert part_idx != -1
            per_part_num = N_TASKS // part_num + 1
            skip_num = per_part_num * part_idx
            env.skip(skip_num)
            N_TASKS = min(per_part_num, N_TASKS - skip_num)
            split_index = range(skip_num, skip_num + N_TASKS)
        else:
            split_index = range(N_TASKS)

        def generator():
            for idx in split_index:
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]

                yield cls(
                    task_id=idx,
                    game_file=game_file,
                    env=env,
                    obs=obs,
                )

        return generator(), N_TASKS
