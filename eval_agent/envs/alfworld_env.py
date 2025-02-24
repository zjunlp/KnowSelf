import re
import json
import logging
from typing import Any, Dict, List, Tuple

from eval_agent.envs import BaseEnv
from eval_agent.tasks import AlfWorldTask
from eval_agent.prompt import prompt_with_icl, prompt_without_icl
from eval_agent.utils.datatypes import State


logger = logging.getLogger("agent_frame")


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()

    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?([^\n]*)")
        action = re.findall(pattern, llm_output)[-1] # get last action
        assert action is not None
        punctuations = [".", "!", "?", ",", ";", ":"]
        while action and action[-1] in punctuations:
            action = action[:-1]
        return action.lower()
    
    def conduct_action(self, action: str):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output) # get action
            observation, reward, done = self.conduct_action(action) # conduct action
        except Exception as e:
            # logger.debug(f"Agent failed with error: {e}")
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state

        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = State()
        self.env.reset_states(game_files)
        self.state.error = self.task.game_file
        cur_task = self.task.observation
        # observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)
        observation, messages = prompt_without_icl(self.instruction, cur_task)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state


class BatchAlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.batch_size = batch_size
        self.state = [State() for i in range(batch_size)]

    def parse_action(self, llm_output: List[str]) -> List[str]:
        llm_output = [x.strip() for x in llm_output]
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = [re.findall(pattern, x)[0] for x in llm_output]
        # assert action is not None
        return action
    
    def conduct_action(self, actions: List[str]):
        observation, reward, done, info = self.env.step(actions)
        outputs = []
        for i in range(self.batch_size):
            observation, reward, done = process_ob(observation[i]), info['won'][i], done[i]
            outputs.append((observation, reward, done))
        return outputs
    
    def step(self, llm_output: List[str]) -> Tuple[str, State]:
        for i in range(self.batch_size):
            self.state[i].history.append({
                "role": "assistant",
                "content": llm_output[i]
            })
        actions = self.parse_action(llm_output)
        
        observations = {}
        correct_idx = []
        
        for i, action in enumerate(actions):
            if action is None:
                self.state[i].success = False
                self.state[i].finished = False
                self.state[i].reward=0
                observation = f"Observation: Error Input. Your input must contains 'Action: '"
                self.state[i].history.append({
                    "role": "user",
                    "content": observation,
                })
                self.state[i].steps += 1
                if self.state[i].steps >= self.max_steps:
                    self.state[i].finished = True
                    self.state[i].success = False
                    self.state[i].terminate_reason = "max_steps"
                    self.state[i].reward = 0
                actions[i] = ""
                observations[i] = observation
            else:
                correct_idx.append(i)
        outputs = self.conduct_action(actions)
        for i in correct_idx:
            observation, reward, done = outputs[i]
            observation = f"Observation: {observation}"
            self.state[i].history.append({
                "role": "user",
                "content": observation,
            })

            self.state[i].steps += 1
            if self.state[i].steps >= self.max_steps:
                self.state[i].finished = True
                self.state[i].success = False
                self.state[i].terminate_reason = "max_steps"
                self.state[i].reward = reward

            if done:
                self.state[i].finished = True
                self.state[i].success = True
                self.state[i].terminate_reason = "success"
                self.state[i].reward = reward
            observations[i] = observation

        return list(observations.values), self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = [State() for i in range(self.batch_size)]
        self.env.reset_states(game_files)
        cur_task = self.task.observation
        for i in range(self.batch_size):
            self.state[i].error = self.task.game_file
            obs = self.env.obs[i]
            obs = "\n".join(obs.split("\n\n")[1:])
            observation, messages = prompt_with_icl(self.instruction, self.raw_icl, obs, 1)
            # observation, messages = prompt_without_icl(self.instruction, obs)
            if self.icl_format == 'first':
                self.state[i].history.append({
                    "role": "user",
                    "content": observation,
                })
            elif self.icl_format == 'conversation':
                self.state[i].history = messages
        return observation, self.state