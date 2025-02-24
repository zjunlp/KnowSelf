import copy
import os

import alfworld
import yaml

import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
from copy import deepcopy

from glob import glob

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
from knowledge_system_construction.state import AlfworldState
import time
import pandas as pd

from os.path import join as pjoin

logger = logging.getLogger("agent_frame")

instruction='''
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.


Your response should use the following format exactly and must always contain both lines:

Thought: <your thoughts here>
Action: <your next action here>

Do not only produce the Action line. You must always produce a Thought line followed immediately by an Action line. Never omit the Thought line. Never produce only the Action line. Generating only the Action is not allowed. No other lines or text should be produced. Please only provide the Thought and Action, do not generate Observation yet.
'''

PROMPT_WITH_ICL_TEMPLATE2 = """{instruction}

{base_prompt}

Now, it's your turn!
"""

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

alfworld_action_list = [
    "go",
    "take",
    "put",
    "open",
    "close",
    "clean",
    "heat",
    "cool",
    "use",
    "examine",
]

# from sharegpt to conv
def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['from'] == "gpt":
            message['role'] = "assistant"
            message['content'] = item['value']
        else:
            message['role'] = "user"
            message['content'] = item['value']
        messages.append(message)
    return messages

def find_original_game_file(core_part, all_game_files):
    for game_file in all_game_files:
        if core_part in game_file:
            return game_file
    return None

def get_prompt(task_type, prompts_file = "knowledge_system_construction/data/alfworld_task_fewshots.json", two_shot=True):
    category = None
    for k, v in PREFIXES.items():
        if k in task_type:
            category = v
            break
        
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    with open(prompts_file, 'r') as f:
        prompts_data = json.load(f)

    react_1_key = f'react_{category}_1'
    react_0_key = f'react_{category}_0'
    
    if react_1_key not in prompts_data or react_0_key not in prompts_data:
        raise ValueError(f"Keys {react_1_key} or {react_0_key} not found in prompts file.")
    
    base_prompt = (
            'Here are two examples.\n'
            + prompts_data[react_1_key]
            + '\n------\n'
            + prompts_data[react_0_key]
    )
    
    prompt = PROMPT_WITH_ICL_TEMPLATE2.format(instruction=instruction, base_prompt=base_prompt)
    if two_shot:
        return [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "OK"
            }
        ]
    else:
        return [
            {
                "role": "user",
                "content": instruction
            },
            {
                "role": "assistant",
                "content": "OK"
            }
        ]

def parse_alfworld_action_and_obj(llm_output: str) -> str:
    if "Action:" not in llm_output:
        logger.info(f"{Fore.RED}No 'Action:' parsing action and obj: {llm_output}{Fore.RESET}")
        raise ValueError("No 'Action:' parsing action and obj")
    try:
        llm_output = llm_output.strip()
        action_str = llm_output.split("Action:")[1].strip()
        action = action_str.split(" ")[0].strip()

        obj = action_str.split(" ")[1].strip()
        if action == "go":
            obj = action_str.split(" ")[2].strip()

        return action, obj
    except Exception as e:
        logger.info(f"{Fore.RED}Error in parsing action and obj: {llm_output}{Fore.RESET}")
        raise ValueError(f"Error {e} in parsing action and obj, llm_output: {llm_output}")


def main(args: argparse.Namespace):
    logging.basicConfig(
        format="%(message)s"
    )
    
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    env_config = exp_config["env_config"]
    task_name = args.exp_config

    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    ignore_task_id = json.load(open("ignore_task_id.json"))
    ignore_task_ids = ignore_task_id[task_name]

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks("train", 1, -1)

    # get all game_files
    all_game_files = json.load(open("alfworld_game_files.json"))

    # get sft data
    golden_data = json.load(open(args.sft_file_path, "r"))

    # save pair data path
    pair_data_save_path = os.path.join(args.save_path, "pair_data")
    folder_path = os.path.join(pair_data_save_path, task_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if args.save_pair_file_name is None:
        save_pair_file_name = "pair_data.json"
    else:
        save_pair_file_name = args.save_pair_file_name
    save_path = os.path.join(folder_path, save_pair_file_name)
    
    # save no find target object data path
    if args.save_no_find_data:
        pair_data_save_path = os.path.join(args.save_path, "no_find_data")
        no_find_data_folder_path = os.path.join(pair_data_save_path, task_name)
        if not os.path.exists(no_find_data_folder_path):
            os.makedirs(no_find_data_folder_path)
        if args.save_no_find_data_file_name is None:
            save_no_find_data_file_name = "no_find_data.json"
        else:
            save_no_find_data_file_name = args.save_no_find_data_file_name
        no_find_target_file = pjoin(no_find_data_folder_path, save_no_find_data_file_name)
    
    # no_find_target_file = pjoin(folder_path, "no_find_target_gemma_1223.json")
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)
        
        # start from the last saved idx
        start = args.start
        if start > 0:
            pair_data = json.load(open(save_path))
            pair_data = [item for item in pair_data if item['idx'] < start]
            
            no_find_target_data = json.load(open(no_find_target_file))
            no_find_target_data = [item for item in no_find_target_data if item['idx'] < start]
        else:
            pair_data = []
            no_find_target_data = []

        # if type_limit > 0, limit the number of each type
        type_limit = args.type_limit
        if type_limit > 0:
            type_num_dict = {}
            if start > 0:
                for d in pair_data:
                    task_type = d["game_file"]
                    task_type = task_type.split("/")[-2].split("-")[0]
                    if task_type not in type_num_dict:
                        type_num_dict[task_type] = 1
                    else:
                        type_num_dict[task_type] += 1
        
        # start to explore
        for idx, task in enumerate(all_tasks):
            # skip the tasks before start
            if idx < start:
                pbar.update(1)
                continue
            
            # if the current task is found in the golden data
            found = False  
            for k, golden_d in enumerate(golden_data):
                game_file = task.game_file.split("/")[-3] + "/" + task.game_file.split("/")[-2]
                golden_game_file = golden_d['game_file'].split("/")[-2] + "/" + golden_d['game_file'].split("/")[-1]
                if game_file == golden_game_file:
                    logger.info(f"{Fore.GREEN}Task {task.game_file} is in the golden data{Fore.RESET}")
                    found = True  # find the task in the golden data
                    break

            if not found:
                logger.info(f"{Fore.RED}Task {task.game_file} is not in the golden data{Fore.RESET}")
                pbar.update(1)
                continue
            
            # get the current task data
            current_data = golden_data[k]
            id = current_data['id']
            
            # skip the ignore task
            if id in ignore_task_ids:
                pbar.update(1)
                continue
            
            task_type = '/'.join(task.game_file.split('/')[-3:-1])
            task_type = task_type.split("-")[0]
            logger.info(f"{Fore.WHITE}Now current task_type is {task_type}{Fore.RESET}")

            # limit the number of each type
            if type_limit > 0:
                if sum(type_num_dict.values()) >= type_limit*6:
                    logger.info(f"{Fore.GREEN}All types have reached the limit{Fore.RESET}")
                    break
                if task_type in type_num_dict and type_num_dict[task_type] >= type_limit:
                    pbar.update(1)
                    continue

            logger.info(f"{Fore.WHITE}Now current task is {id}{Fore.RESET}")
            logger.info(f"{Fore.WHITE}Now current idx is {idx}{Fore.RESET}")
            
            # get the golden trajectory
            golden_traj = current_data['conversations'][2:]
            golden_traj = template_change(golden_traj)

            # initialize the environment
            explore_env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)


            game_file = current_data['game_file']
            core_part = "/".join(game_file.split("/")[3:])
            original_game_file = find_original_game_file(core_part, all_game_files)
            if original_game_file is None:
                logging.info(f"{Fore.RED}Cannot find original game file for {game_file}{Fore.RESET}")
                pbar.update(1)
                continue
            explore_observation, explore_state = explore_env.reset()

            # initialize the alfworld state
            golden_aflworld_state = AlfworldState()
            golden_aflworld_state.parse_reachable_locs(golden_traj[0]['content'])

            # get the task objective
            task_objective = golden_traj[0]['content'].split("Your task is to:")[-1].strip()
            
            # filter the error step
            golden_error = False
            for i in range(1, len(golden_traj)-1, 2):
                try:
                    golden_action = golden_traj[i]['content']
                    golden_observation = golden_traj[i+1]['content']
                    golden_aflworld_state.transition(golden_aflworld_state.parse_action(golden_action, golden_observation))
                except Exception as e:
                    logger.info(f"{Fore.RED}Error in golden step{i}: {e}{Fore.RESET}")
                    golden_error = True
                    break
            if golden_error:
                pbar.update(1)
                continue
            
            # reset the alfworld state
            golden_aflworld_state.reset()
            golden_aflworld_state.parse_reachable_locs(golden_traj[0]['content'])
            
            # skip the search action (no obj in hand)
            golden_step = 0
            for i in range(1, len(golden_traj), 2):
                golden_action = golden_traj[i]['content']
                golden_observation = golden_traj[i+1]['content']
                golden_aflworld_state.transition(golden_aflworld_state.parse_action(golden_action, golden_observation))
                if golden_aflworld_state.obj_in_hand is not None:
                    golden_step = i
                    break
            
            # examine if explore can find the target in 40 steps
            if args.save_no_find_data:
                explore_env.state.history[0] = get_prompt(task_type=task_type, two_shot=args.two_shot)[0]
                find_target = False

                for i in range(40):
                    explore_action: str = agent(explore_env.state.history)
                    logger.info(f"{Fore.GREEN}Find Target Stage, Step {i}, Action {explore_action}{Fore.RESET}")
                    explore_observation, explore_state = explore_env.step(explore_action)
                    try:
                        _explore_action, _explore_obj = parse_alfworld_action_and_obj(explore_action)
                        _golden_action, _golden_obj = parse_alfworld_action_and_obj(golden_action)
                    except Exception as e:
                        logger.info(f"{Fore.RED}Error {e}{Fore.RESET}")
                        continue
                    if _explore_action == _golden_action and _explore_obj == _golden_obj:
                        logger.info(f"{Fore.CYAN}Explore: {explore_action}\nSame as \nGolden:{golden_action}{Fore.RESET}")
                        find_target = True
                        break
                    else:
                        logger.info(f"{Fore.RED}Explore: {explore_action}\nDifferent from \nGolden:{golden_action}{Fore.RESET}")
                    if explore_state.finished and explore_state.success:
                        logger.info(f"{Fore.GREEN}Explore success{Fore.RESET}")
                        find_target = True
                        break
                if not find_target:
                    logger.info(f"{Fore.RED}Explore failed, No find target{Fore.RESET}")
                    no_find_target_data.append({
                        "id": id,
                        "step": 0,
                        "task_type": task_type,
                        "golden_action": golden_action,
                        "golden_observation": golden_observation,
                        "explore_action": explore_action,
                        "explore_observation": explore_observation,
                        "golden_traj": golden_traj,
                        "explore_traj": explore_env.state.history[2:],
                        "full_golden_traj": golden_traj,
                        "game_file": game_file,
                        "idx": idx
                    })
                    json.dump(no_find_target_data, open(no_find_target_file, "w"), indent=4)
            
            # begin to explore step by step
            for i in range(golden_step, len(golden_traj), 2):
                step = int(i/2) + 1
                logger.info(f"{Fore.RED}Start step {step}{Fore.RESET}")
                
                # get the historical context
                golden_context = golden_traj[:i]
                explore_env.reset()
                explore_env.state.history[0] = get_prompt(task_type=task_type, two_shot=args.two_shot)[0]
                
                golden_aflworld_state.reset()
                golden_aflworld_state.parse_reachable_locs(golden_context[0]['content'])
                
                # update the alfworld state and explore environment
                for j in range(1, len(golden_context), 2):
                    golden_action = golden_context[j]['content']
                    golden_observation = golden_context[j+1]['content']
                    explore_env.step(golden_action)
                    golden_aflworld_state.transition(golden_aflworld_state.parse_action(golden_action, golden_observation))

                # get the current golden action and observation
                golden_action = golden_traj[i]['content']
                if i == len(golden_traj) - 1:
                    golden_observation = "Task finished"
                else:
                    golden_observation = golden_traj[i+1]['content']
                
                # skip the search action (find the second obj in puttwo task and find desklamp in examine task)
                find_target = False
                if ("two" in task_objective and golden_aflworld_state.obj_in_hand is None and "Action: take" not in golden_action) or \
                    ("desklamp" in task_objective and "Action: go" in golden_action and ("desk" in golden_action.lower() or "lamp" in golden_action.lower())):
                    if not args.save_no_find_data:
                        logger.info(f"{Fore.GREEN}Jump if no obj in hand{Fore.RESET}")
                        logger.info(f"{Fore.GREEN}golden_action {golden_action}{Fore.RESET}")
                        continue
                    else:
                        if find_target == True:
                            logger.info(f"{Fore.GREEN}Jump if no obj in hand{Fore.RESET}")
                            logger.info(f"{Fore.GREEN}golden_action {golden_action}{Fore.RESET}")
                            continue
                        else:
                            for k in range(40 - step):
                                explore_action: str = agent(explore_env.state.history)
                                logger.info(f"{Fore.GREEN}Find Target Stage, Step {k}, Action {explore_action}{Fore.RESET}")
                                explore_observation, explore_state = explore_env.step(explore_action)
                                if "two" in task_objective and "Observation: You pick up" in explore_observation:
                                    logger.info(f"{Fore.CYAN}Find Target Success: {explore_action}{Fore.RESET}")
                                    logger.info(f"{Fore.CYAN}Take Observation: {explore_observation}{Fore.RESET}")
                                    find_target = True
                                    break
                                if "desklamp" in task_objective and "desklamp" in explore_observation:
                                    logger.info(f"{Fore.CYAN}Find Desklamp Success: {explore_action}{Fore.RESET}")
                                    logger.info(f"{Fore.CYAN}Take Observation: {explore_observation}{Fore.RESET}")
                                    find_target = True
                                    break
                                if explore_state.finished and explore_state.success:
                                    logger.info(f"{Fore.GREEN}Explore success{Fore.RESET}")
                                    find_target = True
                                    break
                            if not find_target:
                                logger.info(f"{Fore.RED}Explore failed, No find target{Fore.RESET}")
                                traj_end = i+2 if i+2 < len(golden_traj) else len(golden_traj)
                                no_find_target_data.append({
                                    "id": id,
                                    "step": step,
                                    "task_type": task_type,
                                    "golden_action": golden_action,
                                    "golden_observation": golden_observation,
                                    "explore_action": explore_action,
                                    "explore_observation": explore_observation,
                                    "golden_traj": golden_traj[:traj_end],
                                    "explore_traj": explore_env.state.history[2:],  # exclude the prompt
                                    "full_golden_traj": golden_traj,
                                    "game_file": game_file,
                                    "idx": idx
                                })
                                json.dump(no_find_target_data, open(no_find_target_file, "w"), indent=4)
                                find_target = True
                            continue
                
                # get the prompt
                prompt = get_prompt(task_type=task_type, two_shot=args.two_shot)
                explore_prompt = prompt + golden_context
                explore_action = agent(explore_prompt)

                # parse action_str to action and obj
                try:
                    _explore_action, _explore_obj = parse_alfworld_action_and_obj(explore_action)
                    _golden_action, _golden_obj = parse_alfworld_action_and_obj(golden_action)
                except Exception as e:
                    logger.info(f"{Fore.RED}Error {e}{Fore.RESET}")
                    continue
                
                # skip close action
                if _golden_action == "close" and _explore_action != "close":
                    logger.info(f"{Fore.GREEN}Jump close action{Fore.RESET}")
                    continue
                
                # change the template of put
                if _explore_action == "put":
                    if " in " in explore_action:
                        explore_action = explore_action.replace(" in ", " in/on ")
                    elif " on " in explore_action:
                        explore_action = explore_action.replace(" on ", " in/on ")

                # get the explore observation and state
                explore_observation, explore_state = explore_env.step(explore_action)

                # cool or heat task
                if "cool" in task_objective or "heat" in task_objective or "hot" in task_objective or "cold" in task_objective:
                    # skip the cool/heat action if the agent successfullly cool/heat the object
                    if "You cool" in explore_observation or "You heat" in explore_observation:
                        logger.info(f"{Fore.GREEN}Jump cool/heat action{Fore.RESET}")
                        continue
                    # skip the (open, cool/heat) or (cool/heat, open) action
                    if _golden_action == "open" and _golden_obj in ["fridge", "microwave"]:
                        if _explore_action in ["cool", "heat"] and "Nothing happens" not in explore_observation:
                            logger.info(f"{Fore.GREEN}Jump open action when explore is cool/heat{Fore.RESET}")
                            continue
                    elif _golden_action in ["cool", "heat"]:
                        if _explore_action == "open" and _explore_obj in ["fridge", "microwave"] and "Nothing happens" not in explore_observation:
                            logger.info(f"{Fore.GREEN}Jump cool/heat action when explore is open{Fore.RESET}")
                            continue
                
                # skip the clean action if the agent successfullly clean the object
                if "clean" in task_objective:
                    if "You clean" in explore_observation:
                        logger.info(f"{Fore.GREEN}Jump clean action{Fore.RESET}")
                        continue
                
                # skip the success action
                if explore_state.finished and explore_state.success:
                    logger.info(f"{Fore.GREEN}Explore success{Fore.RESET}")
                    break
                
                # compare the explore action with the golden action, if not same, save the data
                if _explore_action == _golden_action and _explore_obj == _golden_obj:
                    logger.info(f"{Fore.CYAN}Explore Success{Fore.RESET}")
                    logger.info(f"{Fore.CYAN}Explore: {explore_action}\nSame as \nGolden:{golden_action}{Fore.RESET}")
                else:
                    logger.info(f"{Fore.CYAN}Explore Failed at:\n{explore_action}{Fore.RESET}")
                    traj_end = i+2 if i+2 < len(golden_traj) else len(golden_traj)
                    pair_data.append({
                        "id": id,
                        "step": step,
                        "task_type": task_type,
                        "golden_action": golden_action,
                        "golden_observation": golden_observation,
                        "explore_action": explore_action,
                        "explore_observation": explore_observation,
                        "golden_traj": golden_traj[:traj_end],
                        "explore_traj": explore_env.state.history[2:],  # exclude the prompt
                        "full_golden_traj": golden_traj,
                        "game_file": game_file,
                        "idx": idx
                    })
                    
                    # update the type num dict
                    if type_limit > 0:
                        task_type = '/'.join(task.game_file.split('/')[-3:-1])
                        task_type = task_type.split("-")[0]
                        if task_type not in type_num_dict:
                            type_num_dict[task_type] = 1
                        else:
                            type_num_dict[task_type] += 1
                        logger.info(f"{Fore.GREEN}Type {task_type} now num is {type_num_dict[task_type]}{Fore.RESET}")
                    json.dump(pair_data, open(save_path, "w"), indent=4)

            logger.info(f"{Fore.WHITE}Task {id} finished{Fore.RESET}")
            pbar.update(1)
        pbar.close()
        
        # save the data
        json.dump(pair_data, open(save_path, "w"), indent=4)
        logger.info(f"{Fore.GREEN}All tasks done{Fore.RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="Llama-2-7b-hf-alfworld-sft-explore-0",
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--type_limit",
        type=int,
        default=0
    )
    parser.add_argument(
        "--sft_file_path",
        type=str,
        default="/data/qzs/wkm-p/alfworld_sft.json",
        help="SFT file path."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of the task.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/qzs/wkm-p/pair_data"
    )
    parser.add_argument(
        "--save_pair_file_name",
        type=str,
        default="the save name of the pair data file"
    )
    parser.add_argument(
        "--save_no_find_data",
        action="store_true",
        help="Whether to save the data that the agent cannot find the target object"
    )
    parser.add_argument(
        "--save_no_find_data_file_name",
        type=str,
        default="the save name of the data that the agent cannot find the target object"
    )
    parser.add_argument(
        "--two_shot",
        action="store_true",
        help="Whether to use two-shot prompt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    main(args)