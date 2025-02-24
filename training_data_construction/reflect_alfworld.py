import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.agents as agents

logger = logging.getLogger("agent_frame")

instruction = """
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
"""

PROMPT_WITH_ICL_TEMPLATE2 = """{instruction}

{base_prompt}

Now, it's your turn!
"""

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


alfworld_action_list = [
    "go",
    "take",
    "put",
    "open",
    "close",
    "toggle",
    "clean",
    "heat",
    "cool",
    "use",
]


# from sharegpt to conv
def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item["from"] == "gpt":
            message["role"] = "assistant"
            message["content"] = item["value"]
        else:
            message["role"] = "user"
            message["content"] = item["value"]
        messages.append(message)
    return messages


def find_original_game_file(core_part, all_game_files):
    for game_file in all_game_files:
        if core_part in game_file:
            return game_file
    return None


def get_prompt(
    task_type,
    prompts_file="knowledge_system_construction/data/alfworld_task_fewshots.json",
):
    category = None
    for k, v in PREFIXES.items():
        if k in task_type:
            category = v
            break

    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    with open(prompts_file, "r") as f:
        prompts_data = json.load(f)

    react_1_key = f"react_{category}_1"
    react_0_key = f"react_{category}_0"

    if react_1_key not in prompts_data or react_0_key not in prompts_data:
        raise ValueError(
            f"Keys {react_1_key} or {react_0_key} not found in prompts file."
        )

    base_prompt = (
        "Here are two examples.\n"
        + prompts_data[react_1_key]
        + "\n------\n"
        + prompts_data[react_0_key]
    )

    prompt = PROMPT_WITH_ICL_TEMPLATE2.format(
        instruction=instruction, base_prompt=base_prompt
    )
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": "OK"}]


def parse_alfworld_action_and_obj(llm_output: str) -> str:
    if "Action:" not in llm_output:
        logger.info(
            f"{Fore.RED}No 'Action:' parsing action and obj: {llm_output}{Fore.RESET}"
        )
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
        logger.info(
            f"{Fore.RED}Error in parsing action and obj: {llm_output}{Fore.RESET}"
        )
        raise ValueError("Error in parsing action and obj")


def main(args: argparse.Namespace):
    logging.basicConfig(format="%(message)s")
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config["config"]["model_name"] = args.model_name

    task_name = args.exp_config

    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    # set output path
    output_right_path = (
        pathlib.Path.cwd()
        / "training_data_construction"
        / "reflect"
        / task_name
        / "right"
    )
    output_right_path.mkdir(exist_ok=True, parents=True)
    output_wrong_path = (
        pathlib.Path.cwd()
        / "training_data_construction"
        / "reflect"
        / task_name
        / "wrong"
    )
    output_wrong_path.mkdir(exist_ok=True, parents=True)

    # pair path
    pair_path = args.pair_path
    pair_data = json.load(open(pair_path))

    reflect_right = 0
    reflect_wrong = 0
    reflect_right_list = []
    reflect_wrong_list = []
    n_tasks = len(pair_data)
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)

        for idx, data in enumerate(pair_data):
            try:
                golden_action = data["golden_action"]
                explore_context = get_prompt(data["task_type"]) + data["explore_traj"]
                temp_explore_context = explore_context[-1]["content"]

                # add prompt
                explore_context[-1][
                    "content"
                ] = "Observation: There are something wrong with your action. Your action was not actually executed successfully. Please reconsider your situation and change another action to complete the task. Please response strictly in the format:\n\nThought: Let's think step by step. <your thoughts>\nAction: <your next action>"

                llm_output = agent(explore_context)
                # logger.info(f"{Fore.CYAN}llm_output {llm_output}{Fore.RESET}")

                explore_context[-1]["content"] = temp_explore_context
                data["reflection"] = llm_output
                _explore_action, _explore_obj = parse_alfworld_action_and_obj(
                    llm_output
                )
                _golden_action, _golden_obj = parse_alfworld_action_and_obj(
                    golden_action
                )
                if _explore_action == _golden_action and _explore_obj == _golden_obj:
                    logger.info(
                        f"{Fore.GREEN}Explore action is correct: {llm_output} \nwhen golden action is {_golden_action} {golden_action} {Fore.RESET}"
                    )
                    reflect_right_list.append(data)
                    reflect_right += 1
                else:
                    logger.info(
                        f"{Fore.RED}Explore action is wrong: {llm_output} \nwhen golden action is {_golden_action} {golden_action} {Fore.RESET}"
                    )
                    reflect_wrong_list.append(data)
                    reflect_wrong += 1
            except Exception as e:
                logger.info(f"{Fore.RED}Error in task {idx}: {e}{Fore.RESET}")

            pbar.update(1)
            logger.info(
                f"Reflect right: {reflect_right}, Reflect wrong: {reflect_wrong}"
            )
        pbar.close()

    save_file_name = args.save_file_name
    with open(output_right_path / save_file_name, "w") as f:
        json.dump(reflect_right_list, f, indent=4)
    with open(output_wrong_path / save_file_name, "w") as f:
        json.dump(reflect_wrong_list, f, indent=4)
    logger.info(f"Save right data to {output_right_path / save_file_name}")
    logger.info(f"Save wrong data to {output_wrong_path / save_file_name}")


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
        help="Model name. It will override the 'model_name' in agent_config",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--pair_path",
        type=str,
        default="",
        help="Pair data path generated by sampling.",
    )
    parser.add_argument(
        "--save_file_name", type=str, default="", help="Save file name of reflect data."
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
