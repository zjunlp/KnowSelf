from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
from copy import deepcopy
from os.path import join as pjoin

import eval_agent.agents as agents
from eval_agent.prompt import prompt_with_icl, prompt_without_icl

logger = logging.getLogger("agent_frame")


def conv_to_text(conv):
    return "\n".join(c["content"] for c in conv)


def format_input(task_desc, current_traj, golden_action, explore_action, rules):
    return (
        "The objectve:\n"
        + task_desc
        + "\n\nThe current trajectory:\n"
        + current_traj
        + "\n\nThe correct action:\n"
        + golden_action
        + "\n\nThe wrong action:\n"
        + explore_action
        + "\n\nThe rules:\n"
        + rules
    )


def cal_api(agent: agents.LMAgent):
    def cal_func(input):
        return agent(input)

    return cal_func


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
    output_path = pathlib.Path.cwd() / "training_data_construction" / "rule" / task_name
    output_path.mkdir(exist_ok=True, parents=True)

    # pair path
    pair_path = args.pair_path
    pair_data = json.load(open(pair_path))

    # instruction path
    instruction_path = args.inst_path
    with open(instruction_path) as f:
        instruction_data = f.read()
    # rule path
    rule_path = args.rule_path
    rule_data = json.load(open(rule_path))["all_rules"]
    rules = []
    for k, v in rule_data.items():
        rules.append(f"{k}: {v['rule']}")

    n_tasks = len(pair_data)
    result = []
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)
        start = args.start
        if start > 0:
            pair_data = pair_data[start:]
            result = json.load(open(pjoin(output_path, args.save_file_name)))
            result = result[:start]
        pbar.update(start)

        # Use ThreadPoolExecutor for parallel API calls
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}

            # First, build messages and submit tasks to call the API
            for data in pair_data:
                task_objective = (
                    data["golden_traj"][0]["content"]
                    .split("Instruction: [SEP]")[1]
                    .split("[SEP] Search")[0]
                    .strip()
                )
                current_traj = conv_to_text(data["golden_traj"][1:-1])
                golden_action = data["golden_action"]
                explore_action = data["explore_action"]

                rules = rules.__str__()
                cur_task = format_input(
                    task_objective, current_traj, golden_action, explore_action, rules
                )
                _, input = prompt_without_icl(instruction_data, cur_task)

                logger.info(f"{Fore.GREEN}input: {input}{Fore.RESET}")
                cal_func = cal_api(agent)
                futures[executor.submit(cal_func, input)] = deepcopy(data)

            with tqdm(total=len(futures)) as pbar:
                for future in as_completed(futures):
                    # Get the result of the API call
                    llm_output = future.result()
                    logger.info(f"{Fore.GREEN}LLM output: {llm_output}{Fore.RESET}")
                    rule = None
                    if "[Chosen Rule]:" in llm_output:
                        rule = llm_output.split("[Chosen Rule]:")[1].strip()
                    else:
                        pattern = re.compile(r"(rule_\d+\s*:\s*.+)\n")
                        match = pattern.search(llm_output)
                        if match:
                            rule = match.group(1)
                        else:
                            logger.info(
                                f"{Fore.RED}No rule found in the output.{Fore.RESET}"
                            )

                    result.append(
                        {**futures[future], "rule": rule}
                    )  # Combine query and workflow
                    logger.info(f"{Fore.GREEN}rule: {rule}{Fore.RESET}")
                    with open(pjoin(output_path, args.save_file_name), "w") as f:
                        json.dump(result, f, indent=4)
                    pbar.update(1)
                    print("===========================")

        # Write to file after processing
        # sort by id
        result = sorted(result, key=lambda x: int(x["idx"]))
        with open(pjoin(output_path, args.save_file_name), "w") as f:
            json.dump(result, f, indent=4)

        print("Saved reuslt.json successfully")


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
        "--start",
        type=int,
        default=0,
        help="Start index of the task.",
    )
    parser.add_argument(
        "--inst_path",
        type=str,
        default="training_data_construction/prompt/rule_select_inst_webshop.txt",
        help="Instruction path of selecting rules for WebShop.",
    )
    parser.add_argument(
        "--pair_path",
        type=str,
        help="Path of the pair data that needs to select rules.",
    )
    parser.add_argument(
        "--rule_path",
        type=str,
        help="Path of the rule (knowledge) base.",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        help="Name of the file to save the result.",
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
