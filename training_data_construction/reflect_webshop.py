import copy
import os

import json
import logging
import pathlib
import argparse
import re
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
from copy import deepcopy

# from webshop.web_agent_site.envs import WebAgentTextEnv
from glob import glob

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
import time
import pandas as pd

from os.path import join as pjoin

logger = logging.getLogger("agent_frame")

prompt = [
    {
        "role": "user",
        "content": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]",
    },
    {"role": "assistant", "content": "OK"},
]

two_shot = [
    [
        {
            "role": "user",
            "content": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search",
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]",
        },
        {
            "role": "user",
            "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L'eau D'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L'eau D'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L'Eau d'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L'Eau D'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L'eau D'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L'eau d'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L'eau d'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49",
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]",
        },
        {
            "role": "user",
            "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now",
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]",
        },
        {
            "role": "user",
            "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now",
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]",
        },
    ],
    [
        {
            "role": "user",
            "content": "WebShop [SEP] Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Search",
        },
        {
            "role": "assistant",
            "content": 'Thought: I need to find a grey sectional sofa that fits my living room, so I\'ll start by searching for "grey sectional sofa" to see the available options.\nAction: search[grey sectional sofa]',
        },
        {
            "role": "user",
            "content": 'Observation: Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09GVDTN7S [SEP] Modular Sectional Sofa with Ottoman L-Shaped Couch Reversible Sectional Corner Sofa 6 Seater Sofa Couches Living Room Furniture Sets,Dark Grey [SEP] $1112.0 [SEP] B09R4DNGFZ [SEP] L-Shaped Sectional Sofa Convertible Couch - Modular L Shape Sofa Couch with Chaise Lounge, Down Filled Sectional Sofa Overstuffed, Light Grey Sofa Living Room Furniture Set [SEP] $2099.0 [SEP] B01NADEYMV [SEP] Homelegance Pecos 105" x 117" Leather Gel Manual Reclining Sectional Sofa, Gray [SEP] $100.0 [SEP] B08QMWL422 [SEP] Sectional Sofa Set 52" Small Loveseat with Reversible Ottoman L-Shaped Modular Couch Chaise Loveseat Furniture Light Grey Sectional Couch for Living Room Bedroom Small Space [SEP] $490.0 [SEP] B07FMV7Y5M [SEP] Container Furniture Direct Amelie Linen Upholstered Contemporary Modern Right-Sided Sectional Sofa with Bed, 83.9", Dark Grey [SEP] $579.74 [SEP] B09QGJVJZQ [SEP] HONBAY Convertible Sectional Sofa, L Shape Sectional Couch with Reversible Chaise, 4 Seat Sectional Sofa for Living Room, Dark Grey [SEP] $799.99 [SEP] B09QL6XZQG [SEP] L-Shaped Reversible Sectional Sofa Sleeper - Reversible Sectional Sofa Sleeper 4-Seater Sofas Sectional, Small Apartment Sofabed Furniture, Grey Sectional [SEP] $550.0 [SEP] B0184US6HG [SEP] Moderne Livinf Modern Dark Small Space Reversible Linen Fabric Sectional Sofa in Color Light Grey, Beige, Red [SEP] $669.1 [SEP] B08NBBP225 [SEP] Casa Andrea Milano LLC Modern Sectional Sofa-Small Space Reversible Configurable Couch, Grey Microfiber [SEP] $486.06 [SEP] B092CJZDDZ [SEP] Casa Andrea Milano LLC Modern Sectional Sofa-Reversible Chaise Lounge Perfect for Small Space Dorm or Apartment, Grey Microfiber [SEP] $530.3',
        },
        {
            "role": "assistant",
            "content": "Thought: There are several options, but I'm looking for something affordable and suitable for a small space. The Sectional Sofa Set for $490.0 seems like a good fit for my requirements.\nAction: click[b08qmwl422]",
        },
        {
            "role": "user",
            "content": 'Observation: Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] beige [SEP] dark grey [SEP] dark grey  | 3 seat [SEP] grey [SEP] light grey [SEP] light grey  | 3 seat [SEP] Sectional Sofa Set 52" Small Loveseat with Reversible Ottoman L-Shaped Modular Couch Chaise Loveseat Furniture Light Grey Sectional Couch for Living Room Bedroom Small Space [SEP] Price: $490.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now',
        },
        {
            "role": "assistant",
            "content": "Thought: Before making a purchase, I should confirm the color option to ensure it's the grey I want for my living room.\nAction: click[grey]",
        },
        {
            "role": "user",
            "content": 'Observation: Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] beige [SEP] dark grey [SEP] dark grey  | 3 seat [SEP] grey [SEP] light grey [SEP] light grey  | 3 seat [SEP] Sectional Sofa Set 52" Small Loveseat with Reversible Ottoman L-Shaped Modular Couch Chaise Loveseat Furniture Light Grey Sectional Couch for Living Room Bedroom Small Space [SEP] Price: $490.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now',
        },
        {
            "role": "assistant",
            "content": "Thought: The color and price are right, and it fits the space I have. It's time to proceed with the purchase.\nAction: click[buy now]",
        },
    ],
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


def parse_webshop_action(llm_output: str) -> str:
    if "Action:" not in llm_output:
        logger.info(f"{Fore.RED}No 'Action:' parsing action: {llm_output}{Fore.RESET}")
        raise ValueError("No 'Action:' parsing action")

    llm_output = llm_output.strip()
    action_str = llm_output.split("Action:")[1].strip()
    action_str = action_str.lower()
    return action_str


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
    output_right_path = pathlib.Path.cwd() / "reflect" / "output" / task_name / "right"
    output_right_path.mkdir(exist_ok=True, parents=True)
    output_wrong_path = pathlib.Path.cwd() / "reflect" / "output" / task_name / "wrong"
    output_wrong_path.mkdir(exist_ok=True, parents=True)

    # pair path
    pair_path = args.pair_path
    pair_data = json.load(open(pair_path))

    reflect_right = 0
    reflect_wrong = 0
    reflect_right_list = []
    reflect_wrong_list = []
    n_tasks = len(pair_data)

    reflect_prompt = prompt + two_shot[0] + two_shot[1]
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)

        for idx, data in enumerate(pair_data):
            golden_action = data["golden_action"]
            explore_context = reflect_prompt + data["explore_traj"][:]

            temp_explore_context = copy.deepcopy(explore_context)

            explore_context.append(
                {
                    "role": "user",
                    "content": "OBSERVATION:\nThere are something wrong with your action. Your action was not actually executed successfully. Please reconsider your situation and change another action to complete the task.\nNote that you should align the content you click with the webpage.\nYour previous action is "
                    + parse_webshop_action(data["explore_action"])
                    + "\nPlease response strictly in the format:\n\nThought: Let's think step by step. <your thoughts>\nAction: <your next action>",
                }
            )

            for attemp in range(3):
                try:
                    logger.info(
                        f"{Fore.GREEN}explore_context {explore_context}{Fore.RESET}"
                    )
                    llm_output = agent(explore_context)
                    logger.info(f"{Fore.CYAN}llm_output {llm_output}{Fore.RESET}")

                    _explore_action = parse_webshop_action(llm_output)
                    _golden_action = parse_webshop_action(golden_action)
                    break
                except Exception as e:
                    logger.info(f"{Fore.RED}Error in task {idx}: {e}{Fore.RESET}")
            else:
                logger.info(f"{Fore.RED}Attemp over 3{Fore.RESET}")
                continue

            explore_context = temp_explore_context
            data["reflection"] = llm_output
            if _explore_action == _golden_action:
                logger.info(
                    f"{Fore.GREEN}Explore action is correct: {_explore_action} \nwhen golden action is {_golden_action} {Fore.RESET}"
                )
                reflect_right_list.append(data)
                reflect_right += 1
            else:
                logger.info(
                    f"{Fore.RED}Explore action is wrong: {_explore_action} \nwhen golden action is {_golden_action} {Fore.RESET}"
                )
                reflect_wrong_list.append(data)
                reflect_wrong += 1

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
        "--start",
        type=int,
        default=0,
        help="Start index of the task.",
    )
    parser.add_argument(
        "--type_name",
        type=str,
        default="",
        help="Type name of the task.",
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
