import copy
import os
import re

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
import time
import pandas as pd

from os.path import join as pjoin

logger = logging.getLogger("agent_frame")

prompt = [
    {
        "role": "user",
        "content": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
    },
    {
        "role": "assistant",
        "content": "OK"
    }
]

two_shot = [
    [
        {
            "role": "user",
            "content": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search"
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]"
        },
        {
            "role": "user",
            "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L'eau D'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L'eau D'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L'Eau d'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L'Eau D'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L'eau D'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L'eau d'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L'eau d'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]"
        },
        {
            "role": "user",
            "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]"
        },
        {
            "role": "user",
            "content": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "role": "assistant",
            "content": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]"
        }
    ],
    [
        {
            "role": "user",
            "content": "WebShop [SEP] Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Search"
        },
        {
            "role": "assistant",
            "content": "Thought: I need to find a grey sectional sofa that fits my living room, so I'll start by searching for \"grey sectional sofa\" to see the available options.\nAction: search[grey sectional sofa]"
        },
        {
            "role": "user",
            "content": "Observation: Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B09GVDTN7S [SEP] Modular Sectional Sofa with Ottoman L-Shaped Couch Reversible Sectional Corner Sofa 6 Seater Sofa Couches Living Room Furniture Sets,Dark Grey [SEP] $1112.0 [SEP] B09R4DNGFZ [SEP] L-Shaped Sectional Sofa Convertible Couch - Modular L Shape Sofa Couch with Chaise Lounge, Down Filled Sectional Sofa Overstuffed, Light Grey Sofa Living Room Furniture Set [SEP] $2099.0 [SEP] B01NADEYMV [SEP] Homelegance Pecos 105\" x 117\" Leather Gel Manual Reclining Sectional Sofa, Gray [SEP] $100.0 [SEP] B08QMWL422 [SEP] Sectional Sofa Set 52\" Small Loveseat with Reversible Ottoman L-Shaped Modular Couch Chaise Loveseat Furniture Light Grey Sectional Couch for Living Room Bedroom Small Space [SEP] $490.0 [SEP] B07FMV7Y5M [SEP] Container Furniture Direct Amelie Linen Upholstered Contemporary Modern Right-Sided Sectional Sofa with Bed, 83.9\", Dark Grey [SEP] $579.74 [SEP] B09QGJVJZQ [SEP] HONBAY Convertible Sectional Sofa, L Shape Sectional Couch with Reversible Chaise, 4 Seat Sectional Sofa for Living Room, Dark Grey [SEP] $799.99 [SEP] B09QL6XZQG [SEP] L-Shaped Reversible Sectional Sofa Sleeper - Reversible Sectional Sofa Sleeper 4-Seater Sofas Sectional, Small Apartment Sofabed Furniture, Grey Sectional [SEP] $550.0 [SEP] B0184US6HG [SEP] Moderne Livinf Modern Dark Small Space Reversible Linen Fabric Sectional Sofa in Color Light Grey, Beige, Red [SEP] $669.1 [SEP] B08NBBP225 [SEP] Casa Andrea Milano LLC Modern Sectional Sofa-Small Space Reversible Configurable Couch, Grey Microfiber [SEP] $486.06 [SEP] B092CJZDDZ [SEP] Casa Andrea Milano LLC Modern Sectional Sofa-Reversible Chaise Lounge Perfect for Small Space Dorm or Apartment, Grey Microfiber [SEP] $530.3"
        },
        {
            "role": "assistant",
            "content": "Thought: There are several options, but I'm looking for something affordable and suitable for a small space. The Sectional Sofa Set for $490.0 seems like a good fit for my requirements.\nAction: click[b08qmwl422]"
        },
        {
            "role": "user",
            "content": "Observation: Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] beige [SEP] dark grey [SEP] dark grey  | 3 seat [SEP] grey [SEP] light grey [SEP] light grey  | 3 seat [SEP] Sectional Sofa Set 52\" Small Loveseat with Reversible Ottoman L-Shaped Modular Couch Chaise Loveseat Furniture Light Grey Sectional Couch for Living Room Bedroom Small Space [SEP] Price: $490.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "role": "assistant",
            "content": "Thought: Before making a purchase, I should confirm the color option to ensure it's the grey I want for my living room.\nAction: click[grey]"
        },
        {
            "role": "user",
            "content": "Observation: Instruction: [SEP] i am looking for a grey sectional sofa for my living room [SEP] Back to Search [SEP] < Prev [SEP] color [SEP] beige [SEP] dark grey [SEP] dark grey  | 3 seat [SEP] grey [SEP] light grey [SEP] light grey  | 3 seat [SEP] Sectional Sofa Set 52\" Small Loveseat with Reversible Ottoman L-Shaped Modular Couch Chaise Loveseat Furniture Light Grey Sectional Couch for Living Room Bedroom Small Space [SEP] Price: $490.0 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "role": "assistant",
            "content": "Thought: The color and price are right, and it fits the space I have. It's time to proceed with the purchase.\nAction: click[buy now]"
        }
    ]
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

def parse_webshop_action(llm_output: str) -> str:
    if "Action:" not in llm_output:
        logger.info(f"{Fore.RED}No 'Action:' parsing action: {llm_output}{Fore.RESET}")
        raise ValueError("No 'Action:' parsing action")
        
    llm_output = llm_output.strip()
    action_str = llm_output.split("Action:")[1].strip()
    action_str = action_str.lower()
    return action_str

def main(args: argparse.Namespace):
    logging.basicConfig(
        format="%(message)s"
    )
    
    # load the task config
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    task_name = args.exp_config
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    # get sft data
    golden_datas = json.load(open(args.sft_file_path, "r"))
    n_tasks = len(golden_datas)
    
    # save path
    pair_data_save_path = os.path.join(args.save_path, "pair_data")
    folder_path = os.path.join(pair_data_save_path, task_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if args.save_pair_file_name is None:
        save_pair_file_name = "pair_data.json"
    else:
        save_pair_file_name = args.save_pair_file_name

    save_path = os.path.join(folder_path, save_pair_file_name)

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)
        
        # start from the last saved idx
        start = args.start
        if start > 0:
            pair_data = json.load(open(save_path))
            pair_data = [item for item in pair_data if item['idx'] < start]
        else:
            pair_data = []

        # if limit the number of samples for each type (webshop has only one type)
        type_limit = args.type_limit
        if type_limit > 0:
            sample_num = 0
            if start > 0:
                sample_num = len(pair_data)
        
        # start sampling
        for idx, golden_data in enumerate(golden_datas):
            if idx < start:
                pbar.update(1)
                continue
            
            if type_limit > 0 and sample_num >= type_limit:
                break
            
            id = golden_data["id"]
            golden_traj = golden_data["conversations"][2:]
            golden_traj = template_change(golden_traj)
            golden_reward = golden_data["reward"]
            
            # filter out the bad samples
            if golden_reward < 0.5:
                pbar.update(1)
                continue
            
            # sample step by step
            for i in range(1, len(golden_traj), 2):
                for retry_count in range(3):
                    try:
                        step = int((i+1)/2)
                        
                        golden_context = golden_traj[:i]
                        
                        if args.two_shot:
                            input = prompt + two_shot[0] + two_shot[1] + golden_context
                        else:
                            input = prompt + golden_context
                            
                        logger.info(f"{Fore.GREEN}Input: {input}{Fore.RESET}")
                        
                        golden_action = golden_traj[i]["content"]
                        explore_action = agent(input)
                        
                        _golden_action = parse_webshop_action(golden_action)
                        _explore_action = parse_webshop_action(explore_action)
                        logger.info(f"{Fore.CYAN}Golden: {_golden_action}{Fore.RESET}")
                        logger.info(f"{Fore.CYAN}Explore: {_explore_action}{Fore.RESET}")
                    except Exception as e:
                        logger.error(f"{Fore.RED}Error: {e}{Fore.RESET}")
                    else:
                        break
                else:
                    logger.error(f"{Fore.RED}Failed to generate action after 3 attempts{Fore.RESET}")
                    continue
                
                # skip search action
                if "search" in _explore_action and "search" in _golden_action:
                    logger.info(f"{Fore.CYAN}Skip search action{Fore.RESET}")
                    continue
                
                # compare the actions
                if _golden_action == _explore_action:
                    logger.info(f"{Fore.GREEN}Explore Success{Fore.RESET}")
                    logger.info(f"{Fore.GREEN}Explore: {explore_action}\nSame as \nGolden: {golden_action}{Fore.RESET}")
                else:
                    # if not the same, save the pair data
                    logger.info(f"{Fore.CYAN}Explore Failed at:\n{explore_action}\nWhile Golden: {golden_action}{Fore.RESET}")
                    pair_data.append({
                        "id": id,
                        "step": step,
                        "golden_action": golden_action,
                        "explore_action": explore_action,
                        "golden_traj": golden_traj[:i+1],
                        "explore_traj": golden_traj[:i] + [{"role": "assistant", "content": explore_action}],
                        "full_golden_traj": golden_traj,
                        "idx": idx,
                        "reward": golden_reward
                    })
                    sample_num += 1
                    logger.info(f"{Fore.CYAN}Total samples: {sample_num}{Fore.RESET}")
                    json.dump(pair_data, open(save_path, "w"), indent=4)
            pbar.update(1)
        pbar.close()
        
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
        default="/data/qzs/wkm-p/data/webshop_sft.json",
        help="SFT file path."
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
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
        default=""
    )
    parser.add_argument(
        "--save_pair_file_name",
        type=str,
        default=""
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
