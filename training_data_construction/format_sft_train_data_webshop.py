import argparse
import json
import logging
import pathlib
import re

logger = logging.getLogger("agent_frame")

webshop_prompt = [
    {
        "role": "user",
        "content": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
    },
    {
        "role": "assistant",
        "content": "OK"
    },
]

def format_knowledgeable_traj(golden_context, full_golden_traj, rule):
    if "[Knowledge]<knowledge>" in full_golden_traj[len(golden_context):][0]["content"]:
        return golden_context + full_golden_traj[len(golden_context):]
    full_golden_traj[len(golden_context):][0]["content"] = "[Knowledge]<knowledge>" + rule + "</knowledge>\n" + full_golden_traj[len(golden_context):][0]["content"]
    return golden_context + full_golden_traj[len(golden_context):]

def format_reflect_traj(explore_context, full_golden_traj, reflection):
    if "Action:" in reflection:
        if "Thought:" in reflection:
            thought = reflection.split("Action:")[0].split("Thought:")[1].strip()
        else:
            thought = reflection.split("Action:")[0].strip()
        action = reflection.split("Action:")[1].strip()
    else:
        raise ValueError(f"Reflection does not contain thought and action: {reflection}")
    
    explore_context[-1]["content"] += "\n[Reflection]<reflection>Wait, let's check the action. " + thought + "</reflection>\n" + "Action: " + action
    explore_context[-1]["content"] = explore_context[-1]["content"].replace("\n\n", "\n")
    return explore_context + full_golden_traj[len(explore_context):]

def main(args: argparse.Namespace):
    save_file_name = args.save_file_name
    save_path = f"train/train_data"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_path}/{save_file_name}"

    with open(args.knowledgeable_data, "r") as f:
        knowledgeable_data = json.load(f)

    with open(args.reflect_data, "r") as f:
        reflect_data = json.load(f)

    # generate knowledge data and reflect data
    total_data = knowledgeable_data + reflect_data
    total_data = sorted(total_data, key=lambda x: int(x["id"]))
    result = []
    for data in total_data:
        id = data["id"]
        if len(result) > 0 and result[-1]["id"] == id:
            data["full_golden_traj"] = result[-1]["full_golden_traj"]
        
        if "rule" in data:
            golden_traj = data['golden_traj']
            if golden_traj[-1]['role'] != "user":
                golden_context = golden_traj[:-1]
            else:
                golden_context = golden_traj[:-2]
            
            full_golden_traj = data["full_golden_traj"]
            golden_context = full_golden_traj[:len(golden_context)]
            rule = data["rule"]
            if ": When" in rule:
                rule = "When" + rule.split(": When")[1]
            else:
                rule = rule
                print(f"data id: {id}, rule: {rule}")
            train_data = format_knowledgeable_traj(golden_context, full_golden_traj, rule)
        else:
            explore_context = data["explore_traj"][:]
            full_golden_traj = data["full_golden_traj"]
            explore_context = full_golden_traj[:len(explore_context)]
            reflection = data["reflection"]
            train_data = format_reflect_traj(explore_context, full_golden_traj, reflection)
            
        if len(result) > 0 and result[-1]["id"] == id:
            result[-1]["full_golden_traj"] = train_data
        else:
            result.append({
                "id": id,
                "full_golden_traj": train_data
            })

    # get the sft data
    total_data = json.load(open(args.sft_data))

    for d in total_data:
        conv = d["conversations"]
        for c in conv:
            c["role"] = "user" if c["from"] == "human" else "assistant"
            c.pop("from")
            c["content"] = c.pop("value")
    
    for d in total_data:
        for p in result:
            if p["id"] == d["id"]:
                d["conversations"] = webshop_prompt + p["full_golden_traj"]
                break
            
    json.dump(total_data, open(save_path, "w"), indent=4)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--knowledgeable_data",
        type=str,
        help="The knowledgeable data generated by rule selection.",
    )
    parser.add_argument(
        "--reflect_data",
        type=str,
        help="The reflect data generated by successful reflection.",
    )
    parser.add_argument("--sft_data", type=str, help="The sft data path of the task.")
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
