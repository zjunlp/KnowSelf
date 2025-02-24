import argparse
import json
from util import task_knowledge
import logging
import pathlib

logger = logging.getLogger("agent_frame")

aflworld_prompt = [
    {
        "role": "user",
        "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\nAction: your next action".\n\nThe available actions are:\n1. go to {recep}\n2. take {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. use {obj}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>',
    },
    {"role": "assistant", "content": "OK"},
]

task_prompt = {
    "pick_cool_then_place_in_recep": task_knowledge.alfworld_cool_prompt,
    "pick_two_obj_and_place": task_knowledge.alfworld_puttwo_prompt,
    "look_at_obj_in_light": task_knowledge.alfworld_examine_prompt,
    "pick_clean_then_place_in_recep": task_knowledge.alfworld_clean_prompt,
    "pick_and_place_simple": task_knowledge.alfworld_put_prompt,
    "pick_heat_then_place_in_recep": task_knowledge.alfworld_heat_prompt,
}


def format_knowledgeable_traj(golden_context, full_golden_traj, rule):
    full_golden_traj[len(golden_context) :][0]["content"] = (
        "[Knowledge]<knowledge>"
        + rule
        + "</knowledge>\n"
        + full_golden_traj[len(golden_context) :][0]["content"]
    )
    return golden_context + full_golden_traj[len(golden_context) :]


def format_reflect_traj(explore_context, full_golden_traj, reflection):
    if "Action:" in reflection:
        if "Thought:" in reflection:
            thought = reflection.split("Action:")[0].split("Thought:")[1].strip()
        else:
            thought = reflection.split("Action:")[0].strip()
        action = reflection.split("Action:")[1].strip()
    else:
        raise ValueError(
            f"Reflection does not contain thought and action: {reflection}"
        )

    explore_context[-1]["content"] += (
        "\n[Reflection]<reflection>Wait, let's check the action. "
        + thought
        + "</reflection>\n"
        + "Action: "
        + action
    )
    explore_context[-1]["content"] = explore_context[-1]["content"].replace(
        "\n\n", "\n"
    )
    return explore_context + full_golden_traj[len(explore_context) :]


def main(args: argparse.Namespace):
    save_file_name = args.save_file_name
    save_path = f"train/train_data"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_path}/{save_file_name}"

    with open(args.knowledgeable_data, "r") as f:
        knowledgeable_data = json.load(f)

    with open(args.reflect_data, "r") as f:
        reflect_data = json.load(f)

    with open(args.no_find_data, "r") as f:
        no_find_data = json.load(f)

    # generate knowledge data and reflect data
    total_data = knowledgeable_data + reflect_data
    total_data = sorted(total_data, key=lambda x: int(x["id"]))
    result = []
    for data in total_data:
        id = data["id"]
        if len(result) > 0 and result[-1]["id"] == id:
            data["full_golden_traj"] = result[-1]["full_golden_traj"]

        if "rule" in data:
            golden_traj = data["golden_traj"]
            if golden_traj[-1]["role"] != "user":
                golden_context = golden_traj[:-1]
            else:
                golden_context = golden_traj[:-2]

            full_golden_traj = data["full_golden_traj"]
            golden_context = full_golden_traj[: len(golden_context)]
            rule = data["rule"]
            if ": When" in rule:
                rule = "When" + rule.split(": When")[1]
            else:
                rule = rule
                # print(f"data id: {id}, rule: {rule}")
            train_data = format_knowledgeable_traj(
                golden_context, full_golden_traj, rule
            )
        else:
            explore_context = data["explore_traj"][:-1]
            full_golden_traj = data["full_golden_traj"]
            explore_context = (
                full_golden_traj[: len(explore_context) - 1] + explore_context[-1:]
            )
            reflection = data["reflection"]
            train_data = format_reflect_traj(
                explore_context, full_golden_traj, reflection
            )

        if len(result) > 0 and result[-1]["id"] == id:
            result[-1]["full_golden_traj"] = train_data
        else:
            result.append({"id": id, "full_golden_traj": train_data})

    # Consolidate with the original data
    pratial_data = result
    for d in pratial_data:
        d["conversations"] = aflworld_prompt + d.pop("full_golden_traj")

    # get the sft data
    total_data = json.load(open(args.sft_data))

    result = []
    for d in total_data:
        id = d["id"]
        for c in d["conversations"]:
            if c["from"] == "human":
                c["from"] = "user"
            if c["from"] == "gpt":
                c["from"] = "assistant"
            c["role"] = c.pop("from")
            c["content"] = c.pop("value")
        for p in pratial_data:
            if p["id"] == id:
                d["conversations"] = p["conversations"]
        for n in no_find_data:
            if n["id"] == id:
                if not d["conversations"][3]["content"].startswith("[Knowledge]"):
                    d["conversations"][3]["content"] = (
                        "[Knowledge]<knowledge>"
                        + task_prompt[n["task_type"]]
                        + "</knowledge>\n"
                        + d["conversations"][3]["content"]
                    )

    # Add task knowledge to the ignore data
    diff_ids = json.load(open("ignore_task_id.json"))["alfworld"]
    for d in total_data:
        if d["id"] in diff_ids:
            task_type = d["game_file"]
            prompt = None
            for k, v in task_prompt.items():
                if k in task_type:
                    prompt = v
            if prompt is not None:
                d["conversations"][3]["content"] = (
                    "[Knowledge]<knowledge>"
                    + prompt
                    + "</knowledge>\n"
                    + d["conversations"][3]["content"]
                )

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
    parser.add_argument(
        "--no_find_data",
        type=str,
        help="The no find target data that the agent cannot find the target.",
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
