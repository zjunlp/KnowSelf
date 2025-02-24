import argparse
import json
import logging
import glob

logger = logging.getLogger("agent_frame")

def main(args: argparse.Namespace):
    data_files = glob.glob(args.output_path + "/*.json")
    golden_datas = json.load(open(args.golden_data_path))

    total_data = []

    for file in data_files:
        explore_data = json.load(open(file))
        if explore_data["meta"]["reward"] == None:
            continue
        
        explore_game_file = (
            explore_data["game_file"].split("/")[-3] + "/" + explore_data["game_file"].split("/")[-2]
        )

        for golden_data in golden_datas:
            golden_game_file = (
                golden_data["game_file"].split("/")[-2] + "/" + golden_data["game_file"].split("/")[-1]
            )
            if golden_game_file == explore_game_file:
                if explore_data["meta"]["reward"] < golden_data["reward"]:
                    for k in explore_data["conversations"]:
                        k["role"] = "user" if k["from"] == "human" else "assistant"
                        k.pop("from")
                        k["content"] = k.pop("value")
                    
                    total_data.append(
                        {
                            "id": id,
                            "game_file": explore_game_file,
                            "conversations": golden_data["conversations"][:2],
                            "chosen": golden_data["conversations"][2:],
                            "rejected": explore_data["conversations"][2:],
                            "from": "golden"
                        }
                    )
                break

    total_data.sort(key=lambda x: int(x["id"]))
    json.dump(total_data, open(args.save_file_name, "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--output_path",
        type=str,
        help="The output directory of the data.",
    )
    parser.add_argument(
        "--golden_data_path",
        type=str,
        help="The golden data path.",
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
