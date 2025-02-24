import os
import sys
import json
import yaml
import shutil
import openai
import argparse
import importlib
import alfworld
import alfworld.agents.environment
import traceback
from autobuild_utils import Rule_Manager, Skill_Bank

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["ALFWORLD_DATA"] = "../alfworld/downloaded"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["OPENAI_BASE_URL"]

ENV_NUM = 135

from io import StringIO
class CombinedLogger:
    def __init__(self, filename, stdout=None):
        self.file = open(filename, "a")
        self.terminal = sys.stdout
        self.stdout = stdout if stdout is not None else StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self
        return self.stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        self.file.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examine_tasks", type=str, default=None, help="The type of the examining task, 'pick_and_place', 'pick_clean_then_place', 'pick_heat_then_place', 'pick_cool_then_place', 'pick_two_obj', 'look_at_obj'")
    parser.add_argument('--num_env_per_task', type=int, default=5, help="the number of samples per task type.")
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--model_name", type=str, default='gpt-4-1106-preview', help="The LLM to use. One of 'gpt-4o', 'gpt-4-turbo', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106'")
    parser.add_argument('--assistant_api', action='store_true', help="use openai assistant api")
    parser.add_argument("--agent_type", type=str, default="replan", help="The type of the agent, prompt and examples")
    parser.add_argument("--simple_example", action='store_true', help="Use simplest example")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--start_idx", type=int, default=0, help="If resume, the start env")
    parser.add_argument("--pair_data_path", type=str, help="the path of the pair data")
    args = parser.parse_args()
    return args

issue_tasks = ["pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-26/trial_T20190907_113552_080857",
"pick_cool_then_place_in_recep-Pan-None-Cabinet-5/trial_T20190919_052755_008638",
"look_at_obj_in_light-TissueBox-None-DeskLamp-301/trial_T20190908_072624_007372"]

def get_result_dict(rule_manager):
    '''get the results of each task type from the history in rule_manager'''
    num_successes = 0
    result_dict = {}
    for k, v in rule_manager.global_history.items():
        if "epoch_" in k:
            task_type = v["task_type"]
            num_successes += v["is_success"]
            if task_type not in result_dict:
                result_dict[task_type] = [v["error_step"] if v["is_success"] else -1]
            else:
                result_dict[task_type].append(v["error_step"] if v["is_success"] else -1)
    print(result_dict)
    return num_successes, result_dict

def main(args):
    print("Use Assistant_API:", args.assistant_api)
    logging_dir = args.run_name
    if args.agent_type == "autobuild":
        from autobuild_trail import run
    else:
        raise ValueError(f"{args.agent_type} is invaild.")
    
    from prompts.autobuild_examples import init_rules

    if args.pair_data_path is not None:
        with open(args.pair_data_path) as reader:
            pair_data = json.load(reader)
        
    rule_manager = Rule_Manager(init_rules, save_path=logging_dir)
    skill_bank = Skill_Bank(save_path=logging_dir)

    if args.is_resume:
        # load environment configs
        rule_manager.load(save_path=logging_dir)
        skill_bank.load(save_path=logging_dir)
        epoch_id = rule_manager.cur_epoch
        if rule_manager.global_history[f"epoch_{epoch_id}"]['check_rule']: 
            epoch_id += 1
        num_successes, result_dict = get_result_dict(rule_manager)
    else:
        # Create the run directory
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)
        epoch_id = 0

    print(f"Sending all logs to: {logging_dir} (Run name). Start from epoch {epoch_id}.")
    # set paths to log files
    world_log_path = os.path.join(logging_dir, 'world.log')
    trial_log_path = os.path.join(logging_dir, 'trial.log')

    with open(world_log_path, 'a') as wf:
        wf.write(f'\n\n***** Start building *****\n\n')
    
    # run pair data
    for i, pair in enumerate(pair_data):
        idx = pair["idx"]
        id = pair["id"]
        id_num = int(id.split("_")[-1])
        name = pair['game_file'].split('/')[-2:]
        if name in issue_tasks or idx < args.start_idx: continue
        if args.examine_tasks is not None and all([(t not in name) for t in args.examine_tasks.split(",")]): continue
        # if task_type in result_dict and len(result_dict[task_type]) >= args.num_env_per_task: continue
        
        print(f"using {name}, Environment {id_num}, idx {idx}")
        print(f"starting epoch {epoch_id}")
        
        with CombinedLogger(trial_log_path) as s:
            try:
                consumed_tokens = run(args, epoch_id, pair, rule_manager, skill_bank)
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(f"An error occurred: {e}")
                print("Stack trace:")
                print(stack_trace)
                continue
            
        epoch_id += 1
        with open(world_log_path, 'a') as f:
            f.write(f"Epoch #{epoch_id}, Environment #{id_num}, idx #{idx}\n")
            f.write(consumed_tokens)
            
        # log env results to trial log
        with open(trial_log_path, 'a') as wf:
            wf.write(f'\n#####\n\nEpoch #{epoch_id}, Environment #{id_num}, idx #{idx}\n\n#####\n')

def get_train_list():
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    with open("base_config.yaml") as reader:
        config = yaml.safe_load(reader)
    raw_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="train")
    raw_env = raw_env.init_env(batch_size=1)
    
    train_dict = {}
    num_env_per_task = 9
    for z in range(135):
        ob, info = raw_env.reset()
        ob = '\n'.join(ob[0].split('\n\n')[1:]).replace("put a clean", "put a cleaned")
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
        print(f"using {name}, Environment {z}")
        if name in issue_tasks: continue

        task_type = name.split("-")[0]
        if task_type not in train_dict:
            train_dict[task_type] = [name]
        elif len(train_dict[task_type]) < num_env_per_task:
            train_dict[task_type].append(name)
        
        if len(train_dict)>=6 and all([len(v)>=num_env_per_task for v in train_dict.values()]):
            break
    
    with open("./train_list.json", 'w') as wf:            
        json.dump(train_dict, wf, indent=4)

if __name__ == '__main__':
    args = get_args()
    # get_train_list()
    main(args)
