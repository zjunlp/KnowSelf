import os
import sys
import json
import shutil
import openai
import argparse
import openai
import random
import tempfile
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.base_url = os.environ["OPENAI_BASE_URL"]

from autobuild_utils import Rule_Manager, Skill_Bank

ENV_NUM = 200

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
    parser.add_argument("--run_name", type=str, default="test_run", help="The name of the run")
    parser.add_argument("--examine_sites", type=str, default=None, help="The sites for examining")
    parser.add_argument('--num_env_per_task', type=int, default=7, help="the number of samples per task type.")
    parser.add_argument("--model_name", type=str, default='gpt-4-1106-preview', help="The LLM to use. One of 'gpt-4o', 'gpt-4-turbo', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106'")
    parser.add_argument('--assistant_api', action='store_true', help="use openai assistant api")
    parser.add_argument("--agent_type", type=str, default="replan", help="The type of the agent, prompt and examples")
    # parser.add_argument("--simple_example", action='store_true', help="Use simplest example")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    parser.add_argument("--start_env_id", type=int, default=0, help="If resume, the start env")
    parser.add_argument("--split", type=int, default=None, help="The x-th split of the examine site")
    parser.add_argument("--headless", action='store_true', help="using headless mode or not")
    parser.add_argument("--slow_mo", type=int, default=0, help="Slow down the browser by the specified amount")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--action_set_tag", default="id_accessibility_tree", help="Action type")
    parser.add_argument("--observation_type", choices=["accessibility_tree", "html", "image"], default="accessibility_tree", help="Observation type")
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--pair_data_path", type=str, help="the path of the pair data")
    args = parser.parse_args()
    return args

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
    print("result_dict:", result_dict)
    return num_successes, result_dict

def main(args):
    print("Use Assistant_API:", args.assistant_api)
    logging_dir = args.run_name
    if args.agent_type == "autobuild":
        from autobuild_trail import run
    else:
        raise ValueError(f"{args.agent_type} is invaild.")
    
    if args.pair_data_path is not None:
        with open(args.pair_data_path) as reader:
            pair_data = json.load(reader)

    from prompts.autobuild_examples import init_rules
    
    rule_manager = Rule_Manager(init_rules=init_rules, save_path=logging_dir)
    skill_bank = Skill_Bank(save_path=logging_dir)

    if args.is_resume:
        # load existing rule_manager and skill_bank
        rule_manager.load(save_path=logging_dir)
        if args.start_env_id > 0:
            rule_manager.all_rules = {k: v for k, v in rule_manager.all_rules.items() if int(v["task_id"]) < args.start_env_id}

        num_successes, result_dict = get_result_dict(rule_manager)
    else:
        epoch_id = 0
        num_successes = 0
        result_dict = {}
    
    print(f"Sending all logs to: {logging_dir} (Run name). Start from task_id: {args.start_env_id}.")
    # set paths to log files
    trial_log_path = os.path.join(logging_dir, 'trial.log')
    world_log_path = os.path.join(logging_dir, 'world.log')

    with open(world_log_path, 'a') as wf:
        wf.write(f'\n***** Start building *****\n')

    # run pair data
    for i, pair in enumerate(pair_data):
        # get the task (intent)
        id = pair["id"]
        idx = pair["idx"]
        
        if id < args.start_env_id: continue

        # run one trial
        with CombinedLogger(trial_log_path) as s:
            consumed_tokens = run(args, pair, rule_manager, skill_bank)
        
        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(consumed_tokens)

        # log env results to trial log
        with open(trial_log_path, 'a') as wf:
            wf.write(f'\n#####\n\nEnvironment #{id}, idx #{idx}\n\n#####\n')
    


if __name__ == '__main__':
    args = get_args()
    # Select task_id interval based on the split
    args.min_env, args.max_env = 0, ENV_NUM

    logging_dir = args.run_name
    if not args.is_resume:
        # Create the run directory
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)

    # Create the result directory recording traces
    if not args.result_dir:
        args.result_dir = os.path.join(logging_dir, "results")
    os.makedirs(args.result_dir, exist_ok=True)
    if args.save_trace_enabled:
        os.makedirs(os.path.join(args.result_dir, "traces"), exist_ok=True)

    main(args)
