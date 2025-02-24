import re
from OpenAI_Agent_API import get_ChatGPT_Agent, process_respond
from prompts.builder_case_prompt import *

def cov_to_text(conv):
    return "\n".join(c["content"] for c in conv)

def format_input(task_id, task_desc, current_traj, correct_action, explore_action):
    return "Task ID: " + task_id + "\n\nObjective:\n" + task_desc + "\n\nCurrent epoch's trajectory:\n" + current_traj + "\n\nCorrect Action:\n" + correct_action + "\n\nExplore Action:\n" + explore_action

MAX_REPLAN_STEP = 3
MAX_RULE_NUM = 10
def run(args, pair, rule_manager, skill_bank):

    # Create the Planner and Builder Agent
    from prompts.autobuild_examples import builder_example
    Builder_agent = get_ChatGPT_Agent(args.model_name, "builder", "./prompts/builder_prompt.txt", args.assistant_api, example_list=[builder_example])

    task_id = str(pair['id'])
    task_desc = pair['full_golden_traj'][0]["content"].split("Instruction: [SEP] ")[1].split("[SEP] Search")[0].strip()
    current_traj = cov_to_text(pair['golden_traj'][:-1])

    correct_action = pair['golden_action']
    explore_action = pair['explore_action']

    input = format_input(task_id, task_desc, current_traj, correct_action, explore_action)
    run_autobuild(input, rule_manager, Builder_agent)
    rule_manager.save()
    
    consumed_tokens = Builder_agent.consumed_tokens()

    # If the rule number exceed the maximum, the Consolidator Agent merges or delete rules
    consumed_tokens = run_merge(args, rule_manager, consumed_tokens)

    return consumed_tokens

def run_autobuild(input, rule_manager, Builder_agent):
    print("\nBegin extracting rules from the current epoch.\n")
    Builder_agent.add_prompt(f"Print rule_manager.all_rules:\n{str(rule_manager.all_rules)}\n")
    Builder_agent.add_prompt(f"{input}\n")
    print(input)
    respond = Builder_agent.generate()
    print(respond)
    respond_thought, respond_code = process_respond(respond)
    # execute the solution code
    exec(respond_code)
    tool_messages = rule_manager.report()
    print(tool_messages)

def run_merge(args, rule_manager, consumed_tokens):
    while len(rule_manager.all_rules) > MAX_RULE_NUM:
        print(f"\nToo many rules exist: {len(rule_manager.all_rules)} > {MAX_RULE_NUM}. Begin merging.\n")
        Builder_merge_agent = get_ChatGPT_Agent(args.model_name, "builder_merge", "./prompts/builder_merge_prompt.txt", args.assistant_api, example_list=None)
        message = builder_merge_get_prompt.format(str(rule_manager.all_rules), MAX_RULE_NUM)
        respond = Builder_merge_agent.generate(message)
        print(respond)
        respond_thought, respond_code = process_respond(respond)
        # execute the merging code
        exec(respond_code)
        message = rule_manager.report()
        rule_manager.arrange_rules()
        consumed_tokens += Builder_merge_agent.consumed_tokens()
    
    rule_manager.save()
    return consumed_tokens