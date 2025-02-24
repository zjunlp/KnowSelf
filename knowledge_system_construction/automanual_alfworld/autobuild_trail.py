import re
from OpenAI_Agent_API import get_ChatGPT_Agent, process_respond
from prompts.builder_case_prompt import *

def cov_to_text(conv):
    return "\n".join(c["content"] for c in conv)

def format_input(task_id, task_desc, current_traj, correct_action_feedback, explore_action_feedback):
    return "Task ID: " + task_id + "\n\nObjective:\n" + task_desc + "\n\nCurrent epoch's trajectory:\n" + current_traj + "\n\nCorrect Action and Feedback:\n" + correct_action_feedback + "\n\nExplore Action and Feedback:\n" + explore_action_feedback

MAX_REPLAN_STEP = 3
MAX_RULE_NUM = 24
def run(args, epoch_id, pair, rule_manager, skill_bank):
    epoch_id = epoch_id

    # Create the Planner and Builder Agent
    from prompts.autobuild_examples import worker_example, builder_example

    Builder_agent = get_ChatGPT_Agent(args.model_name, "builder", "./prompts/builder_prompt.txt", args.assistant_api, example_list=[builder_example])

    task_id = pair['id']
    task_desc = pair['full_golden_traj'][0]["content"]
    current_traj = cov_to_text(pair['explore_traj'][1:-2])
    
    correct_action_feedback = pair['golden_action'] + "\n" + pair['golden_observation']
    explore_action_feedback = pair['explore_action'] + "\n" + pair['explore_observation']
    
    input = format_input(task_id, task_desc, current_traj, correct_action_feedback, explore_action_feedback) 
    run_autobuild(input, rule_manager, Builder_agent)
    rule_manager.save()
    
    consumed_tokens = Builder_agent.consumed_tokens()

    # If the rule number exceed the maximum, the Consolidator Agent merges or delete rules
    consumed_tokens = run_merge(args, rule_manager, consumed_tokens)

    return consumed_tokens

def run_autobuild(input, rule_manager, Builder_agent):
    print("\nBegin extracting rules from the current epoch.\n")
    Builder_agent.add_prompt(f"Print rule_manager.all_rules:\n{str(rule_manager.all_rules)}\n\n")
    Builder_agent.add_prompt(f"{input}\n")
    print(f"Prompt added to Builder Agent: {Builder_agent.prompt_messages}")
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
        exec(respond_code)
        message = rule_manager.report()
        rule_manager.arrange_rules()
        consumed_tokens += Builder_merge_agent.consumed_tokens()
    
    rule_manager.save()
    return consumed_tokens