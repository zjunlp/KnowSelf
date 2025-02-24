from functools import partial
import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
from eval_agent.prompt import prompt_with_icl, prompt_without_icl

from util.task_knowledge import alfworld_prompt
from util.templator import Qwen2Templator, Llama3Templator, GemmaTemplator

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import PeftModel


logger = logging.getLogger("agent_frame")

add_knowledge_task = []
add_reflection_task = []


def conv_to_text(cov: List[int]) -> str:
    return "\n".join([c["content"] for c in cov])


def conv_to_text_webshop(cov: List[int]) -> str:
    msg = []
    for c in cov[:-1]:
        if c["role"] == "user":
            msg.append(f"Observation: ...")
        else:
            msg.append(c["content"])
    msg.append(cov[-1]["content"])
    return "\n".join(msg)


def format_input(task_desc, current_traj, rules):
    return (
        "The objectve:\n"
        + task_desc
        + "\n\nThe current trajectory:\n"
        + current_traj
        + "\n\nThe rules:\n"
        + rules
    )

# call model to generate fast response, slow response, or knowledgeable response
def call_model(
    args, task, select_knowledge_agent, model, model_type, tokenizer, messages, mode="adaptive_knowledge"
):
    end_token = ""
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        skip_special_tokens=False,
        stop="\n[Knowledge]",
    )
    # print(messages)
    if model_type == "qwen":
        template = Qwen2Templator()
        input = template.wrap(
            messages,
            add_special_token_when_last_role_is_user=True,
            force_system_prompt=True,
            add_end_splitter=True,
        )
        pass
    elif model_type == "llama3":
        template = Llama3Templator()
        input = template.wrap(
            messages,
            add_special_token_when_last_role_is_user=True,
            add_end_splitter=True,
        )
        end_token = "<|eot_id|>"
    elif model_type == "gemma":
        template = GemmaTemplator()
        input = template.wrap(
            messages,
            add_special_token_when_last_role_is_user=True,
            add_end_splitter=True,
        )
        end_token = "<end_of_turn>"

    pred_text = ""
    if mode != "always_knowledge":
        preds = model.generate(input, sampling_params=sampling_params, use_tqdm=False)
        pred_text = preds[0].outputs[0].text.split(end_token)[0]

    # if "always_knowledge", add knowledge to all responses
    if mode == "always_knowledge":
        add_knowledge = True
    # if "no_knowledge", do not add knowledge to any response
    elif mode == "no_knowledge":
        add_knowledge = False
    # if "adaptive_knowledge", add knowledge to response if it contains [Knowledge]
    else:
        add_knowledge = "[Knowledge]" in pred_text

    if "[Reflection]" in pred_text:
        logger.warning("[Reflection] in preds")
        if task.task_id not in add_reflection_task:
            add_reflection_task.append(task.task_id)
        logger.warning(f"reflection text: {pred_text}")

    if add_knowledge:
        logger.warning("[Knowledge] in preds")
        if task not in add_knowledge_task:
            add_knowledge_task.append(task.task_id)

        if args.exp_config == "alfworld":
            # add task knowledge to the prompt
            if len(messages) == 3:
                prompt = ""
                game_file = task.game_file
                for k, v in alfworld_prompt.items():
                    if k in game_file:
                        prompt = v
                        break
                if prompt == "":
                    logger.warning(f"Game file {game_file} not in alfworld_prompt")
                    return pred_text
                knowledge_augmented_input = (
                    input + f"[Knowledge]<knowledge>{prompt}</knowledge>\n"
                )
                preds = model.generate(
                    knowledge_augmented_input,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                pred_text = preds[0].outputs[0].text.split(end_token)[0]
                pred_text = f"[Knowledge]<knowledge>{prompt}</knowledge>\n" + pred_text
                return pred_text
            else:
                # call select knowledge agent to select knowledge
                with open(args.select_knowledge_inst) as f:
                    prompt = f.read()
                task_desc = messages[2]["content"]
                current_traj = conv_to_text(messages[3:])
                rule_data = json.load(open(args.knowledge_base_path))["all_rules"]
                rules = []
                for k, v in rule_data.items():
                    rules.append(v["rule"])
                rules = rules.__str__()
                cur_task = format_input(task_desc, current_traj, rules)
                _, knowledge_prompt = prompt_without_icl(prompt, cur_task)
                logger.warning(knowledge_prompt)
                select_knowledge_output = select_knowledge_agent(knowledge_prompt)
                logger.warning(f"select_knowledge_output: {select_knowledge_output}")
                if "[Chosen Rule]:" in select_knowledge_output:
                    rule = select_knowledge_output.split("[Chosen Rule]:")[1].strip()
                else:
                    rule = ""
                knowledge_augmented_input = (
                    input + f"[Knowledge]<knowledge>{rule}</knowledge>\n"
                )
                logger.warning(
                    f"knowledge:\n[Knowledge]<knowledge>{rule}</knowledge>\n"
                )
                preds = model.generate(
                    knowledge_augmented_input,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                pred_text = (
                    f"[Knowledge]<knowledge>{rule}</knowledge>\n"
                    + preds[0].outputs[0].text.split(end_token)[0]
                )

        elif args.exp_config == "webshop":
            # call select knowledge agent to select knowledge
            with open(args.select_knowledge_inst) as f:
                prompt = f.read()
            task_desc = (
                messages[2]["content"]
                .split("Instruction: [SEP]")[1]
                .split("[SEP] Search")[0]
                .strip()
            )
            current_traj = conv_to_text_webshop(messages[3:])
            rule_data = json.load(open(args.knowledge_base_path))["all_rules"]
            rules = []
            for k, v in rule_data.items():
                rules.append(v["rule"])
            rules = rules.__str__()
            cur_task = format_input(task_desc, current_traj, rules)
            _, knowledge_prompt = prompt_without_icl(prompt, cur_task)
            logger.warning(knowledge_prompt)
            select_knowledge_output = select_knowledge_agent(knowledge_prompt)
            logger.warning(f"select_knowledge_output: {select_knowledge_output}")
            if "[Chosen Rule]:" in select_knowledge_output:
                rule = select_knowledge_output.split("[Chosen Rule]:")[1].strip()
            else:
                rule = ""
            knowledge_augmented_input = (
                input + f"[Knowledge]<knowledge>{rule}</knowledge>\n"
            )
            logger.warning(f"knowledge:\n[Knowledge]<knowledge>{rule}</knowledge>\n")
            preds = model.generate(
                knowledge_augmented_input,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            pred_text = (
                f"[Knowledge]<knowledge>{rule}</knowledge>\n"
                + preds[0].outputs[0].text.split(end_token)[0]
            )

    return pred_text


def interactive_loop(
    args: argparse.Namespace,
    task: tasks.Task,
    select_knowledge_agent: agents.LMAgent,
    tokenizer: AutoTokenizer,
    model: LLM,
    model_type: str,
    env_config: Dict[str, Any],
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    # reset the environment and set the prompt
    observation, state = env.reset()

    init_msg = observation

    logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    cur_step = 1
    while not state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        # agent act
        try:
            llm_output: str = call_model(
                args=args,
                task=task,
                select_knowledge_agent=select_knowledge_agent,
                model=model,
                model_type=model_type,
                tokenizer=tokenizer,
                messages=state.history,
            )
            # logger.warning(f"\n{Fore.YELLOW}{state.history}{Fore.RESET}")
            # color the action in green
            # logger.info(f"\nLM Agent Action:\n\033[92m{action.value}\033[0m")
            logger.info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break
        # environment step
        observation, state = env.step(llm_output)
        # color the state in blue
        if not state.finished:
            # color the observation in blue
            logger.info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}")

    return state


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)

    with open(os.path.join(args.agent_path, f"{args.select_agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    if args.select_agent_name is not None:
        agent_config["config"]["model_name"] = args.select_agent_name

    task_name = args.exp_config
    model_name = args.model_name_or_path

    model = LLM(
        model=model_name,
        dtype="float16",
        tensor_parallel_size=args.gpu_num,
        gpu_memory_utilization=1.0,
        enable_chunked_prefill=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if args.output_path == "":
        output_path = os.path.join(
            "outputs", "test_knowself" + args.exp_config + args.exp_name
        )
    else:
        output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(output_path, "log.txt"), mode="w", encoding="utf-8"
    )

    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]

    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config["env_class"] == "WebShopEnv":
        from webshop.web_agent_site.envs import WebAgentTextEnv

        env_config["env"] = WebAgentTextEnv(observation_mode="text", human_goals=True)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # initialize the agent
    select_knowledge_agent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    model_type = args.model_type
    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith("json"):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split(".")[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        # calculate metrics
        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")
        return

    # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks

    type_reward_list = {}
    type_success_list = {}
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test 10 tasks in debug mode
            # if args.debug and i == 5:
            #     break

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                args,
                task,
                select_knowledge_agent,
                tokenizer,
                model,
                model_type,
                env_config,
            )

            state_list.append(state)
            json.dump(
                state.to_dict(),
                open(os.path.join(output_path, f"{task.task_id}.json"), "w"),
                indent=4,
            )

            if task_name == "alfworld":
                name = "/".join(task.game_file.split("/")[-3:-1])
                task_type = name.split("-")[0]
                if task_type not in type_reward_list:
                    type_reward_list[task_type] = []
                if task_type not in type_success_list:
                    type_success_list[task_type] = []
                if state.reward is not None:
                    type_reward_list[task_type].append(state.reward)
                type_success_list[task_type].append(state.success)

            logger.warning(f"Knowledge number: {len(add_knowledge_task)}")
            logger.warning(f"Relection number: {len(add_reflection_task)}")
            # calculate metrics
            reward_list = []
            success_list = []
            for state in state_list:
                if state.reward is not None:
                    reward_list.append(state.reward)
                success_list.append(state.success)

            logger.info(type_reward_list)
            if len(reward_list) != 0:
                if type_reward_list is not None:
                    for k, v in type_reward_list.items():
                        logger.warning(
                            f"Task type {k} average reward: {sum(v)/len(v):.4f}"
                        )
                logger.warning(
                    f"Average reward: {sum(reward_list)/len(success_list):.4f}\n"
                )

                if type_success_list is not None:
                    for k, v in type_success_list.items():
                        logger.warning(
                            f"Task type {k} success rate: {sum(v)/len(v):.4f}"
                        )
                logger.warning(
                    f"Success rate: {sum(success_list)/len(success_list):.4f}"
                )
            pbar.update(1)
        pbar.close()

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
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
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--select_agent_config",
        type=str,
        default="deepseek",
        help="Config of select agent.",
    )
    parser.add_argument(
        "--select_agent_name",
        type=str,
        default="deepseek-chat",
        help="Select agent model name. It will override the 'model_name' in agent_config of select agent",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="Config of model.",
    )
    parser.add_argument("--model_type", type=str, default="llama3", help="Model type.")
    parser.add_argument(
        "--select_knowledge_inst",
        type=str,
        default="eval_agent/prompt/instructions/select_knowledge_alfworld.txt",
        help="the instruction for selecting knowledge.",
    )
    parser.add_argument(
        "--knowledge_base_path",
        type=str,
        default="knowledge_system_construction/automanual_alfworld/autobuildcase_logs/rule_manager.json",
        help="Select agent model name. It will override the 'model_name' in agent_config of select agent",
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
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        default=1,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)
