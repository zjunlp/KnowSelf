import logging
import os
import json

from colorama import Fore

logger = logging.getLogger("agent_frame")
PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
{icl_prompt}

{examples}
---

Now, it's your turn.

{task}"""


PROMPT_WITHOUT_ICL_TEMPLATE = """{instruction}

Now, it's your turn.

{task}"""

def prompt_with_icl(instruction, raw_icl, cur_task, icl_num=2):
    examples = ""
    messages = [{
        "role": "user",
        "content": instruction
    }]
    messages.append({
        "role": "assistant",
        "content": "OK"
    })
    assert len(raw_icl) % 2 == 0, "The length of messages should be even."
    raw_icl_len = int(len(raw_icl) / 2)
    icl_num = min(icl_num, raw_icl_len)
    for i in range(len(raw_icl)):
        cur_content = raw_icl[i]['content']
        if i % 2 == 0:
            if i == 0:
                icl_prompt = f"Here are {icl_num} examples." if icl_num > 1 else f"Here is an example."
                messages.append({
                    "role": "user",
                    "content": icl_prompt + "\n" + cur_content
                })
            else:
                messages.append({
                    "role": "user",
                    "content": cur_content
                })
            if icl_num > 1:
                if i == 0:
                    examples += f"Example {i + 1}:\n"
                else:
                    examples += f"\nExample {i + 1}:\n"
            examples += cur_content + '\n'
        elif i % 2 == 1:
            messages.append({
                "role": "assistant",
                "content": cur_content
            })
            examples += '\n' + cur_content + '\n'
                
    icl_prompt = f"Here are {icl_num} examples." if icl_num > 1 else f"Here is an example."
    prompt = PROMPT_WITH_ICL_TEMPLATE.format(instruction=instruction, icl_prompt=icl_prompt, examples=examples, task=cur_task)
    messages.append({
        "role": "user",
        "content": cur_task
    })
    # logger.info(f"{Fore.WHITE}messages: {messages}{Fore.RESET}")
    return prompt, messages


def prompt_without_icl(instruction, cur_task):
    prompt = PROMPT_WITHOUT_ICL_TEMPLATE.format(instruction=instruction, task=cur_task)
    messages = [{
        "role": "user",
        "content": instruction
    }]
    messages.append({
        "role": "assistant",
        "content": "OK"
    })
    messages.append({
        "role": "user",
        "content": cur_task
    })
    return prompt, messages