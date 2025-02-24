# model template

import sys
sys.path.append('../')
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass
import warnings

class Templator:

    @classmethod
    def wrap(self, messages:List, **kwargs) -> str:
        raise NotImplementedError("Please implement the function `wrap` for the class Templator.")
    
    @classmethod
    def generate_structured_input(
        cls, 
        messages: List[Dict],
        system_template:str,
        user_template:str,
        assistant_template:str,
        assistant_template_left:str,
        assistant_template_right:str,
        splitter:str,
        splitter_after_user:str=None,
        final_input:str = None,
        structured_final_input: List[str] = None,
        # just for the inference
        add_special_token_when_last_role_is_user:bool = True,
        add_end_splitter: bool = False
    ) -> List[str]:
        if final_input == None:
            final_input = ""
        for message in messages:
            # Our goal is to get the some part segament that can be concatenated directly and to mask some part segament
            if message['role'] == 'system':
                if system_template == None:
                    warnings.warn("The current templator is not supported the role `system`. Now we ignore the content in the system role.")
                else:
                    final_input = f"{final_input}{system_template.format(prompt=message['content'])}{splitter}"
            elif message['role'] == 'user':
                if splitter_after_user == None:
                    final_input = f"{final_input}{user_template.format(prompt=message['content'])}{splitter}"
                else:
                    final_input = f"{final_input}{user_template.format(prompt=message['content'])}{splitter_after_user}"
            elif message['role'] == 'assistant':
                final_input = f"{final_input}{assistant_template.format(prompt=message['content'])}{splitter}"
            else:
                raise ValueError(f"the role `{message['role']}` is not supported. Our supported role list is `[system, user, assistant]`.")
        if add_end_splitter:
            final_input = f"{final_input}{splitter}"
        else:
            if len(splitter) > 0:
                # remove the last splitter
                assert final_input.endswith(splitter)
                final_input = final_input[:-len(splitter)]
        if add_special_token_when_last_role_is_user and messages[-1]['role'] == 'user':
            final_input = f"{final_input}{splitter}"
        return final_input

class Qwen2Templator(Templator):
    # no explicit system prompt
    """<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
user input 1<|im_end|>
<|im_start|>assistant
model output 1<|im_end|>"""
    # explicit system prompt
    """<|im_start|>system
system prompt 1<|im_end|>
<|im_start|>user
user input 1<|im_end|>
<|im_start|>assistant
model output 1<|im_end|>"""
    @classmethod
    def wrap(cls, messages:List[Dict], add_special_token_when_last_role_is_user:bool=False, force_system_prompt:bool=False, add_end_splitter:bool=False) -> str:
        # no bos and no eos
        default_system_prompt = "You are a helpful assistant."
        system_template = "<|im_start|>system\n{prompt}<|im_end|>"
        user_template = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
        assistant_template = "{prompt}<|im_end|>"
        assistant_template_left = "<|im_start|>assistant\n"
        assistant_template_right = "<|im_end|>"
        splitter = "\n"

        if force_system_prompt:
            is_existed = False
            for message in messages:
                if message['role'] == 'system':
                    is_existed = True
                    break
            if is_existed == False:
                messages = [{"role": "system", "content": default_system_prompt}] + messages

        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user,
            add_end_splitter=add_end_splitter
        )
        

class Llama3Templator(Templator):
    # no explicit system prompt
    """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

user input 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

model output 1<|eot_id|>"""
    # explicit system prompt
    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

system prompt 1<|eot_id|><|start_header_id|>user<|end_header_id|>

user input 1<|eot_id|><|start_header_id|>assistant<|end_header_id|>

model output 1<|eot_id|>"""
    @classmethod
    def wrap(cls, messages:List, add_special_token_when_last_role_is_user:bool=False, add_end_splitter:bool=False, add_bos_token:bool=False) -> List[str]:
        default_system_prompt = None
        system_template = "<|start_header_id|>system<|end_header_id|>\n\n{prompt}<|eot_id|>"
        user_template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_template = "{prompt}<|eot_id|>"
        assistant_template_left = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assistant_template_right = "<|eot_id|>"
        splitter = ""

        if add_bos_token:
            final_input = "<|begin_of_text|>"
        else:
            final_input = ""
        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            final_input=final_input,
            splitter=splitter,
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user,
            add_end_splitter=add_end_splitter
        )

class GemmaTemplator(Templator):
    # no explicit system prompt
    """<bos><start_of_turn>user
user input 1<end_of_turn>
<start_of_turn>model
model output 1<end_of_turn>
<start_of_turn>user
user input 2<end_of_turn>
<start_of_turn>model
model output 2<end_of_turn>"""
    # explicit system prompt
    # system role not supported
    @classmethod
    def wrap(cls, messages:List, add_special_token_when_last_role_is_user:bool=False, add_end_splitter:bool=False, add_bos_token:bool=False) -> List[str]:
        # system role is not supported by official
        default_system_prompt = None
        system_template = "<start_of_turn>system\n{prompt}<end_of_turn>"
        user_template = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"
        assistant_template = "{prompt}<end_of_turn>"
        assistant_template_left = "<start_of_turn>model\n"
        assistant_template_right = "<end_of_turn>"
        splitter = "\n"

        if add_bos_token:
            final_input = "<bos>"
        else:
            final_input = ""
        return cls.generate_structured_input(
            messages=messages,
            system_template=system_template,
            user_template=user_template,
            assistant_template=assistant_template,
            assistant_template_left=assistant_template_left,
            assistant_template_right=assistant_template_right,
            splitter=splitter,
            final_input = final_input,
            add_special_token_when_last_role_is_user=add_special_token_when_last_role_is_user,
            add_end_splitter=add_end_splitter,
        )

