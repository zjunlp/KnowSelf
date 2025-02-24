import os
import re
import json
import time
from copy import copy
import openai
import tiktoken

class ChatGPT_Agent:
    def __init__(self, gpt_version, agent_name=None, prompt_file="./prompts/prompt.txt", tool_list=None, temperature=0.0, stop=None, max_completion_length=500, log_path=None):
        self.gpt_version = gpt_version
        self.agent_name = agent_name
        self.temperature = temperature
        self.stop = stop
        self.log_path = log_path
        self.client = openai.OpenAI()
        if 'instruct' in self.gpt_version:
            assert tool_list is None
            self.text_completion = True
        else:
            self.text_completion = False
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        if 'gpt-3.5-turbo-0613' in gpt_version or 'instruct' in self.gpt_version:
            self.max_token_length = 4000
        elif 'gpt-4-0613' in gpt_version:
            self.max_token_length = 8000
        else:
            self.max_token_length = 15000
        self.max_completion_length = max_completion_length

        self.prompt_messages = []
        if prompt_file is not None:
            # load prompt file
            with open(prompt_file) as f:
                data = f.read()
            data_spilit = re.split(r'(\[user\]|\[system\]|\[assistant\])', data)
            # load role and message
            for i in range(0, len(data_spilit)-1, 2):
                role = data_spilit[i+1].replace('[', '').replace(']', '')
                message = data_spilit[i+2].strip()
                self.prompt_messages.append({'role': role, 'content': message})
        self.tool_list = tool_list
        self.messages = []
        self.last_response = ""
        self.max_used_token = 0
        self.total_token = 0
        self.total_input_token = 0
        self.total_output_token = 0

        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                f.write('\n\n#####New Environment#####\n\n')

    def add_prompt(self, prompt):
        self.prompt_messages.append({'role': 'user', 'content': prompt})

    def get_prompt(self, prompt):
        prompt = self.prompt_messages + prompt
        prompt_content = ""
        for message in prompt:
            prompt_content += f"\('role': {message['role']}, 'content': {message['content']}\)"
        prompt_length = len(self.tokenizer.encode(prompt_content))
        print('prompt length: ' + str(prompt_length))
        if prompt_length > self.max_token_length - self.max_completion_length:
            raise ValueError('prompt too long. need to be truncated.')
        return prompt
    
    def generate(self, message="", tool_messages=None, append_in=True, append_out=True):
        messages = copy(self.messages)
        if len(message)>0:
            messages += [{'role': 'user', 'content': message}]
        if tool_messages is not None:
            for tool_i, tool_message in enumerate(tool_messages):
                messages += [{'role': 'assistant', 'tool_calls': [self.tool_json[tool_i]], 'content': ""}]
                messages += [{"role": "tool", "tool_call_id": self.tool_json[tool_i].id, "name": self.tool_json[tool_i].function.name, "content": tool_message}]
        
        try:
            messages=self.get_prompt(messages)
        except ValueError as e:
            return 'prompt is too long'
        if self.gpt_version == "human":
            print(message)
            return input()
        
        if self.text_completion:
            messages = '\n'.join([m['content'] for m in messages])+'\n'
            response = self.client.completions.create(model=self.gpt_version, 
                                            prompt=messages, 
                                            temperature=self.temperature,
                                            max_tokens=self.max_completion_length,
                                            frequency_penalty=0.15,
                                            stop=self.stop)
            text = response.choices[0].text.strip()
            if text == "": return "null"
        else:
            response = self.client.chat.completions.create(
                model=self.gpt_version,
                messages=messages,
                tools=self.tool_list,
                temperature=self.temperature,
                max_tokens=self.max_completion_length,
                frequency_penalty=0.0,
                presence_penalty=0.0, 
                stop=self.stop)
            text = response.choices[0].message.content
            if text is None: text = ""
        # print("chat completion time:", time.time()-start_time)
        self.last_response = text

        self.total_token += response.usage.total_tokens
        self.max_used_token = max(self.max_used_token, response.usage.total_tokens)
        self.total_input_token += response.usage.prompt_tokens
        self.total_output_token += response.usage.completion_tokens

        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                f.write('\n'+str(self.messages)+'\n'+message)
                f.write('\n'+text)
        if append_in:
            self.messages.append({'role': 'user', 'content': message})
        if append_out:
            self.messages.append({'role': 'assistant', 'content': text})
        if self.tool_list is not None:
            self.tool_json = response.choices[0].message.tool_calls
        return text
    
    def consumed_tokens(self):
        return f"total token {self.total_token}: input token {self.total_input_token}, output token {self.total_output_token}, max token {self.max_used_token}\n"
