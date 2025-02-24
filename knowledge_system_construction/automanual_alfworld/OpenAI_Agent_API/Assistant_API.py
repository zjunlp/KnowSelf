import os
import re
import json
import time
from copy import copy
import openai
from openai import OpenAI
import tiktoken

class ChatGPT_Agent_Assistant_API:
    def __init__(self, gpt_version, agent_name=None, prompt_file="./prompts/prompt.txt", tool_list=None, temperature=0.0, stop=None, max_completion_length=500, log_path=None):
        self.gpt_version = gpt_version
        self.agent_name = agent_name
        self.temperature = temperature
        self.log_path = log_path
        self.client = openai.OpenAI()
        self.thread = None
        self.assistant = None
        self.tool_list = tool_list if tool_list is not None else []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        if 'gpt-3.5-turbo-0613' in gpt_version:
            self.max_token_length = 4000
        elif 'gpt-4-0613' in gpt_version:
            self.max_token_length = 8000
        else:
            self.max_token_length = 15000
        self.max_completion_length = max_completion_length
        
        self.init_messages = []
        self.system_message = ""
        
        self.messages_history = []
        self.run_id = None

        if prompt_file is not None:
            # load prompt file
            with open(prompt_file) as f:
                data = f.read()
            data_spilit = re.split(r'(\[user\]|\[system\]|\[assistant\])', data)
            # load role and message
            for i in range(0, len(data_spilit)-1, 2):
                role = data_spilit[i+1].replace('[', '').replace(']', '')
                message = data_spilit[i+2].strip()
                if role == "system":
                    self.system_message += message
                else:
                    self.init_messages.append({'role': role, 'content': message})
        else:
            self.init_messages = []
        self.assistant = self.client.beta.assistants.create(
            instructions=self.system_message,
            name=self.agent_name,
            tools=self.tool_list,
            model=self.gpt_version,
        )
        self.thread = self.client.beta.threads.create(
            messages=self.init_messages
        )
        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                f.write('\n\n#####New Environment#####\n\n')

    def add_prompt(self, prompt):
        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=prompt
        )
        self.messages_history.append({'role': 'user', 'content': prompt})

    def check_prompt(self, prompt):
        prompt_content = self.system_message
        for message in self.init_messages + self.messages_history:
            prompt_content += message['content']
        prompt_content += prompt
        prompt_length = len(self.tokenizer.encode(prompt_content))
        print('prompt length: ' + str(prompt_length))
        if prompt_length > self.max_token_length - self.max_completion_length:
            raise ValueError('prompt too long. need to be truncated.')
        return prompt_length
    
    def generate(self, message="", tool_messages=None, append_in=True, append_out=True):
        try:
            self.check_prompt(message)
        except ValueError as e:
            return 'prompt is too long'
        
        if len(message)>0:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=message
            )
        if tool_messages is None:
            run = self.client.beta.threads.runs.create(
                thread_id = self.thread.id,
                assistant_id = self.assistant.id,
                temperature = self.temperature # openai==1.16.0 feature
            )
        else:
            tool_outputs=[{"tool_call_id": self.tool_json[i].id, "output": tool_message} for i, tool_message in enumerate(tool_messages)]
            run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=self.run_id,
                    tool_outputs=tool_outputs
            )
        while True:
            run = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
            if run.status not in ["queued", "in_progress"]:
                break
            time.sleep(1)
        self.run_id = run.id
        if run.status == "requires_action":
            self.tool_json = run.required_action.submit_tool_outputs.tool_calls
        else:
            self.tool_json = None
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
            order = "desc",
        )
        response = messages.data[0].content[0].text.value if messages.data[0].role == "assistant" else ""
        response = response.replace("\\_", "_")
        if self.log_path is not None:
            with open(self.log_path, 'a') as f:
                f.write('\n'+message)
                f.write('\n'+response)
        if append_in:
            self.messages_history.append({'role': 'user', 'content': message})
        if append_out:
            self.messages_history.append({'role': 'assistant', 'content': response})
        return response
    
    def consumed_tokens(self):
        return f"total token {self.check_prompt('')}\n"
