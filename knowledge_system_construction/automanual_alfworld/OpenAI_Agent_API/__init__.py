import re
from OpenAI_Agent_API.ChatGPT_API import ChatGPT_Agent
from OpenAI_Agent_API.Assistant_API import ChatGPT_Agent_Assistant_API

def process_respond(respond, check_code=True):
    respond_thought = respond.split("```python")[0]
    matches = re.findall(r"```python\n(.*?)\n```", respond, re.DOTALL)
    respond_code = "\n".join(matches)
    if ("As an AI" in respond or "OpenAI" in respond) or (check_code and respond_code==""):
        raise ValueError("LLM failed to respond with code as expected.")
    return respond_thought, respond_code

def get_ChatGPT_Agent(gpt_version, agent_name, prompt_file, use_assistant_api=True, example_list=None, temperature=0.0, max_completion_length=2000, stop=None):
    if use_assistant_api:
        LLM_agent = ChatGPT_Agent_Assistant_API(gpt_version, agent_name, prompt_file, temperature=temperature, max_completion_length=max_completion_length, stop=stop)
    else:
        LLM_agent = ChatGPT_Agent(gpt_version, agent_name, prompt_file, temperature=temperature, max_completion_length=max_completion_length, stop=stop)
    if example_list is not None:
        example_prompt = f'Here are {len(example_list)} examples.\n'+'\n'.join(example_list)
        LLM_agent.add_prompt(example_prompt)
    return LLM_agent