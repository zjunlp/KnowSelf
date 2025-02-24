import time
import openai
import logging
import backoff
import requests
from .base import LMAgent
from openai import OpenAI

logger = logging.getLogger("agent_frame")
        
# requests interface
class DeepseekAgent(LMAgent):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.5)
        self.max_tokens = config.get("max_tokens", 1024)
        self.top_p = config.get("top_p", 1)
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def __call__(self, messages) -> str:
        response = None
        for _ in range(5):
            try:
                print("Connecting to the model")
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    temperature = self.temperature,
                    max_tokens = self.max_tokens,
                    top_p = self.top_p,
                    messages = messages,
                    stream = False
                )
                return response.choices[0].message.content
            except requests.Timeout as e:
                logger.warning("Timeout error")
                time.sleep(5)
            except ConnectionError as e:
                logger.warning("Connection error")
                time.sleep(5)
            except Exception as e:
                logger.warning(e)
                time.sleep(5)
            
        if response is not None:
            raise Exception(f"Failed to connect to the model, response is {response}")
        else:
            raise Exception("Failed to connect to the model, response is None")
    
    def update(self, config):
        self.api_base = config.get("api_base", self.api_base)
        self.model_name = config.get("model_name", self.model_name)
        self.temperature = config.get("temperature", self.temperature)
        self.max_tokens = config.get("max_tokens", self.max_tokens)
        self.top_p = config.get("top_p", self.top_p)
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        
