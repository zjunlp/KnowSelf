import json
import time
import logging
from typing import List, Dict, Union, Any
import requests
from requests.exceptions import Timeout, ConnectionError

from .base import LMAgent

logger = logging.getLogger("agent_frame")

class LlamaFactoryAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.url = config["url"]
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0.5)
        self.max_tokens = config.get("max_tokens", 1024)
        self.top_p = config.get("top_p", 1)
    
    def __call__(self, messages) -> str:
        response = ""
        for _ in range(3):
            try:
                # Prepend the prompt with the system message
                response = requests.post(
                    url=self.url,
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "stream": False,
                    }
                )
                # logger.info(response.json())
                # print(response.json())
                return response.json()["choices"][0]["message"]["content"]
            except Timeout as e:
                logger.warning("Timeout error")
                time.sleep(1)
            except ConnectionError as e:
                logger.warning("Connection error")
                time.sleep(1)
            except Exception as e:
                logger.error(e)
                break
        raise Exception(f"Failed to connect to the model, response is {response.json()}")
    
    def update(self, config):
        self.url = config.get("url", self.url)
        self.model_name = config.get("model_name", self.model_name)
        self.temperature = config.get("temperature", self.temperature)
        self.max_tokens = config.get("max_tokens", self.max_tokens)
        self.top_p = config.get("top_p", self.top_p)
        
