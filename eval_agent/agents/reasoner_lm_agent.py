import time
import openai
import logging
import backoff
import requests
from openai import OpenAI
from .base import LMAgent

logger = logging.getLogger("agent_frame")

class ReasonerLMAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.api_base = config["api_base"]
        self.api_key = config["api_key"]
        self.model_name = config["model_name"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def __call__(self, messages) -> str:
        response = None
        # print(messages)
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model = self.model_name,
                    messages = messages,
                    stream = False
                )
                # logger.info(response.choices[0].message.reasoning_content)
                logger.info(response.choices[0].message.content)
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(e)
                time.sleep(1)

        if response is not None:
            raise Exception(f"Failed to connect to the model, response is {response.text}")
        else:
            raise Exception("Failed to connect to the model, response is None")
    
    def update(self, config):
        self.api_base = config.get("api_base", self.api_base)
        self.model_name = config.get("model_name", self.model_name)
