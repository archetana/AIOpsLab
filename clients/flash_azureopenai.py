# This is a naive implementation of Flash without tool and TSG.

import asyncio
import logging
import os
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
from openai import AzureOpenAI
from aiopslab.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureGPT4Turbo:
    """Azure OpenAI GPT-4 Turbo client"""
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-turbo")
        
    def run(self, messages: List[Dict[str, str]]) -> List[str]:
        """Run the Azure OpenAI model with the given messages"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            return [response.choices[0].message.content]
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI: {e}")
            return ["Error: Failed to get response from Azure OpenAI"]


class FlashAgent:
    def __init__(self):
        self.history = []
        self.llm = AzureGPT4Turbo()
        self.hindsight_builder = HindsightBuilder()

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        self.system_message = f"""
        Problem Description: {problem_desc}

        Available Telemetry APIs:
        {self._stringify_apis(self.telemetry_apis)}

        Shell API:
        {self._stringify_apis(self.shell_api)}

        Submit API:
        {self._stringify_apis(self.submit_api)}
        """

        self.task_message = instructions
        self.history.append({"role": "system", "content": self.system_message})
        self.history.append({"role": "user", "content": self.task_message})

    def _filter_dict(self, dictionary, filter_func):
        """Helper function to filter the API dictionary."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _stringify_apis(self, apis):
        return "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])

    async def get_action(self, input_text: str) -> str:
        """Determine the next action based on the input, hindsight, and reasoning."""
        hindsight = await self.diagnose_with_hindsight(input_text, self.history)

        combined_input = (
            f"{input_text}\n\nHindsight from Flash agent:\n{hindsight}"
            if hindsight
            else input_text
        )
        self.history.append({"role": "user", "content": combined_input})

        response = self.llm.run(self.history)
        self.history.append({"role": "assistant", "content": response[0]})
        return response[0]

    async def diagnose_with_hindsight(self, input: str, history: dict):
        """Diagnose the incident and integrate hindsight from the environment status."""
        logger.info("Starting diagnosis with hindsight integration...")
        hindsight = self.hindsight_builder.develop_hindsight(input, history)
        if hindsight:
            logger.info(f"Generated Hindsight: {hindsight}")
        else:
            logger.info("No hindsight generated, continuing with normal execution.")
        return hindsight


class HindsightBuilder:
    """Agent hindsight generator."""

    def __init__(self):
        self.llm = AzureGPT4Turbo()

    def generate_prompt(self, input: str, history: dict) -> str:
        """
        Generate a prompt asking the LLM whether the next action should be a submit action
        or if further diagnostic actions like log analysis are required.
        """
        prompt = f"""
        You are a helpful assistant determining the next best action based on the current situation.

        Given the history of the previous action: {history}
            
        and the environment output from last action : {input}

        1. Should the next action be a submit operation? 
        2. If not, please suggest additional diagnostic steps, such as analyzing logs from microservices.

        Thought: Identify whether submitting is the right next step, and if not, propose alternative actions.

        Solution: Provide reasoning and next steps.
        """
        return prompt

    def develop_hindsight(self, input: str, history: dict) -> str:
        """
        Develop hindsight based on the input and provide guidance for the next action.
        """
        prompt = self.generate_prompt(input, history)
        response = self.llm.run([{"role": "user", "content": prompt}])
        return response[0]


if __name__ == "__main__":
    # Verify Azure OpenAI environment variables are set
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please set the following environment variables:")
        logger.info("export AZURE_OPENAI_API_KEY='your-api-key'")
        logger.info("export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        logger.info("export AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment-name'")
        logger.info("export AZURE_OPENAI_API_VERSION='2024-02-15-preview'  # Optional")
        exit(1)

    pids = [
        "k8s_target_port-misconfig-detection-2",
        "k8s_target_port-misconfig-detection-3",
        "user_unregistered_mongodb-detection-2",
        "k8s_target_port-misconfig-localization-2",
        "k8s_target_port-misconfig-localization-3",
        "user_unregistered_mongodb-localization-2",
        "k8s_target_port-misconfig-analysis-2",
        "k8s_target_port-misconfig-analysis-3",
        "user_unregistered_mongodb-analysis-2",
        "k8s_target_port-misconfig-mitigation-2",
        "k8s_target_port-misconfig-mitigation-3",
        "user_unregistered_mongodb-mitigation-2",
    ]

    for pid in pids:
        flash_agent = FlashAgent()
        orchestrator = Orchestrator()

        orchestrator.register_agent(flash_agent, name="flash")

        # pid = "revoke_auth_mongodb-mitigation-2"
        problem_desc, instructions, apis = orchestrator.init_problem(pid)

        flash_agent.init_context(problem_desc, instructions, apis)

        asyncio.run(orchestrator.start_problem(max_steps=20))

