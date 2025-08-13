# CMB Agent implementation replacing Flash Agent with Azure OpenAI
import asyncio
import logging
import os
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
from openai import AzureOpenAI
from aiopslab.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Fixed: was missing __


class CMBAzureAgent:
    """CMB Agent using Azure OpenAI with your specific credentials"""
    
    def __init__(self):  # Fixed: was missing underscores
        # Set your specific Azure OpenAI credentials
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-rq75033025.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_KEY"] = "12be9eb4da2d483ea9d2fe44375a59b8"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
        
        # Initialize Azure OpenAI client with your credentials
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = "iCETS-MSCoE-gpt-4o"  # Your specific model
        
    def analyze_logs(self, log_data: str, context: str = "") -> Dict[str, Any]:
        """Analyze system logs using Azure OpenAI"""
        messages = [
            {"role": "system", "content": "You are an expert AIOps engineer analyzing system logs."},
            {"role": "user", "content": f"Context: {context}\n\nAnalyze these logs:\n{log_data}\n\nProvide structured analysis with severity and recommendations."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            return {"analysis": response.choices[0].message.content, "task": "log_analysis"}
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI for log analysis: {e}")
            return {"analysis": f"Error in log analysis: {e}", "task": "log_analysis"}  # Fixed: indentation
    
    def diagnose_system_issue(self, symptoms: str, metrics: Dict = None) -> Dict[str, Any]:
        """Diagnose system issues using Azure OpenAI"""
        metrics_str = str(metrics) if metrics else "No metrics provided"
        
        messages = [
            {"role": "system", "content": "You are an expert AIOps engineer diagnosing system issues."},
            {"role": "user", "content": f"Diagnose this system issue:\n\nSymptoms: {symptoms}\n\nMetrics: {metrics_str}\n\nProvide root cause analysis and resolution steps."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            return {"diagnosis": response.choices[0].message.content, "task": "system_diagnosis"}
        except Exception as e:
            logger.error(f"Error calling Azure OpenAI for diagnosis: {e}")
            return {"diagnosis": f"Error in diagnosis: {e}", "task": "system_diagnosis"}


class CMBAgentWrapper:
    """CMB Agent wrapper using Azure OpenAI - replaces Flash Agent"""
    
    def __init__(self):  # Fixed: was missing underscores
        self.history = []
        self.cmb_agent = CMBAzureAgent()
        self.hindsight_builder = CMBHindsightBuilder()

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        self.shell_api = self._filter_dict(apis, lambda k, v: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, v: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, v: "exec_shell" not in k and "submit" not in k
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

    def _filter_dict(self, dictionary, filter_func):  # Fixed: added underscores
        """Helper function to filter the API dictionary."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _stringify_apis(self, apis):  # Fixed: added underscores
        return "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])

    async def get_action(self, input_text: str) -> str:
        """Determine the next action using CMB Agent with Azure OpenAI."""
        hindsight = await self.diagnose_with_hindsight(input_text, self.history)

        combined_input = (
            f"{input_text}\n\nHindsight from CMB agent:\n{hindsight}"
            if hindsight
            else input_text
        )
        self.history.append({"role": "user", "content": combined_input})

        # Use CMB Agent with Azure OpenAI for analysis
        result = self.cmb_agent.diagnose_system_issue(
            symptoms=combined_input,
            metrics={
                "available_telemetry_apis": len(self.telemetry_apis),
                "available_shell_apis": len(self.shell_api),
                "available_submit_apis": len(self.submit_api),
                "history_length": len(self.history)
            }
        )
        
        response = result.get('diagnosis', 'Continue with next diagnostic step')
        self.history.append({"role": "assistant", "content": response})
        return response

    async def diagnose_with_hindsight(self, input_text: str, history: dict):
        """Diagnose the incident and integrate hindsight using CMB Agent."""
        logger.info("Starting diagnosis with hindsight integration...")
        hindsight = self.hindsight_builder.develop_hindsight(input_text, history)
        if hindsight:
            logger.info(f"Generated Hindsight: {hindsight}")
        else:
            logger.info("No hindsight generated, continuing with normal execution.")
        return hindsight


class CMBHindsightBuilder:
    """CMB Agent hindsight generator using Azure OpenAI."""

    def __init__(self):  # Fixed: was missing underscores
        self.cmb_agent = CMBAzureAgent()

    def generate_prompt(self, input_text: str, history: dict) -> str:
        """
        Generate a prompt asking CMB Agent whether the next action should be a submit action
        or if further diagnostic actions like log analysis are required.
        """
        prompt = f"""
        You are a helpful assistant determining the next best action based on the current situation.

        Given the history of the previous action: {history}
            
        and the environment output from last action: {input_text}

        1. Should the next action be a submit operation? 
        2. If not, please suggest additional diagnostic steps, such as analyzing logs from microservices.

        Thought: Identify whether submitting is the right next step, and if not, propose alternative actions.

        Solution: Provide reasoning and next steps.
        """
        return prompt

    def develop_hindsight(self, input_text: str, history: dict) -> str:
        """
        Develop hindsight using CMB Agent's analysis capabilities with Azure OpenAI.
        """
        result = self.cmb_agent.analyze_logs(
            log_data=input_text,
            context="Hindsight analysis for next action determination"
        )
        return result.get('analysis', '')


# Keep the same class name for compatibility
FlashAgent = CMBAgentWrapper


if __name__ == "__main__":  # Fixed: was missing underscores
    # No need to check environment variables since they're set in the code
    logger.info("Using hardcoded Azure OpenAI credentials")

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

    # Test one PID at a time - change this line to test different PIDs
    pid = "k8s_target_port-misconfig-detection-2"

    flash_agent = FlashAgent()
    orchestrator = Orchestrator()

    orchestrator.register_agent(flash_agent, name="flash")

    problem_desc, instructions, apis = orchestrator.init_problem(pid)

    flash_agent.init_context(problem_desc, instructions, apis)

    asyncio.run(orchestrator.start_problem(max_steps=20))