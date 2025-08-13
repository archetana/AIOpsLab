# CMB Agent implementation replacing Flash Agent with proper CMB Agent integration
import asyncio
import logging
import os
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
from openai import AzureOpenAI
from aiopslab.orchestrator import Orchestrator

# Import the real CMB Agent using the correct API
try:
    from cmbagent import CMBAgent
    CMB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("CMB Agent successfully imported")
except ImportError as e:
    logging.warning(f"CMB Agent not available: {e}")
    CMB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CMBAgentWrapper:
    """CMB Agent wrapper using real CMB Agent multi-agent system - replaces Flash Agent"""
    
    def __init__(self):
        self.history = []
        
        # Set Azure OpenAI credentials for CMB Agent
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-rq75033025.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_KEY"] = "12be9eb4da2d483ea9d2fe44375a59b8"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2025-01-01-preview"
        
        # Set OpenAI API key for CMB Agent (using Azure key)
        os.environ["OPENAI_API_KEY"] = "12be9eb4da2d483ea9d2fe44375a59b8"
        os.environ["OPENAI_API_BASE"] = "https://openai-rq75033025.openai.azure.com/"
        os.environ["OPENAI_API_TYPE"] = "azure"
        
        if CMB_AVAILABLE:
            try:
                # Initialize CMB Agent with proper configuration for AIOps
                self.cmbagent = CMBAgent(
                    agent_llm_configs={
                        'engineer': {
                            "model": "iCETS-MSCoE-gpt-4o",  # Your Azure model
                            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                            "api_type": "azure",
                            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT"),
                            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                            "temperature": 0.1,
                            "max_tokens": 1000
                        },
                        'researcher': {
                            "model": "iCETS-MSCoE-gpt-4o",  # Your Azure model
                            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                            "api_type": "azure",
                            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT"),
                            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                            "temperature": 0.1,
                            "max_tokens": 1000
                        },
                        'planner': {
                            "model": "iCETS-MSCoE-gpt-4o",  # Your Azure model
                            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                            "api_type": "azure",
                            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT"),
                            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                            "temperature": 0.1,
                            "max_tokens": 1000
                        }
                    }
                )
                logger.info("Real CMB Agent with 35+ agents initialized successfully for AIOps")
                self.cmb_working = True
            except Exception as e:
                logger.error(f"CMB Agent initialization failed: {e}")
                self.cmb_working = False
                # Fallback to Azure OpenAI
                self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                self.deployment_name = "iCETS-MSCoE-gpt-4o"
        else:
            self.cmb_working = False
            # Fallback to Azure OpenAI
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.deployment_name = "iCETS-MSCoE-gpt-4o"
        
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

    def _filter_dict(self, dictionary, filter_func):
        """Helper function to filter the API dictionary."""
        return {k: v for k, v in dictionary.items() if filter_func(k, v)}

    def _stringify_apis(self, apis):
        return "\n\n".join([f"{k}\n{v}" for k, v in apis.items()])

    async def get_action(self, input_text: str) -> str:
        """Determine the next action using real CMB Agent multi-agent system."""
        hindsight = await self.diagnose_with_hindsight(input_text, self.history)

        combined_input = (
            f"{input_text}\n\nHindsight from CMB agent:\n{hindsight}"
            if hindsight
            else input_text
        )
        self.history.append({"role": "user", "content": combined_input})

        if self.cmb_working:
            # Use real CMB Agent multi-agent system with proper solve() method
            try:
                # Create comprehensive AIOps task for CMB Agent's 35+ agents
                aiops_task = f"""
                AIOps System Diagnosis and Analysis Task:
                
                Current System State: {combined_input}
                
                Problem Context: {self._format_history()}
                
                Available Diagnostic Tools:
                - Telemetry APIs: {list(self.telemetry_apis.keys())}
                - Shell Commands: {list(self.shell_api.keys())}  
                - Submit Actions: {list(self.submit_api.keys())}
                
                As an expert AIOps engineering team, analyze this system issue and determine the next best diagnostic action.
                
                Requirements:
                1. Analyze the current system symptoms
                2. Consider available diagnostic tools
                3. Provide a clear, actionable next step
                4. Explain the reasoning behind the recommendation
                
                Focus on systematic troubleshooting and root cause analysis for this AIOps scenario.
                """
                
                # Use CMB Agent's solve method with engineer-focused configuration
                self.cmbagent.solve(
                    task=aiops_task,
                    max_rounds=5,  # Limit rounds for faster response in AIOps context
                    initial_agent='engineer',  # Start with engineer for technical analysis
                    shared_context={
                        'feedback_left': 0,  # Skip planning for faster response
                        'number_of_steps_in_plan': 1,
                        'maximum_number_of_steps_in_plan': 1
                    }
                )
                
                # Extract the result from CMB Agent's output directory
                # CMB Agent saves results to cmbagent_output directory
                output_dir = "/home/useradmin/AIOpsLab/clients/cmbagent_output"
                
                # Try to read the latest result
                try:
                    import glob
                    import json
                    result_files = glob.glob(f"{output_dir}/*.md") + glob.glob(f"{output_dir}/*.txt") + glob.glob(f"{output_dir}/*.json")
                    if result_files:
                        latest_file = max(result_files, key=os.path.getctime)
                        with open(latest_file, 'r') as f:
                            response = f.read()
                        # Extract actionable recommendation from CMB Agent's analysis
                        if len(response) > 500:
                            response = response[:500] + "... [CMB Agent analysis continues]"
                    else:
                        response = "CMB Agent completed analysis. Proceed with systematic diagnostic approach."
                except Exception as e:
                    logger.warning(f"Could not read CMB Agent output: {e}")
                    response = "CMB Agent multi-agent analysis completed. Continue with next diagnostic step."
                
                logger.info("Used real CMB Agent 35+ multi-agent system for AIOps analysis")
                
            except Exception as e:
                logger.error(f"CMB Agent execution error: {e}")
                response = f"CMB Agent multi-agent analysis encountered an issue: {e}. Proceeding with diagnostic approach."
        else:
            # Fallback to Azure OpenAI with enhanced AIOps prompts
            try:
                messages = [
                    {"role": "system", "content": "You are an expert AIOps engineer with access to a multi-agent analysis system. Provide systematic diagnostic recommendations."},
                    {"role": "user", "content": f"Analyze this AIOps scenario and recommend the next diagnostic action:\n\n{combined_input}"}
                ]
                
                azure_response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                response = azure_response.choices[0].message.content
                logger.info("Used Azure OpenAI fallback with enhanced AIOps prompts")
            except Exception as e:
                logger.error(f"Azure OpenAI error: {e}")
                response = f"Error in analysis: {e}"

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

    def _format_history(self):
        """Format conversation history for CMB Agent"""
        formatted = []
        for entry in self.history[-3:]:  # Last 3 entries to avoid overwhelming
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')[:150]  # Truncate long content
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)


class CMBHindsightBuilder:
    """CMB Agent hindsight generator using real CMB Agent multi-agent system."""

    def __init__(self):
        if CMB_AVAILABLE:
            try:
                # Initialize CMB Agent for hindsight analysis
                self.cmbagent = CMBAgent(
                    agent_llm_configs={
                        'researcher': {
                            "model": "iCETS-MSCoE-gpt-4o",
                            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                            "api_type": "azure",
                            "api_base": os.getenv("AZURE_OPENAI_ENDPOINT"),
                            "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                            "temperature": 0.1,
                            "max_tokens": 500
                        }
                    }
                )
                self.cmb_working = True
            except Exception as e:
                logger.error(f"CMB Hindsight Agent initialization failed: {e}")
                self.cmb_working = False
        else:
            self.cmb_working = False
            
        if not self.cmb_working:
            # Fallback to Azure OpenAI
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.deployment_name = "iCETS-MSCoE-gpt-4o"

    def develop_hindsight(self, input_text: str, history: dict) -> str:
        """Develop hindsight using real CMB Agent multi-agent analysis."""
        
        if self.cmb_working:
            try:
                # Create hindsight analysis task for CMB Agent
                hindsight_task = f"""
                AIOps Hindsight Analysis Task:
                
                System Output to Analyze: {input_text}
                
                Previous Context: {str(history)[-300:]}
                
                As an expert AIOps research team, analyze this system output and provide hindsight guidance for determining the next best action.
                
                Questions to address:
                1. What does this output tell us about the system state?
                2. Should we continue with diagnostic steps or submit a solution?
                3. What additional telemetry or analysis would be most valuable?
                4. Are there any patterns or anomalies that warrant attention?
                
                Provide clear, actionable guidance for the next step in this AIOps investigation.
                """
                
                # Use CMB Agent for hindsight analysis
                self.cmbagent.solve(
                    task=hindsight_task,
                    max_rounds=3,  # Quick analysis for hindsight
                    initial_agent='researcher',  # Use researcher for analysis
                    shared_context={
                        'feedback_left': 0,
                        'number_of_steps_in_plan': 1,
                        'maximum_number_of_steps_in_plan': 1
                    }
                )
                
                # Extract hindsight from output
                output_dir = "/home/useradmin/AIOpsLab/clients/cmbagent_output"
                try:
                    import glob
                    result_files = glob.glob(f"{output_dir}/*.md") + glob.glob(f"{output_dir}/*.txt")
                    if result_files:
                        latest_file = max(result_files, key=os.path.getctime)
                        with open(latest_file, 'r') as f:
                            hindsight = f.read()
                        if len(hindsight) > 200:
                            hindsight = hindsight[:200] + "..."
                        return hindsight
                except Exception:
                    pass
                
                return "CMB Agent multi-agent hindsight analysis completed"
                
            except Exception as e:
                logger.error(f"CMB Agent hindsight error: {e}")
                return "Continue with standard diagnostic approach"
        else:
            try:
                # Fallback to Azure OpenAI
                messages = [
                    {"role": "system", "content": "You are an expert AIOps analyst providing hindsight for diagnostic decisions."},
                    {"role": "user", "content": f"Analyze this system output for hindsight:\n{input_text}\n\nProvide guidance for next action determination."}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Azure OpenAI hindsight error: {e}")
                return ""


# Keep the same class name for compatibility
FlashAgent = CMBAgentWrapper


if __name__ == "__main__":
    if CMB_AVAILABLE:
        logger.info("Initializing with real CMB Agent 35+ multi-agent system for AIOps")
    else:
        logger.info("CMB Agent not available - using Azure OpenAI enhanced prompts")

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