# This is a Claude implementation of Flash that works exactly like the Azure version.

import asyncio
import logging
import os
from typing import List, Dict, Tuple, Any
from pydantic import BaseModel
from anthropic import AnthropicBedrock
from anthropic import APIStatusError, APIConnectionError
from aiopslab.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthropicClaudeBedrock:
    """Anthropic Claude Bedrock client"""
    
    def __init__(self):
        self.client = AnthropicBedrock(
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region = os.getenv("AWS_REGION", "us-east-1")
        )
        self.model_name = os.getenv("CLAUDE_BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

    def run(self, messages: List[Dict[str, str]]) -> List[str]:
        claude_messages = []
        system_prompt_content = None

        for msg in messages:
            if msg["role"] == "system":
                system_prompt_content = msg["content"]
            elif msg["role"] == "user" or msg["role"] == "assistant":
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

        # Ensure messages list is not empty and starts with 'user' role
        if not claude_messages:
            return ["Error: No valid messages found."]
        
        # Fix message alternation issues
        claude_messages = self._fix_message_alternation(claude_messages)

        # Determine the final system prompt value to send
        final_system_prompt = system_prompt_content if system_prompt_content else ""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                messages=claude_messages,
                temperature=0.7,
                system=final_system_prompt
            )

            if response.content and isinstance(response.content, list) and len(response.content) > 0 and hasattr(response.content[0], 'text'):
                return [response.content[0].text]
            else:
                return ["Error: Unexpected response format from Anthropic Claude on Bedrock"]

        except APIStatusError as e:
            logger.error(f"Anthropic Bedrock API Error: {e}")
            return [f"Error: Anthropic Bedrock API Error - {e.status_code}"]

        except APIConnectionError as e:
            logger.error(f"Anthropic Bedrock Connection Error: {e}")
            return [f"Error: Anthropic Bedrock API Connection Error - {e}"]

        except Exception as e:
            logger.error(f"Unexpected error with Anthropic Claude: {e}")
            return [f"Error: Failed to get response from Anthropic Claude - {e}"]

    def _fix_message_alternation(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fix message alternation to ensure user/assistant roles alternate properly"""
        if not messages:
            return messages

        fixed_messages = []
        
        # Ensure first message is from user
        if messages[0]["role"] != "user":
            fixed_messages.append({"role": "user", "content": "Please help me with the following task."})
        
        for i, msg in enumerate(messages):
            if not fixed_messages:
                fixed_messages.append(msg)
            else:
                last_role = fixed_messages[-1]["role"]
                current_role = msg["role"]
                
                if last_role == current_role:
                    # Skip duplicate consecutive roles or merge content
                    if current_role == "user":
                        # Merge user messages
                        fixed_messages[-1]["content"] += f"\n\n{msg['content']}"
                    else:
                        # For assistant messages, add a dummy user message
                        fixed_messages.append({"role": "user", "content": "Please continue."})
                        fixed_messages.append(msg)
                else:
                    fixed_messages.append(msg)
        
        return fixed_messages


class FlashAgent:
    def __init__(self):
        self.history = []
        self.llm = AnthropicClaudeBedrock()
        self.hindsight_builder = HindsightBuilder()

    def init_context(self, problem_desc: str, instructions: str, apis: dict):
        self.shell_api = self._filter_dict(apis, lambda k, _: "exec_shell" in k)
        self.submit_api = self._filter_dict(apis, lambda k, _: "submit" in k)
        self.telemetry_apis = self._filter_dict(
            apis, lambda k, _: "exec_shell" not in k and "submit" not in k
        )

        self.system_message = f"""
        You are a diagnostic agent for troubleshooting services. You must respond with ONLY function calls in this exact format.

        Problem Description: {problem_desc}

        Available Telemetry APIs:
        {self._stringify_apis(self.telemetry_apis)}

        Shell API:
        {self._stringify_apis(self.shell_api)}

        Submit API:
        {self._stringify_apis(self.submit_api)}

        CRITICAL: You must respond with ONLY a single function call like these examples:
        - get_logs("test-social-network", "text-service")
        - get_metrics("test-social-network", 5)
        - exec_shell("kubectl get pods -n test-social-network") 
        - submit("Yes") or submit("No")

        The namespace is "test-social-network" and the main service is "text-service".
        
        Do NOT include any explanation, reasoning, or other text. ONLY the function call.
        Do NOT use markdown code blocks or backticks.
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
        raw_response = response[0]
        
        # CRITICAL FIX: Remove markdown code blocks that Claude adds
        cleaned_response = self._clean_claude_response(raw_response)
        
        self.history.append({"role": "assistant", "content": raw_response})
        return cleaned_response  # Return cleaned version
    
    def _clean_claude_response(self, response: str) -> str:
        """Remove markdown code blocks and extra formatting from Claude's response"""
        import re
        
        # First, handle the most common case: code blocks with function calls
        # Pattern: ```optional_language\nfunction_call```
        code_block_pattern = r'```(?:[a-zA-Z]*\n)?(.*?)```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        
        if match:
            # Extract content from inside code blocks
            response = match.group(1).strip()
        else:
            # If no code blocks, just clean up the response
            response = response.strip()
        
        # Remove any remaining backticks that might be standalone
        response = response.replace('`', '')
        
        # If response has multiple lines, take the first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            response = lines[0]
        
        # Final cleanup - remove any leading/trailing whitespace
        response = response.strip()
        
        return response

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
        self.llm = AnthropicClaudeBedrock()

    def generate_prompt(self, input: str, history: dict) -> str:
        """
        Generate a prompt asking the LLM for the next action
        """
        prompt = f"""
        Current situation: {input}

        Based on this, what should be the next diagnostic step?

        Respond with ONLY a single function call in this exact format:
        - get_logs("test-social-network", "text-service") 
        - get_metrics("test-social-network", 5)
        - get_traces("test-social-network", 5)
        - exec_shell("kubectl get pods -n test-social-network")
        - submit("Yes") or submit("No")

        Do NOT include any explanation. ONLY the function call.
        """
        return prompt

    def develop_hindsight(self, input: str, history: dict) -> str:
        """
        Develop hindsight based on the input and provide guidance for the next action.
        """
        prompt = self.generate_prompt(input, history)
        response = self.llm.run([{"role": "user", "content": prompt}])
        raw_response = response[0]
        
        # Clean the hindsight response too
        cleaned_response = self._clean_claude_response(raw_response)
        return cleaned_response
    
    def _clean_claude_response(self, response: str) -> str:
        """Remove markdown code blocks and extra formatting from Claude's response"""
        import re
        
        # Remove markdown code blocks (```...```)
        response = re.sub(r'```[a-zA-Z]*\n?', '', response)
        response = re.sub(r'```', '', response)
        
        # Remove extra whitespace and newlines
        response = response.strip()
        
        # If response has multiple lines, take the first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            response = lines[0]
        
        return response


if __name__ == "__main__":
    # Verify Claude Bedrock environment variables are set
    required_env_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "CLAUDE_BEDROCK_MODEL_ID"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables for Anthropic Claude via Bedrock: {missing_vars}")
        logger.info("Please set the following environment variables:")
        logger.info("export AWS_ACCESS_KEY_ID='your-access-key'")
        logger.info("export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        logger.info("export AWS_REGION='us-east-1'")
        logger.info("export CLAUDE_BEDROCK_MODEL_ID='anthropic.claude-3-sonnet-20240229-v1:0'")
        exit(1)

    pids = ["k8s_target_port-misconfig-detection-2"]

    for pid in pids:
        flash_agent = FlashAgent()
        orchestrator = Orchestrator()

        orchestrator.register_agent(flash_agent, name="flash")

        problem_desc, instructions, apis = orchestrator.init_problem(pid)
        flash_agent.init_context(problem_desc, instructions, apis)

        asyncio.run(orchestrator.start_problem(max_steps=20))
