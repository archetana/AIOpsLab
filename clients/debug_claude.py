#!/usr/bin/env python3
"""
Claude LLM Debug Tracer
This script captures and tests every interaction with Claude LLM step by step
We are trying to debug what is wrong, particularly the responses coming from Claude AI 
"""

import os
import json
from typing import List, Dict
from anthropic import AnthropicBedrock
from anthropic import APIStatusError, APIConnectionError

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DebugClaudeTracer:
    def __init__(self):
        """Initialize Claude client with debug tracking"""
        self.client = AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION", "us-east-1")
        )
        self.model_name = os.getenv("CLAUDE_BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        self.interaction_count = 0
        
    def debug_llm_call(self, messages: List[Dict[str, str]], step_name: str = "Unknown"):
        """Debug a single LLM call with detailed logging"""
        self.interaction_count += 1
        
        print(f"\n{'='*80}")
        print(f"DEBUG STEP #{self.interaction_count}: {step_name}")
        print(f"{'='*80}")
        
        # 1. Log the input
        print(f"INPUT TO CLAUDE:")
        print(f"Model: {self.model_name}")
        print(f"Region: {os.getenv('AWS_REGION')}")
        print(f"Number of messages: {len(messages)}")
        
        for i, msg in enumerate(messages):
            print(f"\n--- Message {i+1} ({msg['role']}) ---")
            content = msg['content']
            if len(content) > 500:
                print(f"{content[:500]}... [TRUNCATED - Total length: {len(content)} chars]")
            else:
                print(content)
        
        # 2. Prepare messages for Claude (fix alternation)
        claude_messages = []
        system_prompt_content = None

        for msg in messages:
            if msg["role"] == "system":
                system_prompt_content = msg["content"]
            elif msg["role"] == "user" or msg["role"] == "assistant":
                claude_messages.append({"role": msg["role"], "content": msg["content"]})

        # Fix message alternation
        if not claude_messages:
            print("ERROR: No valid messages found!")
            return ["Error: No valid messages found."]
        
        claude_messages = self._fix_message_alternation(claude_messages)
        final_system_prompt = system_prompt_content if system_prompt_content else ""
        
        print(f"\nPROCESSED FOR CLAUDE:")
        print(f"System prompt length: {len(final_system_prompt)} chars")
        print(f"Messages after processing: {len(claude_messages)}")
        
        # 3. Make the API call
        print(f"\nMAKING API CALL...")
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                messages=claude_messages,
                temperature=0.7,
                system=final_system_prompt
            )
            
            print(f"API CALL SUCCESSFUL!")
            
            # 4. Log the response
            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                raw_response = response.content[0].text
                
                print(f"\nRAW RESPONSE FROM CLAUDE:")
                print(f"Response length: {len(raw_response)} chars")
                print(f"Response type: {type(raw_response)}")
                print("--- RAW RESPONSE START ---")
                print(repr(raw_response))  # Show with escape characters
                print("--- RAW RESPONSE END ---")
                
                print(f"\nFORMATTED RESPONSE:")
                print("--- FORMATTED RESPONSE START ---")
                print(raw_response)
                print("--- FORMATTED RESPONSE END ---")
                
                # 5. Analyze the response
                self._analyze_response(raw_response)
                
                return [raw_response]
            else:
                print(f"ERROR: Unexpected response format")
                print(f"Response content: {response.content}")
                return ["Error: Unexpected response format from Anthropic Claude on Bedrock"]

        except APIStatusError as e:
            print(f"API STATUS ERROR:")
            print(f"Status Code: {e.status_code}")
            print(f"Error: {e}")
            return [f"Error: Anthropic Bedrock API Error - {e.status_code}"]

        except APIConnectionError as e:
            print(f"API CONNECTION ERROR:")
            print(f"Error: {e}")
            return [f"Error: Anthropic Bedrock API Connection Error - {e}"]

        except Exception as e:
            print(f"UNEXPECTED ERROR:")
            print(f"Error: {e}")
            print(f"Error type: {type(e)}")
            return [f"Error: Failed to get response from Anthropic Claude - {e}"]

    def _analyze_response(self, response: str):
        """Analyze the response for common issues"""
        print(f"\nRESPONSE ANALYSIS:")
        
        # Check for markdown code blocks
        if "```" in response:
            print("WARNING: Response contains markdown code blocks (```)")
            
        # Check for common function calls
        function_patterns = [
            "get_logs", "get_metrics", "get_traces", "exec_shell", "submit"
        ]
        
        found_functions = []
        for pattern in function_patterns:
            if pattern in response:
                found_functions.append(pattern)
        
        if found_functions:
            print(f"Found function calls: {found_functions}")
        else:
            print("No recognizable function calls found")
            
        # Check response structure
        lines = response.strip().split('\n')
        print(f"Response structure:")
        print(f"   - Total lines: {len(lines)}")
        print(f"   - Non-empty lines: {len([l for l in lines if l.strip()])}")
        print(f"   - First line: {repr(lines[0]) if lines else 'None'}")
        print(f"   - Last line: {repr(lines[-1]) if lines else 'None'}")

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


def test_individual_scenarios():
    """Test individual scenarios that would occur in AIOpsLab"""
    tracer = DebugClaudeTracer()
    
    print("TESTING CLAUDE LLM INTEGRATION - STEP BY STEP DEBUG")
    print("=" * 80)
    
    # Test 1: System prompt setup (what happens during init_context)
    print("\n\nTEST 1: SYSTEM PROMPT SETUP")
    system_prompt = """
    You are a diagnostic agent for a social network service. Your goal is to detect anomalies by analyzing the system.

    Problem Description: Kubernetes target port misconfiguration in social network service

    Available Telemetry APIs:
    get_logs: Collects relevant log data from a pod using Kubectl.
    Args: namespace (str), service (str)
    Returns: str | dict | list[dicts]: Log data as a structured object or a string.

    Shell API:
    exec_shell: Execute any shell command in a predefined debugging environment.
    Args: command (str)
    Returns: str: The output of the command.

    Submit API:
    submit: Submit if anomalies are detected to the orchestrator for evaluation.
    Args: has_anomaly (str): Yes if anomalies are detected, No otherwise.
    Returns: SubmissionStatus: The status of the submission.

    CRITICAL: You must respond with ONLY a single function call like these examples:
    - get_logs("test-social-network", "text-service")
    - get_metrics("test-social-network", 5)
    - exec_shell("kubectl get pods -n test-social-network") 
    - submit("Yes") or submit("No")

    Do NOT include any explanation, reasoning, or other text. ONLY the function call.
    """
    
    initial_task = "Please analyze the social network service and start with collecting logs to understand the system state."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_task}
    ]
    
    response1 = tracer.debug_llm_call(messages, "Initial System Setup & Task")
    
    # Test 2: First iteration with input from orchestrator
    print("\n\nTEST 2: FIRST ITERATION WITH ORCHESTRATOR INPUT")
    
    orchestrator_input = "Please take the next action"
    
    messages.append({"role": "assistant", "content": response1[0]})
    messages.append({"role": "user", "content": orchestrator_input})
    
    response2 = tracer.debug_llm_call(messages, "First Orchestrator Interaction")
    
    # Test 3: Error response handling
    print("\n\nTEST 3: ERROR RESPONSE HANDLING")
    
    error_input = "Error parsing response: No API call found!\nPlease take the next action"
    
    messages.append({"role": "assistant", "content": response2[0]})
    messages.append({"role": "user", "content": error_input})
    
    response3 = tracer.debug_llm_call(messages, "Error Response Handling")
    
    # Test 4: Hindsight generation (separate call)
    print("\n\nTEST 4: HINDSIGHT GENERATION")
    
    hindsight_prompt = """
    Current situation: Error parsing response: No API call found!

    Based on this, what should be the next diagnostic step?

    Respond with ONLY a single function call in this exact format:
    - get_logs("test-social-network", "text-service") 
    - get_metrics("test-social-network", 5)
    - get_traces("test-social-network", 5)
    - exec_shell("kubectl get pods -n test-social-network")
    - submit("Yes") or submit("No")

    Do NOT include any explanation. ONLY the function call.
    """
    
    hindsight_messages = [
        {"role": "user", "content": hindsight_prompt}
    ]
    
    hindsight_response = tracer.debug_llm_call(hindsight_messages, "Hindsight Generation")
    
    # Test 5: Simple function call test
    print("\n\nTEST 5: SIMPLE FUNCTION CALL TEST")
    
    simple_prompt = "Respond with exactly this text: get_logs(\"test-social-network\", \"text-service\")"
    
    simple_messages = [
        {"role": "user", "content": simple_prompt}
    ]
    
    simple_response = tracer.debug_llm_call(simple_messages, "Simple Function Call Test")
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY OF ALL RESPONSES:")
    print("="*80)
    
    responses = [
        ("Initial Setup", response1[0]),
        ("First Iteration", response2[0]),
        ("Error Handling", response3[0]),
        ("Hindsight", hindsight_response[0]),
        ("Simple Test", simple_response[0])
    ]
    
    for i, (name, response) in enumerate(responses, 1):
        print(f"\n{i}. {name}:")
        print(f"   Length: {len(response)} chars")
        print(f"   Content: {repr(response[:100])}{'...' if len(response) > 100 else ''}")


def test_credentials():
    """Test if credentials are working"""
    print("\nTESTING AWS BEDROCK CREDENTIALS...")
    
    try:
        tracer = DebugClaudeTracer()
        test_messages = [
            {"role": "user", "content": "Say exactly: CREDENTIALS_TEST_SUCCESS"}
        ]
        
        response = tracer.debug_llm_call(test_messages, "Credential Test")
        
        if "CREDENTIALS_TEST_SUCCESS" in response[0]:
            print("CREDENTIALS WORKING!")
        else:
            print("Credentials work but response unexpected")
            
    except Exception as e:
        print(f"CREDENTIAL TEST FAILED: {e}")


def main():
    """Main debug function"""
    print("STARTING CLAUDE LLM DEBUG SESSION")
    
    # Check environment variables
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "CLAUDE_BEDROCK_MODEL_ID"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"Missing environment variables: {missing}")
        return
    
    print("All environment variables set")
    
    # Test credentials first
    test_credentials()
    
    # Test individual scenarios
    test_individual_scenarios()
    
    print("\nDEBUG SESSION COMPLETE!")
    print("Check the detailed logs above to identify where the issue occurs.")


if __name__ == "__main__":
    main()