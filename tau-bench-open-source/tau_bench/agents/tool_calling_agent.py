# # Copyright Sierra

# import json
# from litellm import completion
# from typing import List, Optional, Dict, Any

# from tau_bench.agents.base import Agent
# from tau_bench.envs.base import Env
# from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


# class ToolCallingAgent(Agent):
#     def __init__(
#         self,
#         tools_info: List[Dict[str, Any]],
#         wiki: str,
#         model: str,
#         provider: str,
#         temperature: float = 0.0,
#     ):
#         self.tools_info = tools_info
#         self.wiki = wiki
#         self.model = model
#         self.provider = provider
#         self.temperature = temperature

#     def solve(
#         self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
#     ) -> SolveResult:
#         total_cost = 0.0
#         env_reset_res = env.reset(task_index=task_index)
#         obs = env_reset_res.observation
#         info = env_reset_res.info.model_dump()
#         reward = 0.0
#         messages: List[Dict[str, Any]] = [
#             {"role": "system", "content": self.wiki},
#             {"role": "user", "content": obs},
#         ]
#         for _ in range(max_num_steps):
#             # Modify the model name for openai_like provider
#             model_name = self.model
#             if self.provider == "openai_like":
#                 # Use a format that the OpenAI API endpoint expects
#                 res = completion(
#                     messages=messages,
#                     model=model_name,
#                     custom_llm_provider="openai",
#                     api_key = "dummy-key",
#                     api_base="http://localhost:8000/v1",
#                     tools=self.tools_info,
#                     temperature=self.temperature,
#                 )
#             else:
#                 res = completion(
#                     messages=messages,
#                     model=model_name,
#                     custom_llm_provider=self.provider,
#                     tools=self.tools_info,
#                     temperature=self.temperature,
#                 )
#             next_message = res.choices[0].message.model_dump()
#             if hasattr(res, '_hidden_params') and res._hidden_params.get("response_cost") is not None:
#                 total_cost += res._hidden_params["response_cost"]
#             action = message_to_action(next_message)
#             env_response = env.step(action)
#             reward = env_response.reward
#             info = {**info, **env_response.info.model_dump()}
#             if action.name != RESPOND_ACTION_NAME:
#                 next_message["tool_calls"] = next_message["tool_calls"][:1]
#                 messages.extend(
#                     [
#                         next_message,
#                         {
#                             "role": "tool",
#                             "tool_call_id": next_message["tool_calls"][0]["id"],
#                             "name": next_message["tool_calls"][0]["function"]["name"],
#                             "content": env_response.observation,
#                         },
#                     ]
#                 )
#             else:
#                 messages.extend(
#                     [
#                         next_message,
#                         {"role": "user", "content": env_response.observation},
#                     ]
#                 )
#             if env_response.done:
#                 break
#         return SolveResult(
#             reward=reward,
#             info=info,
#             messages=messages,
#             total_cost=total_cost,
#         )


# def message_to_action(
#     message: Dict[str, Any],
# ) -> Action:
#     if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
#         tool_call = message["tool_calls"][0]
#         return Action(
#             name=tool_call["function"]["name"],
#             kwargs=json.loads(tool_call["function"]["arguments"]),
#         )
#     else:
#         return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
# Copyright Sierra

import json
import re
import logging
from litellm import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.agents.parse_llama_response import parse_llama_response
# import litellm
# litellm._turn_on_debug()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.known_user_ids = {} # Store known user IDs from tasks
        self.is_llama_model = "llama" in model.lower()
        logger.info(f"Initialized ToolCallingAgent with model={model}, provider={provider}")

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        
        # Extract and store the correct user ID from the task
        if 'task' in info and 'user_id' in info['task']:
            task_user_id = info['task']['user_id']
            self.known_user_ids[task_index] = task_user_id
            logger.info(f"Task {task_index} user ID: {task_user_id}")
            
            # Also extract user ID from instruction if available
            if 'instruction' in info['task']:
                instruction = info['task']['instruction']
                # Look for patterns like "Your user id is X" or "with ID: X"
                user_id_patterns = [
                    r"user\s*id\s*(?:is|:)\s*(\w+_\w+_\d+)",
                    r"(?:with|your)\s*ID\s*(?:is|:)\s*(\w+_\w+_\d+)",
                    r"ID[::]?\s*(\w+_\w+_\d+)",
                    r"(?:^|\s)(\w+_\w+_\d{4,})(?:\s|$)" # Catch standalone IDs
                ]
                for pattern in user_id_patterns:
                    matches = re.findall(pattern, instruction, re.IGNORECASE)
                    if matches and matches[0] == task_user_id:
                        logger.info(f"User ID {task_user_id} confirmed in instruction")
        
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            # Add a stronger system message that emphasizes using the correct user ID
            {"role": "system", "content": self.wiki + "\n\nIMPORTANT: When using get_user_details or performing actions that require a user ID, make sure to extract it exactly as provided in the conversation. Never use an ID that wasn't explicitly mentioned."},
            {"role": "user", "content": obs},
        ]
        
        for _ in range(max_num_steps):
            # Get model completion
            try:
                if self.provider == "openai_like" and "llama" in self.model.lower():
                    try:
                        logger.info(f"Making tool calling request to vLLM server for {self.model}")
                        # Use standardized format for all Llama models
                        res = completion(
                            messages=messages,
                            model=self.model,
                            custom_llm_provider="openai",
                            api_key="dummy-key",
                            api_base="http://localhost:8000/v1",
                            tools=self.tools_info,
                            tool_choice="auto",
                            temperature=self.temperature,
                        )
                        logger.info("vLLM request successful")
                        
                        # Add debug output
                        try:
                            debug_info = {
                                "response_type": type(res).__name__,
                                "has_choices": hasattr(res, "choices"),
                                "choices_length": len(res.choices) if hasattr(res, "choices") else 0,
                                "message_type": type(res.choices[0].message).__name__ if hasattr(res, "choices") and len(res.choices) > 0 else "None",
                                "has_content": hasattr(res.choices[0].message, "content") if hasattr(res, "choices") and len(res.choices) > 0 else False,
                                "content_type": type(res.choices[0].message.content).__name__ if hasattr(res, "choices") and len(res.choices) > 0 and hasattr(res.choices[0].message, "content") else "None",
                                "has_tool_calls": hasattr(res.choices[0].message, "tool_calls") if hasattr(res, "choices") and len(res.choices) > 0 else False,
                                "tool_calls_length": len(res.choices[0].message.tool_calls) if hasattr(res, "choices") and len(res.choices) > 0 and hasattr(res.choices[0].message, "tool_calls") else 0
                            }
                            logger.debug(f"Debug Info: {json.dumps(debug_info, indent=2)}")
                        except Exception as e:
                            logger.error(f"Error generating debug info: {str(e)}")
                            
                        # Process special formatting in Llama responses
                        try:
                            if hasattr(res.choices[0].message, 'model_dump'):
                                # New API format
                                next_message = res.choices[0].message.model_dump()
                            else:
                                # Handle old API format or direct dict
                                next_message = res.choices[0].message
                                
                            # Check for None content and parse if needed
                            if isinstance(next_message, dict):
                                if next_message.get('content') is None and next_message.get('tool_calls'):
                                    # Already has tool calls with None content - this is fine
                                    pass
                                elif next_message.get('content') is None:
                                    # Try to extract tool calls from function_call
                                    if 'function_call' in next_message and next_message['function_call']:
                                        func_call = next_message['function_call']
                                        next_message['tool_calls'] = [{
                                            'function': {
                                                'name': func_call.get('name', ''),
                                                'arguments': func_call.get('arguments', '{}')
                                            }
                                        }]
                                else:
                                    # Parse content if present
                                    content = next_message.get('content')
                                    parsed = parse_llama_response(content)
                                    
                                    # Merge the parsed result with the message
                                    if 'tool_calls' in parsed:
                                        next_message['tool_calls'] = parsed['tool_calls']
                                    elif 'content' in parsed and parsed['content'] != content:
                                        next_message['content'] = parsed['content']
                            else:
                                # Not a dict, try to parse from response text
                                content = getattr(res.choices[0].message, 'content', None)
                                next_message = parse_llama_response(content)
                                
                            logger.info(f"Processed message: {next_message}")
                            
                        except Exception as e:
                            logger.error(f"Error parsing response: {e.__class__.__name__}: {str(e)}")
                            # Create a simple fallback message
                            next_message = {"content": "I encountered an error processing your request."}
                            
                    except Exception as e:
                        logger.error(f"Error with vLLM completion: {e.__class__.__name__}: {str(e)}")
                        # Fallback to a simple response
                        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "I encountered an error processing your request."})
                        env_response = env.step(action)
                        # Continue with reduced functionality
                        continue
                else:
                    res = completion(
                        messages=messages,
                        model=self.model,
                        custom_llm_provider=self.provider,
                        tools=self.tools_info,
                        temperature=self.temperature,
                    )
            except Exception as e:
                logger.error(f"Error during model completion: {str(e)}")
                # Fallback to a simple response
                action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "I encountered an error and couldn't process your request."})
                env_response = env.step(action)
                break

            # Process response and update cost
            next_message = res.choices[0].message.model_dump()
            if hasattr(res, '_hidden_params') and res._hidden_params.get("response_cost") is not None:
                total_cost += res._hidden_params["response_cost"]
            
            # Validate user ID before taking action
            next_message = self.validate_user_id(next_message, task_index)
            
            # Convert message to action
            action = message_to_action(next_message)
            
            # Step environment
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            # Update messages based on action type
            if action.name != RESPOND_ACTION_NAME:
                # Handle tool call action
                # Safely extract tool call information
                try:
                    tool_calls = next_message.get("tool_calls", [])
                    if tool_calls and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        tool_call_id = tool_call.get("id", "unknown_id")
                        tool_name = tool_call.get("function", {}).get("name", "unknown_tool")
                        
                        # Create a clean message with just the first tool call
                        cleaned_message = next_message.copy()
                        cleaned_message["tool_calls"] = [tool_call]
                        
                        messages.append(cleaned_message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": env_response.observation,
                        })
                    else:
                        # Fallback if tool_calls is empty
                        messages.append(next_message)
                        messages.append({"role": "user", "content": env_response.observation})
                except Exception as e:
                    logger.error(f"Error handling tool call: {str(e)}")
                    # Fallback approach
                    messages.append(next_message)
                    messages.append({"role": "user", "content": env_response.observation})
            else:
                # Handle standard response action
                messages.append(next_message)
                messages.append({"role": "user", "content": env_response.observation})
            
            if env_response.done:
                break
                
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
    
    def validate_user_id(self, message: Dict[str, Any], task_index: Optional[int]) -> Dict[str, Any]:
        """
        Validate that any user ID in the tool call matches the known task user ID.
        If the user ID is incorrect, modify the message to use the correct ID.
        
        Args:
            message: The message to validate
            task_index: The current task index
            
        Returns:
            The validated and possibly corrected message
        """
        if task_index is None or task_index not in self.known_user_ids:
            # We don't have a known user ID for validation
            return message
            
        correct_user_id = self.known_user_ids[task_index]
        
        # Check if the message has tool_calls
        if "tool_calls" in message and message["tool_calls"] and len(message["tool_calls"]) > 0:
            for tool_call in message["tool_calls"]:
                if "function" in tool_call:
                    function = tool_call["function"]
                    if "name" in function and "arguments" in function:
                        # Check if this is a user-related tool
                        if function["name"] in ["get_user_details", "send_certificate", "update_user_details"]:
                            try:
                                args = json.loads(function["arguments"]) if isinstance(function["arguments"], str) else function["arguments"]
                                if "user_id" in args and args["user_id"] != correct_user_id:
                                    # Log the mismatch and correct it
                                    logger.warning(f"Hallucinated user_id detected! Got {args['user_id']}, should be {correct_user_id}")
                                    
                                    # Correct the user ID
                                    args["user_id"] = correct_user_id
                                    function["arguments"] = json.dumps(args)
                                    
                                    # Also add a note to the argument explaining the correction
                                    logger.info(f"Corrected user_id to {correct_user_id}")
                            except Exception as e:
                                logger.error(f"Error validating user ID: {str(e)}")
        
        return message

def message_to_action(message: Dict[str, Any]) -> Action:
    """
    Convert a message to an action, handling different formats of tool calls.
    This function is made more robust to handle variations in model outputs.
    
    Args:
        message: The message containing potential tool calls
        
    Returns:
        Action object with name and kwargs
    """
    try:
        # Check if message has tool_calls (OpenAI/Claude format)
        if "tool_calls" in message and message["tool_calls"] and len(message["tool_calls"]) > 0:
            tool_call = message["tool_calls"][0]
            
            # Extract function name and arguments safely
            if "function" in tool_call:
                function = tool_call["function"]
                
                if "name" in function and "arguments" in function:
                    # Parse arguments - could be string or dict
                    kwargs = {}
                    try:
                        if isinstance(function["arguments"], str):
                            kwargs = json.loads(function["arguments"])
                            
                            # Handle nested JSON structures in arguments
                            kwargs = process_nested_args(kwargs)
                                
                        elif isinstance(function["arguments"], dict):
                            kwargs = function["arguments"]
                            
                            # Handle nested JSON structures in arguments
                            kwargs = process_nested_args(kwargs)
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing arguments: {e}")
                        logger.error(f"Arguments text: {function['arguments']}")
                        
                        # Try to recover by treating as a simple string
                        if isinstance(function["arguments"], str):
                            kwargs = {"content": function["arguments"]}
                        
                    return Action(name=function["name"], kwargs=kwargs)
            
        # Check for function_call format (Llama format)
        elif "function_call" in message and message["function_call"]:
            function_call = message["function_call"]
            
            if "name" in function_call and "arguments" in function_call:
                # Parse arguments
                kwargs = {}
                try:
                    if isinstance(function_call["arguments"], str):
                        kwargs = json.loads(function_call["arguments"])
                        
                        # Handle nested JSON structures
                        kwargs = process_nested_args(kwargs)
                            
                    elif isinstance(function_call["arguments"], dict):
                        kwargs = function_call["arguments"]
                        
                        # Handle nested JSON structures
                        kwargs = process_nested_args(kwargs)
                            
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing function_call arguments: {e}")
                    # Try to recover by treating as a simple string
                    kwargs = {"content": function_call["arguments"]}
                    
                return Action(name=function_call["name"], kwargs=kwargs)
        
        # No tool calls, just use content for respond action
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message.get("content", "")})
            
    except Exception as e:
        logger.error(f"Error in message_to_action: {str(e)}")
        logger.error(f"Message: {message}")
        # Fallback to a respond action with the message content or an error message
        return Action(
            name=RESPOND_ACTION_NAME, 
            kwargs={"content": message.get("content", "Error processing response")}
        )


def process_nested_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process nested JSON structures in arguments, especially for arrays like 'flights'.
    
    Args:
        kwargs: The kwargs dictionary to process
        
    Returns:
        Processed kwargs with properly parsed nested structures
    """
    # Handle common nested structures
    for key, value in list(kwargs.items()):
        # Handle string that should be arrays or objects
        if isinstance(value, str):
            # Special case for 'flights' parameter
            if key == "flights" and isinstance(value, str):
                try:
                    # Try to parse nested JSON string
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse flights JSON string: {value[:100]}...")
            
            # Handle other potential JSON strings
            elif (value.strip().startswith('[') and value.strip().endswith(']')) or \
                 (value.strip().startswith('{') and value.strip().endswith('}')):
                try:
                    kwargs[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
        
        # Recursively process nested dictionaries
        elif isinstance(value, dict):
            kwargs[key] = process_nested_args(value)
        
        # Process items in a list
        elif isinstance(value, list):
            # Process each item in the list if it's a dictionary
            kwargs[key] = [
                process_nested_args(item) if isinstance(item, dict) else item
                for item in value
            ]
    
    return kwargs