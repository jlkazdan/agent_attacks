import re
import json
import logging
from typing import Dict, Any, Union, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_llama_response(response_text: Union[str, None]) -> Dict[str, Any]:
    """
    Parse Llama's responses when they contain Python-like tags or other anomalies.
    Returns a cleaned tool call or text response.
    
    Args:
        response_text: The response text from Llama model, could be None
        
    Returns:
        Dict containing either tool_calls or content
    """
    # Handle None content case (when model is making a tool call)
    if response_text is None:
        # This typically means the model intended to make a tool call
        # but the formatting is in a different structure
        logger.debug("Received None response, returning empty tool_calls")
        return {"tool_calls": []}
    
    # Check for Python tags - match more complex formats
    # This improved pattern can handle both simple and complex arguments
    python_tag_pattern = r'<\|python_tag\|>(\w+)\((.*?)\)'
    match = re.search(python_tag_pattern, response_text, re.DOTALL)
    
    if match:
        function_name = match.group(1)
        args_text = match.group(2).strip()
        logger.debug(f"Found Python tag: function={function_name}, args={args_text}")
        
        # Parse arguments based on format
        try:
            # Handle different argument formats
            if args_text.startswith('{') and args_text.endswith('}'):
                # JSON object format
                try:
                    args = json.loads(args_text)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as a string
                    args = parse_quoted_string(args_text)
            elif args_text.startswith('[') and args_text.endswith(']'):
                # JSON array format
                try:
                    args = json.loads(args_text)
                except json.JSONDecodeError:
                    args = {"content": args_text}
            elif args_text.startswith('"') and args_text.endswith('"'):
                # Quoted string format
                args = parse_quoted_string(args_text)
            else:
                # Simple arguments or multiple arguments
                args = parse_function_args(args_text, function_name)
                
            # Special case handlers for common functions
            if function_name == "think" and isinstance(args, dict) and "thought" not in args:
                args = {"thought": next(iter(args.values())) if args else args_text}
            elif function_name == "transfer_to_human_agents" and isinstance(args, dict) and "summary" not in args:
                args = {"summary": next(iter(args.values())) if args else args_text}
                
            # Return properly formatted tool call
            return {
                "tool_calls": [{
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(args)
                    }
                }]
            }
        except Exception as e:
            logger.warning(f"Error parsing Python tag arguments: {str(e)}")
            # Fallback to simple argument format
            return {
                "tool_calls": [{
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps({"content": args_text})
                    }
                }]
            }
    
    # If it's a JSON response, try to parse it
    if isinstance(response_text, str) and response_text.strip().startswith('{') and response_text.strip().endswith('}'):
        try:
            parsed = json.loads(response_text)
            
            # Handle OpenAI-style format
            if "name" in parsed and "parameters" in parsed:
                # Special handling for nested JSON strings within the parameters
                if isinstance(parsed["parameters"], dict):
                    parsed_parameters = process_nested_json_strings(parsed["parameters"])
                    
                return {
                    "tool_calls": [{
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed_parameters)
                        }
                    }]
                }
                
            # Handle direct function call format
            elif "function" in parsed and isinstance(parsed["function"], dict):
                function = parsed["function"]
                if "name" in function and "arguments" in function:
                    # Process nested arguments if needed
                    if isinstance(function["arguments"], str):
                        try:
                            args = json.loads(function["arguments"])
                            # Process any nested JSON strings inside the arguments
                            args = process_nested_json_strings(args)
                            function["arguments"] = json.dumps(args)
                        except json.JSONDecodeError:
                            pass
                    
                    return {"tool_calls": [parsed]}
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse as JSON: {response_text[:100]}...")
    
    # Default to returning as content
    return {"content": response_text}

def parse_quoted_string(text: str) -> Dict[str, str]:
    """Parse a quoted string and return an appropriate argument dict."""
    # Remove surrounding quotes if present
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    # Unescape any escaped quotes
    text = text.replace('\\"', '"')
    return {"content": text}

def parse_function_args(args_text: str, function_name: str) -> Dict[str, Any]:
    """
    Parse function arguments in various formats.
    
    Args:
        args_text: The argument text
        function_name: The function name (used for context-specific parsing)
        
    Returns:
        Dict containing parsed arguments
    """
    # Check if multiple arguments separated by commas
    if ',' in args_text and not (args_text.startswith('"') and args_text.endswith('"')):
        try:
            # Handle key-value pairs
            if ':' in args_text or '=' in args_text:
                args = {}
                # Split by commas not inside quotes or brackets
                parts = split_args(args_text)
                for part in parts:
                    # Check for key:value or key=value
                    if ':' in part:
                        key, value = part.split(':', 1)
                    elif '=' in part:
                        key, value = part.split('=', 1)
                    else:
                        continue
                        
                    key = key.strip().strip('"\'')
                    value = value.strip()
                    
                    # Parse the value
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith('{') and value.endswith('}'):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    elif value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif is_float(value):
                        value = float(value)
                        
                    args[key] = value
                return args
            else:
                # Positional arguments - convert to a dictionary with appropriate keys
                # based on function name context
                parts = [p.strip().strip('"\'') for p in args_text.split(',')]
                if function_name == "get_user_details" and len(parts) == 1:
                    return {"user_id": parts[0]}
                elif function_name in ["get_reservation_details", "cancel_reservation"] and len(parts) == 1:
                    return {"reservation_id": parts[0]}
                else:
                    # Default handling - use positional parameters
                    return {f"param{i}": val for i, val in enumerate(parts)}
        except Exception as e:
            logger.warning(f"Error parsing multiple arguments: {str(e)}")
            
    # If we get here, treat it as a single argument
    return {"content": args_text}

def process_nested_json_strings(params: Dict[str, Any]) -> Dict[str, Any]:
    """Process a dictionary, converting any nested JSON strings to objects."""
    result = {}
    for key, value in params.items():
        if isinstance(value, str):
            # Check if it's a JSON array or object
            if (value.strip().startswith('[') and value.strip().endswith(']')) or \
               (value.strip().startswith('{') and value.strip().endswith('}')):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            result[key] = process_nested_json_strings(value)
        elif isinstance(value, list):
            # Process items in a list
            result[key] = [
                process_nested_json_strings(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result

def split_args(args_text: str) -> List[str]:
    """
    Split arguments by commas, but respect quotes and brackets.
    
    Args:
        args_text: Text containing comma-separated arguments
        
    Returns:
        List of individual arguments
    """
    parts = []
    current = ""
    in_quotes = False
    bracket_level = 0
    
    for char in args_text:
        if char == '"' and (not current or current[-1] != '\\'):
            in_quotes = not in_quotes
            current += char
        elif char == '{' and not in_quotes:
            bracket_level += 1
            current += char
        elif char == '}' and not in_quotes:
            bracket_level -= 1
            current += char
        elif char == ',' and not in_quotes and bracket_level == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
            
    if current.strip():
        parts.append(current.strip())
        
    return parts

def is_float(value: str) -> bool:
    """Check if a string can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False 