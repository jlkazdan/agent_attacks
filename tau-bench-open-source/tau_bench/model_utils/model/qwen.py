from typing import Any, Dict, List, Optional
from tau_bench.model_utils.model.vllm import VLLMChatModel

class QwenModel(VLLMChatModel):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        print("Initializing QwenModel")  # Debug print
        super().__init__(model, base_url, api_key, temperature)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print("Converting messages:", messages)  # Debug print
        converted_messages = []
        
        # First, handle system message if present
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        if system_msg:
            converted_messages.append({
                "role": "system",
                "content": system_msg["content"]
            })
        
        # Then handle the conversation
        for message in messages:
            if message["role"] == "system":
                continue
                
            new_message = {
                "role": message["role"],
                "content": message.get("content", "")
            }
            
            if message["role"] == "assistant":
                new_message["role"] = "developer"
                if "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    if tool_calls and tool_calls[0].function:
                        new_message["function_call"] = {
                            "name": tool_calls[0].function.name,
                            "arguments": tool_calls[0].function.arguments
                        }
            elif message["role"] == "user":
                new_message["role"] = "user"
            
            converted_messages.append(new_message)
        
        print("After conversion:", converted_messages)  # Debug print
        return converted_messages

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        print("Chat method called with:", messages, kwargs)  # Debug print
        
        # Convert tools to functions
        if "tools" in kwargs:
            functions = []
            for tool in kwargs.pop("tools"):
                functions.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "parameters": tool["function"]["parameters"]
                })
            kwargs["functions"] = functions
        
        converted_messages = self._convert_messages(messages)
        print("About to call super().chat with:", converted_messages, kwargs)  # Debug print
        response = super().chat(converted_messages, **kwargs)
        print("Got response:", response)  # Debug print
        
        # Convert response back to OpenAI format
        if response.get("role") == "developer":
            response["role"] = "assistant"
        
        if "function_call" in response:
            func_call = response.pop("function_call")
            response["tool_calls"] = [{
                "type": "function",
                "function": {
                    "name": func_call["name"],
                    "arguments": func_call["arguments"]
                }
            }]
        
        return response 