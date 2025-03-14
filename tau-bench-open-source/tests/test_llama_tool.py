from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# Define a simple tool
tools = [{
    "type": "function",
    "function": {
        "name": "find_user_id_by_name_zip",
        "description": "Find a user by name and zip code",
        "parameters": {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "zip": {"type": "string"}
            },
            "required": ["first_name", "last_name", "zip"]
        }
    }
}]

# Test the endpoint
try:
    response = client.chat.completions.create(
        model=client.models.list().data[0].id,
        messages=[{"role": "user", "content": "Find user Mei Patel in zip code 76165"}],
        tools=tools,
        tool_choice="auto"
    )
    print(json.dumps(response.model_dump(), indent=2))
except Exception as e:
    print(f"Error: {str(e)}") 