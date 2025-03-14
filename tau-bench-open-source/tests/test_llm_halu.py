# test_llm_hallucination.py
import json
import os
import logging
from litellm import completion
import argparse
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_hallucination(model="meta-llama/Llama-3.3-70B-Instruct", api_base="http://localhost:8000/v1"):
    """Test LLM hallucination with the Sophia Silva insurance cancellation scenario."""
    
    print(f"Testing LLM hallucination with model: {model}")
    
    # Full system message from the actual logs - this matches the production environment
    system_message = """# Airline Agent Policy

The current time is 2024-05-15 15:00:00 EST.

As an airline agent, you can help users book, modify, or cancel flight reservations.

- Before taking any actions that update the booking database (booking, modifying flights, editing baggage, upgrading cabin class, or updating passenger information), you must list the action details and obtain explicit user confirmation (yes) to proceed.

- You should not provide any information, knowledge, or procedures not provided by the user or available tools, or give subjective recommendations or comments.

- You should only make one tool call at a time, and if you make a tool call, you should not respond to the user simultaneously. If you respond to the user, you should not make a tool call at the same time.

- You should deny user requests that are against this policy.

- You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions.

## Domain Basic

- Each user has a profile containing user id, email, addresses, date of birth, payment methods, reservation numbers, and membership tier.

- Each reservation has an reservation id, user id, trip type (one way, round trip), flights, passengers, payment methods, created time, baggages, and travel insurance information.

- Each flight has a flight number, an origin, destination, scheduled departure and arrival time (local time), and for each date:
  - If the status is "available", the flight has not taken off, available seats and prices are listed.
  - If the status is "delayed" or "on time", the flight has not taken off, cannot be booked.
  - If the status is "flying", the flight has taken off but not landed, cannot be booked.

## Book flight

- The agent must first obtain the user id, then ask for the trip type, origin, destination.

- Passengers: Each reservation can have at most five passengers. The agent needs to collect the first name, last name, and date of birth for each passenger. All passengers must fly the same flights in the same cabin.

- Payment: each reservation can use at most one travel certificate, at most one credit card, and at most three gift cards. The remaining amount of a travel certificate is not refundable. All payment methods must already be in user profile for safety reasons.

- Checked bag allowance: If the booking user is a regular member, 0 free checked bag for each basic economy passenger, 1 free checked bag for each economy passenger, and 2 free checked bags for each business passenger. If the booking user is a silver member, 1 free checked bag for each basic economy passenger, 2 free checked bag for each economy passenger, and 3 free checked bags for each business passenger. If the booking user is a gold member, 2 free checked bag for each basic economy passenger, 3 free checked bag for each economy passenger, and 3 free checked bags for each business passenger. Each extra baggage is 50 dollars.

- Travel insurance: the agent should ask if the user wants to buy the travel insurance, which is 30 dollars per passenger and enables full refund if the user needs to cancel the flight given health or weather reasons.

## Modify flight

- The agent must first obtain the user id and the reservation id.

- Change flights: Basic economy flights cannot be modified. Other reservations can be modified without changing the origin, destination, and trip type. Some flight segments can be kept, but their prices will not be updated based on the current price. The API does not check these for the agent, so the agent must make sure the rules apply before calling the API!

- Change cabin: all reservations, including basic economy, can change cabin without changing the flights. Cabin changes require the user to pay for the difference between their current cabin and the new cabin class. Cabin class must be the same across all the flights in the same reservation; changing cabin for just one flight segment is not possible.

- Change baggage and insurance: The user can add but not remove checked bags. The user cannot add insurance after initial booking.

- Change passengers: The user can modify passengers but cannot modify the number of passengers. This is something that even a human agent cannot assist with.

- Payment: If the flights are changed, the user needs to provide one gift card or credit card for payment or refund method. The agent should ask for the payment or refund method instead.

## Cancel flight

- The agent must first obtain the user id, the reservation id, and the reason for cancellation (change of plan, airline cancelled flight, or other reasons)

- All reservations can be cancelled within 24 hours of booking, or if the airline cancelled the flight. Otherwise, basic economy or economy flights can be cancelled only if travel insurance is bought and the condition is met, and business flights can always be cancelled. The rules are strict regardless of the membership status. The API does not check these for the agent, so the agent must make sure the rules apply before calling the API!

- The agent can only cancel the whole trip that is not flown. If any of the segments are already used, the agent cannot help and transfer is needed.

- The refund will go to original payment methods in 5 to 7 business days.

## Refund

- If the user is silver/gold member or has travel insurance or flies business, and complains about cancelled flights in a reservation, the agent can offer a certificate as a gesture after confirming the facts, with the amount being $100 times the number of passengers.

- If the user is silver/gold member or has travel insurance or flies business, and complains about delayed flights in a reservation and wants to change or cancel the reservation, the agent can offer a certificate as a gesture after confirming the facts and changing or cancelling the reservation, with the amount being $50 times the number of passengers.

- Do not proactively offer these unless the user complains about the situation and explicitly asks for some compensation. Do not compensate if the user is regular member and has no travel insurance and flies (basic) economy.
"""
    
    # Conversation setup - this now matches how the production system formats the messages
    # Note the task instruction is sent as a user message, not system message
    messages = [
        {"role": "system", "content": system_message},
        # First user message with task instruction
        {"role": "user", "content": "You are Sophia Silva (with ID: sophia_silva_7557), you want to get a refund for the insurance you purchased for your flight (confirmation: H8Q05L) but you don't want to cancel the flight itself. You are not happy with the service you received and you want to cancel the insurance and get a full refund.\n\nHi, I'm looking to cancel the insurance I purchased for my flight."}
    ]
    
    # Full set of tools from the production environment
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get user details using their ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user ID to look up."
                        }
                    },
                    "required": ["user_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_reservation_details",
                "description": "Get reservation details using the reservation ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reservation_id": {
                            "type": "string",
                            "description": "The reservation ID to look up."
                        }
                    },
                    "required": ["reservation_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "think",
                "description": "Think about the problem and reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your thinking process"
                        }
                    },
                    "required": ["thought"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "transfer_to_human_agents",
                "description": "Transfer the customer to a human agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of the issue to provide to the human agent"
                        }
                    },
                    "required": ["summary"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to calculate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    # Track all tool calls like in the production environment
    all_tool_calls = []
    
    # First response - match the production setup
    try:
        print("\n==== Making API request to model ====")
        response1 = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # Match production setting
            custom_llm_provider="openai",
            api_key="dummy-key",
            api_base=api_base,
            temperature=0.2
        )
        
        # Extract and analyze first response
        res1 = response1.choices[0].message
        print("\n==== FIRST RESPONSE ====")
        print(f"Content: {res1.content if res1.content else 'None'}")
        print(f"Has tool calls: {hasattr(res1, 'tool_calls') and len(res1.tool_calls) > 0}")
        
        # Process tool calls similar to production
        if hasattr(res1, 'tool_calls') and res1.tool_calls:
            for tool_call in res1.tool_calls:
                if hasattr(tool_call, 'function'):
                    function = tool_call.function
                    print(f"Tool called: {function.name}")
                    
                    try:
                        args = json.loads(function.arguments)
                        print(f"Arguments: {args}")
                        all_tool_calls.append({
                            "name": function.name,
                            "arguments": args
                        })
                        
                        # Process "think" and "transfer_to_human_agents" specially
                        if function.name == "think":
                            # In production, this returns empty string
                            tool_response = ""
                        elif function.name == "transfer_to_human_agents":
                            tool_response = "Transfer successful"
                        elif function.name == "get_user_details":
                            # Simulate the error response
                            if args.get("user_id") != "sophia_silva_7557":
                                tool_response = "Error: user not found"
                            else:
                                tool_response = json.dumps({
                                    "name": {"first_name": "Sophia", "last_name": "Silva"},
                                    "user_id": "sophia_silva_7557",
                                    "membership": "regular",
                                    "reservations": ["H8Q05L"]
                                })
                        elif function.name == "get_reservation_details":
                            # Simulate error for unrecognized reservations
                            tool_response = "Error: reservation not found"
                        else:
                            tool_response = "Tool response"
                            
                        # Add the tool response to messages like in production
                        messages.append({"role": "assistant", "content": None, "tool_calls": [
                            {"id": f"call_{tool_call.id}", "type": "function", "function": {"name": function.name, "arguments": function.arguments}}
                        ]})
                        messages.append({"role": "tool", "tool_call_id": f"call_{tool_call.id}", "name": function.name, "content": tool_response})
                        
                    except json.JSONDecodeError:
                        print(f"Error parsing arguments: {function.arguments}")
        else:
            # Add regular response
            messages.append({"role": "assistant", "content": res1.content})
            
        # Add user follow-up - matching log format
        messages.append({"role": "user", "content": "I was really not satisfied with the service, and I believe I should be eligible for a refund. Could you help me with that, please?"})
        
        # Second response - match the production setup
        print("\n==== Making second API request ====")
        response2 = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            custom_llm_provider="openai",
            api_key="dummy-key",
            api_base=api_base,
            temperature=0.2
        )
        
        # Extract and analyze second response
        res2 = response2.choices[0].message
        print("\n==== SECOND RESPONSE ====")
        print(f"Content: {res2.content if res2.content else 'No content'}")
        
        if hasattr(res2, 'tool_calls') and res2.tool_calls:
            for tool_call in res2.tool_calls:
                if hasattr(tool_call, 'function'):
                    function = tool_call.function
                    print(f"Tool called: {function.name}")
                    
                    try:
                        args = json.loads(function.arguments)
                        print(f"Arguments: {args}")
                        all_tool_calls.append({
                            "name": function.name,
                            "arguments": args
                        })
                        
                        # Check for ID hallucination specifically in get_user_details
                        if function.name == "get_user_details":
                            if args.get("user_id") != "sophia_silva_7557":
                                print("\n❌ HALLUCINATION DETECTED!")
                                print(f"Expected ID: sophia_silva_7557")
                                print(f"Hallucinated ID: {args.get('user_id')}")
                                print("The model ignored the provided ID and hallucinated a different one.")
                            else:
                                print("\n✅ No hallucination detected in user ID.")
                                
                    except json.JSONDecodeError:
                        print(f"Error parsing arguments: {function.arguments}")
                        
        # Show all tool calls for analysis
        print("\n==== ALL TOOL CALLS ====")
        for i, call in enumerate(all_tool_calls):
            print(f"{i+1}. {call['name']}: {call['arguments']}")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")
        
    return all_tool_calls

def test_multiple_formats():
    """Test multiple formats of presenting the user ID to find which one works best."""
    formats = [
        # Format 1: Original task format (most likely to hallucinate)
        "You are Sophia Silva (with ID: sophia_silva_7557), you want to get a refund for the insurance you purchased for your flight (confirmation: H8Q05L) but you don't want to cancel the flight itself.",
        
        # Format 2: Explicit ID mention first
        "User ID: sophia_silva_7557. You are Sophia Silva, and you want to get a refund for the insurance you purchased for your flight (confirmation: H8Q05L) but you don't want to cancel the flight itself.",
        
        # Format 3: Separate lines with KEY: value format
        "NAME: Sophia Silva\nUSER ID: sophia_silva_7557\nREQUEST: Get a refund for flight insurance without canceling the flight (confirmation: H8Q05L).",
        
        # Format 4: System instruction + user message
        "<<SYSTEM>> You are helping user sophia_silva_7557 (Sophia Silva) <<USER>> I'm looking to cancel the insurance I purchased for my flight (confirmation: H8Q05L)."
    ]
    
    results = {}
    for i, format_text in enumerate(formats):
        print(f"\n\n=== TESTING FORMAT {i+1} ===")
        print(format_text)
        
        model = "meta-llama/Llama-3.3-70B-Instruct"
        api_base = "http://localhost:8000/v1"
        
        # Call test with this format
        # Create full user message with format + request
        user_msg = format_text + "\n\nHi, I'm looking to cancel the insurance I purchased for my flight."
        
        # Configure test for this format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_msg}
        ]
        
        # Run test with this format (implementation omitted for brevity)
        # Record results
        results[f"Format {i+1}"] = "Test results for format"
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLM hallucination")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="Model to test")
    parser.add_argument("--api_base", default="http://localhost:8000/v1", help="API base URL")
    parser.add_argument("--test_formats", action="store_true", help="Test multiple formats for presenting user ID")
    args = parser.parse_args()
    
    if args.test_formats:
        test_multiple_formats()
    else:
        test_llm_hallucination(model=args.model, api_base=args.api_base)