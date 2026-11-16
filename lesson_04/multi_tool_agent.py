
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# =====================================
# 1. CONFIGURATION
# =====================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = "gpt-5.5"

INSTRUCTIONS = """
You are an e-commerce customer support agent.

You can check product inventory, retrieve prices,
and calculate delivery fees.

Rules:
- Use tools whenever the customer asks for business information.
- Never invent prices, stock quantities or delivery fees.
- If a product is unavailable, clearly explain that.
- Combine information from multiple tools when necessary.
- Only answer using information returned by the tools.
- Do not claim that an order has been placed or paid for.
"""

# =====================================
# 2. SAMPLE PRODUCT DATABASE
# =====================================

products = {
    "samsung galaxy s26": {
        "name": "Samsung Galaxy S26",
        "price": 1_250_000,
        "stock": 8
    },
    "iphone 17 pro": {
        "name": "iPhone 17 Pro",
        "price": 2_100_000,
        "stock": 3
    },
    "hp laptop": {
        "name": "HP Laptop",
        "price": 1_800_000,
        "stock": 0
    },
    "wireless headphones": {
        "name": "Wireless Headphones",
        "price": 85_000,
        "stock": 15
    }
}

# =====================================
# 3. DELIVERY DATABASE
# =====================================

delivery_rates = {
    "lagos": 3500,
    "abuja": 5000,
    "port harcourt": 4500
}

# =====================================
# 4. PYTHON TOOLS
# =====================================

def check_inventory(product_name: str) -> dict:

    product = products.get(product_name.lower())

    if product is None:
        return {
            "success": False,
            "message": "Product not found."
        }

    return {
        "success": True,
        "product": product["name"],
        "available": product["stock"] > 0,
        "stock_quantity": product["stock"]
    }


def get_product_price(product_name: str) -> dict:

    product = products.get(product_name.lower())

    if product is None:
        return {
            "success": False,
            "message": "Product not found."
        }

    return {
        "success": True,
        "product": product["name"],
        "price_ngn": product["price"]
    }


def calculate_delivery(city: str) -> dict:

    fee = delivery_rates.get(city.lower())

    if fee is None:
        return {
            "success": False,
            "message": "Delivery rate unavailable for this location."
        }

    return {
        "success": True,
        "destination": city.title(),
        "delivery_fee_ngn": fee
    }


# =====================================
# 5. REGISTER AVAILABLE TOOLS
# =====================================

tools = [
    {
        "type": "function",
        "name": "check_inventory",
        "description": "Check the stock availability of a product.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Exact product name."
                }
            },
            "required": ["product_name"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "get_product_price",
        "description": "Retrieve the current price of a product in NGN.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Exact product name."
                }
            },
            "required": ["product_name"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "calculate_delivery",
        "description": "Calculate delivery fee for a supported Nigerian city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Customer delivery destination."
                }
            },
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# =====================================
# 6. TOOL REGISTRY
# =====================================

# Connect the tool names to actual Python functions.

available_functions = {
    "check_inventory": check_inventory,
    "get_product_price": get_product_price,
    "calculate_delivery": calculate_delivery
}


# =====================================
# 7. EXECUTE A TOOL CALL
# =====================================

def execute_tool(name: str, arguments: dict) -> dict:

    function = available_functions.get(name)

    if function is None:
        return {
            "success": False,
            "message": "Unknown tool requested."
        }

    try:
        return function(**arguments)

    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "Invalid tool arguments."
        }


# =====================================
# 8. AGENT LOOP
# =====================================

def run_agent(user_message: str) -> str:

    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=user_message,
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
        max_output_tokens=300
    )

    # Limit additional tool rounds.
    max_tool_rounds = 3

    for round_number in range(max_tool_rounds):

        # Find all requested function calls.
        function_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        # No more tools requested.
        if not function_calls:
            return response.output_text

        print(f"\n--- TOOL ROUND {round_number + 1} ---")

        tool_outputs = []

        # Execute every requested function.
        for call in function_calls:

            arguments = json.loads(call.arguments)

            print(f"AI requested: {call.name}")
            print(f"Arguments: {arguments}")

            result = execute_tool(
                call.name,
                arguments
            )

            print(f"Result: {result}")

            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })

        # Send the tool results back to the model.
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=[
                *response.output,
                *tool_outputs
            ],
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            max_output_tokens=300
        )

    return "The agent reached its maximum tool-call rounds."


# =====================================
# 9. RUN APPLICATION
# =====================================

if __name__ == "__main__":

    print("\nE-COMMERCE AI AGENT")
    print("Type 'exit' to stop.\n")

    while True:

        message = input("Customer: ").strip()

        if message.lower() in ["exit", "quit"]:
            print("Agent: Goodbye!")
            break

        if not message:
            continue

        try:
            answer = run_agent(message)

            print("\nAI Agent:")
            print(answer)
            print()

        except Exception as error:
            print(f"\nApplication error: {error}\n")
