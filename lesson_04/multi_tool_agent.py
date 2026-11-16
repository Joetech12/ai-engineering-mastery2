
import json

from core.ai_client import get_ai_client

# =====================================
# 1. INITIALIZE AI PROVIDER
# =====================================

client, MODEL, PROVIDER = get_ai_client()

print(f"\nActive AI provider: {PROVIDER}")
print(f"Active model: {MODEL}")


# =====================================
# 2. ASSISTANT INSTRUCTIONS
# =====================================


INSTRUCTIONS = """
You are an e-commerce customer support agent.

You have access to three tools:

1. check_inventory:
   Checks whether a product is available and returns its stock quantity.

2. get_product_price:
   Retrieves the price of a product in Nigerian Naira.

3. calculate_delivery:
   Calculates delivery fees for a supported Nigerian city.

IMPORTANT TOOL USAGE RULES:

- If the customer asks whether a product is available,
  you MUST call check_inventory.

- If the customer asks how much a product costs,
  you MUST call get_product_price.

- If the customer asks about delivery fees or delivery
  cost to a city, you MUST call calculate_delivery.

- If the customer asks about multiple things,
  call ALL tools required to answer the request.

- You may call independent tools together.

- Never invent prices, stock quantities or delivery fees.

- Only provide business information returned by the tools.

- If a tool reports that information is unavailable,
  explain that clearly.

- Do not claim that an order has been placed or paid for.
"""


# =====================================
# 3. SAMPLE PRODUCT DATABASE
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
# 4. DELIVERY DATABASE
# =====================================

delivery_rates = {
    "lagos": 3500,
    "abuja": 5000,
    "port harcourt": 4500
}


# =====================================
# 5. PYTHON FUNCTIONS / TOOLS
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
# 6. TOOL REGISTRY
# =====================================

available_functions = {
    "check_inventory": check_inventory,
    "get_product_price": get_product_price,
    "calculate_delivery": calculate_delivery
}


# =====================================
# 7. TOOL DEFINITIONS
# =====================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Check stock availability of a product.",
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Retrieve a product price in Nigerian Naira.",
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
        }
    },
    {
        "type": "function",
        "function": {
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
    }
]


# =====================================
# 8. EXECUTE REQUESTED TOOL
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
# 9. AGENT LOOP
# =====================================

def run_agent(user_message: str) -> str:

    messages = [
        {
            "role": "system",
            "content": INSTRUCTIONS
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    max_tool_rounds = 3

    for round_number in range(max_tool_rounds + 1):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            max_tokens=400
        )

        assistant_message = response.choices[0].message

        # Save the assistant response in conversation history.
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # If the model has no more tool calls, return its answer.
        if not assistant_message.tool_calls:
            return assistant_message.content or "No response generated."

        if round_number == max_tool_rounds:
            return "The agent reached its maximum tool-call rounds."

        print(f"\n--- TOOL ROUND {round_number + 1} ---")

        # Execute every requested tool.
        for tool_call in assistant_message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"AI requested: {tool_name}")
            print(f"Arguments: {arguments}")

            result = execute_tool(
                tool_name,
                arguments
            )

            print(f"Tool result: {result}")

            # Return tool result to the model.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    return "Unable to complete the request."


# =====================================
# 10. RUN APPLICATION
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
