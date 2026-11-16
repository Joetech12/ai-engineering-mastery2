
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


# =====================================
# 2. SAMPLE ORDER DATABASE
# =====================================

orders = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "product": "HP Laptop",
        "status": "processing",
        "estimated_delivery": "November 20, 2026"
    },

    "ORD-1002": {
        "order_id": "ORD-1002",
        "product": "Samsung Galaxy S26",
        "status": "shipped",
        "estimated_delivery": "November 18, 2026"
    },

    "ORD-1003": {
        "order_id": "ORD-1003",
        "product": "Wireless Headphones",
        "status": "delivered",
        "estimated_delivery": "November 14, 2026"
    }
}


# =====================================
# 3. PYTHON FUNCTION / TOOL
# =====================================

def get_order(order_id: str) -> dict:
    """
    Retrieve an order using its order ID.
    """

    order = orders.get(order_id.upper())

    if order is None:
        return {
            "found": False,
            "message": "Order not found."
        }

    return {
        "found": True,
        "order": order
    }


# =====================================
# 4. EXPOSE FUNCTION TO THE AI
# =====================================

tools = [
    {
        "type": "function",
        "name": "get_order",
        "description": (
            "Retrieve order information using an order ID. "
            "Use this when a customer asks about an order, "
            "its delivery status, or estimated delivery date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The customer's order ID."
                }
            },
            "required": ["order_id"],
            "additionalProperties": False
        },
        "strict": True
    }
]


# =====================================
# 5. CUSTOMER MESSAGE
# =====================================

customer_message = (
    "Please check order ORD-1002. "
    "Has it been delivered yet?"
)


# =====================================
# 6. FIRST AI REQUEST
# =====================================

response = client.responses.create(
    model=MODEL,

    instructions="""
    You are an e-commerce customer support agent.

    You have access to an order lookup tool.

    When a customer asks about an order, use the tool
    to retrieve the actual order information.

    Never invent order details.

    If an order is not found, explain that clearly.
    """,

    input=customer_message,

    tools=tools,
    tool_choice="auto",

    max_output_tokens=250
)


# =====================================
# 7. EXECUTE REQUESTED TOOL
# =====================================

tool_outputs = []

for item in response.output:

    if item.type == "function_call":

        if item.name == "get_order":

            arguments = json.loads(item.arguments)

            order_id = arguments["order_id"]

            print(f"\nAI requested order lookup: {order_id}")

            # Execute our actual Python function
            result = get_order(order_id)

            print("Tool result:", result)

            # Send result back to OpenAI
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result)
            })


# =====================================
# 8. GENERATE FINAL RESPONSE
# =====================================

if tool_outputs:

    final_response = client.responses.create(
        model=MODEL,

        instructions="""
        You are an e-commerce customer support agent.

        Answer the customer using the returned order information.
        Do not invent or change order details.
        """,

        input=[
            *response.output,
            *tool_outputs
        ],

        tools=tools,
        max_output_tokens=200
    )

    print("\nCUSTOMER SUPPORT ASSISTANT:")
    print(final_response.output_text)

else:

    print("\nCUSTOMER SUPPORT ASSISTANT:")
    print(response.output_text)
