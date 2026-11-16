
import json

from core.ai_client import get_ai_client


# ==========================================
# 1. INITIALIZE AI PROVIDER
# ==========================================

client, MODEL, PROVIDER = get_ai_client()

print(f"\nActive AI provider: {PROVIDER}")
print(f"Active model: {MODEL}")


# ==========================================
# 2. ASSISTANT INSTRUCTIONS
# ==========================================

INSTRUCTIONS = """
You are an e-commerce customer support assistant.

Your responsibility is to answer customers using
the business information supplied to you.

Rules:
- Only use the supplied business information.
- Never invent prices, stock quantities or delivery fees.
- If information is unavailable, explain that clearly.
- Be friendly, professional and concise.
- Calculate totals correctly when all necessary values exist.
- Never claim that an order has been placed or paid for.
"""


# ==========================================
# 3. SAMPLE PRODUCT DATABASE
# ==========================================

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


# ==========================================
# 4. DELIVERY DATABASE
# ==========================================

delivery_rates = {
    "lagos": 3500,
    "abuja": 5000,
    "port harcourt": 4500
}


# ==========================================
# 5. PYTHON BUSINESS FUNCTIONS
# ==========================================

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


# ==========================================
# 6. TOOL REGISTRY
# ==========================================

available_functions = {
    "check_inventory": check_inventory,
    "get_product_price": get_product_price,
    "calculate_delivery": calculate_delivery
}


# ==========================================
# 7. VALIDATE AND EXECUTE TOOLS
# ==========================================

def execute_tool(name: str, arguments: dict) -> dict:

    required_arguments = {
        "check_inventory": ["product_name"],
        "get_product_price": ["product_name"],
        "calculate_delivery": ["city"]
    }

    function = available_functions.get(name)

    if function is None:
        return {
            "success": False,
            "message": "Unknown tool requested."
        }

    required = required_arguments[name]

    for argument in required:

        value = arguments.get(argument)

        if not isinstance(value, str) or not value.strip():

            return {
                "success": False,
                "message": f"Missing or invalid argument: {argument}"
            }

    try:
        return function(**arguments)

    except (TypeError, ValueError):
        return {
            "success": False,
            "message": "Invalid tool arguments."
        }


# ==========================================
# 8. UNDERSTAND CUSTOMER REQUEST
# ==========================================

def understand_request(message: str) -> dict:

    text = message.lower()

    # Identify the product mentioned by the customer.
    product_name = None

    for product in products:

        if product in text:
            product_name = product
            break

    # Identify the delivery destination.
    city = None

    for location in delivery_rates:

        if location in text:
            city = location
            break

    # Identify requested operations.
    wants_inventory = any(
        word in text
        for word in [
            "available",
            "availability",
            "in stock",
            "stock",
            "have"
        ]
    )

    wants_price = any(
        word in text
        for word in [
            "price",
            "cost",
            "how much",
            "expensive",
            "cheaper"
        ]
    )

    wants_delivery = any(
        word in text
        for word in [
            "delivery",
            "deliver",
            "shipping",
            "shipping fee",
            "shipping cost"
        ]
    )

    # Build the required operations.
    requested_tools = []

    if wants_inventory:
        requested_tools.append("check_inventory")

    if wants_price:
        requested_tools.append("get_product_price")

    if wants_delivery:
        requested_tools.append("calculate_delivery")

    return {
        "product_name": product_name,
        "city": city,
        "requested_tools": requested_tools
    }


# ==========================================
# 9. ORCHESTRATE BUSINESS OPERATIONS
# ==========================================

def orchestrate_request(message: str) -> dict:

    request = understand_request(message)

    product_name = request["product_name"]
    city = request["city"]

    requested_tools = request["requested_tools"]

    results = {}

    # Product-related operations require a known product.
    product_tools = [
        "check_inventory",
        "get_product_price"
    ]

    for tool_name in requested_tools:

        if tool_name in product_tools:

            if not product_name:
                results[tool_name] = {
                    "success": False,
                    "message": "Please specify a product."
                }
                continue

            arguments = {
                "product_name": product_name
            }

        elif tool_name == "calculate_delivery":

            if not city:
                results[tool_name] = {
                    "success": False,
                    "message": "Please specify a supported delivery city."
                }
                continue

            arguments = {
                "city": city
            }

        else:
            continue

        print(f"\nExecuting tool: {tool_name}")
        print(f"Arguments: {arguments}")

        result = execute_tool(
            tool_name,
            arguments
        )

        results[tool_name] = result

        print(f"Tool result: {result}")

    return results


# ==========================================
# 10. GENERATE CUSTOMER RESPONSE
# ==========================================

def generate_response(
    customer_message: str,
    business_results: dict
) -> str:

    context = {
        "customer_message": customer_message,
        "verified_business_information": business_results
    }

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": INSTRUCTIONS
            },
            {
                "role": "user",
                "content": (
                    "Answer the customer's request using "
                    "the following verified information:\n\n"
                    + json.dumps(context, indent=2)
                )
            }
        ],

        max_tokens=300
    )

    return response.choices[0].message.content


# ==========================================
# 11. MAIN AGENT FUNCTION
# ==========================================

def run_agent(customer_message: str) -> str:

    # Step 1: Understand the request.
    request = understand_request(customer_message)

    # Step 2: Check whether we identified any operations.
    if not request["requested_tools"]:

        return (
            "Could you clarify whether you need product "
            "availability, pricing or delivery information?"
        )

    # Step 3: Execute required business operations.
    results = orchestrate_request(customer_message)

    # Step 4: Generate the final customer response.
    return generate_response(
        customer_message,
        results
    )


# ==========================================
# 12. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\nE-COMMERCE AI ASSISTANT")
    print("Type 'exit' to stop.\n")

    while True:

        message = input("Customer: ").strip()

        if message.lower() in ["exit", "quit"]:

            print("\nAssistant: Goodbye!")
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
