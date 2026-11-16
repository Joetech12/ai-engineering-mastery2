
import os
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = "gpt-5.5"

# Define the expected response structure
classification_schema = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "order_issue",
                "payment_issue",
                "product_enquiry",
                "other"
            ]
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"]
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"]
        },
        "requires_human": {
            "type": "boolean"
        },
        "summary": {
            "type": "string"
        }
    },
    "required": [
        "category",
        "priority",
        "sentiment",
        "requires_human",
        "summary"
    ],
    "additionalProperties": False
}

# Customer message
customer_message = """
I ordered a laptop three days ago, but it hasn't arrived.
I've tried calling your support team and nobody is answering.
"""

# Send request to AI
response = client.responses.create(
    model=MODEL,

    instructions="""
    You are an e-commerce customer support classifier.

    Analyse the customer's message and classify it
    according to the provided JSON schema.

    Be accurate and do not invent customer information.
    """,

    input=customer_message,

    text={
        "format": {
            "type": "json_schema",
            "name": "customer_enquiry",
            "strict": True,
            "schema": classification_schema
        }
    },

    max_output_tokens=150
)

# Convert JSON text into a Python dictionary
result = json.loads(response.output_text)

# Display result
print("\nCUSTOMER CLASSIFICATION")
print(json.dumps(result, indent=4))

print("\nAUTOMATION DECISION")

if result["requires_human"]:
    print("Escalate to human support.")

elif result["priority"] == "high":
    print("Send an urgent notification.")

else:
    print("Handle through normal support workflow.")
