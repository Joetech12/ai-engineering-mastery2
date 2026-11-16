
import os

from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()

# 2. Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# 3. Configure assistant
MODEL = "gpt-5.5"

INSTRUCTIONS = """
You are a helpful AI assistant.

Your job is to:
- Answer questions clearly.
- Explain technical concepts simply.
- Give practical examples.
- Ask clarifying questions when necessary.
"""

# 4. Store the previous response ID
previous_response_id = None

print("\nAI Assistant is ready!")
print("Type 'exit' to end the conversation.\n")

# 5. Start conversation loop
while True:

    user_message = input("You: ").strip()

    # Stop the application
    if user_message.lower() in ["exit", "quit"]:
        print("\nAssistant: Goodbye!")
        break

    # Ignore empty messages
    if not user_message:
        continue

    try:

        # 6. Send message to the AI
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=user_message,
            previous_response_id=previous_response_id
        )

        # 7. Display AI response
        print("\nAssistant:")
        print(response.output_text)
        print()

        # 8. Save response ID for next message
        previous_response_id = response.id

    except Exception as error:
        print(f"\nAn error occurred: {error}\n")
