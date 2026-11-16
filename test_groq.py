
from core.ai_client import get_ai_client

client, model, provider = get_ai_client()

print("Provider:", provider)
print("Model:", model)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Explain what an AI agent is in two sentences."
        }
    ],
    max_tokens=120
)

print("\nAI RESPONSE:")
print(response.choices[0].message.content)
