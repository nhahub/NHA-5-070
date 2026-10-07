import os

from dotenv import load_dotenv
from groq import Groq


# Load the API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


# Create the Groq client
client = Groq(api_key=api_key)


# Send a simple question to the LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one simple sentence."
        }
    ]
)


# Print the LLM answer
print(response.choices[0].message.content)