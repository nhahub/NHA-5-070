import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0,
)


def generate_reply(
    question: str,
    notes: str = "",
    style: str = "",
    profile: str = "",
) -> str:

    prompt = f"""
You are generating a reply for Ahmed's AI twin.

Your job is to answer the instructor's question using the provided
notes, style examples, and profile.

IMPORTANT RULES:

1. Use the provided notes as the main source for technical facts.
2. Use the style examples to imitate Ahmed's way of explaining.
3. Use the profile to understand Ahmed's communication style.
4. Do not invent facts that are not supported by the provided context.
5. Keep the answer short to medium unless the question requires more detail.
6. Answer directly.
7. Preserve technical terms such as RAG, LLM, fine-tuning, embeddings, etc.
8. If the question is in English, answer in English.
9. If the question is in Arabic, answer in Egyptian Arabic.
10. If the question is mixed, naturally mix Egyptian Arabic with English
    technical terms.

QUESTION:
{question}

AHMED'S NOTES:
{notes}

STYLE EXAMPLES:
{style}

PROFILE:
{profile}

Now generate only the final answer that Ahmed would give.
"""

    response = llm.invoke(prompt)

    return response.content