import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from tools import (
    search_my_notes,
    get_profile,
    get_style_examples,
)


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=[
        search_my_notes,
        get_profile,
        get_style_examples,
    ],
    system_prompt=(
        "You are Ahmed's AI assistant.\n\n"

        "Use search_my_notes when you need information "
        "from Ahmed's course notes.\n"

        "Use get_profile when you need information "
        "about Ahmed, his background, language preferences, "
        "or communication style.\n"

        "Use get_style_examples when you need examples of "
        "how Ahmed answers questions in his own style."
    )
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ahmed, explain RAG in the way you normally "
                    "explain technical topics."
                )
            }
        ]
    }
)


print("\n===== AGENT TRACE =====\n")

for message in result["messages"]:

    print("MESSAGE TYPE:", type(message).__name__)
    print("CONTENT:", message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("TOOL CALLS:")

        for tool_call in message.tool_calls:
            print(tool_call)

    print("-" * 60)


print("\n===== FINAL ANSWER =====\n")
print(result["messages"][-1].content)