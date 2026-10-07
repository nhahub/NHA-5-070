from twin_reply import TwinReply


reply = TwinReply(
    addressed_to_me=True,
    action="answer",
    language="en",
    reply_text="RAG retrieves relevant information before the model generates the answer.",
    confidence=0.92,
    tools_used=["search_my_notes"],
    sources=["course_notes.md"],
)

print(reply)
print()
print(reply.model_dump())