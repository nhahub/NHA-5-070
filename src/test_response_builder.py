from twin_reply import TwinReply
from response_builder import build_twin_reply


decision = TwinReply(
    addressed_to_me=True,
    action="answer",
    language="en",
    reply_text="",
    confidence=0.94,
    tools_used=[],
    sources=[],
)

execution = {
    "execution_order": [
        "notes",
        "style",
        "profile",
        "reply",
    ],
    "results": {},
}

final_reply = build_twin_reply(
    decision=decision,
    reply_text="RAG retrieves relevant information before generation.",
    execution=execution,
)

print("===== FINAL TWIN REPLY =====")
print(final_reply)

print("\n===== AS DICT =====")
print(final_reply.model_dump())