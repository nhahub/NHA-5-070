from twin_reply import TwinReply
from response_builder import build_twin_reply


decision = TwinReply(
    addressed_to_me=True,
    action="answer",
    language="ar",
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
        "glossary",
        "reply",
    ],
    "results": {},
}

final_reply = build_twin_reply(
    decision=decision,
    reply_text="الـRAG بيعمل retrieval للمعلومات المناسبة قبل ما الـLLM يولد الإجابة.",
    execution=execution,
)

print("===== FINAL TWIN REPLY =====")
print(final_reply)

print("\n===== AS DICT =====")
print(final_reply.model_dump())