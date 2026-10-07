from planner import make_plan
from executor import execute_plan
from reply_generator import generate_reply


question = "Ahmed, explain RAG in the way you normally explain technical topics."

plan = make_plan(question)

result = execute_plan(
    question=question,
    plan=plan,
)

reply = generate_reply(
    question=question,
    notes=result["results"].get("notes", ""),
    style=result["results"].get("style", ""),
    profile=result["results"].get("profile", ""),
)

print("\n===== FINAL REPLY =====")
print(reply)