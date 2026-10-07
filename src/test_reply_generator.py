from planner import make_plan
from executor import execute_plan
from reply_generator import generate_reply
from memory import RollingMemory


memory = RollingMemory(max_turns=3)


# ============================================================
# TURN 1
# ============================================================

question_1 = "Ahmed, explain RAG in the way you normally explain technical topics."

plan_1 = make_plan(question_1)

result_1 = execute_plan(
    question=question_1,
    plan=plan_1,
)

reply_1 = generate_reply(
    question=question_1,
    notes=result_1["results"].get("notes", ""),
    style=result_1["results"].get("style", ""),
    profile=result_1["results"].get("profile", ""),
    memory_context=memory.get_context(),
)

memory.add_turn(
    question=question_1,
    answer=reply_1,
)

print("\n===== TURN 1 =====")
print("Question:")
print(question_1)

print("\nTwin Answer:")
print(reply_1)


# ============================================================
# TURN 2 - FOLLOW-UP
# ============================================================

question_2 = "Ahmed, and how is that different from what you said earlier?"

plan_2 = make_plan(question_2)

result_2 = execute_plan(
    question=question_2,
    plan=plan_2,
)

reply_2 = generate_reply(
    question=question_2,
    notes=result_2["results"].get("notes", ""),
    style=result_2["results"].get("style", ""),
    profile=result_2["results"].get("profile", ""),
    memory_context=memory.get_context(),
)

memory.add_turn(
    question=question_2,
    answer=reply_2,
)

print("\n===== TURN 2 =====")
print("Question:")
print(question_2)

print("\nMemory Used:")
print(memory.get_context())

print("\nTwin Follow-up Answer:")
print(reply_2)