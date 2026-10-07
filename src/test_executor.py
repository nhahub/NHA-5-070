from planner import make_plan
from executor import execute_plan


question = "يا أحمد، إيه هو الـRAG ببساطة؟"
language = "ar"

plan = make_plan(question)

print("===== PLAN =====")
print(plan.model_dump())

result = execute_plan(
    question=question,
    plan=plan,
    language=language,
)

print("\n===== EXECUTION ORDER =====")
print(result["execution_order"])

print("\n===== RESULTS =====")

for name, value in result["results"].items():
    print(f"\n--- {name.upper()} ---")
    print(value)