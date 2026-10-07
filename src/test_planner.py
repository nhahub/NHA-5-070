from planner import make_plan


questions = [
    "What is RAG?",
    "What is Ahmed's background?",
    "Ahmed, explain RAG in the way you normally explain technical topics.",
]


for question in questions:

    plan = make_plan(question)

    print("\nQUESTION:")
    print(question)

    print("\nPLAN:")
    print(plan.model_dump())

    print("-" * 60)