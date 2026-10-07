from memory import RollingMemory


memory = RollingMemory(max_turns=3)


memory.add_turn(
    question="Ahmed, what is RAG?",
    answer="RAG retrieves relevant information before generating the answer.",
)

memory.add_turn(
    question="Ahmed, what is fine-tuning?",
    answer="Fine-tuning adapts a pretrained model to a specific task.",
)

print("===== MEMORY AFTER 2 TURNS =====")
print(memory.get_context())

print("\n===== NUMBER OF TURNS =====")
print(len(memory))


memory.add_turn(
    question="Ahmed, what is an AI agent?",
    answer="An AI agent can reason about a task and use tools.",
)

memory.add_turn(
    question="Ahmed, what is pruning?",
    answer="Pruning removes unnecessary branches from a decision process.",
)

print("\n===== MEMORY AFTER 4 TURNS =====")
print(memory.get_context())

print("\n===== NUMBER OF TURNS =====")
print(len(memory))