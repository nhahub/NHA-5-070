from collections import deque


class RollingMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str):
        self.turns.append(
            {
                "question": question,
                "answer": answer,
            }
        )

    def get_context(self) -> str:
        if not self.turns:
            return "No previous conversation."

        context_parts = []

        for index, turn in enumerate(self.turns, start=1):
            context_parts.append(
                f"Turn {index}:\n"
                f"Question: {turn['question']}\n"
                f"Twin answer: {turn['answer']}"
            )

        return "\n\n---\n\n".join(context_parts)

    def clear(self):
        self.turns.clear()

    def __len__(self):
        return len(self.turns)