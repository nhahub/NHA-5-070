from pydantic import BaseModel


class Plan(BaseModel):
    need_notes: bool
    need_style: bool
    need_profile: bool


def make_plan(question: str) -> Plan:
    question_lower = question.lower()

    about_ahmed = any(
        keyword in question_lower
        for keyword in [
            "ahmed",
            "ahmad",
            "your background",
            "your profile",
            "about you",
        ]
    )

    asks_for_style = any(
        phrase in question_lower
        for phrase in [
            "your way",
            "your style",
            "normally explain",
            "how you explain",
            "in your own words",
        ]
    )

    asks_about_knowledge = any(
        keyword in question_lower
        for keyword in [
            "rag",
            "reg",
            "fine-tuning",
            "fine tuning",
            "agent",
            "pruning",
            "decision tree",
            "retrieval",
            "embedding",
        ]
    )

    need_notes = asks_about_knowledge

    # Style examples are useful for normal answerable questions,
    # not only when the instructor explicitly asks about Ahmed's style.
    need_style = asks_about_knowledge or asks_for_style

    need_profile = about_ahmed

    return Plan(
        need_notes=need_notes,
        need_style=need_style,
        need_profile=need_profile,
    )