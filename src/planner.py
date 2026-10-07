from pydantic import BaseModel


class Plan(BaseModel):
    need_notes: bool
    need_style: bool
    need_profile: bool
    need_glossary: bool


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
            "أحمد",
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
            "بطريقتك",
            "طريقتك",
            "بطريقة شرحك",
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
            "الـrag",
            "الـagent",
            "الـpruning",
        ]
    )

    asks_for_definition = any(
        phrase in question_lower
        for phrase in [
            "what is",
            "what's",
            "what does",
            "define",
            "definition of",
            "what do you mean by",
            "إيه هو",
            "ايه هو",
            "ما هو",
            "يعني إيه",
            "يعني ايه",
            "تعريف",
        ]
    )

    asks_for_comparison = any(
        phrase in question_lower
        for phrase in [
            "difference between",
            "difference",
            "compare",
            "comparison",
            "الفرق بين",
            "ايه الفرق",
            "إيه الفرق",
            "مقارنة",
        ]
    )

    glossary_terms = [
        "rag",
        "reg",
        "fine-tuning",
        "fine tuning",
        "agent",
        "pruning",
        "الـrag",
        "الـagent",
        "الـpruning",
    ]

    asks_about_glossary_term = any(
        term in question_lower
        for term in glossary_terms
    )

    need_notes = asks_about_knowledge

    # Style examples are useful for normal answerable questions,
    # not only when the instructor explicitly asks about Ahmed's style.
    need_style = asks_about_knowledge or asks_for_style

    need_profile = about_ahmed

    # Use the glossary only for direct definition questions.
    # Comparison questions should rely on notes instead.
    need_glossary = (
        asks_for_definition
        and asks_about_glossary_term
        and not asks_for_comparison
    )

    return Plan(
        need_notes=need_notes,
        need_style=need_style,
        need_profile=need_profile,
        need_glossary=need_glossary,
    )