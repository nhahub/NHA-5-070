from concurrent.futures import ThreadPoolExecutor

from planner import Plan
from reply_graph import get_execution_order
from tools import (
    search_my_notes,
    get_profile,
    get_style_examples,
    get_course_glossary,
)


def extract_glossary_term(question: str) -> str:
    """Extract a known glossary term from the question."""

    question_lower = question.lower()

    glossary_terms = [
        "fine-tuning",
        "fine tuning",
        "rag",
        "reg",
        "agent",
        "pruning",
    ]

    for term in glossary_terms:
        if term in question_lower:
            return term

    return question.strip()


def run_tool(
    step: str,
    question: str,
    language: str = "en",
):
    if step == "notes":
        return search_my_notes.invoke(
            {"query": question}
        )

    if step == "style":
        return get_style_examples.invoke(
            {
                "question": question,
                "language": language,
            }
        )

    if step == "profile":
        return get_profile.invoke({})

    if step == "glossary":
        term = extract_glossary_term(question)

        return get_course_glossary.invoke(
            {"term": term}
        )

    return None


def execute_plan(
    question: str,
    plan: Plan,
    language: str = "en",
):
    execution_order = get_execution_order(
        need_notes=plan.need_notes,
        need_style=plan.need_style,
        need_profile=plan.need_profile,
        need_glossary=plan.need_glossary,
    )

    results = {}

    # These branches are independent,
    # so run them in parallel.
    independent_steps = [
        step
        for step in execution_order
        if step in {
            "notes",
            "style",
            "profile",
            "glossary",
        }
    ]

    if independent_steps:
        with ThreadPoolExecutor(
            max_workers=len(independent_steps)
        ) as executor:

            futures = {
                step: executor.submit(
                    run_tool,
                    step,
                    question,
                    language,
                )
                for step in independent_steps
            }

            for step, future in futures.items():
                results[step] = future.result()

    # Reply must wait for all independent branches.
    if "reply" in execution_order:
        results["reply"] = (
            "Reply generation handled separately."
        )

    return {
        "execution_order": execution_order,
        "results": results,
    }