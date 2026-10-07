from twin_reply import TwinReply


TOOL_NAMES = {
    "notes": "search_my_notes",
    "style": "get_style_examples",
    "profile": "get_profile",
    "glossary": "get_course_glossary",
}

SOURCE_NAMES = {
    "notes": "course_notes.md",
    "style": "style_examples.jsonl",
    "profile": "profile.json",
    "glossary": "course_glossary",
}


def build_twin_reply(
    decision: TwinReply,
    reply_text: str,
    execution: dict,
) -> TwinReply:
    executed_steps = [
        step
        for step in execution["execution_order"]
        if step in TOOL_NAMES
    ]

    tools_used = [
        TOOL_NAMES[step]
        for step in executed_steps
    ]

    sources = [
        SOURCE_NAMES[step]
        for step in executed_steps
    ]

    return TwinReply(
        addressed_to_me=decision.addressed_to_me,
        action=decision.action,
        language=decision.language,
        reply_text=reply_text,
        confidence=decision.confidence,
        tools_used=tools_used,
        sources=sources,
    )