import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from decision_scores import DecisionScores
from twin_reply import TwinReply
from tools import search_my_notes


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0,
)


def cheap_checks(
    transcript: str,
    language: str | None,
    quality: float,
) -> TwinReply | None:

    text = transcript.strip().lower()

    # --------------------------------------------------
    # 1. Very low audio quality
    # --------------------------------------------------

    if quality < 0.40:
        return TwinReply(
            addressed_to_me=False,
            action="ask_to_repeat",
            language="en",
            reply_text="Sorry, could you repeat the question?",
            confidence=0.95,
            tools_used=[],
            sources=[],
        )

    # --------------------------------------------------
    # 2. Clearly addressed to another student
    # --------------------------------------------------

    other_students = [
        "sara",
        "john",
        "mohamed",
        "ahmed ali",
    ]

    for name in other_students:
        if text.startswith(name):
            return TwinReply(
                addressed_to_me=False,
                action="stay_silent",
                language="en",
                reply_text="",
                confidence=0.95,
                tools_used=[],
                sources=[],
            )

    # --------------------------------------------------
    # 3. Clearly addressed to Ahmed
    # --------------------------------------------------

    ahmed_names = [
        "ahmed",
        "ahmad",
        "أحمد",
    ]

    for name in ahmed_names:
        if name in text:
            return None

    # --------------------------------------------------
    # 4. Unclear -> continue to LLM stage
    # --------------------------------------------------

    return None


def score_with_llm(
    transcript: str,
    language: str | None,
    quality: float,
    notes: str,
) -> DecisionScores:

    prompt = f"""
You are the decision layer for Ahmed's AI twin.

Evaluate the instructor's question and assign a score from 0 to 1
to each possible action.

Possible actions:

- answer:
  The question can be answered using the available evidence
  about Ahmed and his course knowledge.

- ask_to_repeat:
  The audio or question is unclear.

- defer:
  The question requires information that is not supported
  by the available evidence. The twin must not invent an answer.

- stay_silent:
  The question is clearly addressed to someone else
  or the twin should not respond.

IMPORTANT:

Use the provided notes as evidence.

If the question asks about something that is NOT supported
by the notes, strongly prefer "defer".

Do not assume that Ahmed knows something just because the
question is addressed to Ahmed.

Question:
{transcript}

Detected language:
{language}

Audio quality:
{quality}

Relevant notes:
{notes}

Return only the four scores.
"""

    structured_llm = llm.with_structured_output(
        DecisionScores
    )

    return structured_llm.invoke(prompt)


def select_action(
    scores: DecisionScores,
    threshold: float = 0.70,
) -> str:

    score_map = scores.model_dump()

    best_action = max(
        score_map,
        key=score_map.get,
    )

    best_score = score_map[best_action]

    if best_score < threshold:
        return "ask_to_repeat"

    return best_action


def decide(
    transcript: str,
    language: str | None,
    quality: float,
) -> TwinReply:

    # --------------------------------------------------
    # Step 1: Cheap checks
    # --------------------------------------------------

    cheap_result = cheap_checks(
        transcript=transcript,
        language=language,
        quality=quality,
    )

    if cheap_result is not None:
        return cheap_result

    # --------------------------------------------------
    # Step 2: Retrieve relevant notes
    # --------------------------------------------------

    notes = search_my_notes.invoke(
        {
            "query": transcript,
        }
    )

    # --------------------------------------------------
    # Step 3: LLM scoring with evidence
    # --------------------------------------------------

    scores = score_with_llm(
        transcript=transcript,
        language=language,
        quality=quality,
        notes=notes,
    )

    # --------------------------------------------------
    # Step 4: Apply threshold
    # --------------------------------------------------

    action = select_action(scores)

    # --------------------------------------------------
    # Step 5: Build reply text
    # --------------------------------------------------

    if action == "stay_silent":
        reply_text = ""

    elif action == "ask_to_repeat":
        reply_text = "Sorry, could you repeat the question?"

    elif action == "defer":
        reply_text = (
            "I'm not sure about that, "
            "so I don't want to guess."
        )

    else:
        reply_text = ""

    # --------------------------------------------------
    # Step 6: Determine language
    # --------------------------------------------------

    if language:
        language_lower = language.lower()

        if language_lower.startswith("en"):
            output_language = "en"

        elif language_lower.startswith("ar"):
            output_language = "ar"

        else:
            output_language = "mixed"

    else:
        output_language = "mixed"

    # --------------------------------------------------
    # Step 7: Confidence
    # --------------------------------------------------

    confidence = max(
        scores.model_dump().values()
    )

    # --------------------------------------------------
    # Step 8: Structured result
    # --------------------------------------------------

    return TwinReply(
        addressed_to_me=(
            action == "answer"
        ),
        action=action,
        language=output_language,
        reply_text=reply_text,
        confidence=confidence,
        tools_used=["search_my_notes"],
        sources=["course_notes.md"],
    )