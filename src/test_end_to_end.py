from transcriber import transcribe
from decision_policy import decide
from planner import make_plan
from executor import execute_plan
from reply_generator import generate_reply
from response_builder import build_twin_reply


audio_path = "twin_data/test_audio/question1.m4a"


print("=== STEP 1: TRANSCRIPTION ===")

result = transcribe(audio_path)

print("Text:", result["text"])
print("Language:", result["language"])
print("Quality:", result["quality"])


print("\n=== STEP 2: DECISION ===")

decision = decide(
    transcript=result["text"],
    language=result["language"],
    quality=result["quality"],
)

print(decision)


if decision.action == "answer":

    print("\n=== STEP 3: PLANNING ===")

    plan = make_plan(result["text"])

    print("Plan:", plan)


    print("\n=== STEP 4: EXECUTING PLAN ===")

    execution = execute_plan(
        question=result["text"],
        plan=plan,
        language=(
            "en"
            if result["language"].lower().startswith("en")
            else "ar"
            if result["language"].lower().startswith("ar")
            else "mixed"
        ),
    )

    print("Execution order:", execution["execution_order"])


    print("\n=== STEP 5: GENERATING REPLY ===")

    notes = execution["results"].get("notes", "")
    style = execution["results"].get("style", "")
    profile = execution["results"].get("profile", "")

    reply = generate_reply(
        question=result["text"],
        notes=notes,
        style=style,
        profile=profile,
    )


    print("\n=== STEP 6: BUILDING FINAL TWIN REPLY ===")

    final_reply = build_twin_reply(
        decision=decision,
        reply_text=reply,
        execution=execution,
    )

    print("\n===== FINAL TWIN REPLY =====")
    print(final_reply)

    print("\n===== AS DICT =====")
    print(final_reply.model_dump())


else:

    print("\nNo reply generated because action is:", decision.action)