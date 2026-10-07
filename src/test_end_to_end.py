from transcriber import transcribe
from decision_policy import decide
from planner import make_plan
from executor import execute_plan
from reply_generator import generate_reply


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

    print("\n===== FINAL REPLY =====")
    print(reply)

else:

    print("\nNo reply generated because action is:", decision.action)