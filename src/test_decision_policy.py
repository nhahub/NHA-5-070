from decision_policy import decide


tests = [
    {
        "name": "Ahmed question",
        "transcript": (
            "Ahmed, what is the difference "
            "between RAG and fine-tuning?"
        ),
        "language": "English",
        "quality": 0.90,
    },
    {
        "name": "Another student",
        "transcript": (
            "Sara, can you share your screen?"
        ),
        "language": "English",
        "quality": 0.90,
    },
    {
        "name": "Low quality",
        "transcript": (
            "Ahmed, what is RAG?"
        ),
        "language": "English",
        "quality": 0.20,
    },
    {
        "name": "Outside knowledge",
        "transcript": (
            "Ahmed, how did your assignment "
            "use Kubernetes?"
        ),
        "language": "English",
        "quality": 0.90,
    },
]


for test in tests:

    print("\n====================")
    print(test["name"])
    print("====================")

    result = decide(
        transcript=test["transcript"],
        language=test["language"],
        quality=test["quality"],
    )

    print(result)
    print()
    print(result.model_dump())