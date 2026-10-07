from decision_scores import DecisionScores


scores = DecisionScores(
    answer=0.90,
    ask_to_repeat=0.05,
    defer=0.03,
    stay_silent=0.02,
)

print(scores)
print()
print(scores.model_dump())