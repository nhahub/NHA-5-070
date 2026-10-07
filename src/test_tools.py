from tools import get_style_examples


result = get_style_examples.invoke(
    {
        "question": "What is RAG?",
        "language": "en"
    }
)

print("===== STYLE EXAMPLES =====")
print(result)