from retriever import NotesRetriever


retriever = NotesRetriever(
    "twin_data/knowledge/course_notes.md",
    top_k=3
)


results = retriever.search("What is RAG?")


print("Search results:")
print()

for rank, result in enumerate(results, start=1):
    print(f"Rank {rank}")
    print(f"Score: {result['score']:.4f}")
    print(result["text"])
    print("-" * 60)