from tools import embed_style_texts

texts = [
    "passage: What is RAG?",
    "passage: What is an AI agent?",
    "passage: How does memory help the agent?",
    "passage: What is a vector database?",
]

embeddings = embed_style_texts(texts)

query = embed_style_texts([
    "query: What is RAG?"
])

similarities = query @ embeddings.T

for text, score in zip(texts, similarities[0]):
    print(f"{score.item():.4f}  {text}")