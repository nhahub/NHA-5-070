from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "intfloat/multilingual-e5-small"


# --------------------------------------------------
# 1. Load the E5 model
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def embed(texts):
    """
    Convert texts into normalized 384-dimensional embeddings.
    """

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask

    embeddings = (
        masked_embeddings.sum(dim=1)
        / attention_mask.sum(dim=1)
    )

    return F.normalize(embeddings, p=2, dim=1)


# --------------------------------------------------
# 2. Read the knowledge file
# --------------------------------------------------

knowledge_path = Path(
    "twin_data/knowledge/course_notes.md"
)

knowledge_text = knowledge_path.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# 3. Split the document into chunks
# --------------------------------------------------

chunks = knowledge_text.split("\n## ")

chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]

# Remove the document title from the chunks
if chunks and chunks[0].startswith("# Agentic AI Course Notes"):
    chunks = chunks[1:]

# --------------------------------------------------
# 4. Prepare chunks for E5
# --------------------------------------------------

passages = [
    "passage: " + chunk
    for chunk in chunks
]


# --------------------------------------------------
# 5. Create embeddings for all knowledge chunks
# --------------------------------------------------

embeddings = embed(passages)


print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print()


# --------------------------------------------------
# 6. User question
# --------------------------------------------------

query = "What is RAG?"


# --------------------------------------------------
# 7. Create query embedding
# --------------------------------------------------

query_embedding = embed([
    "query: " + query
])


# --------------------------------------------------
# 8. Calculate similarity with all chunks
# --------------------------------------------------

similarities = torch.matmul(
    query_embedding,
    embeddings.T
).squeeze(0)


# --------------------------------------------------
# 9. Get the Top-K results
# --------------------------------------------------

K = 3

top_scores, top_indices = torch.topk(
    similarities,
    k=K
)


# --------------------------------------------------
# 10. Display the results
# --------------------------------------------------

print("Question:")
print(query)
print()

print(f"Top {K} results:")
print()


for rank, (score, index) in enumerate(
    zip(top_scores, top_indices),
    start=1
):
    print(f"Rank {rank}")
    print(f"Chunk: {index.item() + 1}")
    print(f"Similarity: {score.item():.4f}")
    print()
    print(chunks[index.item()])
    print()
    print("-" * 60)
    print()