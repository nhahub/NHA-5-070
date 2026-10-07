from pathlib import Path
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from groq import Groq


MODEL_NAME = "intfloat/multilingual-e5-small"


# --------------------------------------------------
# 1. Load models and API key
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


# --------------------------------------------------
# 2. Embedding function
# --------------------------------------------------

def embed(texts):

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
# 3. Read knowledge
# --------------------------------------------------

knowledge_path = Path(
    "twin_data/knowledge/course_notes.md"
)

knowledge_text = knowledge_path.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# 4. Create chunks
# --------------------------------------------------

chunks = knowledge_text.split("\n## ")

chunks = [
    chunk.strip()
    for chunk in chunks
    if chunk.strip()
]


# Remove document title
if chunks and chunks[0].startswith("# Agentic AI Course Notes"):
    chunks = chunks[1:]


# --------------------------------------------------
# 5. Create embeddings for knowledge
# --------------------------------------------------

passages = [
    "passage: " + chunk
    for chunk in chunks
]

embeddings = embed(passages)


# --------------------------------------------------
# 6. User question
# --------------------------------------------------

query = "What is RAG?"


# --------------------------------------------------
# 7. Retrieve Top-K chunks
# --------------------------------------------------

query_embedding = embed([
    "query: " + query
])

similarities = torch.matmul(
    query_embedding,
    embeddings.T
).squeeze(0)

K = 3

top_scores, top_indices = torch.topk(
    similarities,
    k=K
)


# --------------------------------------------------
# 8. Build the context
# --------------------------------------------------

retrieved_chunks = []

for index in top_indices:
    retrieved_chunks.append(
        chunks[index.item()]
    )


context = "\n\n---\n\n".join(
    retrieved_chunks
)


# --------------------------------------------------
# 9. Send question + context to the LLM
# --------------------------------------------------

prompt = f"""
Answer the question using the provided context.

If the context does not contain enough information,
say that you do not have enough information.

Context:
{context}

Question:
{query}
"""


response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# --------------------------------------------------
# 10. Print the result
# --------------------------------------------------

answer = response.choices[0].message.content

print("Question:")
print(query)

print("\nRetrieved Context:")
print(context)

print("\nLLM Answer:")
print(answer)