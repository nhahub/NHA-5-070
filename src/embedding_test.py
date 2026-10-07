import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "intfloat/multilingual-e5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


texts = [
    "query: What is RAG?",
    "query: Explain Retrieval-Augmented Generation.",
    "query: How do I cook pasta?"
]


inputs = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True
)


with torch.no_grad():
    outputs = model(**inputs)


# Mean pooling while ignoring padding tokens
token_embeddings = outputs.last_hidden_state
attention_mask = inputs["attention_mask"].unsqueeze(-1)

masked_embeddings = token_embeddings * attention_mask

embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

# Normalize embeddings for cosine similarity
embeddings = F.normalize(embeddings, p=2, dim=1)


similarity = F.cosine_similarity(
    embeddings[0].unsqueeze(0),
    embeddings[1].unsqueeze(0)
).item()


similarity_different = F.cosine_similarity(
    embeddings[0].unsqueeze(0),
    embeddings[2].unsqueeze(0)
).item()


print("Vector shape:", embeddings.shape)
print()
print("Similarity between RAG questions:", similarity)
print("Similarity between RAG and pasta:", similarity_different)