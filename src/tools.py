from pathlib import Path
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.tools import tool

from retriever import NotesRetriever


# -----------------------------
# Notes Retriever
# -----------------------------

retriever = NotesRetriever(
    "twin_data/knowledge/course_notes.md",
    top_k=3
)


@tool
def search_my_notes(query: str) -> str:
    """
    Search Ahmed's personal course notes and return the most relevant information.
    """
    results = retriever.search(query)

    formatted_results = []

    for rank, result in enumerate(results, start=1):
        formatted_results.append(
            f"Result {rank} "
            f"(similarity: {result['score']:.4f}):\n"
            f"{result['text']}"
        )

    return "\n\n---\n\n".join(formatted_results)


# -----------------------------
# Ahmed's Profile
# -----------------------------

@tool
def get_profile() -> str:
    """
    Get Ahmed's profile, background, language preferences, and communication style.
    """
    profile_path = Path("twin_data/profile.json")

    return profile_path.read_text(encoding="utf-8")


# -----------------------------
# Style Examples Retriever
# -----------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"

style_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
style_model = AutoModel.from_pretrained(MODEL_NAME)


def embed_style_texts(texts):
    inputs = style_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = style_model(**inputs)

    token_embeddings = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask

    embeddings = (
        masked_embeddings.sum(dim=1)
        / attention_mask.sum(dim=1)
    )

    return F.normalize(embeddings, p=2, dim=1)


@tool
def get_style_examples(question: str, language: str = "en") -> str:
    """
    Retrieve the most similar examples of how Ahmed answers questions.
    """

    examples_path = Path("twin_data/style_examples.jsonl")

    examples = []

    with examples_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            examples.append(json.loads(line))

    filtered_examples = [
        example
        for example in examples
        if example.get("language") == language
    ]

    if not filtered_examples:
        filtered_examples = examples

    if not filtered_examples:
        return "No style examples available."

    texts = [
        "passage: " + example["question"]
        for example in filtered_examples
    ]

    example_embeddings = embed_style_texts(texts)

    query_embedding = embed_style_texts(
        ["query: " + question]
    )

    similarities = torch.matmul(
        query_embedding,
        example_embeddings.T
    ).squeeze(0)

    top_k = min(3, len(filtered_examples))

    top_scores, top_indices = torch.topk(
        similarities,
        k=top_k
    )

    selected_examples = []

    for score, index in zip(top_scores, top_indices):
        example = filtered_examples[index.item()]

        selected_examples.append(
            {
                "similarity": round(float(score.item()), 4),
                "question": example["question"],
                "answer": example["answer"],
                "language": example["language"]
            }
        )

    return json.dumps(
        selected_examples,
        ensure_ascii=False,
        indent=2
    )