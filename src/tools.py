from pathlib import Path
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langchain_core.tools import tool

from retriever import NotesRetriever


# ============================================================
# Notes Retriever
# ============================================================

retriever = NotesRetriever(
    "twin_data/knowledge/course_notes.md",
    top_k=3
)


@tool
def search_my_notes(query: str) -> str:
    """Search Ahmed's course notes for information relevant to the question."""
    results = retriever.search(query)

    formatted_results = []

    for rank, result in enumerate(results, start=1):
        formatted_results.append(
            f"Result {rank} "
            f"(similarity: {result['score']:.4f}):\n"
            f"{result['text']}"
        )

    return "\n\n---\n\n".join(formatted_results)


# ============================================================
# Profile Tool
# ============================================================

@tool
def get_profile() -> str:
    """Return Ahmed's profile and communication style information."""
    profile_path = Path("twin_data/profile.json")

    return profile_path.read_text(
        encoding="utf-8"
    )


# ============================================================
# Style Retrieval
# ============================================================

MODEL_NAME = "intfloat/multilingual-e5-small"

style_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

style_model = AutoModel.from_pretrained(
    MODEL_NAME
)


def embed_style_texts(texts):
    inputs = style_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = style_model(**inputs)

    token_embeddings = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = (
        token_embeddings * attention_mask
    )

    embeddings = (
        masked_embeddings.sum(dim=1)
        / attention_mask.sum(dim=1)
    )

    return F.normalize(
        embeddings,
        p=2,
        dim=1,
    )


@tool
def get_style_examples(
    question: str,
    language: str = "en",
) -> str:
    """Retrieve examples similar to the question to imitate Ahmed's communication style."""

    examples_path = Path(
        "twin_data/style_examples.jsonl"
    )

    examples = []

    with examples_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            examples.append(
                json.loads(line)
            )

    if not examples:
        return "No style examples available."

    # Prefer examples in the requested language.
    preferred_examples = [
        example
        for example in examples
        if example.get("language") == language
    ]

    # Mixed examples are useful for Ahmed's natural
    # Egyptian Arabic + English technical style.
    mixed_examples = [
        example
        for example in examples
        if example.get("language") == "mixed"
    ]

    # Keep other languages as fallback candidates.
    other_examples = [
        example
        for example in examples
        if example.get("language")
        not in {language, "mixed"}
    ]

    candidate_examples = (
        preferred_examples
        + mixed_examples
        + other_examples
    )

    texts = [
        "passage: " + example["question"]
        for example in candidate_examples
    ]

    example_embeddings = embed_style_texts(
        texts
    )

    query_embedding = embed_style_texts(
        ["query: " + question]
    )

    similarities = torch.matmul(
        query_embedding,
        example_embeddings.T,
    ).squeeze(0)

    top_k = min(
        3,
        len(candidate_examples),
    )

    top_scores, top_indices = torch.topk(
        similarities,
        k=top_k,
    )

    selected_examples = []

    for score, index in zip(
        top_scores,
        top_indices,
    ):

        example = candidate_examples[
            index.item()
        ]

        selected_examples.append(
            {
                "similarity": round(
                    float(score.item()),
                    4,
                ),
                "question": example["question"],
                "answer": example["answer"],
                "language": example["language"],
            }
        )

    return json.dumps(
        selected_examples,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Course Glossary
# ============================================================

COURSE_GLOSSARY = {
    "rag": (
        "RAG stands for Retrieval-Augmented Generation. "
        "It retrieves relevant information from a knowledge base "
        "before the language model generates the answer."
    ),
    "fine-tuning": (
        "Fine-tuning means training a pretrained model further "
        "on a specific dataset to adapt its behavior or performance."
    ),
    "agent": (
        "An AI agent is a system that can reason about a task "
        "and decide which tools or actions are needed to complete it."
    ),
    "pruning": (
        "Pruning is the process of removing unnecessary branches "
        "or options from a decision process to reduce the search space."
    ),
}


@tool
def get_course_glossary(term: str) -> str:
    """Return a concise definition of an Agentic AI course term."""

    normalized_term = term.strip().lower()

    if normalized_term in COURSE_GLOSSARY:
        return COURSE_GLOSSARY[normalized_term]

    return f"No glossary definition found for: {term}"