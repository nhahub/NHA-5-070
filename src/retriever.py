from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "intfloat/multilingual-e5-small"


class NotesRetriever:

    def __init__(self, knowledge_path: str, top_k: int = 3):

        self.top_k = top_k

        # Load E5 model
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        # Read knowledge
        path = Path(knowledge_path)

        knowledge_text = path.read_text(
            encoding="utf-8"
        )

        # Split into chunks
        chunks = knowledge_text.split("\n## ")

        chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

        # Remove document title
        if chunks and chunks[0].startswith(
            "# Agentic AI Course Notes"
        ):
            chunks = chunks[1:]

        self.chunks = chunks

        # Create passage embeddings
        passages = [
            "passage: " + chunk
            for chunk in self.chunks
        ]

        self.embeddings = self.embed(passages)

    def embed(self, texts):

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state

        attention_mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
        )

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
            dim=1
        )

    def search(self, query: str):

        # Convert question into query embedding
        query_embedding = self.embed([
            "query: " + query
        ])

        # Compare query with all chunks
        similarities = torch.matmul(
            query_embedding,
            self.embeddings.T
        ).squeeze(0)

        # Get Top-K
        top_scores, top_indices = torch.topk(
            similarities,
            k=min(self.top_k, len(self.chunks))
        )

        results = []

        for score, index in zip(
            top_scores,
            top_indices
        ):

            results.append({
                "text": self.chunks[index.item()],
                "score": float(score.item())
            })

        return results