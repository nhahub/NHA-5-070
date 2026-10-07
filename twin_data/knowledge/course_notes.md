# Agentic AI Course Notes

## RAG

RAG stands for Retrieval-Augmented Generation.

The system retrieves relevant information from a knowledge base and provides that information to the language model before generating the final answer.

RAG is useful when the model needs access to specific or updated information without changing the model itself.

## Fine-Tuning

Fine-tuning means training a pretrained model further on a specific dataset.

It can be used to adapt model behavior, style, or performance for a particular task.

Unlike RAG, fine-tuning does not simply retrieve external information at query time.

## AI Agent

An AI agent is a system that can reason about a task and decide which tools or actions are needed to complete it.

A tool-using agent can select tools, use their results, and continue processing before producing a final answer.

## Decision Tree

A decision tree represents decisions as a sequence of conditions.

In the Class Twin, the decision policy can consider whether the question is addressed to Ahmed, whether the audio quality is acceptable, and whether the system has enough information to answer.

Possible actions include:

- answer
- ask_to_repeat
- defer
- stay_silent

## Pruning

Pruning removes branches that are unlikely or unnecessary before expensive processing.

In the Class Twin, cheap checks should happen first so that the LLM does not score every possible action unnecessarily.

## Memory

The Class Twin keeps a rolling memory of the meeting transcript and previous answers.

This allows it to handle follow-up questions that refer to something said earlier.