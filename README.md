# Maintenance Action Generation with LLM Agent

This project compares a vanilla LLM baseline with an agent-based workflow
for generating maintenance actions from fault descriptions.

## Task
Input: Fault description (Problem)
Output: Maintenance action (text generation)

## Dataset
Aircraft Historical Maintenance Logs (Fault → Action)
(https://www.kaggle.com/datasets/merishnasuwal/aircraft-historical-maintenance-dataset)

## Methods
- Baseline: Vanilla LLM Prompting
- Agent: RAG-based generation + re-ranking

## How to Run

1. Start Ollama:
```bash
ollama run llama3.2:3b
```
2. run python code :
```bash
python src/run_experiment.py
```
