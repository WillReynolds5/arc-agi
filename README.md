# Animus - ARC AGI Benchmark with Reinforcement Learning

This project aims to solve the Abstraction and Reasoning Corpus (ARC) benchmark using reinforcement learning techniques.

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the ARC dataset and place it in the `data/` directory

## Project Structure

- `data/`: ARC dataset files
- `src/`: Source code
  - `utils/`: Utility functions for data processing and visualization
  - `inference.py`: LLM inference implementation
  - `prompts.py`: Prompt generation and parsing
- `notebooks/`: Jupyter notebooks for exploration and analysis

## Usage

More instructions will be added as the project develops. 



## currsorrules
ARC Solver: Building an AI System for Abstract Reasoning
This repository implements a comprehensive framework for tackling the Abstraction and Reasoning Corpus (ARC) challenge - a benchmark designed to measure AI systems' ability to perform general reasoning tasks. We're building a self-improving system that leverages large language models to solve abstract visual pattern recognition problems. The framework follows a multi-stage approach: first, we use a powerful LLM (like Gemini) to generate solutions for ARC tasks through a two-stage prompting strategy - one prompt to solve the puzzle and another to extract structured output. We then evaluate these solutions using custom metrics like mean Intersection over Union (IoU) to quantify performance. The system runs multiple inference attempts per task, analyzes the distribution of scores, and selectively saves high-quality solutions as training examples. These examples are formatted as conversation pairs and used to fine-tune specialized models (potentially using techniques like LoRA on models such as Gemma-27B). This creates a feedback loop where the system continuously improves its reasoning capabilities by learning from its own successful solutions. The architecture is designed to run autonomously, systematically working through the entire dataset to generate training data, train improved models, and repeat the process - essentially bootstrapping increasingly better abstract reasoning capabilities.


hf token:
pip install huggingface_hub
huggingface-cli login
add token with hf_ and then this BsZuzRxQbbnSJXlDXblIYuPayYgpIxRDhV