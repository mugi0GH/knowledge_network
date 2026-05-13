# Knowledge Network

A personal AI/ML knowledge base covering Computer Vision, NLP/LLM, Reinforcement Learning, and foundational programming (C++17, CUDA).

Built as a structured study path with implementations, derivations, and notes.

## Repository Structure

```
knowledge_network/
├── basic/                          # Foundations
│   ├── cpp17/                      C++17 knowledge map
│   ├── cuda/                       CUDA programming (RAII, memory management)
│   └── nlp/                        Transformer implementation from scratch (PyTorch)
│
└── specific/                       # Domain-specific study
    ├── CV/                         Computer Vision
    │   ├── 1.image_classification/ LeNet → AlexNet → VGG → GoogLeNet → ResNet
    │   ├── 2.object_detection/     Object detection (Fast R-CNN series)
    │   └── clip/                   CLIP multimodal
    ├── NLP/
    │   ├── llm_framework/          LLM inference framework (vLLM, RAG, Agent)
    │   └── 2.LLMs/                 LLM fundamentals
    ├── Reinforcement_Learning/     RL algorithm implementations (DQN → TRPO → PPO → Actor-Critic)
    └── Timeseries/                 Time-series analysis
```

## Quick Navigation

|Directory                                                          |Content                                      |Stack           |
|-------------------------------------------------------------------|---------------------------------------------|----------------|
|[basic/cpp17](basic/cpp17/)                                        |C++17 knowledge tree                         |Markdown        |
|[basic/cuda](basic/cuda/)                                          |CUDA fundamentals + RAII patterns            |CUDA C++        |
|[basic/nlp/Transformer](basic/nlp/2.Mechanism/Transformer/)        |Transformer built from scratch               |PyTorch         |
|[specific/CV](specific/CV/)                                        |Image classification → Detection → Multimodal|PyTorch         |
|[specific/Reinforcement_Learning](specific/Reinforcement_Learning/)|RL implementations with derivations          |PyTorch         |
|[specific/NLP/llm_framework](specific/NLP/llm_framework/)          |Local LLM inference stack                    |vLLM, RAG, Agent|

## Highlights

**Reinforcement Learning** — Covers DQN, TRPO, PPO, and Actor-Critic with full implementations.
The TRPO section includes a rigorous mathematical derivation: KL-divergence constraints, Natural Gradient, Fisher Information Matrix, Conjugate Gradient, and Line Search for step-size backtracking.

**Transformer from Scratch** — Full PyTorch implementation of the attention mechanism and encoder-decoder architecture.

**CUDA + C++17** — Low-level programming foundations directly relevant to edge inference optimization work.

## Dependencies

- Python 3.x + PyTorch
- pixi (package manager)
- CUDA Toolkit (optional, required only for `basic/cuda/`)
