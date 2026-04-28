# Knowledge Network

个人 AI / ML 知识库，覆盖计算机视觉、自然语言处理、强化学习、基础编程语言等。

## 目录结构

```
knowledge_network/
├── basic/           # 基础知识
│   ├── cpp17/      C++17 知识架构
│   ├── cuda/       CUDA 编程入门
│   └── nlp/        NLP 基础（Transformer 手写实现）
│
├── specific/       # 各领域实践
│   ├── CV/         计算机视觉
│   │   ├── 1.image_classification/  LeNet → AlexNet → VGG → GoogLeNet → ResNet
│   │   ├── 2.object_detection/      目标检测（Fast R-CNN 等）
│   │   └── clip/                    CLIP 多模态
│   ├── NLP/        NLP / LLM
│   │   ├── llm_framework/           LLM 推理框架（vLLM, RAG, Agent）
│   │   └── 2.LLMs/                  LLM 教程
│   ├── Reinforcement_Learning/      强化学习（DQN → TRPO → PPO → Actor-Critic）
│   └── Timeseries/                  时序数据
```

## 快速导航

| 目录 | 内容 | 技术栈 |
|------|------|--------|
| [basic/cpp17](basic/cpp17/) | C++17 知识树 | Markdown |
| [basic/cuda](basic/cuda/) | CUDA 入门 + RAII | CUDA C++ |
| [basic/nlp/2.Mechanism/Transformer](basic/nlp/2.Mechanism/Transformer/) | Transformer 从零实现 | PyTorch |
| [specific/CV](specific/CV/) | 图像分类 → 检测 → 多模态 | PyTorch |
| [specific/NLP/llm_framework](specific/NLP/llm_framework/) | LLM 推理框架 | vLLM, RAG, Agent |
| [specific/Reinforcement_Learning](specific/Reinforcement_Learning/) | RL 算法实现 | PyTorch |

## Dependencies

- Python 3.x + PyTorch
- pixi（包管理）
- CUDA Toolkit（optional，仅 `basic/cuda/` 需要）
