<h2 align="center">🎬 VideoExplorer: Thinking with Video for Long-Form Understanding </a></h2>

<p align="center">
    <a href="https://arxiv.org/pdf/2506.10821">
            <img alt="Build" src="http://img.shields.io/badge/cs.CV-arXiv%3A2406.04264-B31B1B.svg">
    </a>
</p>

## 👉 Introduction

**VideoExplorer** is a novel framework for long-video understanding  that moves beyond single-pass reasoning. Inspired by the "thinking with video" principle, it performs **faithful, efficient, and interpretable reasoning** by dynamically exploring video content.


## 🎉 News 
2025.10.16 - We released the newest version of VideoDeepResearch called VideoExplorer! It's smaller, cheaper, but just as effective in long video understanding. Details refer to our [updated paper]("https://arxiv.org/pdf/2506.10821"). ✨
2025.06.10 - We released the first version of VideoDeepResearch. 🎬

## 🚀 Overview

Long-video understanding is challenging. Existing methods often sacrifice detail by downsampling or rely on task-agnostic representations, limiting their perception.

VideoExplorer solves this by **intertwining planning, temporal grounding, and scalable perception** into a coherent, iterative loop:
1.  **Formulates** a sub-question.
2.  **Locates** the relevant moments.
3.  **Performs** task-oriented, fine-grained perception.
4.  **Repeats** until the final answer is reached.

## 💡 Key Features

*   **Iterative Reasoning:** Dynamically explores video content instead of relying on a static context.
*   **Task-Oriented Perception:** Focuses computational resources on relevant moments, enabling scalable analysis.
*   **Interpretable Trajectories:** Each step of the reasoning process is transparent and traceable.

## 🏛️ Framework & Training

To overcome the lack of LVU training data, we constructed a high-quality dataset using **difficulty-adaptive sampling**. Our training pipeline consists of:
1.  **Supervised Trajectory Initialization**
2.  **Trajectory-level Preference Optimization**

This two-stage approach encourages adaptive temporal grounding and iterative information integration guided by downstream rewards.

## 📈 Results

Extensive evaluations on popular long-video benchmarks show that VideoExplorer achieves **significant performance advantages** over existing baselines, demonstrating its robustness, adaptability, and efficiency.


---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/yhy-2000/VideoDeepResearch.git
cd VideoDeepResearch

# Install dependencies
pip install -r requirements.txt
```


**Project Layout:**

```
VideoDeepResearch/
├── requirements.txt              # Python dependencies
├── eval/                         # Code for evaluating benchmarks
├── train/                        # Code for supervised finetuning (SFT) and trajectory-based direct preference optimization (TDPO)
├── asset/                        # Assets used in the demo
├── data/
│   ├── videos/                   # Raw video files
│   ├── clips/                    # Generated video clips
│   ├── dense_frames/             # Extracted key frames
└── README.md                     # This documentation
```


## Launch Demo

```bash
base eval/demo.sh
```

## Evaluation on Benchmarks
```bash
base eval/eval.sh
```

## Training

Our training dataset is available at https://huggingface.co/datasets/avery00/VideoExplorer-Dataset/tree/main. To set up:

1. Place dpo_marathon.json in train/LLaMA-Factory-dpo/data.

2. Place the remaining two files in train/LLaMA-Factory-sft/data.

## Environment Setting
```bash
mv train/LLaMA-Factory-sft train/LLaMA-Factory-main
cd train/LLaMA-Factory-main
pip install -e ".[torch,metrics]" --no-build-isolation
mv train/LLaMA-Factory-main train/LLaMA-Factory-sft
```

## Supervised Finetuning
```bash
cd train

# load the right code
mv train/LLaMA-Factory-sft train/LLaMA-Factory-main

# finetuning planner
bash sft_planner.sh

# finetuning temporal grounder
bash sft_temporal_grounding_agent.sh

mv train/LLaMA-Factory-main train/LLaMA-Factory-sft
```

## Trajectory-based Direct Preference Optimization
```bash
# load the right code
mv train/LLaMA-Factory-dpo train/LLaMA-Factory-main

# Trajectory-based DPO
bash train/dpo_planner.sh

mv train/LLaMA-Factory-main train/LLaMA-Factory-dpo
```

---
## 📬 Contact

Encounter issues or have questions? Reach out to:

> **H.Y. Yuan**
> Email: [hyyuan@ruc.edu.cn](mailto:hyyuan@ruc.edu.cn)


## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
@misc{yuan2025thinkvideosagenticlongvideo,
      title={Think With Videos For Agentic Long-Video Understanding}, 
      author={Huaying Yuan and Zheng Liu and Junjie Zhou and Hongjin Qian and Yan Shu and Nicu Sebe and Ji-Rong Wen and Zhicheng Dou},
      year={2025},
      eprint={2506.10821},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.10821}, 
}
```
