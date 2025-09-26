🚀 **Note to Reviewers:** This anonymous GitHub repository accompanies our paper submission. We chose this format to effectively showcase dynamic visualizations (e.g., animated GIFs) of our forecasting results, which are best understood in motion. The repository is fully anonymized and contains no information that could identify the authors. We thank you for your time and careful consideration in reviewing our work.

---

This repository contains the official PyTorch implementation for our paper, "**Spatiotemporal Forecasting as Planning: A Model-Based Reinforcement Learning Approach with Generative World Models**", submitted for review.

# Spatiotemporal Forecasting as Planning (SFP)

We introduce **SFP (Spatiotemporal Forecasting as Planning)**, a novel paradigm that reframes spatiotemporal forecasting as a Model-Based Reinforcement Learning (MBRL) planning problem. SFP is designed to address the critical challenges faced by conventional deep learning methods: optimizing for **non-differentiable**, domain-specific metrics (e.g., CSI in extreme weather prediction) and performing robustly in data-scarce scenarios.

---

## 🚀 Core Idea: From Supervised Learning to Planning

### The Dilemma of Conventional Methods

Conventional forecasting models (Fig. 1a) rely on differentiable proxy losses like Mean Squared Error (MSE) for end-to-end optimization. However, in many scientific domains, true performance is measured by domain-specific metrics that are often **non-differentiable** (e.g., Critical Success Index (CSI), Turbulent Kinetic Energy (TKE) spectrum). This creates a **"Fundamental Disconnect"** between the training objective and the actual evaluation criteria, leading to models that, despite low MSE, fail to capture critical extreme events or maintain physical consistency.

### Our New Paradigm: SFP

The SFP framework (Fig. 1b) breaks this limitation. We treat the forecasting model as an **Agent** that learns a policy. Instead of directly outputting a final prediction, the agent generates an "intention" or **Action**. This action guides a learned **Generative World Model**, which performs forward exploration in its "imagination space" to generate a diverse set of high-fidelity future possibilities. A **Planning Algorithm** then leverages the non-differentiable domain metric as a **Reward Function** to identify the future trajectory with the highest return. Finally, this high-reward trajectory serves as a high-quality **pseudo-label** to update the agent's policy via an iterative **self-training** loop.

<p align="center">
  <img src="Figure/github_intro.png" width="900" alt="SFP Paradigm">
</p>
<p align="center">
  <b>Figure 1</b>: (a) The conventional supervised learning paradigm vs. (b) our SFP planning paradigm. SFP forms a closed-loop learning system that allows the agent to optimize directly for the true objectives of the task.
</p>

---

## ✨ Framework & Highlights

1.  **A New Paradigm**: We are the first to systematically formalize spatiotemporal forecasting as an MBRL problem, providing a principled pathway to directly optimize for non-differentiable scientific metrics.

2.  **An Innovative Framework**: We design and implement SFP, which creatively integrates:
    *   A VQ-VAE-based **Generative World Model (GWM)** to simulate the stochastic dynamics of physical systems (Fig. 2).
    *   A **Beam Search-based Planning Algorithm** for efficient exploration within the "imagined" future.
    *   A **closed-loop self-training process** that distills knowledge from non-differentiable rewards into the agent's policy (Fig. 3).

<p align="center">
  <img src="Figure/stage1.png" width="900" alt="Generative World Model Architecture">
</p>
<p align="center">
  <b>Figure 2</b>: Architecture of our Generative World Model (GWM).
</p>
    
<p align="center">
  <img src="Figure/stage2.png" width="900" alt="Policy Optimization Loop">
</p>
<p align="center">
  <b>Figure 3</b>: Iterative Policy Optimization via Planning and Self-Training.
</p>

3.  **State-of-the-Art Performance**: On several challenging benchmarks (e.g., extreme weather, turbulence, combustion), SFP demonstrates significant improvements:
    *   Achieved **up to 39% MSE reduction** across various tasks.
    *   Boosted the CSI by an average of **29.7%** on the SEVIR dataset and reduced the TKE spectrum error by an average of **57.3%** on the NSE task.
    *   Accurately predicted high-intensity cores of marine heatwaves that were completely missed by baseline models.

<p align="center">
  <img src="Figure/Table1.png" width="900" alt="Performance Table">
</p>
  
---

## 🛠️ Getting Started

Follow these steps to set up the environment, prepare the data, and run the complete SFP two-stage training pipeline.

### 1. Environment Setup

We recommend using Conda to set up the environment:

```bash
# Clone the repository
git clone https://github.com/easylearningscores/SFP.git
cd SFP

# Create and activate the Conda environment
conda create -n sfp python=3.8
conda activate sfp

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation
Please download the required datasets and place them under a unified data/ directory. You will need to specify the path to your data in the configuration files.



### 3. SFP Two-Stage Training Pipeline
Our framework is trained in two separate stages. Each stage is executed by a main script, with parameters managed through YAML configuration files located in the configs/ directory.




