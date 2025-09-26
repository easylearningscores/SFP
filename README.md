🚀 **Note to Reviewers:** This anonymous GitHub repository accompanies our paper submission. We chose this format to effectively showcase dynamic visualizations (e.g., animated GIFs) of our forecasting results, which are best understood in motion. The repository is fully anonymized and contains no information that could identify the authors. We thank you for your time and careful consideration in reviewing our work.


This repository contains the official PyTorch implementation for our paper, "**Spatiotemporal Forecasting as Planning: A Model-Based Reinforcement Learning Approach with Generative World Models**", submitted for review.



# Spatiotemporal Forecasting as Planning (SFP)

We introduce **SFP (Spatiotemporal Forecasting as Planning)**, a novel paradigm that reframes spatiotemporal forecasting as a Model-Based Reinforcement Learning (MBRL) planning problem. SFP is designed to address the critical challenges faced by conventional deep learning methods: optimizing for **non-differentiable**, domain-specific metrics (e.g., CSI in extreme weather prediction) and performing robustly in data-scarce scenarios.


---

## 🚀 Core Idea: From Supervised Learning to Planning

### The Dilemma of Conventional Methods

Conventional forecasting models (Fig. a) rely on differentiable proxy losses like Mean Squared Error (MSE) for end-to-end optimization. However, in many scientific domains, true performance is measured by domain-specific metrics that are often **non-differentiable** (e.g., Critical Success Index (CSI), Turbulent Kinetic Energy (TKE) spectrum). This creates a **"Fundamental Disconnect"** between the training objective and the actual evaluation criteria, leading to models that, despite low MSE, fail to capture critical extreme events or maintain physical consistency.

### Our New Paradigm: SFP

The SFP framework (Fig. b) breaks this limitation. We treat the forecasting model as an **Agent** that learns a policy. Instead of directly outputting a final prediction, the agent generates an "intention" or **Action**. This action guides a learned **Generative World Model**, which performs forward exploration in its "imagination space" to generate a diverse set of high-fidelity future possibilities. A **Planning Algorithm** then leverages the non-differentiable domain metric as a **Reward Function** to identify the future trajectory with the highest return. Finally, this high-reward trajectory serves as a high-quality **pseudo-label** to update the agent's policy via an iterative **self-training** loop.

<p align="center">
  <img src="Figure/github_intro.png" width="900" alt="SFP Paradigm">
</p>
<p align="center">
  <b>Figure 1</b>: (a) The conventional supervised learning paradigm vs. (b) our SFP planning paradigm. SFP forms a closed-loop learning system that allows the agent to optimize directly for the true objectives of the task.
</p>


---

## ✨ Key Contributions & Highlights

1.  **A New Paradigm**: We are the first to systematically formalize spatiotemporal forecasting as an MBRL problem, providing a principled pathway to directly optimize for non-differentiable scientific metrics.

2.  **An Innovative Framework**: We design and implement SFP, which creatively integrates:
    *   A VQ-VAE-based **Generative World Model** to simulate the stochastic dynamics of physical systems.
    *   A **Beam Search-based Planning Algorithm** for efficient exploration within the "imagined" future.
    *   A **closed-loop self-training process** that distills knowledge from non-differentiable rewards into the agent's policy.

<p align="center">
  <img src="Figure/stage1.png" width="900" alt="stage1 Paradigm">
</p>
<p align="center">
  <b>Figure 2: Architecture of our Generative World Model.
</p>
    

<p align="center">
  <img src="Figure/stage2.png" width="900" alt="stage2 Paradigm">
</p>
<p align="center">
  <b>Figure 3: Iterative Policy Optimization via Planning and Self-Training.
</p>

3.  **State-of-the-Art Performance**: On several challenging benchmarks (e.g., extreme weather, turbulence, combustion), SFP demonstrates significant improvements:
    *   **Significant Error Reduction**: Achieved **up to 39% MSE reduction** across various tasks.
    *   **Optimized Domain Metrics**: Boosted the Critical Success Index (CSI) by an average of **29.7%** on the SEVIR dataset and reduced the Turbulent Kinetic Energy (TKE) spectrum error by an average of **57.3%** on the NSE task.
    *   **Enhanced Physical Consistency**: Generated more realistic, fine-grained vortex structures in turbulence simulations with energy spectra that closely match ground truth.
    *   **Superior Extreme Event Capture**: Accurately predicted high-intensity cores of marine heatwaves that were completely missed by baseline models, especially in data-scarce settings.

<p align="center">
  <img src="Figure/Table1.png" width="900" alt="stage2 Paradigm">
</p>
<p align="center">
</p>
  
---

## 🛠️ Getting Started

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

### 2. Stage 1 Reproduction: Pre-training the Generative World Model

In the first stage of the SFP framework, our primary goal is to train a high-quality **Generative World Model (GWM)**. This model, based on a Conditional VQ-VAE architecture with a Top-K quantization mechanism, is designed to learn the fundamental dynamics and data distribution of the physical system. It is trained through a **reconstruction task**: learning to encode an input spatiotemporal state into a discrete latent representation and then accurately decode it back to its original state.

The success of this stage is foundational for the subsequent planning task (Stage 2), as it provides the agent with a high-fidelity and explorable "imagination space."

#### Step 1: Understanding the Data Loader (`dataloader_rec.py`)

We use the `dataloader_rec.py` script to prepare the training data. Its key features are:

*   **Designed for Reconstruction**: The `__getitem__` function returns `(data, data.clone())`, meaning the model's input and target are identical, which aligns with the autoencoder training paradigm.
*   **On-Demand Loading**: Instead of loading the entire dataset into memory at once, data is read from `.nc` files on disk as needed. This is ideal for handling large-scale climate datasets.
*   **Flexibility**: It supports filtering data by year, variable, and time step, facilitating various experimental configurations.

#### Step 2: Understanding the Model Architecture (`generative_world_model.py`)

The core model architecture is defined in `generative_world_model.py`. It consists of several key components:

1.  **`AtmosphericEncoder` & `AtmosphericDecoder`**: A multi-scale convolutional encoder-decoder pair with skip connections, designed to efficiently capture spatial features at different scales.
2.  **`VectorQuantizerEMA`**: The core Top-K vector quantization module. It maps the continuous features from the encoder to the *K* nearest codebook vectors, thereby discretizing the latent space. This not only compresses information but also provides the basis for generating diversity through the Top-K selections.
3.  **`Generative_World_Model`**: The top-level wrapper class. Its forward pass returns three critical outputs:
    *   `pred`: The primary reconstructed image, based on the single closest (Top-1) codebook vector.
    *   `top_k_features`: A list containing multiple versions of the reconstructed image, each decoded from one of the Top-K nearest codebook vectors. This embodies the "generative" capability of the model and provides diverse future possibilities for planning.
    *   `vq_loss`: The loss incurred during the quantization process, used to optimize the codebook.

You can easily instantiate and test the model as shown below:

```python
if __name__ == '__main__':
    # --- Instantiate and Test the Model ---
    # Create a sample input tensor: batch size of 1, 69 climate variables, 180x360 resolution
    input_tensor = torch.rand(1, 69, 180, 360)
    
    # Initialize the Generative World Model
    # in_channel: Number of input channels (i.e., number of variables)
    # res_layers: Number of layers in the encoder/decoder
    # embedding_nums: Size of the codebook (K)
    # embedding_dim: Dimensionality of each codebook vector (D)
    # top_k: The number of nearest neighbors to select during quantization
    model = Generative_World_Model(in_channel=69,
                                   res_layers=2,
                                   embedding_nums=1024, 
                                   embedding_dim=256,
                                   top_k=10)
    
    # Perform a forward pass
    pred, top_k_features, vq_loss = model(input_tensor)
    
    # Print the shapes of the outputs
    print(f"Shape of the primary reconstruction: {pred.shape}")
    print(f"Number of Top-K generated samples: {len(top_k_features)}")
    print(f"Shape of the first Top-K sample: {top_k_features[0].shape}")
    print(f"VQ-Loss: {vq_loss.item()}")
```

#### Step 3: Running the Training (`pretrain_multi_scale.py`)

The `pretrain_multi_scale.py` script is the main entry point for executing the Stage 1 training. It integrates the data loader and model definition and includes a complete multi-GPU distributed training (DDP) workflow.

**How to Run the Training:**

This script is designed for a multi-GPU environment. You need to launch it using `torchrun` or a similar utility. Assuming you have `N` GPUs available, run the following command in your terminal:

```bash
# Replace NUM_GPUS with the number of GPUs you wish to use (e.g., 4 or 8)
torchrun --nproc_per_node=NUM_GPUS pretrain_multi_scale.py
```

**What the Script Does:**

1.  **Initializes Distributed Environment**: Sets up a process for each GPU.
2.  **Prepares Distributed Datasets**: Uses `DistributedSampler` to efficiently shard the data across all GPUs.
3.  **Builds the Model**: Initializes `Generative_World_Model` and wraps it with `DistributedDataParallel`.
4.  **Starts the Training Loop**:
    *   The model is trained on the training set. The total loss is a weighted sum of the **Reconstruction Loss (MSE)** and the **Codebook Loss (VQ Loss)**.
    *   After each epoch, performance is evaluated on the validation set.
    *   Training logs are saved to the `./logs/` directory.
    *   The best-performing model checkpoint is saved to `./checkpoints/beamvq_reconstruction_v1_best_model.pth`.
5.  **Final Testing**: Once training is complete, the script automatically loads the best model checkpoint, runs a final evaluation on the test set, and saves the results (inputs, targets, and outputs) as `.npy` files in the `./results/` directory.

The reconstruction results are as shown in the figure below:
<p align="center">
  <img src="Figure/var_01_com.png" width="1000" alt="reconstruction results Paradigm">
</p>
<p align="center">
  <b>Figure 4: reconstruction results.
</p>

    
Upon successful completion of these steps, you will have a pre-trained, high-quality Generative World Model ready for use in Stage 2 of the SFP framework.
