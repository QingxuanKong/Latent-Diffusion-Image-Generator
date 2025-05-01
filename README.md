# HW5-README

# Objective

This project aims to implement a Latent Diffusion Model (LDM) enhanced U-Net framework, which performs denoising in a compressed latent space using a pre-trained Variational Autoencoder (VAE) to reduce computational overhead while preserving fidelity. To further enhance the generation quality, we incorporate Classifier-Free Guidance (CFG)  for class-conditional generation and integrate Transformer blocks into the U-Net bottleneck to improve the model’s capacity for capturing long-range dependencies. Our objective is to develop a scalable, computationally efficient, and high-fidelity image generation pipeline.

# Wandb Log

https://wandb.ai/idl-/IDL-hw5-Ivy

https://wandb.ai/islakong-carnegie-mellon-university/dl-hw5

https://wandb.ai/idl-/dl-hw5

# Methodology

## U-Net Architecture

The U-Net is the backbone of our diffusion model, designed to predict noise at each timestep. It uses a symmetric encoder-decoder structure with skip connections, where the encoder reduces spatial resolution and increases channels, and the decoder reverses this process. Middle layers handle low-resolution features for better pattern recognition. Each level includes ResBlocks with normalization and residuals to support training. A sinusoidal embedding encodes the timestep, allowing the model to adjust to noise levels. Channel depth increases as resolution decreases to improve multi-scale feature learning.

## Schedulers

The DDPM scheduler defines the noise variance schedule during both forward and reverse diffusion. We also experimented with DDIM. The DDIM scheduler accelerates sampling by redefining the reverse process without requiring a strict Markov chain. This allows the reverse process to skip multiple steps, significantly reducing the inference time. 

## Variational Autoencoder (VAE)

Variational Autoencoder (VAE) compresses input images into a lower-dimensional latent space through a probabilistic mapping to improve training efficiency.

Operating in the latent space provides the following computational benefits:

- A 16× reduction in space significantly decreases memory usage and computational cost.
- Compression focuses modeling capacity on semantically meaningful variations rather than redundant pixel-level details.
- Smoother latent distributions lead to more stable training dynamics and faster convergence.

## Classifier-Free Guidance (CFG)

Classifier-Free Guidance (CFG) improves alignment between generated samples and conditioning inputs (such as class labels or text prompts) without requiring a separate classifier.

## Transformer Block

The transformer Block is incorporated in the U-Net bottleneck module in place of the traditional ResNet block to further enhance the model's capacity. This modification allows the model to better capture long-range dependencies across spatial features at the lowest resolution, where computation is more efficient.

The transformer block first flattens the feature map into a sequence of tokens and adds learnable positional embeddings. Each block then applies Adaptive Layer Normalization (AdaLN) conditioned on the timestep and optional class embedding, followed by Multi-Head Self-Attention and a two-layer MLP. Residual connections are used around both the attention and MLP sub-layers.

# Results