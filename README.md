# ResNet From Scratch – PyTorch Implementation

This project implements the ResNet architecture completely from scratch in PyTorch to understand deep residual learning and skip connections in deep convolutional neural networks.

The objective was to manually construct residual blocks and train a deep CNN without using pretrained models, in order to analyze convergence behavior, gradient stability, and loss reduction across epochs.

## Architecture

The network includes:

- Convolution layers
- Batch Normalization
- ReLU activations
- Residual (skip) connections
- Global Average Pooling
- Fully connected classification layer

Each residual block follows the formulation:

H(x) = F(x) + x

where F(x) represents the learned residual mapping and x is the identity shortcut connection.

## Implementation Details

- Framework: PyTorch
- Optimizer: Adam
- Loss Function: Cross Entropy
- Custom training loop
- Manual forward pass implementation
- Loss monitoring across epochs

## Project Structure

resnet-from-scratch-pytorch/
│
├── model.py
├── resnet50_from_scratch.py
├── dataset.py
├── transforms.py
├── train.py
├── config.py
└── README.md

## Key Learnings

- Understanding residual learning
- Mitigating vanishing gradient problems
- Deep CNN training stability
- Optimization and loss convergence behavior

This architectural understanding was later applied in a separate PPE detection system using Faster R-CNN with a ResNet backbone.
