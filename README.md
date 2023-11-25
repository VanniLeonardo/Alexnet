# AlexNet (2012) - Pure NumPy Implementation

A complete implementation of the AlexNet convolutional neural network architecture from scratch using only NumPy, faithfully reproducing the original 2012 paper by Krizhevsky, Sutskever, and Hinton.

## Features

- **Pure NumPy**: No external deep learning libraries (TensorFlow, PyTorch, etc.)
- **Complete Architecture**: All 8 layers with exact specifications
- **Grouped Convolutions**: Authentic dual-GPU architecture simulation
- **Training Ready**: Forward and backward propagation implemented
- **Well Documented**: Clear code with extensive comments

## Architecture

| Layer | Type | Input → Output | Parameters | Groups |
|-------|------|----------------|------------|---------|
| Conv1 | Standard | 227×227×3 → 55×55×96 | 34,944 | 1 |
| Conv2 | **Grouped** | 27×27×96 → 27×27×256 | 307,456 | 2 |
| Conv3 | Standard | 13×13×256 → 13×13×384 | 885,120 | 1 |
| Conv4 | **Grouped** | 13×13×384 → 13×13×384 | 663,936 | 2 |
| Conv5 | **Grouped** | 13×13×384 → 13×13×256 | 442,624 | 2 |
| FC1 | Dense | 9216 → 4096 | 37,752,832 | - |
| FC2 | Dense | 4096 → 4096 | 16,781,312 | - |
| FC3 | Dense | 4096 → 1000 | 4,097,000 | - |

**Total Parameters**: 60,965,224 (~61M parameters, 233 MB)

## Key Components

- **ReLU Activation**: Applied after each convolutional and fully connected layer
- **Max Pooling**: 3×3 kernels with stride 2 after conv1, conv2, and conv5
- **Local Response Normalization (LRN)**: After pool1 and pool2 layers
- **Dropout**: 50% probability after FC1 and FC2 during training
- **Grouped Convolutions**: 2 groups each in conv2, conv4, and conv5
- **Softmax Output**: 1000-class classification

## Quick Start

### Requirements
```bash
pip install numpy
```

### Run the Demo
```bash
python alexnet.py
```

## Performance

- **Forward Pass**: ~5-6 seconds (CPU, batch size 2)
- **Backward Pass**: ~0.2-0.4 seconds  
- **Memory Usage**: ~233 MB for model weights
- **Parameter Efficiency**: 50% reduction in grouped conv layers

## Implementation Details

### Grouped Convolution Benefits
- **Parameter Reduction**: 50% fewer parameters in conv2, conv4, conv5
- **Memory Efficiency**: Originally enabled dual-GPU training in 2012
- **Regularization Effect**: Forces feature specialization per group
- **Single Hardware**: Works on single CPU/GPU (no dual-GPU required)

### Original Paper Specifications
- **Input**: 227×227×3 RGB images
- **Weight Initialization**: Gaussian (μ=0, σ=0.01)
- **Bias Initialization**: 1 for conv2/conv4/FC layers, 0 for others
- **LRN Parameters**: depth_radius=2, α=1e-4, β=0.75, bias=1
- **Dropout**: p=0.5 for training

## Example Output

```
AlexNet initialized with 1000 output classes
Dropout probability: 0.5

Generated synthetic input data: (2, 3, 227, 227)
Input shape: (2, 3, 227, 227)
Conv1 + ReLU shape: (2, 96, 55, 55)
Pool1 shape: (2, 96, 27, 27)
LRN1 shape: (2, 96, 27, 27)
Conv2 + ReLU shape: (2, 256, 27, 27)
Pool2 shape: (2, 256, 13, 13)
LRN2 shape: (2, 256, 13, 13)
Conv3 + ReLU shape: (2, 384, 13, 13)
Conv4 + ReLU shape: (2, 384, 13, 13)
Conv5 + ReLU shape: (2, 256, 13, 13)
Pool3 shape: (2, 256, 6, 6)
Flattened shape: (2, 9216)
FC1 + ReLU shape: (2, 4096)
FC1 + Dropout shape: (2, 4096)
FC2 + ReLU shape: (2, 4096)
FC2 + Dropout shape: (2, 4096)
FC3 shape: (2, 1000)
Output shape: (2, 1000)

Forward pass completed in 5.47 seconds
Final output shape: (2, 1000)
Output probabilities sum: [1. 1.]
Top-5 predicted classes for first image: [717 877 775 364 928]

Cross-entropy loss: 8.8756

Backward pass completed in 0.43 seconds
Gradients computed:
  fc3_W: (4096, 1000)
  fc3_b: (1000,)
  fc2_W: (4096, 4096)
  fc2_b: (4096,)
  fc1_W: (9216, 4096)
  fc1_b: (4096,)
Conv1: 34,944 parameters
Conv2: 307,456 parameters
Conv3: 885,120 parameters
Conv4: 663,936 parameters
Conv5: 442,624 parameters
FC1: 37,752,832 parameters
FC2: 16,781,312 parameters
FC3: 4,097,000 parameters

Total parameters: 60,965,224
Model size (approx): 232.6 MB (float32)
Parameter comparison (Grouped vs. Non-grouped):
Conv2: 307,456 (grouped) vs 614,656 (normal) - 50.0% reduction
Conv4: 663,936 (grouped) vs 1,327,488 (normal) - 50.0% reduction
Conv5: 442,624 (grouped) vs 884,992 (normal) - 50.0% reduction

Total conv params: 1,414,016 (grouped) vs 2,827,136 (normal)
Overall reduction: 50.0% of normal parameters
Input shape: (1, 4, 8, 8)
Normal conv output shape: (1, 8, 8, 8)
Normal conv parameters: 296
Grouped conv output shape: (1, 8, 8, 8)
Grouped conv parameters: 152

Parameter reduction: 51.4%
```

## Reference

**ImageNet Classification with Deep Convolutional Neural Networks**  
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
*NIPS 2012*

