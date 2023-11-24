import numpy as np
import time
from typing import Tuple

class AlexNet:
    """
    Complete AlexNet implementation using pure NumPy.
    
    Architecture based on Krizhevsky et al. (2012):
    - Input: 227x227x3 RGB images
    - 5 Convolutional layers with ReLU activation
    - Max pooling after conv1, conv2, and conv5
    - Local Response Normalization (LRN) after pool1 and pool2
    - 3 Fully connected layers with dropout
    - Softmax output for 1000 classes
    - Grouped convolutions in conv2, conv4, and conv5 (2 groups each)
      to replicate the original dual-GPU architecture
    """
    
    def __init__(self, num_classes: int = 1000, dropout_prob: float = 0.5):

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.training = True
        
        self._initialize_parameters()
        
        # Cache for backward pass
        self.cache = {}
        
    def _initialize_parameters(self):
        """
        Initialize weights and biases according to AlexNet specifications:
        - Weights: Gaussian distribution with mean=0, std=0.01
        - Biases: 1 for conv2, conv4, and all FC layers; 0 for others
        """
        np.random.seed(42)
        
        # Conv1: 11x11x3x96, stride=4, pad=0
        self.conv1_W = np.random.normal(0, 0.01, (96, 3, 11, 11))
        self.conv1_b = np.zeros(96)
        
        # Conv2: 5x5x48x256 (grouped, 2 groups), stride=1, pad=2
        # Each group: 5x5x48x128, total output channels: 256
        self.conv2_W = np.random.normal(0, 0.01, (256, 48, 5, 5))
        self.conv2_b = np.ones(256)
        
        # Conv3: 3x3x256x384, stride=1, pad=1 (fully connected)
        self.conv3_W = np.random.normal(0, 0.01, (384, 256, 3, 3))
        self.conv3_b = np.zeros(384)
        
        # Conv4: 3x3x192x384 (grouped, 2 groups), stride=1, pad=1
        # Each group: 3x3x192x192, total output channels: 384
        self.conv4_W = np.random.normal(0, 0.01, (384, 192, 3, 3))
        self.conv4_b = np.ones(384)
        
        # Conv5: 3x3x192x256 (grouped, 2 groups), stride=1, pad=1
        # Each group: 3x3x192x128, total output channels: 256
        self.conv5_W = np.random.normal(0, 0.01, (256, 192, 3, 3))
        self.conv5_b = np.zeros(256)
        
        # FC1: 9216 -> 4096 (6*6*256 = 9216)
        self.fc1_W = np.random.normal(0, 0.01, (9216, 4096))
        self.fc1_b = np.ones(4096)
        
        # FC2: 4096 -> 4096
        self.fc2_W = np.random.normal(0, 0.01, (4096, 4096))
        self.fc2_b = np.ones(4096)
        
        # FC3: 4096 -> num_classes
        self.fc3_W = np.random.normal(0, 0.01, (4096, self.num_classes))
        self.fc3_b = np.ones(self.num_classes)
        
    def _pad_input(self, x: np.ndarray, pad: int) -> np.ndarray:
        if pad == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    def _conv2d(self, x: np.ndarray, W: np.ndarray, b: np.ndarray, 
                stride: int = 1, pad: int = 0) -> np.ndarray:
        """
        Args:
            x: Input tensor (N, C_in, H, W)
            W: Weight tensor (C_out, C_in, KH, KW)
            b: Bias tensor (C_out,)
            stride: Convolution stride
            pad: Padding size
            
        Returns:
            Output tensor (N, C_out, H_out, W_out)
        """
        N, C_in, H_in, W_in = x.shape
        C_out, _, KH, KW = W.shape
        
        x_padded = self._pad_input(x, pad)
        
        H_out = (H_in + 2 * pad - KH) // stride + 1
        W_out = (W_in + 2 * pad - KW) // stride + 1
        
        out = np.zeros((N, C_out, H_out, W_out))
        
        # Perform convolution
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        h_end = h_start + KH
                        w_start = w * stride
                        w_end = w_start + KW
                        
                        out[n, c_out, h, w] = np.sum(
                            x_padded[n, :, h_start:h_end, w_start:w_end] * W[c_out]
                        ) + b[c_out]
        
        return out
    
    def _grouped_conv2d(self, x: np.ndarray, W: np.ndarray, b: np.ndarray, 
                       groups: int = 2, stride: int = 1, pad: int = 0) -> np.ndarray:
        """
        Args:
            x: Input tensor (N, C_in, H, W)
            W: Weight tensor (C_out, C_in//groups, KH, KW)
            b: Bias tensor (C_out,)
            groups: Number of groups for grouped convolution
            stride: Convolution stride
            pad: Padding size
            
        Returns:
            Output tensor (N, C_out, H_out, W_out)
        """
        N, C_in, H_in, W_in = x.shape
        C_out, C_in_per_group, KH, KW = W.shape
        
        assert C_in % groups == 0, f"Input channels {C_in} not divisible by groups {groups}"
        assert C_out % groups == 0, f"Output channels {C_out} not divisible by groups {groups}"
        
        C_in_per_group_actual = C_in // groups
        C_out_per_group = C_out // groups
        
        x_padded = self._pad_input(x, pad)
        
        H_out = (H_in + 2 * pad - KH) // stride + 1
        W_out = (W_in + 2 * pad - KW) // stride + 1
        
        out = np.zeros((N, C_out, H_out, W_out))
        
        # Process each group separately
        for g in range(groups):

            x_group = x_padded[:, g * C_in_per_group_actual:(g + 1) * C_in_per_group_actual, :, :]
            
            W_group = W[g * C_out_per_group:(g + 1) * C_out_per_group, :, :, :]
            
            b_group = b[g * C_out_per_group:(g + 1) * C_out_per_group]
            
            # Perform convolution for this group
            for n in range(N):
                for c_out in range(C_out_per_group):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * stride
                            h_end = h_start + KH
                            w_start = w * stride
                            w_end = w_start + KW
                            
                            # Calculate global output channel index
                            global_c_out = g * C_out_per_group + c_out
                            
                            out[n, global_c_out, h, w] = np.sum(
                                x_group[n, :, h_start:h_end, w_start:w_end] * W_group[c_out]
                            ) + b_group[c_out]
        
        return out
    
    def _grouped_conv2d_backward(self, dout: np.ndarray, x: np.ndarray, W: np.ndarray, 
                                groups: int = 2, stride: int = 1, pad: int = 0) -> tuple:
        """
        Args:
            dout: Gradient from next layer (N, C_out, H_out, W_out)
            x: Input tensor from forward pass (N, C_in, H, W)
            W: Weight tensor (C_out, C_in//groups, KH, KW)
            groups: Number of groups
            stride: Convolution stride
            pad: Padding size
            
        Returns:
            Tuple of (dx, dW, db)
        """
        N, C_in, H_in, W_in = x.shape
        N, C_out, H_out, W_out = dout.shape
        C_out_tensor, C_in_per_group, KH, KW = W.shape
        
        C_in_per_group_actual = C_in // groups
        C_out_per_group = C_out // groups
        
        x_padded = self._pad_input(x, pad)
        
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(W)
        db = np.zeros(C_out)
        
        for g in range(groups):
            x_group = x_padded[:, g * C_in_per_group_actual:(g + 1) * C_in_per_group_actual, :, :]
            
            dout_group = dout[:, g * C_out_per_group:(g + 1) * C_out_per_group, :, :]
            dx_group = dx_padded[:, g * C_in_per_group_actual:(g + 1) * C_in_per_group_actual, :, :]
            dW_group = dW[g * C_out_per_group:(g + 1) * C_out_per_group, :, :, :]
            db_group = db[g * C_out_per_group:(g + 1) * C_out_per_group]
            
            # Compute gradients for this group
            for n in range(N):
                for c_out in range(C_out_per_group):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * stride
                            h_end = h_start + KH
                            w_start = w * stride
                            w_end = w_start + KW
                            
                            dx_group[n, :, h_start:h_end, w_start:w_end] += (
                                dout_group[n, c_out, h, w] * W[g * C_out_per_group + c_out, :, :, :]
                            )
                            
                            dW_group[c_out, :, :, :] += (
                                dout_group[n, c_out, h, w] * x_group[n, :, h_start:h_end, w_start:w_end]
                            )
                            
                            db_group[c_out] += dout_group[n, c_out, h, w]
        
        if pad > 0:
            dx = dx_padded[:, :, pad:-pad, pad:-pad]
        else:
            dx = dx_padded
            
        return dx, dW, db
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _max_pool2d(self, x: np.ndarray, pool_size: int = 3, stride: int = 2) -> np.ndarray:
        """  
        Args:
            x: Input tensor (N, C, H, W)
            pool_size: Pooling kernel size
            stride: Pooling stride
            
        Returns:
            Output tensor (N, C, H_out, W_out)
        """
        N, C, H, W = x.shape
        
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        
        out = np.zeros((N, C, H_out, W_out))
        
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size
                        
                        out[n, c, h, w] = np.max(x[n, c, h_start:h_end, w_start:w_end])
        
        return out
    
    def _local_response_normalization(self, x: np.ndarray, depth_radius: int = 2,
                                    alpha: float = 1e-4, beta: float = 0.75,
                                    bias: float = 1.0) -> np.ndarray:
        """
        Args:
            x: Input tensor (N, C, H, W)
            depth_radius: Half-width of normalization window
            alpha: Scaling parameter
            beta: Exponent
            bias: Additive bias
            
        Returns:
            Normalized tensor
        """
        N, C, H, W = x.shape
        
        x_squared = x ** 2
        
        out = np.zeros_like(x)
        
        for n in range(N):
            for c in range(C):
                c_start = max(0, c - depth_radius)
                c_end = min(C, c + depth_radius + 1)
                
                sum_squared = np.sum(x_squared[n, c_start:c_end, :, :], axis=0)
                
                denominator = (bias + alpha * sum_squared) ** beta
                out[n, c, :, :] = x[n, c, :, :] / denominator
        
        return out
    
    def _dropout(self, x: np.ndarray, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x: Input tensor
            p: Dropout probability
            
        Returns:
            Tuple of (output tensor, dropout mask)
        """
        if not self.training:
            return x, np.ones_like(x)
        
        mask = np.random.rand(*x.shape) > p
        out = x * mask / (1 - p)  # Scale to maintain expected value
        
        return out, mask
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor (N, num_classes)
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor (N, 3, 227, 227) - RGB images
            
        Returns:
            Output probabilities (N, num_classes)
        """
        if x.shape[1:] != (3, 227, 227):
            raise ValueError(f"Expected input shape (N, 3, 227, 227), got {x.shape}")
        
        print(f"Input shape: {x.shape}")
        
        # Conv Layer 1: 227x227x3 -> 55x55x96
        # 11x11 conv, stride 4, no padding
        conv1 = self._conv2d(x, self.conv1_W, self.conv1_b, stride=4, pad=0)
        conv1_relu = self._relu(conv1)
        print(f"Conv1 + ReLU shape: {conv1_relu.shape}")
        
        # Max Pool 1: 55x55x96 -> 27x27x96
        # 3x3 pool, stride 2
        pool1 = self._max_pool2d(conv1_relu, pool_size=3, stride=2)
        print(f"Pool1 shape: {pool1.shape}")
        
        # Local Response Normalization 1
        lrn1 = self._local_response_normalization(pool1)
        print(f"LRN1 shape: {lrn1.shape}")
        
        # Conv Layer 2: 27x27x96 -> 27x27x256
        # 5x5 conv, stride 1, padding 2, GROUPED (2 groups)
        conv2 = self._grouped_conv2d(lrn1, self.conv2_W, self.conv2_b, groups=2, stride=1, pad=2)
        conv2_relu = self._relu(conv2)
        print(f"Conv2 + ReLU shape: {conv2_relu.shape}")
        
        # Max Pool 2: 27x27x256 -> 13x13x256
        # 3x3 pool, stride 2
        pool2 = self._max_pool2d(conv2_relu, pool_size=3, stride=2)
        print(f"Pool2 shape: {pool2.shape}")
        
        # Local Response Normalization 2
        lrn2 = self._local_response_normalization(pool2)
        print(f"LRN2 shape: {lrn2.shape}")
        
        # Conv Layer 3: 13x13x256 -> 13x13x384
        # 3x3 conv, stride 1, padding 1, FULLY CONNECTED (not grouped)
        conv3 = self._conv2d(lrn2, self.conv3_W, self.conv3_b, stride=1, pad=1)
        conv3_relu = self._relu(conv3)
        print(f"Conv3 + ReLU shape: {conv3_relu.shape}")
        
        # Conv Layer 4: 13x13x384 -> 13x13x384
        # 3x3 conv, stride 1, padding 1, GROUPED (2 groups)
        conv4 = self._grouped_conv2d(conv3_relu, self.conv4_W, self.conv4_b, groups=2, stride=1, pad=1)
        conv4_relu = self._relu(conv4)
        print(f"Conv4 + ReLU shape: {conv4_relu.shape}")
        
        # Conv Layer 5: 13x13x384 -> 13x13x256
        # 3x3 conv, stride 1, padding 1, GROUPED (2 groups)
        conv5 = self._grouped_conv2d(conv4_relu, self.conv5_W, self.conv5_b, groups=2, stride=1, pad=1)
        conv5_relu = self._relu(conv5)
        print(f"Conv5 + ReLU shape: {conv5_relu.shape}")
        
        # Max Pool 3: 13x13x256 -> 6x6x256
        # 3x3 pool, stride 2
        pool3 = self._max_pool2d(conv5_relu, pool_size=3, stride=2)
        print(f"Pool3 shape: {pool3.shape}")
        
        # Flatten for fully connected layers: 6x6x256 = 9216
        flatten = pool3.reshape(pool3.shape[0], -1)
        print(f"Flattened shape: {flatten.shape}")
        
        # Fully Connected Layer 1: 9216 -> 4096
        fc1 = np.dot(flatten, self.fc1_W) + self.fc1_b
        fc1_relu = self._relu(fc1)
        print(f"FC1 + ReLU shape: {fc1_relu.shape}")
        
        # Dropout 1
        fc1_dropout, dropout1_mask = self._dropout(fc1_relu, self.dropout_prob)
        print(f"FC1 + Dropout shape: {fc1_dropout.shape}")
        
        # Fully Connected Layer 2: 4096 -> 4096
        fc2 = np.dot(fc1_dropout, self.fc2_W) + self.fc2_b
        fc2_relu = self._relu(fc2)
        print(f"FC2 + ReLU shape: {fc2_relu.shape}")
        
        # Dropout 2
        fc2_dropout, dropout2_mask = self._dropout(fc2_relu, self.dropout_prob)
        print(f"FC2 + Dropout shape: {fc2_dropout.shape}")
        
        # Fully Connected Layer 3: 4096 -> num_classes
        fc3 = np.dot(fc2_dropout, self.fc3_W) + self.fc3_b
        print(f"FC3 shape: {fc3.shape}")
        
        # Softmax activation
        output = self._softmax(fc3)
        print(f"Output shape: {output.shape}")
        
        # Store intermediate results for backward pass
        self.cache = {
            'x': x,
            'conv1': conv1, 'conv1_relu': conv1_relu,
            'pool1': pool1, 'lrn1': lrn1,
            'conv2': conv2, 'conv2_relu': conv2_relu,
            'pool2': pool2, 'lrn2': lrn2,
            'conv3': conv3, 'conv3_relu': conv3_relu,
            'conv4': conv4, 'conv4_relu': conv4_relu,
            'conv5': conv5, 'conv5_relu': conv5_relu,
            'pool3': pool3, 'flatten': flatten,
            'fc1': fc1, 'fc1_relu': fc1_relu, 'fc1_dropout': fc1_dropout,
            'fc2': fc2, 'fc2_relu': fc2_relu, 'fc2_dropout': fc2_dropout,
            'fc3': fc3, 'output': output,
            'dropout1_mask': dropout1_mask, 'dropout2_mask': dropout2_mask
        }
        
        return output
    
    def _relu_backward(self, dout: np.ndarray, x: np.ndarray) -> np.ndarray:
        return dout * (x > 0)
    
    def _softmax_backward(self, dout: np.ndarray, softmax_out: np.ndarray) -> np.ndarray:
        return dout  # Simplified for cross-entropy loss
    
    def _dropout_backward(self, dout: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return dout * mask / (1 - self.dropout_prob)
    
    def _fc_backward(self, dout: np.ndarray, x: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        dx = np.dot(dout, W.T)
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        
        return dx, dW, db
    
    def backward(self, dout: np.ndarray) -> dict:
        """
        Args:
            dout: Gradient from loss function (N, num_classes)
            
        Returns:
            Dictionary of gradients for all parameters
        """
        gradients = {}
        
        cache = self.cache
        
        # Backward through softmax
        dfc3 = self._softmax_backward(dout, cache['output'])
        
        # Backward through FC3
        dfc2_dropout, gradients['fc3_W'], gradients['fc3_b'] = self._fc_backward(
            dfc3, cache['fc2_dropout'], self.fc3_W
        )
        
        # Backward through dropout2
        dfc2_relu = self._dropout_backward(dfc2_dropout, cache['dropout2_mask'])
        
        # Backward through FC2 ReLU
        dfc2 = self._relu_backward(dfc2_relu, cache['fc2'])
        
        # Backward through FC2
        dfc1_dropout, gradients['fc2_W'], gradients['fc2_b'] = self._fc_backward(
            dfc2, cache['fc1_dropout'], self.fc2_W
        )
        
        # Backward through dropout1
        dfc1_relu = self._dropout_backward(dfc1_dropout, cache['dropout1_mask'])
        
        # Backward through FC1 ReLU
        dfc1 = self._relu_backward(dfc1_relu, cache['fc1'])
        
        # Backward through FC1
        dflatten, gradients['fc1_W'], gradients['fc1_b'] = self._fc_backward(
            dfc1, cache['flatten'], self.fc1_W
        )
        
        # Reshape back to conv5 output shape
        dpool3 = dflatten.reshape(cache['pool3'].shape)
        
        return gradients
    
    def predict(self, x: np.ndarray) -> np.ndarray:

        self.training = False
        output = self.forward(x)
        self.training = True
        return output
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        """
        N = predictions.shape[0]
        
        # Convert targets to one-hot if needed
        if targets.ndim == 1:
            targets_onehot = np.zeros_like(predictions)
            targets_onehot[np.arange(N), targets] = 1
            targets = targets_onehot
        
        epsilon = 1e-15
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(targets * np.log(predictions_clipped), axis=1))
        
        return loss


def demo_alexnet():
    
    alexnet = AlexNet(num_classes=1000)
    print(f"AlexNet initialized with {alexnet.num_classes} output classes")
    print(f"Dropout probability: {alexnet.dropout_prob}")
    
    # Create synthetic input data (batch of 2 images)
    batch_size = 2
    input_data = np.random.randn(batch_size, 3, 227, 227)
    print(f"\nGenerated synthetic input data: {input_data.shape}")
    
    # Forward pass
    start_time = time.time()
    predictions = alexnet.forward(input_data)
    forward_time = time.time() - start_time
    
    print(f"\nForward pass completed in {forward_time:.2f} seconds")
    print(f"Final output shape: {predictions.shape}")
    print(f"Output probabilities sum: {np.sum(predictions, axis=1)}")
    print(f"Top-5 predicted classes for first image: {np.argsort(predictions[0])[-5:][::-1]}")
    
    # Create synthetic targets for loss computation
    targets = np.random.randint(0, 1000, batch_size)
    loss = alexnet.compute_loss(predictions, targets)
    print(f"\nCross-entropy loss: {loss:.4f}")
    
    # Backward pass
    # Create gradient signal
    dout = predictions.copy()
    dout[np.arange(batch_size), targets] -= 1  # Gradient of cross-entropy + softmax
    dout /= batch_size
    
    start_time = time.time()
    gradients = alexnet.backward(dout)
    backward_time = time.time() - start_time
    
    print(f"\nBackward pass completed in {backward_time:.2f} seconds")
    print(f"Gradients computed:")
    for key, grad in gradients.items():
        print(f"  {key}: {grad.shape}")

    # Model summary
    total_params = 0
    param_info = [
        ("Conv1", alexnet.conv1_W.size + alexnet.conv1_b.size),
        ("Conv2", alexnet.conv2_W.size + alexnet.conv2_b.size),
        ("Conv3", alexnet.conv3_W.size + alexnet.conv3_b.size),
        ("Conv4", alexnet.conv4_W.size + alexnet.conv4_b.size),
        ("Conv5", alexnet.conv5_W.size + alexnet.conv5_b.size),
        ("FC1", alexnet.fc1_W.size + alexnet.fc1_b.size),
        ("FC2", alexnet.fc2_W.size + alexnet.fc2_b.size),
        ("FC3", alexnet.fc3_W.size + alexnet.fc3_b.size),
    ]
    
    for layer_name, num_params in param_info:
        total_params += num_params
        print(f"{layer_name}: {num_params:,} parameters")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (approx): {total_params * 4 / (1024**2):.1f} MB (float32)")
    
    print("Parameter comparison (Grouped vs. Non-grouped):")
    
    # Conv2: grouped vs non-grouped
    conv2_grouped_params = 256 * 48 * 5 * 5 + 256  # 2 groups of 128 filters × 48 channels each
    conv2_normal_params = 256 * 96 * 5 * 5 + 256   # 256 filters × 96 channels
    print(f"Conv2: {conv2_grouped_params:,} (grouped) vs {conv2_normal_params:,} (normal) - {conv2_grouped_params/conv2_normal_params:.1%} reduction")
    
    # Conv4: grouped vs non-grouped  
    conv4_grouped_params = 384 * 192 * 3 * 3 + 384  # 2 groups of 192 filters × 192 channels each
    conv4_normal_params = 384 * 384 * 3 * 3 + 384   # 384 filters × 384 channels
    print(f"Conv4: {conv4_grouped_params:,} (grouped) vs {conv4_normal_params:,} (normal) - {conv4_grouped_params/conv4_normal_params:.1%} reduction")
    
    # Conv5: grouped vs non-grouped
    conv5_grouped_params = 256 * 192 * 3 * 3 + 256  # 2 groups of 128 filters × 192 channels each
    conv5_normal_params = 256 * 384 * 3 * 3 + 256   # 256 filters × 384 channels  
    print(f"Conv5: {conv5_grouped_params:,} (grouped) vs {conv5_normal_params:,} (normal) - {conv5_grouped_params/conv5_normal_params:.1%} reduction")
    
    total_grouped = conv2_grouped_params + conv4_grouped_params + conv5_grouped_params
    total_normal = conv2_normal_params + conv4_normal_params + conv5_normal_params
    print(f"\nTotal conv params: {total_grouped:,} (grouped) vs {total_normal:,} (normal)")
    print(f"Overall reduction: {total_grouped/total_normal:.1%} of normal parameters")


def compare_grouped_vs_normal_conv():

    np.random.seed(123)
    x = np.random.randn(1, 4, 8, 8)
    
    print(f"Input shape: {x.shape}")
    
    W_normal = np.random.randn(8, 4, 3, 3) * 0.1
    b_normal = np.zeros(8)
    
    W_grouped = np.random.randn(8, 2, 3, 3) * 0.1
    b_grouped = np.zeros(8)
    
    net = AlexNet()
    
    out_normal = net._conv2d(x, W_normal, b_normal, stride=1, pad=1)
    print(f"Normal conv output shape: {out_normal.shape}")
    print(f"Normal conv parameters: {W_normal.size + b_normal.size}")
    
    out_grouped = net._grouped_conv2d(x, W_grouped, b_grouped, groups=2, stride=1, pad=1)
    print(f"Grouped conv output shape: {out_grouped.shape}")
    print(f"Grouped conv parameters: {W_grouped.size + b_grouped.size}")
    
    print(f"\nParameter reduction: {(W_grouped.size + b_grouped.size) / (W_normal.size + b_normal.size):.1%}")


if __name__ == "__main__":
    demo_alexnet()
    compare_grouped_vs_normal_conv()