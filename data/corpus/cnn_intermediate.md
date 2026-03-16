# Convolutional Neural Networks (CNN)

## The Convolution Operation

A Convolutional Neural Network (CNN) uses a mathematical operation called convolution to extract features from input data, most commonly images. Instead of connecting every neuron to every input (as in a fully connected layer), a CNN applies a small learnable filter — called a kernel — that slides across the input, computing a dot product at each position.

For a 2D image input I and a kernel K of size k×k, the output at position (i,j) is: (I * K)(i,j) = ΣΣ I(i+m, j+n) · K(m,n). The kernel moves with a configurable stride (how many pixels to skip between positions) and can be padded (zeros added around the border) to control output dimensions.

Each kernel detects one specific spatial pattern — edges, corners, textures — regardless of where that pattern appears in the image. This property is called translation equivariance: if the input pattern shifts, the activation map shifts by the same amount. Because the same kernel weights are reused at every position (weight sharing), a convolutional layer has far fewer parameters than a fully connected layer on the same input size.

Multiple kernels are applied in parallel, each producing a separate feature map. The depth of the output tensor equals the number of kernels, allowing the network to detect many patterns simultaneously.

Interview note: A 3×3 kernel applied to a 224×224 image with 64 filters produces a 222×222×64 output (without padding). Be able to compute output spatial dimensions: output = (input − kernel + 2·padding) / stride + 1.

## Pooling and Spatial Downsampling

Pooling layers reduce the spatial dimensions of feature maps after a convolutional layer. This achieves two goals: it reduces computational cost for deeper layers, and it provides a degree of translation invariance — small shifts in the input produce the same pooled output.

Max pooling, the most common variant, takes the maximum value within each non-overlapping window (typically 2×2 with stride 2). A 224×224 feature map becomes 112×112 after one 2×2 max pooling layer. Average pooling computes the mean instead of the maximum and is more common in the final layers of modern architectures.

Pooling has no learnable parameters — it is a fixed deterministic operation. The key insight is that the exact position of a feature matters less than whether the feature is present at all. A cat ear detected two pixels to the left or right should not change the classification — pooling discards this positional precision while retaining the feature's presence.

One tradeoff: pooling discards spatial information permanently. For tasks requiring precise localization (object detection, semantic segmentation), architectures like U-Net and Feature Pyramid Networks use upsampling or skip connections to recover spatial detail that pooling would otherwise destroy.

Interview note: Global average pooling (GAP), used in architectures like ResNet before the final classifier, averages each entire feature map to a single value — replacing large fully connected layers and significantly reducing parameter count.

## Feature Maps and Hierarchical Representations

CNNs learn hierarchical representations by stacking convolutional layers. Early layers learn low-level features (edges, color gradients, corners); middle layers combine these into textures and shapes; deep layers respond to high-level semantic concepts (faces, wheels, eyes).

This hierarchical structure mirrors how the visual cortex is believed to process images — simple cells respond to oriented edges, complex cells to combinations of edges, and higher areas to object parts. Each feature map in a CNN layer captures the spatial distribution of one detected pattern across the input.

As depth increases, each neuron's receptive field — the region of the original input it is influenced by — grows. A neuron in the third convolutional layer (using 3×3 kernels) has a 7×7 effective receptive field in the input, even though its direct inputs are from a 3×3 region of the previous layer. This allows deep layers to reason about large-scale spatial structure without ever using large kernels.

Transfer learning exploits this hierarchy: a CNN trained on ImageNet has learned general visual features in early and middle layers. Freezing these layers and fine-tuning only the final classifier on a new task often outperforms training from scratch, especially when labeled data is scarce.

Interview note: Visualizing what neurons in different layers respond to (using techniques like gradient ascent on the input) confirms this hierarchy empirically and is an important tool for CNN interpretability.

## LeNet and AlexNet Architectures

**LeNet-5 (LeCun et al., 1998)** was the first CNN to demonstrate practical success, applied to handwritten digit recognition (MNIST). Its architecture — two convolutional layers followed by average pooling, then three fully connected layers — established the pattern still followed today. LeNet had only ~60,000 parameters and ran on CPUs, but proved that learned convolutional filters could outperform handcrafted features.

**AlexNet (Krizhevsky et al., 2012)** won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a 10-percentage-point margin over the second-best entry, marking the beginning of the deep learning era in computer vision. Key innovations: five convolutional layers with 3×3 kernels, ReLU activations (replacing sigmoid/tanh, solving vanishing gradients), dropout regularization (0.5 in fully connected layers), data augmentation, and training on two GPUs in parallel. AlexNet had ~60 million parameters.

The leap from LeNet to AlexNet illustrates three enabling factors: more data (ImageNet: 1.2M images), more compute (GPUs), and deeper architectures. AlexNet demonstrated that depth — not feature engineering — was the key to visual recognition performance.

Interview note: AlexNet's use of ReLU was a critical practical contribution. The authors explicitly noted that ReLU trained 6× faster than tanh on CIFAR-10, directly enabling the deep network that won ImageNet.
