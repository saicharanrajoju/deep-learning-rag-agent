# Artificial Neural Networks (ANN)

## Forward Propagation

An Artificial Neural Network (ANN) processes input data by passing it forward through a series of layers. Each layer consists of neurons, where every neuron receives weighted inputs from the previous layer, sums them, adds a bias term, and applies a non-linear activation function to produce an output. This sequence — weighted sum followed by activation — is called forward propagation.

For a single neuron, the computation is: z = Wx + b, then a = f(z), where W is the weight matrix, x is the input, b is the bias, and f is the activation function. The output of one layer becomes the input to the next. The final layer produces the network's prediction — a class probability for classification, or a real-valued output for regression.

Forward propagation is purely feedforward — information moves only from input to output with no cycles. The architecture (number of layers, neurons per layer, activation functions) is fixed before training and determines the hypothesis space the network can represent. A network with no hidden layers is a linear model; hidden layers allow the network to learn non-linear feature representations.

Interview note: Be prepared to walk through the math for a two-layer network by hand: given input x, compute z¹ = W¹x + b¹, a¹ = f(z¹), z² = W²a¹ + b², ŷ = softmax(z²).

## Backpropagation

Backpropagation is the algorithm that computes the gradient of the loss function with respect to every weight in the network. It applies the chain rule of calculus, starting from the output layer and working backwards through each layer to the input.

After forward propagation produces a prediction ŷ, a loss function L (such as cross-entropy) measures how wrong the prediction was. Backpropagation first computes dL/dŷ, then propagates this error signal backwards: for each layer, it computes how much each weight contributed to the error by multiplying the incoming gradient by the local derivative of the layer's operation.

For a fully connected layer z = Wx + b with activation a = f(z), the gradients are: dL/dW = (dL/da)(da/dz)·xᵀ and dL/dx = Wᵀ·(dL/da)(da/dz). These gradients are then used by an optimizer (SGD, Adam) to update weights: W ← W − η·(dL/dW), where η is the learning rate.

Modern frameworks (PyTorch, JAX) build a computational graph during the forward pass and traverse it in reverse during backpropagation — called automatic differentiation. This means backprop is O(n) in the number of parameters, not O(n²).

Interview note: If asked why backprop requires storing intermediate activations, explain that computing dL/dW for each layer requires the activations from the forward pass — this is the memory cost of training versus inference.

## Activation Functions

Activation functions introduce non-linearity into the network. Without them, stacking multiple linear layers collapses to a single linear transformation, regardless of depth — the network loses all representational power beyond a linear model.

The most common activation functions are:

**ReLU (Rectified Linear Unit):** f(x) = max(0, x). Fast to compute, sparse activations, does not saturate for positive values. The default choice for hidden layers in most modern networks. Suffers from the dying ReLU problem: neurons whose input is always negative produce zero gradients and stop learning.

**Sigmoid:** f(x) = 1/(1+e⁻ˣ). Outputs between 0 and 1, historically used for binary classification output layers. Saturates at both ends, causing vanishing gradients in deep networks — rarely used in hidden layers today.

**Tanh:** f(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ). Outputs between −1 and 1, zero-centered (unlike sigmoid). Still saturates, but better than sigmoid for hidden layers in shallow networks.

**Softmax:** Applied at the final layer for multi-class classification. Converts raw logits into a probability distribution that sums to 1. Not used in hidden layers — it creates competition between neurons, which is not desirable mid-network.

Interview note: Be ready to explain why ReLU's non-differentiability at x=0 is not a practical problem — subgradients (typically set to 0 at x=0) work fine in practice, and the probability of a weight landing exactly at 0 is negligible.

## Vanishing Gradients in ANNs

The vanishing gradient problem occurs when gradients become exponentially small as they are propagated backwards through many layers during backpropagation. When gradients approach zero, the weights in early layers receive near-zero updates and effectively stop learning — the network trains only its later layers.

The root cause is repeated multiplication. During backpropagation, gradients are multiplied by the derivative of each layer's activation function at every step. Sigmoid and tanh derivatives are bounded between 0 and 0.25 and 0 and 1 respectively. When these small values are multiplied together across dozens of layers, the product shrinks exponentially — a gradient that starts at 1.0 can become 10⁻²⁰ by the time it reaches the first layer.

ReLU mitigates this because its derivative is either 0 or 1 for positive inputs — multiplying by 1 does not shrink the gradient. However, ReLU introduces the dying neuron problem (gradient = 0 for negative inputs). Modern solutions include: careful weight initialization (He/Xavier), batch normalization, residual connections (ResNets), and gradient clipping.

Interview note: Vanishing gradients are distinct from exploding gradients (where products grow unboundedly). Exploding gradients are typically addressed with gradient clipping (cap gradients at a maximum norm) while vanishing gradients require architectural solutions.
