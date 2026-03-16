# Recurrent Neural Networks (RNN)

## Hidden State and Sequence Processing

A Recurrent Neural Network (RNN) is designed for sequential data — text, time series, audio, or any input where order matters. Unlike a feedforward network that processes each input independently, an RNN maintains a hidden state that carries information from previous time steps. This hidden state acts as the network's memory of what it has seen so far in the sequence.

At each time step t, the RNN receives the current input xₜ and the previous hidden state hₜ₋₁, and produces a new hidden state hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁ + b). The same weight matrices Wₓ and Wₕ are used at every time step — this is parameter sharing across time, analogous to how CNNs share weights across space. An optional output yₜ = Wᵧ·hₜ can be produced at each step for sequence-to-sequence tasks, or only at the final step for sequence classification.

This recurrent connection creates a feedback loop: the network's output influences its own future processing. In theory, this allows an RNN to model dependencies of arbitrary length in a sequence — the hidden state at time T can in principle encode information from time step 1. In practice, learning long-range dependencies is very difficult due to the vanishing gradient problem.

Interview note: The hidden state hₜ is a fixed-size vector regardless of how long the sequence is. This information bottleneck is why LSTMs were developed — they add a separate cell state and gating mechanisms to selectively preserve long-range information.

## Backpropagation Through Time (BPTT)

Training an RNN requires computing gradients with respect to all weight matrices (Wₓ, Wₕ, Wᵧ) and biases. Because the same weights are used at every time step, computing the gradient requires unrolling the network across all T time steps and applying the chain rule — this is called Backpropagation Through Time (BPTT).

The gradient of the loss with respect to Wₕ sums contributions from all time steps: dL/dWₕ = Σₜ dLₜ/dWₕ. Computing dLₜ/dWₕ for time step t requires multiplying Jacobians all the way back from step T to step t: ∏ₖ₌ₜᵀ (∂hₖ/∂hₖ₋₁). This product of Jacobian matrices is the source of both vanishing and exploding gradients.

Full BPTT is memory-intensive and computationally expensive for long sequences — storing all intermediate hidden states for a 1000-step sequence requires 1000× more memory than a single forward pass. Truncated BPTT addresses this by backpropagating only k steps back at a time, treating earlier hidden states as constants. This trades gradient accuracy for feasibility on long sequences.

Interview note: Be ready to explain the tradeoff in truncated BPTT. Shorter truncation windows mean faster training and lower memory use, but the network cannot learn dependencies longer than k steps. Choosing k is a hyperparameter that depends on the expected dependency length in the data.

## Vanishing Gradients in RNNs: Why They Are Worse Than in ANNs

RNNs suffer from vanishing gradients more severely than ANNs because the same weight matrix Wₕ is multiplied at every time step during BPTT. The gradient flowing back k steps involves Wₕᵏ — the k-th power of the hidden-to-hidden weight matrix. If the largest eigenvalue of Wₕ is less than 1, this product shrinks exponentially with k. If it exceeds 1, gradients explode.

In a feedforward ANN, different weight matrices are used at each layer, so the product of Jacobians involves different matrices — while vanishing gradients are still possible, they are not guaranteed. In an RNN, the repeated application of the same Wₕ makes exponential decay or growth almost inevitable for long sequences.

The practical consequence is that vanilla RNNs typically fail to learn dependencies longer than 10-20 time steps. A network trying to learn that "The cat, which sat on the mat every day for years, was hungry" requires connecting "cat" and "was hungry" across roughly 15 tokens — often beyond the effective memory of a vanilla RNN.

Long Short-Term Memory (LSTM) networks, introduced by Hochreiter and Schmidhuber (1997), solve this with a separate cell state and multiplicative gates that allow gradients to flow through time unchanged when needed — the gradient highway that vanilla RNNs lack.

Interview note: A key interview question: "Why does an LSTM solve vanishing gradients?" The answer is the cell state's additive update rule: cₜ = fₜ⊙cₜ₋₁ + iₜ⊙g̃ₜ. The forget gate fₜ can be set close to 1, making the cell state update additive rather than multiplicative across time — preserving gradients for many steps.
