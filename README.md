# From Autograd Engine to CNN (99.20% Test Accuracy on MNIST)

This repo builds neural networks from first principles in PyTorch. The finale is a convolutional neural network at 99.19% test accuracy on MNIST. [`foundations/`](foundations/) holds the experimentation and implementation of the fundamental concepts neccessary for the progression of this repo: reverse-mode autodiff engine, activations and manual gradient descent - before reaching for the library version.

## Seeded Results Table

| Model | Architecture | Test Accuracy |
| :---- | :----------- | :-----------  |
| MLP   |
| CNN   |

## Building a Shallow Neural Network

- Before using Pytorch's API to build a MLP baseline at 97.66%, it was key to practice the mathematics and pipelines underlying Pytorch's framework:
- *class Value* ([`foundations/micrograd_mlp.py`](foundations//micrograd_mlp.py)) tracked all its operations and children, then traversed backwards from the output by applying local derivatives rules pertaining to each operation.
- *backward()* ([`foundations/micrograd_mlp.py`](foundations//micrograd_mlp.py)) builds a DFS ordering of the network's computation graphs so that every parent node is "visited" or evaluated before its children, then move in exact reverse after the output gradient seeded at 1.0.
- *class Neuron* ([`foundations/micrograd_mlp.py`](foundations//micrograd_mlp.py)) calculates from-scratch the raw pre-activation of weighted sum and bias (forward pass), before feeding the equation through tanh()
- In training: 
    - *MSE Loss* was written as a comprehension over predictions and targets
    - Zeroed gradients without using Pytorch by looping over every parameter and setting its gradient to zero before *loss.backward()*
    - Manual gradient descent by stepping/ updating each parameter by hand

- [`foundations/mnist_shallow_net.py`](foundations//mnist_shallow_net.py) built a feedforward **MLP** using Pytorch's API. 
- It run on CPU, was trained and tested on MNIST using the dataset's own fixed test/train split (60,000 images/ 10,000 MNIST images).
- **The MLP Loss Curve:**
| Epoch 0 | 1.2169
| Epoch 9 | 0.2625
| Epoch 29| 0.1477
| Epoch 69| 0.0754
| Epoch 99| 0.0537
- Upon reaching Epoch 99 the training loss had not plateaued, indicating uncertainty of whether increasing the number of epochs would improve test performance or simply overfit...
- This shallow FNN script reflected low accuracy over predicting the image targets since it lacked a validation set, which the CNN script fixed

## Building a Convolutional Neural Network
- [`mnist_cnn.py`](mnist_cnn.py) was a network of two convolutional blocks (the backbone) and two linear layers (the classifier head)
- Each convolutional block applies a 3x3 kernel size with padding1, batch normalisation, ReLU and 2d pooling - so at each block image halved whilst feature depth doubled: 28x28x1 --> 14x14x16 --> 7x7x32.
 to prevent spatial dimensions of the feature map, so 2d pooling was the only building block that downsampled.
- The result is flattened into a 1D vector of 1568 features, and passed to the first (hidden) linear layer with 128 neurons with 0.4 dropout before the final 10-class output.
- `padding=1` with a 3x3 kernel preserved spatial dimensions, leaving 2-D pooling the only building block the downsampled
- Run on MPS. Trained on MNIST's 60,000-image training set and 10,000-image test split 50/50 into validation and held-out test.



