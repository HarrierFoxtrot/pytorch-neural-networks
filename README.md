# From Autograd Engine to CNN (99.20% Test Accuracy on MNIST)

This repo builds neural networks from first principles in PyTorch. The finale is a convolutional neural network at 99.20% test accuracy on MNIST. [`foundations/`](foundations/) holds the experimentation and implementation of the fundamental concepts necessary for the progression of this repo: reverse-mode autodiff engine, activations and manual gradient descent - before reaching for the library version.

## Seeded Results Table:
[`foundations/mnist_shallow_net.py`](foundations/mnist_shallow_net.py)
[`mnist_cnn.py`](mnist_cnn.py)

| Model | Test Accuracy | Precision | Recall |
|:------|--------------:|----------:|-------:|
| MLP   |    97.66%     |  97.65%   | 97.64% |
| CNN   |    99.20%     |  99.20%   | 99.19% |

*Accuracy micro-averaged, precision and recall macro-averaged; `SEED = 42`*

- Gap between MLP and CNN: 1.54pp, so CNN cut the error rate by ~66% (2.34% --> 0.80%).


## Building a Shallow Neural Network

- Before using Pytorch's API to build a MLP baseline at 97.66%, it was key to practice the mathematics and pipelines underlying Pytorch's framework:
- *class Value* ([`foundations/micrograd_mlp.py`](foundations/micrograd_mlp.py)) tracked all its operations and children, then traversed backwards from the output by applying local derivatives rules pertaining to each operation.
- *backward()* ([`foundations/micrograd_mlp.py`](foundations/micrograd_mlp.py)) builds a topological ordering by depth first search, appending each node only after visiting its children, then `reversed(topo)` walks that list in exact reverse after seeding the output gradient at 1.0.
- *class Neuron* ([`foundations/micrograd_mlp.py`](foundations/micrograd_mlp.py)) calculates from-scratch the raw pre-activation of weighted sum and bias (forward pass), before feeding the equation through tanh().
- In training:
    - *MSE Loss* was written as a comprehension over predictions and targets.
    - Zeroed gradients without using Pytorch by looping over every parameter and setting its gradient to zero before *loss.backward()*.
    - Manual gradient descent by stepping/ updating each parameter by hand.

- [`foundations/mnist_shallow_net.py`](foundations/mnist_shallow_net.py) built a feedforward **MLP** using Pytorch's API.
- It ran on CPU, was trained and tested on MNIST using the dataset's own fixed test/train split (60,000 images/ 10,000 MNIST images).

### The MLP Loss Curve:

| Epoch | Avg training loss |
|------:|------------------:|
|     0 | 1.2169 |
|     9 | 0.2625 |
|    29 | 0.1477 |
|    69 | 0.0754 |
|    99 | 0.0537 |

- Upon reaching Epoch 99 the training loss was still falling, so it was unclear whether increasing the number of epochs would improve test performance or overfit.
- This script has no validation set, so again: no signal to detect overfitting during training or to inform stopping.
- The MLP treated 784 pixels as unrelated, discarding spatial relationships within the image -- which was what the CNN script fixed: detecting edges and corners in small patches during early filters, seeing wider regions in later layers.

## Building a Convolutional Neural Network

- [`mnist_cnn.py`](mnist_cnn.py) was a network of two convolutional blocks (the backbone) and two linear layers (the classifier head).
- Each convolutional block applies a 3x3 kernel size with `padding=1`, batch normalisation, ReLU and 2d pooling - so at each block image halved whilst feature depth grew: 28x28x1 --> 14x14x16 --> 7x7x32.
- The result is flattened into a 1D vector of 1568 features, and passed to the first (hidden) linear layer with 128 neurons with 0.4 dropout before the final 10-class output.
- `padding=1` with a 3x3 kernel preserved spatial dimensions, so 2-D pooling was the only building block that downsampled.
- It ran on MPS. Trained on MNIST's 60,000-image training set and 10,000-image test split 50/50 into validation and held-out test.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python mnist_cnn.py                        # CNN, ~3 min on MPS
python foundations/mnist_shallow_net.py    # MLP, ~8 min on CPU
```
