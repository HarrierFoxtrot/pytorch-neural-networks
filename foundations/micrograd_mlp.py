import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
  
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
    
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward 
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
    
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
  
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
    
        def _backward():
            self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = _backward
        return out
  
    def backward(self):
    
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
    
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

# requires_grad=True to track operations on this tensor so gradients can be computed for it during backward()
x1 = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.double, requires_grad=True)
w1 = torch.tensor([-3.0], dtype=torch.double, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
b = torch.tensor([6.8813735870195432], dtype=torch.double, requires_grad=True)
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print('---')
print('x2', x2.grad.item())
print('w2', w2.grad.item())
print('x1', x1.grad.item())
print('w1', w1.grad.item())

class Neuron:
    # Neuron(2) means 2 inputs --> 2 weights
    def __init__(self, nin):
        # creates nin Value objects, each with a random starting weight
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # creates a single Value for bias 
        self.b = Value(random.uniform(-1,1))
    def __call__(self, x):
        # w * x + b 
        # raw act
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b] # concatenate list of weights and biases tgt
    def weights(self):
        return self.w 
    
class Layer: 
    # hidden layer
    def __init__(self, nin, nout):
        # Layer(nin, nout) --> nout Neurons, each with nin weights
        # range(nouts) controls how many times the list comprehnesion runs
        # so nounts = 3 means run Neuron(2) 3 times
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        # same as class Neuron, but accordinly to how many Neuron objects in the self.neurons list 
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        # for each neuron in self.neuron list:
        # for each Value object (weighht & bias), 
        # loop over the list and get the values of individual parameters each
        return [p for neuron in self.neurons for p in neuron.parameters()]
    def weights(self):
        return [wts for neuron in self.neurons for wts in neuron.weights()]
    
class MLP:
  
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
  
    def parameters(self):
        # same thing as above
        return [p for layer in self.layers for p in layer.parameters()]
    def weights(self):
        return [wts for layer in self.layers for wts in layer.weights()]


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
o = n(x)
print(o)

print(n.parameters())

print("No. of parameters across all layers:",len(n.parameters()))
print("No. of weights across all layers:",len(n.weights()))
print("weights across all layers:",n.weights())

grad_first_layer = n.layers[0].neurons[0].w[0]
print("First weight of the first neuron:", grad_first_layer)


draw_dot(o).render('graph', format='png', view=True)

# 4 separate inputs, each with 3 features
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

# run MLP on each one and get 4 predictions
# then compare all 4 to ys to compute the loss

ys = [1.0, -1.0, -1.0, 1.0] # desired target
# call n(x) 4 times, one per input

min_loss = float("inf")
min_step = 0
for i in range (80):
    print(f"Step{i}")

    #forward pass
    ypred = [n(x) for x in xs]
    print("Y hat:", ypred)
    print("MSE: ")
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    print("loss", loss.data)

    # zeroed gradients to prevent .grad from accumulating
    for p in n.parameters():
        p.grad = 0.0
    # backward pass
    loss.backward()

    # paramaters update
    for p in n.parameters():
        p.data -= 0.05 * p.grad

    if loss.data < min_loss:
        min_loss = loss.data
        min_step = i


    print("=======\n")

print(f"min loss: {min_loss} at step {min_step}")



