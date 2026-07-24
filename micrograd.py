import math
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph

# building a tiny autograd engine
# manual back prop

a = 2.0
b = -3.0
c = 10.0
d = a*b + c

print(d)

h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b +c
c += h # moving vertically up in y-axis
d2 = a*b + c 
print("d1", d1)
print("d2", d2)
print("slope", (d2-d1) / h) 

class Value: 
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        # out is output
        out = Value(self.data + other.data, (self, other), '+')
    
        def _backward():
            # backward prop (chain rule backwards) for addition is just the output gradient x 1
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
    
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward   
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')
    
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward 
        return out
  
    def backward(self): 
    # builds back prop for the whole expression graph 
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


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
print(L)

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

# back prop is recurive application of the chain rule backwards through the computational graph
L.grad = 1.0
a.grad = 6.0
b.grad = -4.0
c.grad = -2.0
d.grad = -2.0
e.grad = -2.0
f.grad = 4.0

draw_dot(L).render('graph', format='png', view=True)

def lol():
    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    b.data += h
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data 

    print((L2 - L1)/h)

a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
L = d * f

print(L.data)

plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid();
plt.show()

# maunual back prop of a simple 2D neuron
#inputs
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights w1,w2
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
# bias of the neuron
b = Value(6.8813735870195432, label="b")
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = "x1w1"
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# n on line 191 is still the raw cell body without the activation function
n = x1w1x2w2 + b; n.label = 'n'
draw_dot(n).render('graph', format='png', view=True)
# take n through the activation function, we use tanh() which requires exponential function & division. not just addition & multiplication
# o is output
o = n.tanh(); o.label = "o"

# now compute the back prop starting from o

o.backward()
draw_dot(o).render('graph', format='png', view=True)


'''''''''

# back prop through tanh 
# o = tanh(n)
# do/dn = 1 - tanh(n)**2
# do/dn = 1 - o**2 
print( 1 - o.data**2) 
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()

draw_dot(o).render('graph', format='png', view=True)


n.grad = 0.5
x1w1x2w2.grad = 0.5 
# and by symmetry, b will be the same
b.grad = 0.5 
# and again by symmetry, derivative will flow to both nodes x1w1 and x2w2(the same)
x1w1.grad = 0.5
x2w2.grad = 0.5
x2.grad = 0.5
w2.grad = 0
x1.grad = -1.5
w1.grad = 1.0
'''''''''''''''


