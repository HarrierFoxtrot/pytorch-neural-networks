import torch

# a simple computation: z = (x * y) + y
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

z = (x * y) + y
print("z:", z)

x.grad = None
y.grad = None

z.backward()

print("dz/dx:", x.grad)
print("dz/dy:", y.grad)

# A slightly more realistic omputation: loss = (prediction - target) ** 2
prediction = torch.tensor(2.5, requires_grad=True)
target = torch.tensor(4.0)

loss = (prediction - target) ** 2
print("\nloss is \n", loss)
print(loss.backward())

print("dloss/dprediction: ", prediction.grad)