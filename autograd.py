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