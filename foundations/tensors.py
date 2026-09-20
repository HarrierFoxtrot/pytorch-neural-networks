import torch 

# 1. Create a tensor 
x = torch.tensor([1.0, 2.0, 3.0])
print(f"1D tensor: {x}")

# 2. Create a 2D tensor (matrix)
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("2D tensor:\n", A)

# 3. Shape
print("Shape:", A.shape)

# 4. Arithmetic
print("x * 2:", x * 2)
print("A + A:\n", A + A)

# 5. Indexing
print(f"First element of x: {x[0]}")
print(f"First row of A: {A[0]}")
print(f"Element at row 1, col 0: {A[1, 0]}")

# 6. Reshape
B = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
print("OG shape: ", B.shape)
B_reshaped = B.view(2,3)
print("reshaped:\n", B_reshaped)

# 7. Move to GPU Apple M2 (MPS)
x_cpu = x.device
print(x_cpu)
device = torch.device("mps")
x_gpu = x.to(device)
print("x on GPU:", x_gpu)
print("device:", x_gpu.device)
