import torch
import torch.nn as nn 
import torch.optim as optim

# define the model 
model = nn.Linear(1,1) # 1 input, 1 output 

# data
x = torch.tensor([[2.0]]) # shape (1, 1) --> nn.Linear expects 2D output
target = torch.tensor([[10.0]])

#loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training loop
for i in range (100):
    optimizer.zero_grad() #reset gradients
    output = model(x) # forward prop
    loss = criterion(output, target) #compute loss
    loss.backward() # backprop
    print(model.weight.grad)
    print(model.bias.grad)
    optimizer.step() # update parameters

    if i % 10 == 0:
        print(f"Step {i}: loss={loss.item():.4f}")
    
print(f"\nLearned weight: {model.weight.item():.4f}")
print(f"Learned bias: {model.bias.item():.4f}")

