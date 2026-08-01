import torch 

# parameters : weight and bias (what neuron learns)

weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

# data

x = torch.tensor(2.0)
target = torch.tensor(10.0)

# training loop 
for i in range(100):
    # forward pass
    output = weight * x + bias  # records into the graph 
                                # can use nn.Linear (Pytorch built in linear layer)

    # forward pass: calculating MSE loss 
    loss = (output - target) ** 2 # records into the graph
                                # can use nn.MSELoss() - (Pytorch built in loss functino)

    # backward pass
    loss.backward() # compute gradients from that graph 
    print(weight.grad)
    print(bias.grad)

    # optimiser step (manual SGD)
    with torch.no_grad(): # just arithmetic update of parameters
        weight -= 0.01 * weight.grad # new weight
        bias -= 0.01 * bias.grad # new bias 

    # reset gradients
    weight.grad.zero_() # need a new gradient for each epoch 
    bias.grad.zero_()

    if i % 10 == 0:
        print(f"Step {i}: loss={loss.item():.4f}, weight={weight.item():.4f}, bias={bias.item():.4f}")