import torch

checkpoint = torch.load("cnn_v2_checkpoint.pt")
print(checkpoint["test_metrics"])