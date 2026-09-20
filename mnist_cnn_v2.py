import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import torch
from torch import rand
from torchvision.datasets import MNIST
from torch import nn
from torchvision.transforms import v2 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchsummary import summary
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from torchmetrics import MetricCollection
raw_train_data = MNIST(root="./data",
                     train=True, 
                     download=True
                     )
raw_test_data = MNIST(root="./data",
                     train=True, 
                     download=True
                     )
print("========Set-up Prep=======")
print(type(raw_train_data[0]))
print(len(raw_train_data))
print(raw_train_data[0][0], raw_train_data[0][1])
print(raw_train_data[1][0], raw_train_data[1][1])

##
## every data sample is a tuple: (image, label)
## fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
## for ax, value in zip (axes.flatten(), np.random.randint(0, 60000, size=(2,3)).flatten()):
##    image, label = raw_train_data[value]
##   ax.imshow(image, cmap="gray")
##    ax.set_title(f"Training sample #{value}\n Label {label}")
##   ax.axis(("off"))

## plt.tight_layout()
##


training_data = MNIST(
    root="./data",
    train=True,
    download=False,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
    )
test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
            lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
        ),
)

##
## visualising data sample after transformation (should be the same)
## print(training_data[0][1])
## img, lab = training_data[0]
## plt.imshow(img.squeeze(), cmap="gray")
## plt.title(f"Label: {lab.argmax()}")
## plt.show()
##
print(training_data[0][0].shape)
print(len(test_data))
cv_size = int(0.5 * len(test_data))
print(cv_size)
test_size = len(test_data) - cv_size
print(test_size)

test_dataset, cv_dataset = random_split(test_data, [test_size, cv_size])

print(len(test_dataset))
print(len(cv_dataset))
print(type(training_data))
print(type(cv_dataset)) # subset 
print(type(test_dataset)) # subset 

feature, label = training_data[0]
print(feature.shape)
print(label)

print(training_data[0][0])

class CNN_v2(nn.Module):
    def __init__(self):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2) # if only kernel size is specify, kernel size = stride = 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # hidden layer with 128 units / neurons 
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        # Layer 1: Conv --> Batch Norm --> Relu --> MaxPool
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        # Layer 2: conv --> Batch Norm --> Relu --> MaxPool
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        # Flatten the feature maps into a 1D vector
        x = self.flatten(x)
        # Fully Connected Layer 1 with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Final Output Layer
        x = self.fc2(x)
        return x

model = CNN_v2()

summary(model, input_size=(1, 28, 28))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CNN_v2().to(device)
print(type(cv_dataset))

# smoke test on 5 cv image samples into our untrained network
cv_x = torch.stack([cv_dataset[i][0] for i in range(5)])
cv_y = torch.stack([cv_dataset[i][1] for i in range(5)])
                                      
model.eval()
with torch.no_grad():
    # passing in only 5 images
    untrained_preds = model(cv_x.to(device))

print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Lenght of test samples;: {len(cv_y)}, Shape: {cv_y.shape}")
print(f"\nFirst 5 predictions:\n{torch.round(untrained_preds[:5])}")
print(f"\nFirst 5 test labels:\n{cv_y}")
print(device)

print("\n\n========End of Set-up Prep=======\n\n")

print(f"Model structure: {model}\n\n")
for name, p in model.named_parameters():
    print(name, p.shape)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.LinearLR(optimizer)

print("\n\n")


training_metrics = MetricCollection({
    "accuracy": MulticlassAccuracy(num_classes=10),
    "precision": MulticlassPrecision(num_classes=10),
    "recall": MulticlassRecall(num_classes=10)
}).to(device)

cv_metrics = MetricCollection({
    "accuracy": MulticlassAccuracy(num_classes=10),
    "precision": MulticlassPrecision(num_classes=10),
    "recall": MulticlassRecall(num_classes=10)
}).to(device)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
cv_dataloader = DataLoader(cv_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

def train_loop(model, epoch_nums, loss_fn, optimizer, train_dataloader, cv_dataloader, training_metrics, cv_metrics):
    epochs = epoch_nums
    for epoch in range(epochs):
        model.train()  

        epoch_train_loss = 0
        epoch_cv_loss = 0

        for batch, (input, target) in enumerate(train_dataloader):
            input, target = input.to(device), target.to(device)
            train_pred = model(input)

            train_loss = loss_fn(train_pred, target)
            epoch_train_loss = train_loss

            training_metrics.update(train_pred, target.argmax(dim=1))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_results = training_metrics.compute()
        training_metrics.reset()

        model.eval()
        with torch.inference_mode():
            for batch, (input, target) in enumerate(cv_dataloader):
                input, target = input.to(device), target.to(device)
                cv_pred = model(input)

                cv_loss = loss_fn(cv_pred, target)
                epoch_cv_loss = cv_loss

                cv_metrics.update(cv_pred, target.argmax(dim=1))
        cv_results = cv_metrics.compute()
        cv_metrics.reset()

        print(f"EPOCH {epoch+1}\n------------------------------------------")
        print(f"Training Loss: {epoch_train_loss:.4f} | CV Loss: {epoch_cv_loss:.4f}\n")
        print(f"Train Metrics: {train_results}\n")
        print(f"CV Metrics: {cv_results}")

train_loop(model, 10, loss_fn, optimizer, train_dataloader, cv_dataloader, training_metrics, cv_metrics)

# Final evaluation on the held-out test set (touched here for the first time, using
# the same weights train_loop just produced, no retraining/reloading needed).
test_metrics = MetricCollection({
    "accuracy": MulticlassAccuracy(num_classes=10),
    "precision": MulticlassPrecision(num_classes=10),
    "recall": MulticlassRecall(num_classes=10)
}).to(device)

model.eval()
with torch.inference_mode():
    for batch, (input, target) in enumerate(test_dataloader):
        input, target = input.to(device), target.to(device)
        test_pred = model(input)
        test_loss = loss_fn(test_pred, target)
        test_metrics.update(test_pred, target.argmax(dim=1))

test_results = test_metrics.compute()
test_metrics.reset()

print("\n\n========Final Test Evaluation (unseen data)=======\n")
print(f"Test Loss (last batch): {test_loss:.4f}\n")
print(f"Test Metrics: {test_results}")

# Add a checkpoint save after training: bundle model_state_dict, optimizer_state_dict, epoch and the final test? cv metrics into a dict
# save checkpoint with torch.save() to a uniquely named .pt file
# save the checkpoint where the script lives (same directory)
# Bundle everything needed to reload this exact trained model later, or resume training.
checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_v2_checkpoint.pt")
torch.save({
    "epoch": 10,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "test_metrics": test_results,
}, checkpoint_path)

print(f"\nSaved checkpoint to {checkpoint_path}")


