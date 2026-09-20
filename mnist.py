import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Ff 


# covert raw pixel values from MNIST to normalised Pytorch tensor
# nn trains on tensors (smaller pixel values 0 -1)

# conversion object
transform = transforms.ToTensor()

# MNIST comes pre-split
# download the images if not alr on disk
# 60,000 training images
train_data = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# 10,0000 testing images
test_data = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 28 x 28 pixels
# would the input be 28 then 28 and output 1
# multi class classification but then there would be 10 ouput neurons
# s can we use softmax at the end output layer, 
# cross entropy loss function
# what kind of labels would it be? true labels ? 

class ShallowNetwork(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(ShallowNetwork, self).__init__()
        # map input layer to hidden layers
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass input(s) through hidden layers
        x = self.hidden(x)
        # pass through activation layer
        x = self.relu(x)
        # Pass through to the output layer
        x = self.output(x)
        return x 

model = ShallowNetwork(input_size=784, hidden_size=128, output_size=10)
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.01)

# loop over train_loader
# each batch consists of (images, labels)
for epoch in range(100):
    total_loss = 0 
    for images, labels in train_loader:
        images = images.view(images.size(0), -1)
        out = model(images)
        # y hat here is raw logits
        # y (ground truth) here is integers 0-9
        loss = loss_fn(out, labels)

        optimiser.zero_grad() 
        loss.backward() 
        optimiser.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1)
        out = model(images)
        # out.data is a 2d tensor shape [64, 10]
        # 64 rows so 64 images, each row is one image
        # walk through every row along the column to find the max value of the row (look across)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (labels==predicted).sum().item()
accuracy = (correct / total ) * 100 
print(f"Accuracy of this model on the 10000 test images: {accuracy}")

''''''''''
    if epoch % 10 == 0:

        for p in model.parameters():
            print(p.grad)
        print(f"Epoch {epoch}: loss={loss.item():.4f}")
'''''''''''

