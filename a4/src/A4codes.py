# Dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

'''
    COMP3105 Intro to Machine Learning
    Assignment 4
    Group 51:
        Andrew Wallace - 101210291
        Christer Henrysson - 101260693
    Due: November 30th, 2025
'''

# Preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Verify data load
def imshow(img):
    img = img / 2 + 0.5               # unnormalize if necessary
    npimg = img.permute(1, 2, 0).numpy()  # C,H,W → H,W,C
    plt.imshow(npimg)
    plt.axis('off')
    # plt.show() 
    plt.savefig("img_load.png")
    plt.clf()

dataiter = iter(trainloader)
images, labels = next(dataiter)
labels
imshow(torchvision.utils.make_grid(images))


# Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        """
            Building blocks of convolutional neural network.
            Parameters:
                * num_classes: Number of classes to predict
                * in_channels: Number of channels in the input image (for grayscale images, 1, for rgb, 3)
       """
       
        super(CNN, self).__init__()
        # 2 convolutional layers with 1 max pooling layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=1)
        # Fully connected layers
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        """
            Define the forward pass of the neural network.
            Parameters:
                x: Input tensor.
            Returns:
                torch.Tensor
                The output tensor after passing through the network.
        """
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.view(-1, 16 * 6 * 6) # Flatten
        # x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Move into GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=3, num_classes=10).to(device)
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs=10
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # Move data into device
        inputs, labels = data
        inputs = inputs.to(device) 
        labels = labels.to(device) 

        # Outputs & loss from model
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Optimize 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate loss
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Test model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

# Save the model
torch.save(model.state_dict(), 'MulticlassCNN.pth')

# Create a new model
loaded_model = CNN(in_channels=1, num_classes=10)

# Load the saved model
loaded_model.load_state_dict(torch.load('MulticlassCNN.pth'))
print(loaded_model)
