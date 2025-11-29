# Dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.io import read_image
import os
from torch.utils.data import Dataset
from PIL import Image



'''
    COMP3105 Intro to Machine Learning
    Assignment 4
    Group 51:
        Andrew Wallace - 101210291
        Christer Henrysson - 101260693
    Due: November 30th, 2025
'''


class UnlabelledDataset(Dataset):
    """
        Custom dataset for unlabelled images
    """
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg'))]

    def __len__(self):
        return len(self.img_files)

    # def __getitem__(self, idx):
    #     img_path = os.path.join(self.img_dir, self.img_files[idx])
    #     image = read_image(img_path).float() / 255.0  # convert to [0,1]
    #     if self.transform:
    #         image = self.transform(image)
    #     return image

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img = Image.open(f"{self.img_dir}/{file_name}").convert("RGB")  # RETURN PIL IMAGE
        return self.transform(img)

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

in_domain_train_dir = "/home/andrew/cu/comp3105/a4/src/A4data/in-domain-train"
in_domain_eval_dir = "/home/andrew/cu/comp3105/a4/src/A4data/in-domain-eval"
out_domain_train_dir = "/home/andrew/cu/comp3105/a4/src/A4data/out-domain-train"
out_domain_eval_dir = "/home/andrew/cu/comp3105/a4/src/A4data/out-domain-eval"

# ImageFolder automatically assigns class indices from subdirectory names
in_domain_train = ImageFolder(root=in_domain_train_dir, transform=transform)
in_domain_eval  = ImageFolder(root=in_domain_eval_dir, transform=transform)
out_domain_train = UnlabelledDataset(f"{out_domain_train_dir}/unlabelled", transform=transform)
out_domain_eval= ImageFolder(root=out_domain_eval_dir, transform=transform)

batch_size = 32

in_train_loader = DataLoader(in_domain_train, batch_size=batch_size, shuffle=True, drop_last=True)
in_eval_loader = DataLoader(in_domain_eval, batch_size=batch_size, shuffle=True)
out_train_loader = DataLoader(out_domain_train, batch_size=batch_size, shuffle=True, drop_last=True)
out_eval_loader = DataLoader(out_domain_eval , batch_size=batch_size, shuffle=True)

# Verify data load
def imshow(img, dest_dir="", filename="img"):
    img = img / 2 + 0.5 # unormalize
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"{dest_dir}/{filename}.jpg")
    plt.clf()

dest_dir = "/home/andrew/cu/comp3105/a4/src/A4verify"
# # Verify train image
# print(f"IN_TRAIN")
# dataiter = iter(in_train_loader)
# images, labels = next(dataiter)
# print(f"Labels: {labels}")
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="in_train_image")

# # Verify test image
# print(f"IN_TEST")
# dataiter = iter(in_eval_loader)
# images, labels = next(dataiter)
# print(f"Labels: {labels}")
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="in_test_image")

# # Verify out train
# print(f"OUT_TRAIN")
# dataiter = iter(out_train_loader)
# images = next(dataiter)
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="out_train_image")

# # Verify out test 
# print(f"OUT_TEST")
# dataiter = iter(out_eval_loader)
# images, labels = next(dataiter)
# print(f"Labels: {labels}")
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="out_eval_image")


# Convolution Neural Network
class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        """
            Building blocks of convolutional neural network.
            Parameters:
                * num_classes: Number of classes to predict
                * in_channels: Number of channels in the input image (for grayscale images, 1, for rgb, 3)
       """
        
        super().__init__()

        # Define feature extractor 
        self.feature_extractor = nn.Sequential(

            # First convolution layer
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)), # Reduces spatial dimensions to 1x1
            nn.Flatten(),  # Flattens the [32, 64, 4, 4] tensor
        )

        # Flattened feature size
        self.feature_dim = 64 * 4 * 4 

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2) # 0 for in, 1 for out domaiin
        )

    def forward(self, x):
        """
            The forward pass of the nn
            Args:
                x: input tensor.
            Returns:
                The output tensor after passing through the network
        """
        # 32: Batch size, 64: Number of channels, 56: Height, 56: Width
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        domain_logits = self.domain_classifier(features)

        return class_logits, domain_logits, features


# Move into GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=3, num_classes=10).to(device)
print(model)

# Hyperparameters
# lambda_domain = 0.05
# beta_separation = 1.0

criterion_class = nn.CrossEntropyLoss()       # only in-domain
# criterion_domain = nn.CrossEntropyLoss()      # binary domain classifier
# margin_loss = nn.TripletMarginLoss(margin=1.0)

optimizer = optim.Adam(model.parameters(), lr=0.001)

in_iter = iter(in_train_loader)
out_iter = iter(out_train_loader)

num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    running_loss = 0.0

    # Make both loaders iterate equally
    # for step in range(min(len(in_train_loader), len(out_train_loader))):
    for step in range(len(in_train_loader)):

        try:
            x_in, y_in = next(in_iter)
        except StopIteration:
            in_iter = iter(in_train_loader)
            x_in, y_in = next(in_iter)

        # try:
        #     x_out = next(out_iter)
        # except StopIteration:
        #     out_iter = iter(out_train_loader)
        #     x_out = next(out_iter)

        x_in, y_in = x_in.to(device), y_in.to(device)
        # x_out = x_out.to(device)

        # Domain labels
        domain_in = torch.zeros(len(x_in), dtype=torch.long).to(device)
        # domain_out = torch.ones(len(x_out), dtype=torch.long).to(device)

        # ---- FORWARD PASS ----
        class_logits_in, domain_logits_in, feat_in = model(x_in)
        # _, domain_logits_out, feat_out = model(x_out)

        # ---- LOSSES ----
        # 1. Classification loss (in-domain only)
        class_loss = criterion_class(class_logits_in, y_in)

        # 2. Domain loss
        # domain_logits_all = torch.cat([domain_logits_in, domain_logits_out], dim=0)
        # domain_labels_all = torch.cat([domain_in, domain_out], dim=0)
        # domain_loss = criterion_domain(domain_logits_all, domain_labels_all)

        # 3. Triplet loss (push OUT far from IN)
        # Anchor: in-domain features
        # Positive: different in-domain features
        # Negative: out-domain features
        # anchors = feat_in
        # positives = torch.zeros_like(feat_in)
        # for i in range(len(feat_in)):
            # same_class = (y_in == y_in[i]).nonzero().squeeze()

            # remove self
            # same_class = same_class[same_class != i]

            # if len(same_class) == 0:
                # positives[i] = feat_in[i]
            # else:
                # j = same_class[torch.randint(len(same_class), (1,))]
                # positives[i] = feat_in[j]

        # negatives = feat_out[torch.randperm(len(feat_out))]

        # separation_loss = margin_loss(anchors, positives, negatives)

        # Combined objective
        # loss = class_loss + lambda_domain * domain_loss + beta_separation * separation_loss
        

        # ---- OPTIMIZE ----
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # running_loss += loss.item()

        class_loss = criterion_class(class_logits_in, y_in)
        optimizer.zero_grad()
        class_loss.backward()
        optimizer.step()
        running_loss += class_loss.item()

    if epoch % 2 == 0:
        print(f"[INFO]: [{epoch+1}, step {step+1}] loss: {running_loss:.3f}")
        running_loss = 0.0
print("[INFO]: Finished Training")

# Evalutation - in domain
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in in_eval_loader:
        images, labels = images.to(device), labels.to(device)

        class_logits, _, _ = model(images)

        _, predicted = torch.max(class_logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"[INFO]: Accuracy on IN DOMAIN test images: {100 * correct / total:.2f}%")

# Evalutation - out domain
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in out_eval_loader:
        images, labels = images.to(device), labels.to(device)

        class_logits, _, _ = model(images)

        _, predicted = torch.max(class_logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"[INFO]: Accuracy on OUT DOMAIN test images: {100 * correct / total:.2f}%")


# Save the model
torch.save(model.state_dict(), 'MulticlassCNN.pth')

# Create a new model
loaded_model = CNN(in_channels=3, num_classes=10)

# Load the saved model
loaded_model.load_state_dict(torch.load('MulticlassCNN.pth'))
print(loaded_model)
