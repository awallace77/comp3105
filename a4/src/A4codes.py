
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from datetime import datetime
import time
import torch.nn.functional as F
import random

'''
    COMP3105 Intro to Machine Learning
    Assignment 4
    Group 51:
        Andrew Wallace - 101210291
        Christer Henrysson - 10126069
    Due: November 30th, 2025
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    """ Convolution Neural Network w ResNet18 """

    def __init__(self, num_classes=10, fine_tune=True):
        super().__init__()

        # Load ResNet model as our CNN (removing final fully connected layer)
        # resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet = models.resnet18()
        resnet_feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity() # output feature vector instead of class logits

        # Freeze the resnet params (so general features are not collapsed)
        # for param in resnet.parameters():
        #     param.requires_grad = False

        # # Unfreeze last resnet layer params for domain tuning
        # if fine_tune:
        #     for name, param in resnet.named_parameters():
        #         if "layer2" in name or "layer3" in name or "layer4" in name:  # last three blocks
        #             param.requires_grad = True

        self.resnet = resnet # Convolution layer
       
        self.class_classifier = nn.Sequential( # Classifier (fully connected)
            nn.Linear(resnet_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        self.domain_classifier = nn.Sequential( # Domain classifier (fully connected)
            nn.Linear(resnet_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
            The forward pass of the nn
            Args:
                x: input tensor
            Returns:
                The output tensor after passing through the network
        """
        # 32: Batch size, 64: Number of channels, 56: Height, 56: Width
        features = self.resnet(x)
        class_logits = self.class_classifier(features)
        domain_logits = self.domain_classifier(features)

        return class_logits, domain_logits, features  

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        class_logits = self.classifier(feat)
        domain_logits = self.domain_classifier(feat)
        return class_logits, domain_logits, feat  

class MediumCNN(nn.Module):
    """Improved MediumCNN with GAP, more channels, and regularization."""

    def __init__(self, num_classes=10):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        # Feature extractor
        self.features = nn.Sequential(
            block(3, 32),    # 224 → 112
            block(32, 64),   # 112 → 56
            block(64, 128),  # 56 → 28
            block(128, 256), # 28 → 14
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # [B, 256, 1, 1]
        self.flatten_dim = 256

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # Domain classification head
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.gap(feat)           # [B, 256, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, 256]
        class_logits = self.classifier(feat)
        domain_logits = self.domain_classifier(feat)
        return class_logits, domain_logits, feat


class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        # 224 → 112 → 56 → 28 → 14 → 7
        self.features = nn.Sequential(
            block(3, 32),     # 112x112
            block(32, 64),    # 56x56
            block(64, 128),   # 28x28
            block(128, 256),  # 14x14
            block(256, 512),  # 7x7
        )

        self.flatten_dim = 512 * 7 * 7

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        class_logits = self.classifier(feat)
        domain_logits = self.domain_classifier(feat)
        return class_logits, domain_logits, feat   

class UnlabelledDataset(Dataset):
    """ Custom dataset for the unlabelled images """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img = Image.open(f"{self.img_dir}/{file_name}").convert("RGB")  # Convert to PIL image 
        return self.transform(img)


def log(level, msg):
    """ For logging messages """
    print(f"{datetime.now().replace(microsecond=0)} [{level}]: {msg}")


# Preprocess data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Data directories
in_domain_train_dir = "/home/andrew/cu/comp3105/a4/src/A4data/in-domain-train"
in_domain_eval_dir = "/home/andrew/cu/comp3105/a4/src/A4data/in-domain-eval"
out_domain_train_dir = "/home/andrew/cu/comp3105/a4/src/A4data/out-domain-train"
out_domain_eval_dir = "/home/andrew/cu/comp3105/a4/src/A4data/out-domain-eval"

# Create datasets (ImageFolder automatically assigns class indices from subdirectory names [such cool wow])
in_domain_train = ImageFolder(root=in_domain_train_dir, transform=transform)
in_domain_eval  = ImageFolder(root=in_domain_eval_dir, transform=transform)
out_domain_train = UnlabelledDataset(f"{out_domain_train_dir}/unlabelled", transform=transform)
out_domain_eval= ImageFolder(root=out_domain_eval_dir, transform=transform)

batch_size = 32
# Create data loaders
in_train_loader = DataLoader(in_domain_train, batch_size=batch_size, shuffle=True, drop_last=True)
in_eval_loader = DataLoader(in_domain_eval, batch_size=batch_size, shuffle=True)
out_train_loader = DataLoader(out_domain_train, batch_size=batch_size, shuffle=True, drop_last=True)
out_eval_loader = DataLoader(out_domain_eval , batch_size=batch_size, shuffle=True)


# Verify data load
dest_dir = "/home/andrew/cu/comp3105/a4/src/A4verify"
def imshow(img, dest_dir="", filename="img"):
    img = img / 2 + 0.5 # unormalize
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"{dest_dir}/{filename}.jpg")
    plt.clf()

# # Verify in-domain train image
# print(f"IN_TRAIN")
# dataiter = iter(in_train_loader)
# images, labels = next(dataiter)
# print(f"Labels: {labels}")
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="in_train_image")

# # Verify in-domain test image
# print(f"IN_TEST")
# dataiter = iter(in_eval_loader)
# images, labels = next(dataiter)
# print(f"Labels: {labels}")
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="in_test_image")

# # Verify out-domain train
# print(f"OUT_TRAIN")
# dataiter = iter(out_train_loader)
# images = next(dataiter)
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="out_train_image")

# # Verify out-domain test image
# print(f"OUT_TEST")
# dataiter = iter(out_eval_loader)
# images, labels = next(dataiter)
# print(f"Labels: {labels}")
# imshow(torchvision.utils.make_grid(images), dest_dir=dest_dir, filename="out_eval_image")
# def learn(path_to_in_domain, path_to_out_domain):

def train_model(
        learning_rate=0.0001, 
        num_epochs=10,
        lambda_ood=2.0):
    """ Trains the CNN model """

    # Define CNN and move into GPU (if available) or CPU
    # model = MediumCNN().to(device)
    model = MediumCNN().to(device)

    # Define optimizer (only optimize parameters that require grad)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define losses
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    # Trainin model
    in_iter = iter(in_train_loader)
    out_iter = iter(out_train_loader)
    for epoch in range(num_epochs):
        
        model.train() # Model in training mode
        running_loss = 0.0

        # Equalize in/out iterations
        for step in range(min(len(in_train_loader), len(out_train_loader))):

            # Load in domain batch 
            try:
                x_in, y_in = next(in_iter)
            except StopIteration:
                in_iter = iter(in_train_loader)
                x_in, y_in = next(in_iter)
            
            x_in, y_in = x_in.to(device), y_in.to(device)

            # Load out domain batch
            try:
                x_out = next(out_iter)
            except StopIteration:
                out_iter = iter(out_train_loader)
                x_out = next(out_iter)

            # Move data to device
            x_out = x_out.to(device)

            # Forward pass
            class_in_logits, domain_in_logits, feat_in = model(x_in)
            class_out_logits, domain_out_logits, feat_out = model(x_out)

            # Classification loss
            class_in_loss = criterion_class(class_in_logits, y_in)

            # Domain classification 
            domain_labels_in = torch.zeros(len(x_in), dtype=torch.long).to(device)
            domain_labels_out = torch.ones(len(x_out), dtype=torch.long).to(device)
            domain_loss = (
                criterion_domain(domain_in_logits, domain_labels_in) +
                criterion_domain(domain_out_logits, domain_labels_out)
            )

            # Out-domain loss (force out of domain to uniform classification)
            p_out = F.log_softmax(class_out_logits, dim=1)
            uniform = torch.ones_like(class_out_logits) / class_out_logits.size(1)
            ood_loss = F.kl_div(p_out, uniform, reduction="batchmean")
            # print("Max OOD softmax prob:", class_out_logits.softmax(1).max(dim=1)[0].mean().item())

            # Total loss 
            loss = class_in_loss + 0.1 * domain_loss + lambda_ood * ood_loss
            # loss = class_in_loss + 0.5 * domain_loss
            # loss = class_in_loss + lambda_ood * ood_loss

            # Backprop 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        log("INFO", f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(in_train_loader):.4f}")

    return model

def save_model(model, dest_dir="", name="model"):
    path = f"{dest_dir}/{name}.pth"
    torch.save(model.state_dict(), path)
    log("INFO", f"Saved model to {path}")

def load_model(model_full_path="model.pth"):
    model = CNN()
    model.load_state_dict(torch.load(model_full_path))
    return model

def evaluate(loader, model, device):
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            class_logits, domain_logits, _ = model(images)

            # Domain prediction - 0 = in, 1 = out
            domain_pred = domain_logits.argmax(dim=1)

            # For in-domain: use classification prediction
            class_pred = class_logits.argmax(dim=1)

            # For out-domain choose random label
            for i in range(len(domain_pred)):
                if domain_pred[i].item() == 1:  # OUT
                    class_pred[i] = random.randint(0, 10 - 1)

            # Compute accuracy
            correct += (class_pred == labels).sum().item()
            total += images.size(0)

    return correct / total

def evaluate_reg(loader, model, device):
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            class_logits, domain_logits, _ = model(images)

            # Domain prediction - 0 = in, 1 = out
            # domain_pred = domain_logits.argmax(dim=1)

            # For in-domain: use classification prediction
            class_pred = class_logits.argmax(dim=1)

            # For out-domain choose random label
            # for i in range(len(domain_pred)):
                # if domain_pred[i].item() == 1:  # OUT
                    # class_pred[i] = random.randint(0, 10 - 1)

            # Compute accuracy
            correct += (class_pred == labels).sum().item()
            total += images.size(0)

    return correct / total


def run_experiments():
    """ Runs experiments for model training """
    max_accuracy = -1
    max_model = None
    final_lr = None
    num_epochs = 10
    # learning_rates = [0.01, 0.001, 0.0005, 0.0001, 0.00005]
    learning_rates = [0.001, 0.0005, 0.0001, 0.00005]
    lambda_ood = [1.0, 0.5, 0.3, 0.1]
    
    for lr in learning_rates:

        for lo in lambda_ood:

            log("INFO", f"Training model\n\
                learning_rate = {lr}\n\
                lambda_ood = {lo}\n")

            start_time = time.time()
            # model = train_model(learning_rate=lr, 
                                # lambda_domain=0, 
                                # lambda_triplet=lt,
                                # margin=margin,
                                # num_epochs=num_epochs)
            model = train_model(learning_rate=lr, num_epochs=num_epochs, lambda_ood=lo)
            
            log("INFO", f"Training finished in {time.time() - start_time:.3f} seconds")

            # Evaluate 
            in_acc = evaluate(in_eval_loader, model, device) * 100
            out_acc = evaluate(out_eval_loader, model, device) * 100
            in_acc_reg = evaluate_reg(in_eval_loader, model, device) * 100
            out_acc_reg = evaluate_reg(out_eval_loader, model, device) * 100
            log("INFO", f"In-domain Accuracy: {in_acc:.2f}%")
            log("INFO", f"Out-domain Accuracy: {out_acc:.2f}%\n")
            log("INFO", f"In-domain Accuracy [REG]: {in_acc_reg:.2f}%")
            log("INFO", f"Out-domain Accuracy [REG]: {out_acc_reg:.2f}%\n")

            # Find best model
            if in_acc > max_accuracy:
                max_model = model
                final_lr = lr
                max_accuracy = in_acc
        
    # Save model w best accuracy
    model_path = "/home/andrew/cu/comp3105/a4"
    model_name = f"resnet_cnn_lr{final_lr}"
    save_model(max_model, model_path, model_name)

if __name__ == "__main__":
    # model_path = "/home/andrew/cu/comp3105/a4"
    # model_name = "resnet_cnn"
    # model_full_path = f"{model_path}/{model_name}.pth"
    run_experiments()