import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import os
import random
from datetime import datetime
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------
# Custom small CNN
# ----------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 3 convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # input size depends on 224x224 input
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # flatten
        features = F.relu(self.fc1(x))
        logits = self.fc2(features)
        return logits, features


# ----------------------
# Custom unlabelled dataset for OOD
# ----------------------
class UnlabelledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        img = Image.open(f"{self.img_dir}/{file_name}").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ----------------------
# Logging helper
# ----------------------
def log(level, msg):
    print(f"{datetime.now().replace(microsecond=0)} [{level}]: {msg}")


# ----------------------
# Data preprocessing
# ----------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# ----------------------
# Load datasets
# ----------------------
in_domain_train_dir = "/home/andrew/cu/comp3105/a4/src/A4data/in-domain-train"
in_domain_eval_dir = "/home/andrew/cu/comp3105/a4/src/A4data/in-domain-eval"
out_domain_train_dir = "/home/andrew/cu/comp3105/a4/src/A4data/out-domain-train"
out_domain_eval_dir = "/home/andrew/cu/comp3105/a4/src/A4data/out-domain-eval"

in_train_ds = ImageFolder(root=in_domain_train_dir, transform=transform_train)
in_eval_ds = ImageFolder(root=in_domain_eval_dir, transform=transform_eval)
out_train_ds = UnlabelledDataset(f"{out_domain_train_dir}/unlabelled", transform=transform_train)
out_eval_ds = ImageFolder(root=out_domain_eval_dir, transform=transform_eval)

batch_size = 32
in_train_loader = DataLoader(in_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
in_eval_loader = DataLoader(in_eval_ds, batch_size=batch_size, shuffle=False)
out_train_loader = DataLoader(out_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
out_eval_loader = DataLoader(out_eval_ds, batch_size=batch_size, shuffle=False)


# ----------------------
# Training function
# ----------------------
def train_model(num_epochs=10, learning_rate=0.001):
    model = SmallCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in in_train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        log("INFO", f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(in_train_loader):.4f}")

    return model


# ----------------------
# Evaluation function
# ----------------------
def evaluate(loader, model, threshold=0.5):
    """Evaluate in-domain classification and OOD detection via confidence"""
    model.eval()
    total = 0
    correct = 0
    ood_total = 0
    ood_correct = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, list) or len(batch) == 2:
                # In-domain: images + labels
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
            else:
                # Out-domain: images only
                images = batch.to(device)
                labels = None

            logits, _ = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            max_conf = probs.max(dim=1).values

            if labels is not None:
                # Standard in-domain accuracy
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                # OOD detection: low confidence = correct
                ood_correct += (max_conf < threshold).sum().item()
                ood_total += images.size(0)

    in_acc = 100 * correct / total if total > 0 else 0
    ood_acc = 100 * ood_correct / ood_total if ood_total > 0 else 0
    return in_acc, ood_acc


# ----------------------
# Run experiments
# ----------------------
def run():
    num_epochs = 15
    learning_rate = 0.001
    model = train_model(num_epochs=num_epochs, learning_rate=learning_rate)

    in_acc, ood_acc = evaluate(in_eval_loader, model)
    log("INFO", f"In-domain Accuracy: {in_acc:.2f}%")
    log("INFO", f"OOD Detection Accuracy (confidence < 0.5): {ood_acc:.2f}%")

    # Evaluate on out-domain using confidence
    _, ood_acc = evaluate(out_eval_loader, model)
    log("INFO", f"OOD Detection Accuracy on out-domain: {ood_acc:.2f}%")


if __name__ == "__main__":
    run()
