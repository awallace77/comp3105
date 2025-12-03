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

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------
# Logging
# ----------------------
def log(level, msg):
    print(f"{datetime.now().replace(microsecond=0)} [{level}]: {msg}")


# ----------------------
# Small CNN from scratch
# ----------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(x))
        logits = self.fc2(features)
        return logits, features


# ----------------------
# Unlabelled dataset for OOD
# ----------------------
class UnlabelledDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(f"{self.img_dir}/{self.img_files[idx]}").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


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
# Feature-level OOD loss
# ----------------------
def compute_centroids(features, labels, num_classes=10):
    centroids = []
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            centroids.append(features[mask].mean(0))
        else:
            centroids.append(torch.zeros(features.size(1)).to(device))
    return torch.stack(centroids)


# ----------------------
# Training function with OOD regularization
# ----------------------
def train_model(model, in_loader, out_loader, num_epochs=15, lr=1e-3, lambda_ood=0.5, margin=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        out_iter = iter(out_loader)

        for images, labels in in_loader:
            images, labels = images.to(device), labels.to(device)

            try:
                x_out = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                x_out = next(out_iter)
            x_out = x_out.to(device)

            optimizer.zero_grad()

            logits_in, feat_in = model(images)
            _, feat_out = model(x_out)

            # In-domain classification
            class_loss = criterion(logits_in, labels)

            # Compute centroids for this batch
            with torch.no_grad():
                centroids = compute_centroids(feat_in, labels)

            # Feature OOD loss: push out-domain features away from centroids
            dists = torch.cdist(feat_out, centroids)  # [batch_out, num_classes]
            min_dists, _ = dists.min(dim=1)
            ood_loss = F.relu(margin - min_dists).mean()

            # Total loss
            loss = class_loss + lambda_ood * ood_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        log("INFO", f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(in_loader):.4f}")

    return model


# ----------------------
# Evaluation function
# ----------------------
def evaluate(model, loader, centroids=None, threshold=0.5):
    model.eval()
    total, correct = 0, 0
    ood_total, ood_correct = 0, 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, list) or len(batch) == 2:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits, features = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            else:
                images = batch.to(device)
                logits, features = model(images)
                probs = F.softmax(logits, dim=1)
                max_conf = probs.max(dim=1).values
                ood_correct += (max_conf < threshold).sum().item()
                ood_total += images.size(0)

    in_acc = 100 * correct / total if total > 0 else 0
    ood_acc = 100 * ood_correct / ood_total if ood_total > 0 else 0
    return in_acc, ood_acc


# ----------------------
# Run full experiment
# ----------------------
def run_experiment():
    model = SmallCNN().to(device)
    model = train_model(model, in_train_loader, out_train_loader, num_epochs=15, lr=0.001,
                        lambda_ood=0.5, margin=1.0)

    in_acc, _ = evaluate(model, in_eval_loader)
    log("INFO", f"In-domain Accuracy: {in_acc:.2f}%")

    _, ood_acc = evaluate(model, out_eval_loader)
    log("INFO", f"OOD Detection Accuracy (confidence < 0.5): {ood_acc:.2f}%")


if __name__ == "__main__":
    run_experiment()
