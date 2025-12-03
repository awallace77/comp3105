import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import random
from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ----------------------
# Logging
# ----------------------
def log(level, msg):
    print(f"{datetime.now().replace(microsecond=0)} [{level}]: {msg}")


# ----------------------
# Small CNN
# ----------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*28*28, 256)
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
# In-memory dataset
# ----------------------
class InMemoryDataset(Dataset):
    def __init__(self, img_dir, transform=None, labelled=True, device='cpu'):
        self.transform = transform
        self.device = device
        self.images = []
        self.labels = []

        if labelled:
            for label_idx, label_name in enumerate(sorted(os.listdir(img_dir))):
                class_dir = os.path.join(img_dir, label_name)
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(".jpg"):
                        img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
                        if transform:
                            img = transform(img)
                        self.images.append(img.to(device))  # <- GPU tensor
                        self.labels.append(label_idx)
        else:
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(".jpg"):
                    img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                    if transform:
                        img = transform(img)
                    self.images.append(img.to(device))


# ----------------------
# Data transforms
# ----------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

transform_eval = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(1.5,0.5,0.5))
])


# ----------------------
# Load datasets into RAM
# ----------------------
direc = "/content/drive/MyDrive/A4data"
in_train_ds = InMemoryDataset(f"{direc}/in-domain-train", transform=transform_train, labelled=True)
in_eval_ds = InMemoryDataset(f"{direc}/in-domain-eval", transform=transform_eval, labelled=True)
out_train_ds = InMemoryDataset(f"{direc}/out-domain-train/unlabelled", transform=transform_train, labelled=False)
out_eval_ds = InMemoryDataset(f"{direc}/out-domain-eval", transform=transform_eval, labelled=True)

batch_size = 32
in_train_loader = DataLoader(in_train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
in_eval_loader = DataLoader(in_eval_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
out_train_loader = DataLoader(out_train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
out_eval_loader = DataLoader(out_eval_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)


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
# Training function
# ----------------------
def train_model(model, in_loader, out_loader, num_epochs=15, lr=1e-3, lambda_ood=0.5, margin=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        out_iter = iter(out_loader)

        for images, labels in in_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            try:
                x_out = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                x_out = next(out_iter)
            x_out = x_out.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits_in, feat_in = model(images)
            _, feat_out = model(x_out)

            # In-domain classification loss
            class_loss = criterion(logits_in, labels)

            # Centroids for current batch
            with torch.no_grad():
                centroids = compute_centroids(feat_in, labels)

            # OOD loss: push out-domain features away from centroids
            dists = torch.cdist(feat_out, centroids)
            min_dists, _ = dists.min(dim=1)
            ood_loss = F.relu(margin - min_dists).mean()

            loss = class_loss + lambda_ood * ood_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        log("INFO", f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(in_loader):.4f}")

    return model


# ----------------------
# Evaluation
# ----------------------
def evaluate(model, loader, threshold=0.5):
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
# Run experiment
# ----------------------
def run_experiment():
    model = SmallCNN().to(device)
    model = train_model(model, in_train_loader, out_train_loader, num_epochs=15, lr=0.001, lambda_ood=0.5, margin=1.0)

    in_acc, _ = evaluate(model, in_eval_loader)
    log("INFO", f"In-domain Accuracy: {in_acc:.2f}%")

    _, ood_acc = evaluate(model, out_eval_loader)
    log("INFO", f"OOD Detection Accuracy (confidence < 0.5): {ood_acc:.2f}%")


if __name__ == "__main__":
    run_experiment()
