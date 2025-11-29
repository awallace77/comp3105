
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
'''
    COMP3105 Intro to Machine Learning
    Assignment 4
    Group 51:
        Andrew Wallace - 101210291
        Christer Henrysson - 101260693
    Due: November 30th, 2025
'''

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    """ Convolution Neural Network w ResNet18 """

    def __init__(self, num_classes=10, fine_tune=True):
        super().__init__()

        # Load ResNet model as our CNN (removing final fully connected layer)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet_feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity() # output feature vector instead of class logits

        # Freeze the resnet params (so general features are not collapsed)
        for param in resnet.parameters():
            param.requires_grad = False

        # Unfreeze last resnet layer params for domain tuning
        if fine_tune:
            for name, param in resnet.named_parameters():
                if "layer4" in name:  # last block
                    param.requires_grad = True

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
    transforms.Resize((64, 64)),
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


def train_model(learning_rate=0.001, num_epochs=10):
    """ Trains the CNN model """

    # Define CNN and move into GPU (if available) or CPU
    model = CNN().to(device)

    # Define losses and optimizer
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    margin_loss = nn.TripletMarginLoss(margin=1.0)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Only optimize parameters that require grad (classifier + layer4)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Trainin model
    in_iter = iter(in_train_loader)
    out_iter = iter(out_train_loader)
    for epoch in range(num_epochs):
        log("INFO", f"[Epoch {epoch + 1}/{num_epochs}]")

        # Freeze resnet 
        # model.resnet.eval()
        model.train()
        running_loss = 0.0

        # Equalize in/out iterations
        for step in range(min(len(in_train_loader), len(out_train_loader))):

            # Load in domain batch 
            try:
                x_in, y_in = next(in_iter)
            except StopIteration:
                in_iter = iter(in_train_loader)
                x_in, y_in = next(in_iter)

            # Load out domain batch
            try:
                x_out = next(out_iter)
            except StopIteration:
                out_iter = iter(out_train_loader)
                x_out = next(out_iter)

            # Move data to device
            x_in, y_in, x_out = x_in.to(device), y_in.to(device), x_out.to(device)

            # Forward pass
            class_logits, domain_logits, feat_in = model(x_in)
            with torch.no_grad(): # don't compute gradients for out domain
                feat_out = model.resnet(x_out) 

            # Triplet loss for domain separation
            anchors = [] # base data (in-domain features)
            positives = [] # other in-domain sample of the same class (to encourage in-domain classification)
            negatives = [] # out-domain sample

            # for i in range(len(feat_in)):
            #     anchors.append(feat_in[i])

            #     # Ensure same class
            #     same_class_idx = (y_in == y_in[i]).nonzero(as_tuple=True)[0]
            #     same_class_idx = same_class_idx[same_class_idx != i]

            #     if len(same_class_idx) > 0:
            #         pos_idx = same_class_idx[torch.randint(len(same_class_idx), (1,)).item()]
            #         positives.append(feat_in[pos_idx])
            #     else:
            #         positives.append(feat_in[i])  # fallback if no same-class in batch
            # Populate negative (random out domain)
            # neg_indices = torch.randperm(len(feat_out))
            # negatives = [feat_out[idx] for idx in neg_indices[:len(feat_in)]]

            # anchors = torch.stack(anchors)
            # positives = torch.stack(positives)
            # negatives = torch.stack(negatives)

            # Compute pairwise distances
            # feat_in: in-domain features
            # feat_out: out-domain features (detached)
            feat_out_detached = feat_out.detach()  # ensure no gradients

            for i in range(len(feat_in)):
                anchor = feat_in[i]
                anchors.append(anchor)

                # Positive: another in-domain feature of same class
                same_class_idx = (y_in == y_in[i]).nonzero(as_tuple=True)[0]
                same_class_idx = same_class_idx[same_class_idx != i]
                if len(same_class_idx) > 0:
                    pos_idx = same_class_idx[torch.randint(len(same_class_idx), (1,)).item()]
                    positive = feat_in[pos_idx]
                else:
                    positive = feat_in[i]  # fallback
                positives.append(positive)

                # Hard negative - closest out-domain feature
                distances = torch.norm(anchor - feat_out_detached, dim=1)  # L2 distance
                hard_neg_idx = torch.argmin(distances)
                negatives.append(feat_out_detached[hard_neg_idx])

            # Stack tensors
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives) 

            # Classification loss
            class_loss = criterion_class(class_logits, y_in)

            # Separation loss (encourage domain separation)
            separation_loss = margin_loss(anchors, positives, negatives)

            # Combine to get total loss
            lambda_triplet = 1.0 # For separation loss
            loss = class_loss + lambda_triplet * separation_loss

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

# Evaluate model
def evaluate(loader, model, device):
    model.resnet.eval()
    model.class_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_experiments():
    """ Runs experiments for model training """

    max_accuracy = -1
    max_model = None
    final_lr = None
    num_epochs = 20
    learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    for lr in learning_rates:

        log("INFO", f"Training model (lr = {lr})")
        model = train_model(learning_rate=lr, num_epochs=num_epochs)
        in_acc = evaluate(in_eval_loader, model, device)
        out_acc = evaluate(out_eval_loader, model, device)
        log("INFO", f"In-domain Accuracy: {in_acc:.2f}%")
        log("INFO", f"Out-domain Accuracy: {out_acc:.2f}%")

        if in_acc > max_accuracy:
            max_model = model
            final_lr = lr
    
    # Save model w best accuracy
    model_path = "/home/andrew/cu/comp3105/a4"
    model_name = f"resnet_cnn_lr{final_lr}"
    save_model(max_model, model_path, model_name)

if __name__ == "__main__":
    model_path = "/home/andrew/cu/comp3105/a4"
    model_name = "resnet_cnn"
    model_full_path = f"{model_path}/{model_name}.pth"

    # log("INFO", f"Training model")
    # model = train_model()
    # save_model(model, model_path, model_name)
    # log("INFO", f"Done training model")
    # ## Or load the model
    # # log("INFO", f"Loading model")
    # # model = load_model(model_full_path)

    # log("INFO", f"Evaluating model")
    # in_acc = evaluate(in_eval_loader, model, device)
    # out_acc = evaluate(out_eval_loader, model, device)
    # log("INFO", f"Done evaluating model")

    # log("INFO", f"In-domain Accuracy: {in_acc:.2f}%")
    # log("INFO", f"Out-domain Accuracy: {out_acc:.2f}%")
    run_experiments()
