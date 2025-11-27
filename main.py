import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import matplotlib.pyplot as plt
import random
import numpy as np

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"[SEED] Fixed to {seed}")

set_seed(42)

# ==========================================
# 2. DATASET & LOADERS
# ==========================================
print("\n🔹 Loading ASL Alphabet Dataset...")

data_dir = './asl_alphabet_train/asl_alphabet_train'
if not os.path.exists(data_dir):
    print("❌ ERROR: Dataset folder not found!")
    exit()

# Proper RGB normalization
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Augmentation for training only
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

# Load dataset with different transforms
train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
test_dataset_full = datasets.ImageFolder(root=data_dir, transform=test_transform)

# Train/Val/Test split (80/10/10)
total_count = len(train_dataset_full)
train_count = int(0.8 * total_count)
val_count = int(0.1 * total_count)
test_count = total_count - train_count - val_count

indices = torch.randperm(total_count).tolist()
train_idx = indices[:train_count]
val_idx   = indices[train_count:train_count+val_count]
test_idx  = indices[train_count+val_count:]

train_data = Subset(train_dataset_full, train_idx)
val_data   = Subset(test_dataset_full, val_idx)
test_data  = Subset(test_dataset_full, test_idx)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"✓ Dataset Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test\n")

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class ASLRobot(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLRobot, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# FORCE CPU (RTX 5060 requires it)
device = torch.device("cpu")
print(f"⚠️ Training on CPU only (GPU unsupported)\n")

model = ASLRobot().to(device)

# ==========================================
# 4. OPTIMIZER, LOSS, SCHEDULER
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ==========================================
# 5. EARLY STOPPING
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"   ⚠️ EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopper = EarlyStopping(patience=5)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
num_epochs = 25
train_losses, val_losses, val_accuracies = [], [], []

print("🏫 Training Started!\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    val_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.2f}%")

    scheduler.step()
    early_stopper(val_loss)
    if early_stopper.stop:
        print("🛑 Early stopping triggered.\n")
        break

# ==========================================
# 7. TEST EVALUATION
# ==========================================
print("\n🎓 Final Exam: Testing Model...")

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total
print(f"🎯 FINAL TEST ACCURACY: {final_acc:.2f}%")

# ==========================================
# 8. SAVE MODEL & PLOTS
# ==========================================
torch.save(model.state_dict(), "asl_robot_brain.pth")
print("💾 Model saved as asl_robot_brain.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")

plt.figure(figsize=(10, 5))
plt.plot(val_accuracies, color='green', label="Validation Accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("accuracy_curve.png")

print("📊 Saved loss_curve.png and accuracy_curve.png")
print("\n✅ Training Complete!")