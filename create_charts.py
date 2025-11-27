import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

# ==========================================
# 1. MODEL ARCHITECTURE (Matches main.py)
# ==========================================
class ASLRobot(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLRobot, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
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
        return self.classifier(x)


# ==========================================
# 2. REPORT GENERATION
# ==========================================
def create_report():
    print("\n📊 Generating Advanced Report Charts...\n")

    # Load Dataset
    data_dir = './asl_alphabet_train/asl_alphabet_train'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = full_dataset.classes

    # Same split as training
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    _, _, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Load model
    print("🧠 Loading robot brain...")
    device = torch.device("cpu")
    model = ASLRobot(num_classes=len(classes)).to(device)

    try:
        model.load_state_dict(torch.load("asl_robot_brain.pth", map_location=device))
        print("✅ Brain loaded correctly!\n")
    except:
        print("❌ ERROR: 'asl_robot_brain.pth' not found. Train the model first!")
        return

    model.eval()

    all_preds = []
    all_labels = []
    mistakes = []

    print("📝 Running performance evaluation...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Save 20 real mistakes
            if len(mistakes) < 20:
                wrong_idx = (preds != labels).nonzero()
                for idx in wrong_idx:
                    if len(mistakes) >= 20:
                        break
                    idx = idx.item()
                    img = images[idx].cpu()
                    t_lbl = classes[labels[idx].item()]
                    p_lbl = classes[preds[idx].item()]
                    mistakes.append((img, t_lbl, p_lbl))

    # ==========================================
    # 3. CONFUSION MATRIX
    # ==========================================
    print("🎨 Creating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(20, 15))
    sns.heatmap(cm, cmap='viridis',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix", fontsize=20)
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ==========================================
    # 4. PER CLASS ACCURACY
    # ==========================================
    print("📊 Creating per-class accuracy chart...")

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    accuracies = cm_norm.diagonal()

    plt.figure(figsize=(15, 8))
    sns.barplot(x=classes, y=accuracies, palette="magma")
    plt.title("Accuracy Per Class", fontsize=16)
    plt.ylim(0, 1.0)
    plt.savefig("class_accuracy_chart.png")
    plt.close()

    # ==========================================
    # 5. TEXT REPORT
    # ==========================================
    print("📄 Writing metrics_report.txt...")
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open("metrics_report.txt", "w") as f:
        f.write(report)

    # ==========================================
    # 6. ERROR GALLERY
    # ==========================================
    print("📸 Saving example misclassified images...")

    gallery_dir = "error_gallery"
    if os.path.exists(gallery_dir):
        shutil.rmtree(gallery_dir)
    os.makedirs(gallery_dir)

    for i, (img_tensor, t_lbl, p_lbl) in enumerate(mistakes):
        img_tensor = img_tensor * 0.5 + 0.5
        img_pil = transforms.ToPILImage()(img_tensor)
        img_pil.save(f"{gallery_dir}/Mistake_{i}_True_{t_lbl}_Pred_{p_lbl}.jpg")

    print("\n==========================================")
    print("🎉 REPORT COMPLETE!")
    print("Generated files:")
    print(" - confusion_matrix.png")
    print(" - class_accuracy_chart.png")
    print(" - metrics_report.txt")
    print(" - /error_gallery (20 misclassified images)")
    print("==========================================\n")


if __name__ == "__main__":
    create_report()