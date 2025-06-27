
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
import numpy as np
import pandas as pd
import random
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

# 固定随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_pth = './data'
data_dir = '/code/image'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = sorted(os.listdir(data_dir))
        self.file_list = []
        self.label_list = []
        for label in self.labels:
            label_path = os.path.join(data_dir, label)
            for file in os.listdir(label_path):
                self.file_list.append(os.path.join(label_path, file))
                self.label_list.append(self.labels.index(label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.label_list[index]

class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=8, num_features=2048):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = F.relu(self.resnet(x))
        logits = self.classifier(features)
        return features, logits

def calculate_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    sensitivities, specificities = [], []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        sens = tp / (tp + fn) if (tp + fn) else 0
        spec = tn / (tn + fp) if (tn + fp) else 0
        sensitivities.append(sens)
        specificities.append(spec)
    acc = np.mean(y_true == y_pred)
    return auc, acc, np.mean(sensitivities), np.mean(specificities)

dataset = PatchDataset(data_dir, transform=transform)
indices = list(range(len(dataset)))
labels = [dataset.label_list[i] for i in indices]

# 划分训练集与测试集
train_idx, test_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
train_labels = [labels[i] for i in train_idx]
test_dataset = Subset(dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# 十折交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
best_val_acc = 0.0

for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(train_idx, train_labels)):
    print(f"Fold {fold+1}")
    train_subset = Subset(dataset, [train_idx[i] for i in train_fold_idx])
    val_subset = Subset(dataset, [train_idx[i] for i in val_fold_idx])

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4)

    model = FeatureExtractor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        auc, acc, sens, spec = calculate_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        print(f"Fold {fold+1} Epoch {epoch+1} | ACC: {acc:.4f} AUC: {auc:.4f} SENS: {sens:.4f} SPEC: {spec:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), output_pth)
            print("Best model saved.")

# 测试集评估
best_model = FeatureExtractor().to(device)
best_model.load_state_dict(torch.load(output_pth))
best_model.eval()

test_labels, test_preds, test_probs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, outputs = best_model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())

auc, acc, sens, spec = calculate_metrics(
    np.array(test_labels), np.array(test_preds), np.array(test_probs)
)
print(f"=== Final Test Set ===\nACC: {acc:.4f} AUC: {auc:.4f} SENS: {sens:.4f} SPEC: {spec:.4f}")
