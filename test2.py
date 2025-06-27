import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# === 路径与设备配置 ===
model_path = 'best_model.pth'
val_data_dir = 'image'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 图像预处理 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === 数据集定义 ===
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

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx]).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# === 模型结构 ===
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

# === 指标计算函数 ===
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
        sensitivities.append(tp / (tp + fn) if (tp + fn) else 0)
        specificities.append(tn / (tn + fp) if (tn + fp) else 0)
    acc = np.mean(y_true == y_pred)
    return auc, acc, np.mean(sensitivities), np.mean(specificities)

# === bootstrap 计算95%置信区间 ===
def bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    accs, aucs, senss, specs = [], [], [], []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), n, replace=True)
        acc = np.mean(y_true[idx] == y_pred[idx])
        try:
            y_true_bin = np.eye(np.max(y_true)+1)[y_true[idx]]
            auc = roc_auc_score(y_true_bin, y_prob[idx], multi_class='ovr', average='macro')
        except:
            auc = np.nan
        cm = confusion_matrix(y_true[idx], y_pred[idx])
        sens_list, spec_list = [], []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            sens_list.append(tp / (tp + fn) if (tp + fn) else 0)
            spec_list.append(tn / (tn + fp) if (tn + fp) else 0)
        accs.append(acc)
        aucs.append(auc)
        senss.append(np.mean(sens_list))
        specs.append(np.mean(spec_list))

    def ci(arr):
        arr = np.array(arr)
        return np.nanmean(arr), np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)

    return {
        "ACC": ci(accs),
        "AUC": ci(aucs),
        "SENS": ci(senss),
        "SPEC": ci(specs)
    }

# === 数据加载与评估 ===
val_dataset = PatchDataset(val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=8)

model = FeatureExtractor(num_features=2048).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_labels, all_preds, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        _, outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# === 输出指标 ===
auc, acc, sens, spec = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
print("[VALIDATION]")
print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC: {auc:.4f}")
print(f"Sensitivity: {sens:.4f}")
print(f"Specificity: {spec:.4f}")

print("\n[BOOTSTRAP 95% CI]")
ci_result = bootstrap_ci(np.array(all_labels), np.array(all_preds), np.array(all_probs))
for k, (mean, low, high) in ci_result.items():
    print(f"{k}: {mean:.4f} [{low:.4f}, {high:.4f}]")
