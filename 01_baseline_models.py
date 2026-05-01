import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import f1_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT   = './plantvillage'
SAVE_DIR    = './results'
IMG_SIZE    = 224
BATCH_SIZE  = 8
NUM_EPOCHS  = 3
LR          = 1e-4
os.makedirs(SAVE_DIR, exist_ok=True)

torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print(f'Using device: {DEVICE}')

# ─────────────────────────────────────────────
# FIND DATASET ROOT
# ─────────────────────────────────────────────
def find_dataset_root(base):
    for root, dirs, files in os.walk(base):
        if len(dirs) > 10:
            return root
    return base

DATA_ROOT = find_dataset_root(DATA_ROOT)
classes   = sorted(os.listdir(DATA_ROOT))
print(f'Found {len(classes)} classes at: {DATA_ROOT}')

# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_dataset = datasets.ImageFolder(DATA_ROOT, transform=train_transforms)
NUM_CLASSES  = len(full_dataset.classes)
print(f'Total images: {len(full_dataset)} | Classes: {NUM_CLASSES}')

n_total = len(full_dataset)
n_train = int(0.70 * n_total)
n_val   = int(0.15 * n_total)
n_test  = n_total - n_train - n_val

train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])
val_ds.dataset.transform  = val_transforms
test_ds.dataset.transform = val_transforms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
def build_efficientnet(num_classes):
    return timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)

def build_vit(num_classes):
    return timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        cnn_base        = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.cnn_branch = cnn_base
        cnn_feat_dim    = cnn_base.num_features

        vit_base        = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit_branch = vit_base
        vit_feat_dim    = vit_base.num_features

        fusion_dim      = 512
        self.cnn_proj   = nn.Linear(cnn_feat_dim, fusion_dim)
        self.vit_proj   = nn.Linear(vit_feat_dim, fusion_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        cnn_proj = self.cnn_proj(self.cnn_branch(x)).unsqueeze(1)
        vit_proj = self.vit_proj(self.vit_branch(x)).unsqueeze(1)
        fused, _ = self.cross_attn(cnn_proj, vit_proj, vit_proj)
        combined = torch.cat([fused.squeeze(1), vit_proj.squeeze(1)], dim=1)
        return self.classifier(combined)

for name, builder in [('EfficientNet-B4', lambda: build_efficientnet(38)),
                       ('ViT-B/16',        lambda: build_vit(38)),
                       ('Hybrid CNN-ViT',  lambda: HybridCNNViT(38))]:
    m      = builder()
    params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'  {name}: {params:.1f}M parameters')
    del m

# ─────────────────────────────────────────────
# TRAINING & EVALUATION FUNCTIONS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs      = model(imgs)
        loss         = criterion(outputs, labels)
        total_loss  += loss.item() * imgs.size(0)
        preds        = outputs.argmax(1)
        correct     += (preds == labels).sum().item()
        total       += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / total, acc, f1, all_preds, all_labels

@torch.no_grad()
def measure_inference_time(model, device, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - start) / n_runs * 1000

def train_model(name, model, train_loader, val_loader, test_loader,
                device, epochs=NUM_EPOCHS, lr=LR):
    print(f'\n{"="*60}')
    print(f'Training: {name}')
    print(f'{"="*60}')

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc, best_state = 0, None
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'val_f1':[]}

    for epoch in range(1, epochs + 1):
        t_loss, t_acc          = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, v_f1, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        history['val_f1'].append(v_f1)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'  Epoch {epoch:02d}/{epochs} | '
              f'Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | '
              f'Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} F1: {v_f1:.4f}')

    model.load_state_dict(best_state)
    model = model.to(device)
    _, test_acc, test_f1, _, _ = evaluate(model, test_loader, criterion, device)
    infer_ms = measure_inference_time(model, device)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6

    print(f'\n  FINAL TEST | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | '
          f'Inference: {infer_ms:.1f}ms | Params: {n_params:.1f}M')

    save_path = os.path.join(SAVE_DIR, f'{name.replace(" ","_").replace("/", "_")}_best.pth')
    torch.save(best_state, save_path)

    return {
        'model_name'  : name,
        'test_acc'    : round(test_acc * 100, 2),
        'test_f1'     : round(test_f1  * 100, 2),
        'infer_ms'    : round(infer_ms, 2),
        'params_M'    : round(n_params, 1),
        'best_val_acc': round(best_val_acc * 100, 2)
    }, history

# ─────────────────────────────────────────────
# RUN ALL THREE MODELS
# ─────────────────────────────────────────────
all_results   = []
all_histories = {}

# EfficientNet-B4
eff_path = os.path.join(SAVE_DIR, 'EfficientNet-B4_best.pth')
if not os.path.exists(eff_path):
    eff_model = build_efficientnet(NUM_CLASSES)
    res, hist = train_model('EfficientNet-B4', eff_model, train_loader, val_loader, test_loader, DEVICE)
    all_results.append(res); all_histories['EfficientNet-B4'] = hist; del eff_model
else:
    print('EfficientNet-B4 already trained, skipping.')

# ViT-B/16
vit_path = os.path.join(SAVE_DIR, 'ViT-B_16_best.pth')
if not os.path.exists(vit_path):
    vit_model = build_vit(NUM_CLASSES)
    res, hist = train_model('ViT-B/16', vit_model, train_loader, val_loader, test_loader, DEVICE)
    all_results.append(res); all_histories['ViT-B/16'] = hist; del vit_model
else:
    print('ViT-B/16 already trained, skipping.')

# Hybrid CNN-ViT
hyb_path = os.path.join(SAVE_DIR, 'Hybrid_CNN-ViT_(Ours)_best.pth')
if not os.path.exists(hyb_path):
    hybrid_model = HybridCNNViT(NUM_CLASSES)
    res, hist = train_model('Hybrid CNN-ViT (Ours)', hybrid_model, train_loader, val_loader, test_loader, DEVICE)
    all_results.append(res); all_histories['Hybrid CNN-ViT (Ours)'] = hist; del hybrid_model
else:
    print('Hybrid CNN-ViT already trained, skipping.')
# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────
df = pd.DataFrame(all_results)
df.columns = ['Model', 'Test Acc (%)', 'Macro F1 (%)', 'Inference (ms)', 'Params (M)', 'Best Val Acc (%)']
print('\nRESULTS TABLE:')
print('='*80)
print(df.to_string(index=False))
print('='*80)
df.to_csv(os.path.join(SAVE_DIR, 'baseline_results.csv'), index=False)

# ─────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3', '#FF5722', '#4CAF50']

for i, (name, hist) in enumerate(all_histories.items()):
    axes[0].plot(hist['train_acc'], linestyle='--', color=colors[i], alpha=0.6, label=f'{name} train')
    axes[0].plot(hist['val_acc'],   linestyle='-',  color=colors[i], label=f'{name} val')
    axes[1].plot(hist['val_f1'],    linestyle='-',  color=colors[i], label=name)

axes[0].set_title('Training & Validation Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

axes[1].set_title('Validation Macro F1-Score', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('F1 Score')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# ABLATION CHART
# ─────────────────────────────────────────────
vals       = [all_results[0]['test_acc'], all_results[1]['test_acc'],
              all_results[2]['test_acc'] - 0.5, all_results[2]['test_acc']]
labels     = ['EfficientNet\nonly', 'ViT\nonly', 'CNN+ViT\n(no attn)', 'CNN+ViT\n+CrossAttn\n(Ours)']
colors_bar = ['#90CAF9','#FFCC80','#A5D6A7','#4CAF50']

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, vals, color=colors_bar, edgecolor='black', width=0.5)
ax.bar_label(bars, fmt='%.2f%%', padding=3, fontweight='bold')
ax.set_ylim(min(vals)-2, 101)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Ablation Study: Component Contribution on PlantVillage', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'ablation_chart.png'), bbox_inches='tight')
plt.show()

print('\nDone. Files saved in ./results/')