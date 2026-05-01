import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT      = './plantvillage'
PLANTDOC_ROOT  = './PlantDoc/test'
SAVE_DIR       = './results'
BATCH_SIZE     = 8
EPOCHS         = 4
LR             = 1e-4
LAMBDA_ENTROPY = 0.01
LAMBDA_USAGE   = 0.01
LAMBDA_ORTH    = 0.001
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# ─────────────────────────────────────────────
# PLANTDOC DOWNLOAD (if not present)
# ─────────────────────────────────────────────
if not os.path.exists('./PlantDoc-Dataset'):
    os.system('git clone https://github.com/pratikkayal/PlantDoc-Dataset.git')

if not os.path.exists(PLANTDOC_ROOT):
    for root, dirs, files in os.walk('./PlantDoc-Dataset'):
        if len(dirs) > 5 and any('Apple' in d or 'Tomato' in d for d in dirs):
            PLANTDOC_ROOT = root
            break

print(f'PlantDoc path: {PLANTDOC_ROOT}')

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
pv_dataset  = datasets.ImageFolder(DATA_ROOT, transform=train_tf)
PV_CLASSES  = pv_dataset.classes
NUM_CLASSES = len(PV_CLASSES)

n_total = len(pv_dataset)
n_train = int(0.80*n_total); n_val = n_total - n_train
train_ds, val_ds = random_split(pv_dataset, [n_train, n_val])
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

pd_dataset      = datasets.ImageFolder(PLANTDOC_ROOT, transform=val_tf)
PD_CLASSES      = pd_dataset.classes
plantdoc_loader = DataLoader(pd_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'PlantVillage train: {len(train_ds)} | val: {len(val_ds)}')
print(f'PlantDoc: {len(pd_dataset)} images | {len(PD_CLASSES)} classes')

# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────
class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.net(x)

class MoEClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=6, top_k=3, hidden_dim=512):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.experts     = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, num_classes)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        gate_logits         = self.gate(x)
        topk_vals, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        sparse              = torch.full_like(gate_logits, float('-inf'))
        sparse.scatter_(1, topk_idx, topk_vals)
        gw           = torch.softmax(sparse, dim=-1)
        expert_outs  = torch.stack([e(x) for e in self.experts], dim=1)
        out          = (gw.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out, gw

    def regularisation_losses(self, gw):
        eps      = 1e-8
        L_ent    = -(-( gw * (gw+eps).log()).sum(dim=-1).mean())
        avg_use  = gw.mean(dim=0)
        L_use    = ((avg_use - 1.0/self.num_experts)**2).sum()
        W        = torch.stack([e.net[0].weight for e in self.experts])
        W_flat   = W.view(self.num_experts, -1)
        gram     = W_flat @ W_flat.T
        identity = torch.eye(self.num_experts, device=gram.device)
        L_orth   = ((gram - identity)**2).sum()
        return L_ent, L_use, L_orth

class PlantGuard(nn.Module):
    def __init__(self, num_classes, num_experts=6, top_k=3):
        super().__init__()
        cnn_base        = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.cnn_branch = cnn_base
        vit_base        = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit_branch = vit_base
        fusion_dim      = 512
        self.cnn_proj   = nn.Linear(cnn_base.num_features, fusion_dim)
        self.vit_proj   = nn.Linear(vit_base.num_features, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, 8, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(fusion_dim*2)
        self.moe        = MoEClassifier(fusion_dim*2, num_classes, num_experts, top_k)

    def forward(self, x):
        cp       = self.cnn_proj(self.cnn_branch(x)).unsqueeze(1)
        vp       = self.vit_proj(self.vit_branch(x)).unsqueeze(1)
        fused, _ = self.cross_attn(cp, vp, vp)
        combined = self.layer_norm(torch.cat([fused.squeeze(1), vp.squeeze(1)], dim=1))
        return self.moe(combined)

    def count_params(self):
        return sum(p.numel() for p in self.parameters()) / 1e6

model = PlantGuard(NUM_CLASSES).to(DEVICE)
print(f'PlantGuard parameters: {model.count_params():.1f}M')

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
model_path = os.path.join(SAVE_DIR, 'plantguard_best.pth')
if os.path.exists(model_path):
    print('PlantGuard already trained, loading...')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    best_val_acc, best_state = 0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_correct, train_total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, gw = model(imgs)
            L_class    = criterion(logits, labels)
            L_ent, L_use, L_orth = model.moe.regularisation_losses(gw)
            loss = (L_class
                    + LAMBDA_ENTROPY * L_ent
                    + LAMBDA_USAGE   * L_use
                    + LAMBDA_ORTH    * L_orth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_correct += (logits.argmax(1)==labels).sum().item()
            train_total   += len(labels)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits, _    = model(imgs)
                val_correct += (logits.argmax(1)==labels).sum().item()
                val_total   += len(labels)

        va = val_correct/val_total
        ta = train_correct/train_total
        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        print(f'Epoch {epoch:02d}/{EPOCHS} | Train: {ta:.4f} | Val: {va:.4f} | Best: {best_val_acc:.4f}')

    torch.save(best_state, os.path.join(SAVE_DIR, 'plantguard_best.pth'))

# ─────────────────────────────────────────────
# CROSS-DOMAIN EVALUATION
# ─────────────────────────────────────────────
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# In-domain (PlantVillage val)
pv_correct, pv_total = 0, 0
pv_preds, pv_labels  = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits, _    = model(imgs)
        preds        = logits.argmax(1)
        pv_correct  += (preds==labels).sum().item()
        pv_total    += len(labels)
        pv_preds.extend(preds.cpu().numpy())
        pv_labels.extend(labels.cpu().numpy())

pv_acc = pv_correct/pv_total
pv_f1  = f1_score(pv_labels, pv_preds, average='macro', zero_division=0)

# Cross-domain (PlantDoc)
def normalize_name(n):
    return n.lower().replace('_',' ').replace('-',' ').strip()

pv_map     = {normalize_name(c): i for i,c in enumerate(PV_CLASSES)}
pd_to_pv   = {}
for pd_cls in PD_CLASSES:
    norm = normalize_name(pd_cls)
    if norm in pv_map:
        pd_to_pv[pd_cls] = pv_map[norm]
    else:
        for pv_name, pv_idx in pv_map.items():
            if any(w in pv_name for w in norm.split() if len(w)>4):
                pd_to_pv[pd_cls] = pv_idx
                break

print(f'Mapped {len(pd_to_pv)}/{len(PD_CLASSES)} PlantDoc classes')

pd_correct, pd_total = 0, 0
pd_preds_l, pd_labels_l = [], []
with torch.no_grad():
    for imgs, labels in plantdoc_loader:
        imgs = imgs.to(DEVICE)
        logits, _ = model(imgs)
        preds     = logits.argmax(1)
        for pred, label in zip(preds.cpu(), labels):
            pd_cls = PD_CLASSES[label.item()]
            if pd_cls in pd_to_pv:
                true_idx    = pd_to_pv[pd_cls]
                pd_correct += (pred.item() == true_idx)
                pd_total   += 1
                pd_preds_l.append(pred.item())
                pd_labels_l.append(true_idx)

pd_acc = pd_correct/pd_total if pd_total > 0 else 0
pd_f1  = f1_score(pd_labels_l, pd_preds_l, average='macro', zero_division=0) if pd_total > 0 else 0

print(f'\nRESULTS:')
print(f'  PlantVillage (in-domain):   Acc={pv_acc*100:.2f}% | F1={pv_f1*100:.2f}%')
print(f'  PlantDoc    (cross-domain): Acc={pd_acc*100:.2f}% | F1={pd_f1*100:.2f}%')
print(f'  Domain gap:                 {(pv_acc-pd_acc)*100:.2f}pp')
print(f'\n  Salman et al. 2025 SOTA: 68.00%')
if pd_acc*100 > 68:
    print(f'  YOUR MODEL BEATS SOTA by {pd_acc*100-68:.2f}pp')
else:
    print(f'  Gap to SOTA: {68-pd_acc*100:.2f}pp')

# ─────────────────────────────────────────────
# FULL COMPARISON TABLE
# ─────────────────────────────────────────────
try:
    b = pd.read_csv(os.path.join(SAVE_DIR,'baseline_results.csv'))
    eff_acc = b[b['Model']=='EfficientNet-B4']['Test Acc (%)'].values[0]
    vit_acc = b[b['Model']=='ViT-B/16']['Test Acc (%)'].values[0]
    hyb_acc = b[b['Model'].str.contains('Hybrid')]['Test Acc (%)'].values[0]
except:
    eff_acc = vit_acc = hyb_acc = 'Run NB1'

comparison = pd.DataFrame([
    {'Method':'Mohanty et al. [2016]',     'PV Acc (%)':99.35, 'PD Acc (%)':'~84.0', 'Params(M)':6.8},
    {'Method':'Atila et al. [2021]',        'PV Acc (%)':99.97, 'PD Acc (%)':'N/A',   'Params(M)':19.3},
    {'Method':'Salman et al. [2025] SOTA',  'PV Acc (%)':99.96, 'PD Acc (%)':68.0,    'Params(M)':'~86'},
    {'Method':'EfficientNet-B4 (Ours)',     'PV Acc (%)':eff_acc,'PD Acc (%)':'—',     'Params(M)':19.3},
    {'Method':'ViT-B/16 (Ours)',            'PV Acc (%)':vit_acc,'PD Acc (%)':'—',     'Params(M)':86.6},
    {'Method':'Hybrid CNN-ViT (Ours)',      'PV Acc (%)':hyb_acc,'PD Acc (%)':'—',     'Params(M)':'~106'},
    {'Method':'PlantGuard / CNN+ViT+MoE (Ours)',
                                            'PV Acc (%)':round(pv_acc*100,2),
                                            'PD Acc (%)':round(pd_acc*100,2),
                                            'Params(M)':round(model.count_params(),1)},
])
print('\nFULL COMPARISON TABLE:')
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(SAVE_DIR,'full_comparison_table.csv'), index=False)

# ─────────────────────────────────────────────
# PER-CLASS F1 CHART
# ─────────────────────────────────────────────
if pd_total > 0:
    unique_labels = sorted(set(pd_labels_l))
    report = classification_report(
        pd_labels_l, pd_preds_l,
        labels=unique_labels,
        target_names=[PV_CLASSES[i] for i in unique_labels],
        output_dict=True, zero_division=0
    )
    rdf = pd.DataFrame(report).T
    rdf = rdf[~rdf.index.isin(['accuracy','macro avg','weighted avg'])]
    rdf = rdf.sort_values('f1-score', ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(6, len(rdf)*0.4)))
    colors  = ['#4CAF50' if v>=0.7 else '#FF9800' if v>=0.5 else '#F44336'
               for v in rdf['f1-score']]
    ax.barh(range(len(rdf)), rdf['f1-score'], color=colors)
    ax.set_yticks(range(len(rdf)))
    ax.set_yticklabels(rdf.index, fontsize=8)
    ax.set_xlabel('F1-Score')
    ax.set_title('PlantGuard Per-Class F1 on PlantDoc (Cross-Domain)', fontweight='bold')
    ax.axvline(x=pd_f1, color='black', linestyle='--', alpha=0.7, label=f'Macro F1={pd_f1*100:.1f}%')
    ax.legend(); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,'plantdoc_perclass_f1.png'), bbox_inches='tight', dpi=150)
    plt.show()

print('\nDone. Files saved in ./results/')