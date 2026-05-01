import matplotlib
matplotlib.use('Agg')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from collections import Counter
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT        = './plantvillage'
SAVE_DIR         = './results'
SYNTH_DIR        = './synthetic_leaves'
DIFF_IMG_SIZE    = 64
DIFF_EPOCHS      = 30
DIFF_BATCH       = 16
DIFF_LR          = 1e-4
SYNTH_PER_CLASS  = 200
BATCH_SIZE       = 4
EPOCHS           = 3
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)

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

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_dataset = datasets.ImageFolder(DATA_ROOT, transform=base_transform)
CLASS_NAMES  = full_dataset.classes
NUM_CLASSES  = len(CLASS_NAMES)

# ─────────────────────────────────────────────
# IDENTIFY MINORITY CLASSES
# ─────────────────────────────────────────────
label_counts  = Counter(full_dataset.targets)
threshold     = np.percentile(list(label_counts.values()), 25)
MINORITY_CLASSES = [idx for idx, cnt in label_counts.items() if cnt < threshold]
print(f'Minority classes (count < {threshold:.0f}): {len(MINORITY_CLASSES)} classes')

# ─────────────────────────────────────────────
# DIFFUSION DATASET (minority only)
# ─────────────────────────────────────────────
diff_transform = transforms.Compose([
    transforms.Resize((DIFF_IMG_SIZE, DIFF_IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

minority_indices = [i for i,(_, label) in enumerate(full_dataset.samples)
                    if label in MINORITY_CLASSES]
diff_dataset     = datasets.ImageFolder(DATA_ROOT, transform=diff_transform)
minority_subset  = Subset(diff_dataset, minority_indices)
diff_loader      = DataLoader(minority_subset, batch_size=DIFF_BATCH, shuffle=True, num_workers=0)

print(f'Diffusion training images: {len(minority_subset)}')

# ─────────────────────────────────────────────
# DIFFUSION MODEL
# ─────────────────────────────────────────────
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

unet = UNet2DModel(
    sample_size=DIFF_IMG_SIZE,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=('DownBlock2D','AttnDownBlock2D','AttnDownBlock2D','AttnDownBlock2D'),
    up_block_types=('AttnUpBlock2D','AttnUpBlock2D','AttnUpBlock2D','UpBlock2D')
).to(DEVICE)

optimizer_diff    = torch.optim.AdamW(unet.parameters(), lr=DIFF_LR)
lr_scheduler_diff = get_cosine_schedule_with_warmup(
    optimizer_diff,
    num_warmup_steps=100,
    num_training_steps=DIFF_EPOCHS * len(diff_loader)
)

print(f'UNet params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M')
print('Fine-tuning diffusion model...')

unet_path = os.path.join(SAVE_DIR, 'diffusion_unet.pth')
if os.path.exists(unet_path):
    print('Diffusion model already trained, loading...')
    unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
else:
    print('Fine-tuning diffusion model...')
    unet.train()
    for epoch in range(1, DIFF_EPOCHS + 1):
        epoch_loss = 0
        for batch in diff_loader:
            clean_images = batch[0].to(DEVICE)
            noise        = torch.randn_like(clean_images)
            bsz          = clean_images.shape[0]
            timesteps    = torch.randint(0, noise_scheduler.num_train_timesteps,
                                         (bsz,), device=DEVICE).long()
            noisy        = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred   = unet(noisy, timesteps, return_dict=False)[0]
            loss         = F.mse_loss(noise_pred, noise)
            optimizer_diff.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer_diff.step()
            lr_scheduler_diff.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0:
            print(f'  Epoch {epoch:02d}/{DIFF_EPOCHS} | Loss: {epoch_loss/len(diff_loader):.4f}')
    torch.save(unet.state_dict(), unet_path)
    print('Diffusion model saved.')


# ─────────────────────────────────────────────
# GENERATE SYNTHETIC IMAGES
# ─────────────────────────────────────────────
unet.eval()

@torch.no_grad()
def generate_batch(unet, scheduler, n, device, img_size=64):
    s    = DDPMScheduler(num_train_timesteps=1000)
    imgs = torch.randn(n, 3, img_size, img_size).to(device)
    for t in s.timesteps:
        pred = unet(imgs, t, return_dict=False)[0]
        imgs = s.step(pred, t, imgs, return_dict=False)[0]
    return (imgs.clamp(-1,1) + 1) / 2

print('Generating synthetic images...')
for cls_idx in MINORITY_CLASSES:
    cls_name = CLASS_NAMES[cls_idx]
    cls_dir  = os.path.join(SYNTH_DIR, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    existing = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])
    if existing >= SYNTH_PER_CLASS:
        print(f'  {cls_name}: already done ({existing} images), skipping.')
        continue
    generated = 0
    while generated < SYNTH_PER_CLASS:
        batch_n  = min(8, SYNTH_PER_CLASS - generated)
        imgs     = generate_batch(unet, noise_scheduler, batch_n, DEVICE)
        imgs_up  = F.interpolate(imgs, size=(224,224), mode='bilinear', align_corners=False)
        for j, img_t in enumerate(imgs_up):
            TF.to_pil_image(img_t.cpu()).save(
                os.path.join(cls_dir, f'synth_{generated+j:04d}.png'))
        generated += batch_n
    print(f'  {cls_name}: {generated} images')

# Save sample grid
sample = generate_batch(unet, noise_scheduler, 8, DEVICE)
grid   = make_grid(sample, nrow=4)
plt.figure(figsize=(10,4))
plt.imshow(grid.permute(1,2,0).cpu().numpy())
plt.axis('off')
plt.title('Generated Synthetic Disease Leaf Images', fontweight='bold')
plt.savefig(os.path.join(SAVE_DIR,'synthetic_samples.png'), bbox_inches='tight', dpi=150)
plt.show()

# ─────────────────────────────────────────────
# HYBRID MODEL DEFINITION
# ─────────────────────────────────────────────
class HybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        cnn_base        = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
        self.cnn_branch = cnn_base
        vit_base        = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit_branch = vit_base
        fusion_dim      = 512
        self.cnn_proj   = nn.Linear(cnn_base.num_features, fusion_dim)
        self.vit_proj   = nn.Linear(vit_base.num_features, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(fusion_dim, 8, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim*2), nn.Linear(fusion_dim*2, 512),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
    def forward(self, x):
        cp = self.cnn_proj(self.cnn_branch(x)).unsqueeze(1)
        vp = self.vit_proj(self.vit_branch(x)).unsqueeze(1)
        f, _ = self.cross_attn(cp, vp, vp)
        return self.classifier(torch.cat([f.squeeze(1), vp.squeeze(1)], dim=1))

# ─────────────────────────────────────────────
# DATASETS WITH AND WITHOUT AUGMENTATION
# ─────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

real_ds  = datasets.ImageFolder(DATA_ROOT, transform=train_tf)
n_total  = len(real_ds)
n_train  = int(0.70*n_total); n_val = int(0.15*n_total)
n_test   = n_total - n_train - n_val
train_ds, val_ds, test_ds = random_split(real_ds, [n_train, n_val, n_test])
val_ds.dataset.transform = val_tf
test_ds.dataset.transform = val_tf

synth_ds = datasets.ImageFolder(SYNTH_DIR, transform=train_tf)
synth_class_to_real = {}
for sc, si in synth_ds.class_to_idx.items():
    if sc in real_ds.class_to_idx:
        synth_class_to_real[si] = real_ds.class_to_idx[sc]
synth_ds.targets = [synth_class_to_real.get(t, t) for t in synth_ds.targets]
synth_ds.samples = [(p, synth_class_to_real.get(l, l)) for p, l in synth_ds.samples]

aug_train_ds = ConcatDataset([train_ds, synth_ds])

loader_no_aug   = DataLoader(train_ds,     batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_with_aug = DataLoader(aug_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader      = DataLoader(val_ds,       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader     = DataLoader(test_ds,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────
# TRAIN WITH / WITHOUT AUGMENTATION
# ─────────────────────────────────────────────
def quick_train(name, train_loader, num_classes, device, epochs=EPOCHS):
    model     = HybridCNNViT(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = 0, None

    for epoch in range(1, epochs+1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss(label_smoothing=0.1)(model(imgs), labels).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1)==labels).sum().item()
                total   += len(labels)
        va = correct/total
        if va > best_acc:
            best_acc   = va
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        print(f'  [{name}] Epoch {epoch:02d}/{epochs} | Val Acc: {va:.4f}')

    model.load_state_dict(best_state); model = model.to(device)
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            all_p.extend(model(imgs).argmax(1).cpu().numpy())
            all_l.extend(labels.cpu().numpy())

    acc      = sum(p==l for p,l in zip(all_p,all_l)) / len(all_l)
    f1_all   = f1_score(all_l, all_p, average='macro', zero_division=0)
    f1_minor = f1_score(all_l, all_p, labels=MINORITY_CLASSES, average='macro', zero_division=0)
    print(f'\n  [{name}] Acc: {acc:.4f} | Macro F1: {f1_all:.4f} | Minority F1: {f1_minor:.4f}')
    return acc, f1_all, f1_minor

print('\nTraining WITHOUT augmentation...')
acc_no, f1_no, mf1_no = quick_train('No Aug', loader_no_aug, NUM_CLASSES, DEVICE)

print('\nTraining WITH augmentation...')
acc_w, f1_w, mf1_w = quick_train('With DDPM', loader_with_aug, NUM_CLASSES, DEVICE)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
aug_df = pd.DataFrame({
    'Configuration': ['Hybrid CNN-ViT (No Aug)', 'Hybrid CNN-ViT + DDPM (Ours)'],
    'Overall Acc (%)':        [round(acc_no*100,2), round(acc_w*100,2)],
    'Macro F1 (%)':           [round(f1_no*100,2),  round(f1_w*100,2)],
    'Minority-Class F1 (%)':  [round(mf1_no*100,2), round(mf1_w*100,2)]
})
print('\nAUGMENTATION RESULTS:')
print(aug_df.to_string(index=False))
aug_df.to_csv(os.path.join(SAVE_DIR,'augmentation_results.csv'), index=False)

fig, ax = plt.subplots(figsize=(9,5))
metrics = ['Overall Acc (%)', 'Macro F1 (%)', 'Minority-Class F1 (%)']
x = np.arange(len(metrics)); w = 0.35
b1 = ax.bar(x-w/2, aug_df.iloc[0][metrics], w, label='Without DDPM', color='#FF8A65')
b2 = ax.bar(x+w/2, aug_df.iloc[1][metrics], w, label='With DDPM (Ours)', color='#66BB6A')
ax.bar_label(b1, fmt='%.2f', padding=2, fontsize=9)
ax.bar_label(b2, fmt='%.2f', padding=2, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0,105); ax.set_ylabel('Score (%)')
ax.set_title('Impact of DDPM Augmentation', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'augmentation_comparison.png'), bbox_inches='tight', dpi=150)
plt.show()

print('\nDone. Files saved in ./results/')