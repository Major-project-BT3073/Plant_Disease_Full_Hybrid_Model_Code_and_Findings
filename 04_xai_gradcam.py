import matplotlib
matplotlib.use('Agg')
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT  = './plantvillage'
SAVE_DIR   = './results'
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
        self.num_experts = num_experts; self.top_k = top_k
        self.experts     = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, num_classes) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
    def forward(self, x):
        gl = self.gate(x)
        tv, ti = gl.topk(self.top_k, dim=-1)
        sp = torch.full_like(gl, float('-inf'))
        sp.scatter_(1, ti, tv)
        gw = torch.softmax(sp, dim=-1)
        return (gw.unsqueeze(-1) * torch.stack([e(x) for e in self.experts], dim=1)).sum(1), gw

class PlantGuard(nn.Module):
    def __init__(self, num_classes, num_experts=6, top_k=3):
        super().__init__()
        cnn_base        = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.cnn_branch = cnn_base
        vit_base        = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
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

# Load class names
dummy_ds    = datasets.ImageFolder(DATA_ROOT)
CLASS_NAMES = dummy_ds.classes
NUM_CLASSES = len(CLASS_NAMES)

model = PlantGuard(NUM_CLASSES).to(DEVICE)
state = torch.load(os.path.join(SAVE_DIR, 'plantguard_best.pth'), map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f'PlantGuard loaded | Classes: {NUM_CLASSES}')

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.grads = None
        self.acts  = None
        self.h1    = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))
        self.h2    = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def __call__(self, img_tensor, class_idx=None):
        self.model.zero_grad()
        logits, _ = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2,3), keepdim=True)
        cam     = torch.relu((weights * self.acts).sum(dim=1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def remove(self):
        self.h1.remove(); self.h2.remove()

# ─────────────────────────────────────────────
# SCORE-CAM
# ─────────────────────────────────────────────
class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts  = None
        self.h     = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))

    @torch.no_grad()
    def __call__(self, img_tensor, class_idx=None):
        logits, _ = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        acts    = self.acts
        n_ch    = acts.shape[1]
        H, W    = img_tensor.shape[2], img_tensor.shape[3]
        bl, _   = self.model(torch.zeros_like(img_tensor))
        base_sc = torch.softmax(bl, dim=1)[0, class_idx].item()
        weights = []
        for start in range(0, n_ch, 16):
            end   = min(start+16, n_ch)
            batch = []
            for k in range(start, end):
                ak   = acts[0, k]
                ak   = (ak - ak.min()) / (ak.max() - ak.min() + 1e-8)
                mask = torch.nn.functional.interpolate(
                    ak.unsqueeze(0).unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False)
                batch.append(img_tensor * mask)
            bt       = torch.cat(batch, dim=0)
            logits_b, _ = self.model(bt)
            scores_b = torch.softmax(logits_b, dim=1)[:, class_idx]
            weights.extend((scores_b - base_sc).cpu().tolist())
        weights = torch.tensor(weights)
        cam = torch.relu((weights.to(DEVICE).view(-1,1,1) * acts.squeeze(0)).sum(dim=0)).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

    def remove(self):
        self.h.remove()

target_layer = model.cnn_branch.conv_head
gradcam      = GradCAM(model, target_layer)
scorecam     = ScoreCAM(model, target_layer)
print('Grad-CAM and Score-CAM initialized')

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def overlay_heatmap(img_np, cam, alpha=0.5):
    cam_r   = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cm.jet(cam_r)[:,:,:3]
    overlay = alpha * heatmap + (1-alpha) * img_np/255.0
    return (overlay * 255).astype(np.uint8)

# ─────────────────────────────────────────────
# SELECT SAMPLE IMAGES
# ─────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

TARGET_CLASSES = [
    'Tomato___Early_blight',
    'Grape___Black_rot',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Apple___Apple_scab',
    'Potato___Late_blight',
]
available = [c for c in TARGET_CLASSES if c in CLASS_NAMES]
if not available:
    available = CLASS_NAMES[:5]

sample_imgs = []
for cls_name in available:
    cls_dir = os.path.join(DATA_ROOT, cls_name)
    if os.path.exists(cls_dir):
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if imgs:
            pil = Image.open(os.path.join(cls_dir, imgs[0])).convert('RGB')
            sample_imgs.append((cls_name, pil))

print(f'Visualizing {len(sample_imgs)} disease classes')

# ─────────────────────────────────────────────
# GENERATE XAI FIGURE
# ─────────────────────────────────────────────
n     = len(sample_imgs)
fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
if n == 1:
    axes = [axes]

for j, t in enumerate(['Original Image','Grad-CAM','Score-CAM','Grad-CAM Overlay']):
    axes[0][j].set_title(t, fontweight='bold', fontsize=12)

for i, (cls_name, pil_img) in enumerate(sample_imgs):
    img_tensor = val_tf(pil_img).unsqueeze(0).to(DEVICE)
    img_np     = np.array(pil_img.resize((224,224)))

    img_tensor.requires_grad_(True)
    gcam, pred_idx = gradcam(img_tensor)
    img_tensor     = img_tensor.detach()

    scam, _ = scorecam(img_tensor)

    gcam_224     = cv2.resize(gcam, (224,224))
    scam_224     = cv2.resize(scam, (224,224))
    gcam_overlay = overlay_heatmap(img_np, gcam_224)

    axes[i][0].set_ylabel(
        cls_name.replace('___','\n').replace('_',' '),
        fontsize=8, rotation=0, labelpad=80, va='center'
    )
    axes[i][0].imshow(img_np)
    axes[i][1].imshow(gcam_224, cmap='jet')
    axes[i][2].imshow(scam_224, cmap='jet')
    axes[i][3].imshow(gcam_overlay)

    with torch.no_grad():
        logits, _ = model(img_tensor)
        conf      = torch.softmax(logits, dim=1)[0, pred_idx].item()
    axes[i][3].set_title(
        f'Pred: {CLASS_NAMES[pred_idx].split("___")[-1]}\nConf: {conf:.1%}', fontsize=8)

    for j in range(4):
        axes[i][j].axis('off')

plt.suptitle('PlantGuard XAI: Grad-CAM vs Score-CAM', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'xai_gradcam_scorecam.png'), bbox_inches='tight', dpi=150)
plt.show()
print('Saved xai_gradcam_scorecam.png')

# ─────────────────────────────────────────────
# MoE EXPERT UTILIZATION
# ─────────────────────────────────────────────
val_ds      = datasets.ImageFolder(DATA_ROOT, transform=val_tf)
val_loader  = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

expert_usage    = defaultdict(lambda: defaultdict(float))
n_per_class     = defaultdict(int)
num_experts     = 6

model.eval()
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        if batch_idx > 20: break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        _, gw        = model(imgs)
        for gw_i, label in zip(gw, labels):
            cls = CLASS_NAMES[label.item()]
            for e_idx, w in enumerate(gw_i.cpu()):
                expert_usage[cls][e_idx] += w.item()
            n_per_class[cls] += 1

n_show       = min(20, len(expert_usage))
usage_matrix = np.zeros((n_show, num_experts))
row_labels   = []

for i, (cls, usage) in enumerate(list(expert_usage.items())[:n_show]):
    n = n_per_class[cls]
    for e in range(num_experts):
        usage_matrix[i, e] = usage[e] / n
    row_labels.append(cls.replace('___',' ').replace('_',' ')[:30])

fig, ax = plt.subplots(figsize=(10, max(6, n_show*0.5)))
im      = ax.imshow(usage_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(num_experts))
ax.set_xticklabels([f'Expert {i+1}' for i in range(num_experts)], fontweight='bold')
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_title('MoE Expert Utilization per Disease Class', fontweight='bold')
plt.colorbar(im, ax=ax, label='Average Gate Weight')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR,'moe_expert_utilization.png'), bbox_inches='tight', dpi=150)
plt.show()
print('Saved moe_expert_utilization.png')

gradcam.remove()
scorecam.remove()
print('\nDone. Files saved in ./results/')