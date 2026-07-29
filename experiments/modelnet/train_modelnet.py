"""Train the paper's ModelNet10/40 SPM, ASP, and distillation pipeline.

Full runs require a CUDA GPU.  A small CPU-only forward-pass check is provided
through ``--smoke`` so reviewers can validate the model without downloading a
dataset or allocating a training GPU.
"""

import argparse, json, math, os, random, shutil, time
from contextlib import nullcontext
import warnings
warnings.filterwarnings("ignore")

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import kagglehub
except ImportError:
    kagglehub = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--datasets",      default="ModelNet10,ModelNet40")
parser.add_argument("--epochs",        type=int,   default=300)
parser.add_argument("--batch",         type=int,   default=64)
parser.add_argument("--points",        type=int,   default=1024)
parser.add_argument("--dim",           type=int,   default=384)
parser.add_argument("--depth",         type=int,   default=12)
parser.add_argument("--timestep",      type=int,   default=2)
parser.add_argument("--groups",        type=int,   default=128)
parser.add_argument("--group_size",    type=int,   default=32)
parser.add_argument("--asp_steps",     type=int,   default=4)
parser.add_argument("--expand",        type=float, default=1.1)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--weight_decay",  type=float, default=0.1)
parser.add_argument("--warmup_ep",     type=int,   default=30)
parser.add_argument("--drop_path",     type=float, default=0.3)
parser.add_argument("--label_smooth",  type=float, default=0.2)
parser.add_argument("--kd_temp",       type=float, default=4.0)
parser.add_argument("--teacher_epochs",type=int,   default=150)
parser.add_argument("--vote",          type=int,   default=5)
parser.add_argument("--exit_thr",      type=float, default=0.45)
parser.add_argument("--ckpt_dir",      default="./a100_ckpts")
parser.add_argument("--data_dir",      default="./data")
parser.add_argument("--no_kd",         action="store_true")
parser.add_argument("--no_compile",    action="store_true")
parser.add_argument("--no_bf16",       action="store_true")
parser.add_argument("--resume",        action="store_true", default=True)
parser.add_argument("--no_resume",     action="store_false", dest="resume")
parser.add_argument("--seed",           type=int, default=42)
parser.add_argument(
    "--smoke",
    action="store_true",
    help="run a CPU/GPU forward-pass check without data or training",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda" and not args.smoke:
    raise RuntimeError("Full ModelNet training requires a CUDA GPU; use --smoke for the CPU check.")

if DEVICE == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("Device  : CPU smoke test")
print(f"PyTorch : {torch.__version__}")
print(f"BF16    : {not args.no_bf16 and DEVICE == 'cuda'}")
print(f"compile : {not args.no_compile and not args.smoke}")

EPOCHS        = args.epochs
BATCH         = args.batch
GRAD_ACCUM    = 1          # A100 can hold batch=64 natively
NUM_POINTS    = args.points
TIMESTEP      = args.timestep
TRANS_DIM     = args.dim
DEPTH         = args.depth
NUM_GROUP     = args.groups
GROUP_SIZE    = args.group_size
ASP_STEPS     = args.asp_steps
EXPAND        = args.expand
LR            = args.lr
WEIGHT_DECAY  = args.weight_decay
WARMUP_EP     = args.warmup_ep
DROP_PATH     = args.drop_path
LABEL_SMOOTH  = args.label_smooth
KD_TEMP       = args.kd_temp
KD_CE_WEIGHT  = 0.5
KD_LOGIT_WEIGHT = 0.5
KD_AUX_WEIGHT = 0.1
TEACHER_EPOCHS = args.teacher_epochs
N_VOTE        = args.vote
EXIT_THR      = args.exit_thr
USE_KD        = not args.no_kd
USE_BF16      = DEVICE == "cuda" and not args.no_bf16 and torch.cuda.is_bf16_supported()
USE_COMPILE   = DEVICE == "cuda" and not args.no_compile and not args.smoke
NUM_WORKERS   = 0 if args.smoke else 8
DATASET_NAMES = [d.strip() for d in args.datasets.split(",")]
CKPT_DIR      = args.ckpt_dir
DATA_DIR      = args.data_dir
RESUME        = args.resume

if not args.smoke:
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

assert NUM_GROUP % ASP_STEPS == 0
assert NUM_GROUP <= NUM_POINTS, "--groups cannot exceed --points"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

print("\n[Config]")
print(f"  datasets={DATASET_NAMES}  epochs={EPOCHS}  batch={BATCH}")
print(f"  dim={TRANS_DIM}  depth={DEPTH}  timestep={TIMESTEP}")
print(f"  groups={NUM_GROUP}  group_size={GROUP_SIZE}  expand={EXPAND}")
print(f"  asp_steps={ASP_STEPS}  kd={USE_KD}  kd_temp={KD_TEMP}")
print(f"  ckpt={CKPT_DIR}  data={DATA_DIR}")

AMP_CTX = (
    (lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16))
    if DEVICE == "cuda"
    else nullcontext
)
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE == "cuda" and not USE_BF16),
)  # BF16 does not need a scaler.

# ── Dataset ───────────────────────────────────────────────────────────────────
def _download(name, slug):
    if kagglehub is None:
        raise RuntimeError("kagglehub is required for automatic downloads; install requirements.txt.")
    folder = os.path.join(DATA_DIR, name)
    if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
        print(f"  {name}: cached at {folder}")
        return folder
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    print(f"  Downloading {name} ...")
    path = kagglehub.dataset_download(slug)
    for root, dirs, _files in os.walk(path):
        if name in dirs:
            shutil.copytree(os.path.join(root, name), folder)
            print(f"  {name} -> {folder}")
            return folder
        subdirs_with_train = [d for d in dirs if os.path.isdir(os.path.join(root, d, "train"))]
        if len(subdirs_with_train) >= 5:
            shutil.copytree(root, folder)
            print(f"  {name} -> {folder}")
            return folder
    return path


def _augment(pts):
    n = pts.shape[0]
    keep = max(int(n * random.uniform(0.875, 1.0)), 1)
    idx = np.random.choice(n, keep, replace=False)
    pts2 = pts[idx]
    if keep < n:
        pad = np.random.choice(keep, n - keep, replace=True)
        pts2 = np.vstack([pts2, pts2[pad]])
    pts2 = pts2 * np.random.uniform(0.8, 1.25)
    pts2 = pts2 + np.random.uniform(-0.1, 0.1, (1, 3))
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pts2 = pts2 @ rz.T
    pts2 += np.clip(np.random.randn(*pts2.shape).astype(np.float32) * 0.01, -0.05, 0.05)
    return pts2.astype(np.float32)


class ModelNetDataset(Dataset):
    def __init__(self, root, num_points=1024, split="train"):
        self.num_points = num_points
        self.split = split
        self.files = self._scan(root)
        print(f"  [{split}] Loading {len(self.files)} files ...")
        self.data, self.labels = self._load_all()
        print(f"  [{split}] Loaded. Shape: {self.data.shape}")

    def _scan(self, root):
        items = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for cls in classes:
            path = os.path.join(root, cls, self.split)
            if not os.path.isdir(path):
                continue
            label = classes.index(cls)
            for fname in os.listdir(path):
                if fname.lower().endswith((".npy", ".txt", ".off")):
                    items.append((os.path.join(path, fname), label))
        return items

    def _load_pts(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)[:, :3]
        if path.endswith(".txt"):
            return np.loadtxt(path, delimiter=",").astype(np.float32)[:, :3]
        if trimesh is None:
            raise RuntimeError("trimesh is required to sample .off meshes.")
        mesh = trimesh.load(path, force="mesh")
        pts, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return pts.astype(np.float32)

    def _load_all(self):
        all_pts, all_lbl = [], []
        failed = []
        for path, label in self.files:
            try:
                pts = self._load_pts(path)
                n = pts.shape[0]
                if n >= self.num_points:
                    pts = pts[np.random.choice(n, self.num_points, replace=False)]
                else:
                    pad = np.random.choice(n, self.num_points - n, replace=True)
                    pts = np.vstack([pts, pts[pad]])
                all_pts.append(pts)
                all_lbl.append(label)
            except (OSError, ValueError, RuntimeError) as exc:
                if len(failed) < 5:
                    failed.append(f"{path}: {exc}")
        if failed:
            warnings.warn("Skipped unreadable point-cloud files:\n" + "\n".join(failed))
        if not all_pts:
            raise RuntimeError(f"No readable {self.split} samples found under the dataset root.")
        return np.asarray(all_pts, dtype=np.float32), np.asarray(all_lbl, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pts = self.data[idx].copy()
        pts -= pts.mean(axis=0)
        pts /= np.max(np.linalg.norm(pts, axis=1)) + 1e-8
        if self.split == "train":
            pts = _augment(pts)
        np.random.shuffle(pts)
        return torch.tensor(pts, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ── Grouping ──────────────────────────────────────────────────────────────────
def index_points(points, idx):
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(b, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx]


def farthest_point_sample_batched(xyz, npoint):
    b, n, _ = xyz.shape
    npoint = min(npoint, n)
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device)
    farthest = torch.randint(0, n, (b,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(b, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1).indices
    return centroids


class OfficialLikeGroup(nn.Module):
    def __init__(self, num_group, group_size, expand=1.1, timestep=2):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.expand = expand
        self.timestep = timestep

    def _moving_centers(self, pts):
        b, n, _ = pts.shape
        step_f = int((self.expand - 1.0) * self.num_group / self.timestep * 2)
        step_b = int((self.expand - 1.0) * self.num_group)
        total = min(max(self.num_group + (step_f + step_b) * (self.timestep - 1), self.num_group), n)
        center_idx = farthest_point_sample_batched(pts.contiguous(), total)
        pool = index_points(pts, center_idx)
        if total < self.num_group + (step_f + step_b) * (self.timestep - 1):
            repeat = math.ceil((self.num_group + (step_f + step_b) * (self.timestep - 1)) / total)
            pool = pool.repeat(1, repeat, 1)
        centers = []
        for i in range(self.timestep):
            first = pool[:, i * step_f : i * step_f + (self.num_group - step_b)]
            start = (i - 1) * step_b + self.num_group + (self.timestep - 1) * step_f
            end = i * step_b + self.num_group + (self.timestep - 1) * step_f
            second = pool[:, start:end]
            cur = torch.cat([first, second], dim=1)
            if cur.shape[1] < self.num_group:
                pad = cur[:, -1:].repeat(1, self.num_group - cur.shape[1], 1)
                cur = torch.cat([cur, pad], dim=1)
            centers.append(cur[:, : self.num_group])
        return torch.stack(centers, dim=0)

    def forward(self, pts):
        b, n, _ = pts.shape
        centers = self._moving_centers(pts)
        flat_centers = centers.reshape(self.timestep * b, self.num_group, 3)
        flat_pts = pts.unsqueeze(0).expand(self.timestep, -1, -1, -1).reshape(self.timestep * b, n, 3)
        k = min(self.group_size, n)
        dist = torch.cdist(flat_centers, flat_pts)
        idx = dist.topk(k, dim=-1, largest=False).indices
        grouped = index_points(flat_pts, idx).reshape(self.timestep, b, self.num_group, k, 3)
        grouped = grouped - centers.unsqueeze(3)
        return grouped.contiguous(), centers.contiguous()


# ── SNN layers ────────────────────────────────────────────────────────────────
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + x.abs()) ** 2

spike_fn = SurrogateSpike.apply


class SpikeAct(nn.Module):
    def __init__(self, vth=0.5):
        super().__init__()
        self.vth = vth
        self.register_buffer("spike_sum", torch.tensor(0.0))
        self.register_buffer("elem_count", torch.tensor(0.0))

    def forward(self, x):
        y = spike_fn(x - self.vth)
        with torch.no_grad():
            self.spike_sum += y.sum()
            self.elem_count += torch.tensor(y.numel(), dtype=torch.float32, device=y.device)
        return y

    def rate(self):
        return (self.spike_sum / self.elem_count).item() if self.elem_count.item() else 0.0


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(torch.full((x.shape[0],) + (1,) * (x.ndim - 1), keep, device=x.device)) / keep
        return x * mask


class TokenBNSpike(nn.Module):
    def __init__(self, dim, vth=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.spike = SpikeAct(vth)

    def forward(self, x):
        b, l, c = x.shape
        return self.spike(self.bn(x.reshape(b * l, c)).reshape(b, l, c))


class OfficialLikeEncoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.spk1 = SpikeAct()
        self.spk2 = SpikeAct()
        self.spk3 = SpikeAct()
        self.first_conv1 = nn.Conv2d(3, 128, 1)
        self.first_bn1 = nn.BatchNorm2d(128)
        self.first_conv2 = nn.Conv2d(128, 256, 1)
        self.first_bn2 = nn.BatchNorm2d(256)
        self.second_conv1 = nn.Conv2d(512, 512, 1)
        self.second_bn1 = nn.BatchNorm2d(512)
        self.second_conv2 = nn.Conv2d(512, encoder_channel, 1)
        self.second_bn2 = nn.BatchNorm2d(encoder_channel)

    def forward(self, point_groups):
        t, b, g, k, _ = point_groups.shape
        x = point_groups.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x = self.spk1(self.first_bn1(self.first_conv1(x)))
        x = self.first_bn2(self.first_conv2(x))
        x_global = x.max(dim=3, keepdim=True).values
        x = torch.cat([x_global.expand(-1, -1, -1, k), x], dim=1)
        x = self.spk2(x)
        x = self.spk3(self.second_bn1(self.second_conv1(x)))
        x = self.second_bn2(self.second_conv2(x))
        x = x.max(dim=3).values.transpose(1, 2).contiguous()
        return x.reshape(t, b, g, -1)


class PosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), SpikeAct(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim),
        )

    def forward(self, centers):
        t, b, g, _ = centers.shape
        x = centers.flatten(0, 1).permute(0, 2, 1).contiguous()
        return self.net(x).permute(0, 2, 1).contiguous().reshape(t, b, g, -1)


class MambaLiteMixer(nn.Module):
    def __init__(self, dim, expand=2):
        super().__init__()
        inner = dim * expand
        self.in_proj  = nn.Linear(dim, inner * 2)
        self.dwconv   = nn.Conv1d(inner, inner, 3, padding=1, groups=inner)
        self.scan_proj = nn.Linear(inner, inner)
        self.out_proj  = nn.Linear(inner, dim)

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.dwconv(u.transpose(1, 2)).transpose(1, 2)
        u = F.silu(u)
        steps = torch.arange(1, u.shape[1] + 1, device=u.device, dtype=u.dtype).view(1, -1, 1)
        state = torch.cumsum(u, dim=1) / steps
        u = u + self.scan_proj(state)
        return self.out_proj(u * torch.sigmoid(gate))


class OfficialLikeBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm_lif  = TokenBNSpike(dim)
        self.mixer     = MambaLiteMixer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, hidden_states, residual=None):
        residual = self.drop_path(hidden_states) + residual if residual is not None else hidden_states
        hidden_states = self.norm_lif(residual)
        return self.mixer(hidden_states), residual


class OfficialLikeMixerModel(nn.Module):
    def __init__(self, dim, depth, timestep, drop_path=0.3):
        super().__init__()
        self.timestep = timestep
        self.layers = nn.ModuleList([OfficialLikeBlock(dim, drop_path) for _ in range(depth)])

    def forward(self, tokens, pos):
        t, b, l, c = tokens.shape
        x = (tokens + pos).reshape(t * b, l, c)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x = x + residual if residual is not None else x
        return x.reshape(t, b, l, c)


class OfficialLikeHead(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            SpikeAct(), nn.Conv1d(dim, 256, 1), nn.BatchNorm1d(256),
            SpikeAct(), nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128),
            SpikeAct(), nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x):
        t, b, _l, c = x.shape
        pooled = x.mean(dim=2).reshape(t * b, c, 1)
        return self.net(pooled).reshape(t, b, -1, 1).mean(dim=0).squeeze(-1)


class OfficialLikeSPM(nn.Module):
    def __init__(self, num_classes, dim=384, depth=12, num_group=128,
                 group_size=32, timestep=2, expand=1.1, drop_path=0.3):
        super().__init__()
        self.num_classes  = num_classes
        self.dim          = dim
        self.depth        = depth
        self.num_group    = num_group
        self.group_size   = group_size
        self.timestep     = timestep
        self.group_divider = OfficialLikeGroup(num_group, group_size, expand, timestep)
        self.encoder      = OfficialLikeEncoder(dim)
        self.pos_embed    = PosEmbed(dim)
        self.blocks       = OfficialLikeMixerModel(dim, depth, timestep, drop_path)
        self.drop_out     = nn.Dropout(0.0)
        self.cls_head     = OfficialLikeHead(dim, num_classes)

    def encode_groups(self, pts):
        neighborhoods, centers = self.group_divider(pts)
        return self.encoder(neighborhoods), self.pos_embed(centers), centers

    def forward_tokens(self, tokens, pos):
        return self.cls_head(self.blocks(self.drop_out(tokens), pos))

    def forward(self, pts):
        tokens, pos, _ = self.encode_groups(pts)
        return self.forward_tokens(tokens, pos)

    def mean_firing_rate(self):
        rates = [m.rate() for m in self.modules() if isinstance(m, SpikeAct)]
        return sum(rates) / max(1, len(rates))


# ── Teacher ───────────────────────────────────────────────────────────────────
class AnalogGroupEncoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.first_conv1  = nn.Conv2d(3, 128, 1)
        self.first_bn1    = nn.BatchNorm2d(128)
        self.first_conv2  = nn.Conv2d(128, 256, 1)
        self.first_bn2    = nn.BatchNorm2d(256)
        self.second_conv1 = nn.Conv2d(512, 512, 1)
        self.second_bn1   = nn.BatchNorm2d(512)
        self.second_conv2 = nn.Conv2d(512, encoder_channel, 1)
        self.second_bn2   = nn.BatchNorm2d(encoder_channel)

    def forward(self, point_groups):
        t, b, g, k, _ = point_groups.shape
        x = point_groups.flatten(0, 1).permute(0, 3, 1, 2).contiguous()
        x = F.gelu(self.first_bn1(self.first_conv1(x)))
        x = F.gelu(self.first_bn2(self.first_conv2(x)))
        x_global = x.max(dim=3, keepdim=True).values
        x = torch.cat([x_global.expand(-1, -1, -1, k), x], dim=1)
        x = F.gelu(self.second_bn1(self.second_conv1(x)))
        x = self.second_bn2(self.second_conv2(x))
        return x.max(dim=3).values.transpose(1, 2).contiguous().reshape(t, b, g, -1)


class AnalogPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim),
        )

    def forward(self, centers):
        t, b, g, _ = centers.shape
        x = centers.flatten(0, 1).permute(0, 2, 1).contiguous()
        return self.net(x).permute(0, 2, 1).contiguous().reshape(t, b, g, -1)


class PointTransformerTeacher(nn.Module):
    def __init__(self, num_classes, dim=384, depth=8, heads=8,
                 num_group=128, group_size=32, expand=1.1):
        super().__init__()
        self.num_classes   = num_classes
        self.group_divider = OfficialLikeGroup(num_group, group_size, expand, timestep=1)
        self.encoder       = AnalogGroupEncoder(dim)
        self.pos_embed     = AnalogPosEmbed(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm   = nn.LayerNorm(dim)
        self.head   = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(dim, num_classes),
        )

    def forward(self, pts):
        neighborhoods, centers = self.group_divider(pts)
        x = self.encoder(neighborhoods) + self.pos_embed(centers)
        x = self.norm(self.blocks(x.squeeze(0)))
        return self.head(torch.cat([x.mean(1), x.max(1).values], dim=-1))


# ── ASP ───────────────────────────────────────────────────────────────────────
class SliceSelectionPolicy(nn.Module):
    def __init__(self, mem_dim, geo_dim=7, hidden=128):
        super().__init__()
        self.mem_proj = nn.Linear(mem_dim, hidden, bias=False)
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden, bias=False))
        self.scale    = math.sqrt(hidden)

    def forward(self, belief, geo, visited_mask=None):
        scores = torch.bmm(self.geo_proj(geo), self.mem_proj(belief).unsqueeze(-1)).squeeze(-1) / self.scale
        if visited_mask is not None:
            scores = scores.masked_fill(visited_mask.clone(), float("-inf"))
        return scores


class OfficialLikeASP(nn.Module):
    def __init__(self, base_model, asp_steps=4, d_ssp=128):
        super().__init__()
        self.base_model = base_model
        self.asp_steps  = asp_steps
        self.chunk_size = base_model.num_group // asp_steps
        self.ssp        = SliceSelectionPolicy(base_model.dim, geo_dim=7, hidden=d_ssp)
        self.belief_proj = nn.Sequential(
            nn.Linear(base_model.num_classes, base_model.dim),
            nn.GELU(),
            nn.Linear(base_model.dim, base_model.dim),
        )
        self.register_buffer("gumbel_tau", torch.tensor(1.0))

    @property
    def num_classes(self):
        return self.base_model.num_classes

    def set_gumbel_tau(self, tau):
        self.gumbel_tau.fill_(tau)

    def mean_firing_rate(self):
        return self.base_model.mean_firing_rate()

    def _chunkify(self, tokens, pos, centers, pts):
        t, b, g, c = tokens.shape
        s, k = self.asp_steps, self.chunk_size
        tokens_c  = tokens.reshape(t, b, s, k, c)
        pos_c     = pos.reshape(t, b, s, k, c)
        centers_b = centers.mean(dim=0).reshape(b, s, k, 3)
        chunk_center = centers_b.mean(dim=2)
        centroid  = pts.mean(dim=1, keepdim=True)
        anchor_dist = (chunk_center - centroid).norm(dim=-1, keepdim=True)
        spread    = (centers_b - chunk_center.unsqueeze(2)).norm(dim=-1).mean(dim=2, keepdim=True)
        order     = torch.linspace(0, 1, s, device=pts.device).view(1, s, 1).expand(b, -1, -1)
        geo       = torch.cat([chunk_center, anchor_dist, spread, torch.ones(b, s, 1, device=pts.device), order], dim=-1)
        return tokens_c, pos_c, geo

    def _gather_chunk(self, chunks, idx):
        t, b, _s, k, c = chunks.shape
        gi = idx.view(1, b, 1, 1, 1).expand(t, b, 1, k, c)
        return chunks.gather(2, gi).squeeze(2)

    def forward_active_train(self, pts):
        tokens, pos, centers = self.base_model.encode_groups(pts)
        tok_c, pos_c, geo   = self._chunkify(tokens, pos, centers, pts)
        b, device = pts.shape[0], pts.device
        visited  = torch.zeros(b, self.asp_steps, dtype=torch.bool, device=device)
        belief   = torch.zeros(b, self.base_model.dim, device=device)
        selected_tokens, selected_pos, logits_all = [], [], []
        for _ in range(self.asp_steps):
            scores = self.ssp(belief, geo, visited)
            w = F.gumbel_softmax(scores, tau=float(self.gumbel_tau.item()), hard=True)
            idx = w.detach().argmax(-1)
            visited.scatter_(1, idx.unsqueeze(1), True)
            tok = (w.view(1, b, self.asp_steps, 1, 1) * tok_c).sum(2)
            ps  = (w.view(1, b, self.asp_steps, 1, 1) * pos_c).sum(2)
            selected_tokens.append(tok)
            selected_pos.append(ps)
            logits = self.base_model.forward_tokens(torch.cat(selected_tokens, 2), torch.cat(selected_pos, 2))
            logits_all.append(logits)
            belief = self.belief_proj(logits.detach().softmax(-1))
        return logits_all[-1], logits_all

    @torch.no_grad()
    def forward_active_infer(self, pts, threshold=0.45):
        tokens, pos, centers = self.base_model.encode_groups(pts)
        tok_c, pos_c, geo   = self._chunkify(tokens, pos, centers, pts)
        b, device = pts.shape[0], pts.device
        visited  = torch.zeros(b, self.asp_steps, dtype=torch.bool, device=device)
        belief   = torch.zeros(b, self.base_model.dim, device=device)
        selected_tokens, selected_pos = [], []
        last_logits = None
        for step in range(self.asp_steps):
            idx = self.ssp(belief, geo, visited).argmax(-1)
            visited.scatter_(1, idx.unsqueeze(1), True)
            selected_tokens.append(self._gather_chunk(tok_c, idx))
            selected_pos.append(self._gather_chunk(pos_c, idx))
            logits = self.base_model.forward_tokens(torch.cat(selected_tokens, 2), torch.cat(selected_pos, 2))
            last_logits = logits
            belief = self.belief_proj(logits.softmax(-1))
            top2 = logits.softmax(-1).topk(2, -1).values
            if (top2[:, 0] - top2[:, 1]).min().item() > threshold:
                return logits, step + 1
        return last_logits, self.asp_steps


# ── Losses / schedulers / checkpoints ────────────────────────────────────────
def smooth_ce(logits, labels):
    return F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTH)


def kd_ce(logits, labels, teacher_logits=None):
    ce = smooth_ce(logits, labels)
    if teacher_logits is None or KD_LOGIT_WEIGHT <= 0:
        return ce
    kd = F.kl_div(
        F.log_softmax(logits / KD_TEMP, -1),
        F.softmax(teacher_logits.detach() / KD_TEMP, -1),
        reduction="batchmean",
    ) * KD_TEMP ** 2
    return KD_CE_WEIGHT * ce + KD_LOGIT_WEIGHT * kd


def active_loss(logits_final, logits_all, labels, model, teacher_logits=None):
    loss = kd_ce(logits_final, labels, teacher_logits)
    if len(logits_all) > 1:
        aux  = sum(kd_ce(l, labels, teacher_logits) for l in logits_all[:-1])
        loss = loss + KD_AUX_WEIGHT * aux / (len(logits_all) - 1)
    for i, l in enumerate(logits_all):
        w    = (len(logits_all) - i) / len(logits_all)
        loss = loss + 0.05 * w * (1.0 - l.softmax(-1).max(-1).values).mean() / len(logits_all)
    return loss + 0.01 * model.mean_firing_rate()


def gumbel_tau(epoch, tau_0=1.0, tau_min=0.1, rate=0.04):
    return max(tau_min, tau_0 * math.exp(-rate * epoch))


def make_scheduler(optimizer, epochs, warmup_epochs):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(1, warmup_epochs)
        t = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(1e-2, 0.5 * (1.0 + math.cos(math.pi * t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def save_ckpt(path, model, opt, sch, epoch, best, history):
    raw = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "epoch": epoch, "model": raw,
        "optimizer": opt.state_dict(), "scheduler": sch.state_dict(),
        "best": best, "history": history,
        "config": {"dim": TRANS_DIM, "depth": DEPTH, "timestep": TIMESTEP,
                   "num_group": NUM_GROUP, "group_size": GROUP_SIZE},
    }, path)


def load_ckpt(path, model, opt, sch):
    if not RESUME or not os.path.isfile(path):
        return 0, 0.0, []
    try:
        ckpt = _torch_load(path)
        raw  = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sch.load_state_dict(ckpt["scheduler"])
        ep   = int(ckpt["epoch"])
        print(f"  [CKPT] resumed {os.path.basename(path)} epoch {ep}")
        return ep, float(ckpt.get("best", 0.0)), ckpt.get("history", [])
    except Exception as exc:
        print(f"  [CKPT] resume skipped for {path}: {exc}")
        return 0, 0.0, []


def save_best(path, model):
    raw = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save(raw, path)


# ── Model factories ───────────────────────────────────────────────────────────
def make_spm(num_classes):
    m = OfficialLikeSPM(
        num_classes, dim=TRANS_DIM, depth=DEPTH, num_group=NUM_GROUP,
        group_size=GROUP_SIZE, timestep=TIMESTEP, expand=EXPAND, drop_path=DROP_PATH,
    ).to(DEVICE)
    if USE_COMPILE:
        try:
            m = torch.compile(m)
        except Exception:
            pass
    return m


def make_asp(num_classes):
    return OfficialLikeASP(
        OfficialLikeSPM(num_classes, dim=TRANS_DIM, depth=DEPTH, num_group=NUM_GROUP,
                        group_size=GROUP_SIZE, timestep=TIMESTEP, expand=EXPAND, drop_path=DROP_PATH).to(DEVICE),
        asp_steps=ASP_STEPS,
    ).to(DEVICE)


def make_teacher(num_classes):
    heads = 8 if TRANS_DIM % 8 == 0 else 4
    m = PointTransformerTeacher(
        num_classes, dim=TRANS_DIM, depth=8, heads=heads,
        num_group=NUM_GROUP, group_size=GROUP_SIZE, expand=EXPAND,
    ).to(DEVICE)
    if USE_COMPILE:
        try:
            m = torch.compile(m)
        except Exception:
            pass
    return m


# ── Training loops ────────────────────────────────────────────────────────────
def train_teacher_epoch(model, loader, opt):
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss = total_acc = n = 0
    for step, (pts, labels) in enumerate(loader):
        pts, labels = pts.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with AMP_CTX():
            logits = model(pts)
            loss   = smooth_ce(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        b = pts.shape[0]
        total_loss += loss.item() * b
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n += b
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_teacher(model, loader, n_vote=1):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        prob_sum = torch.zeros(pts.shape[0], model.num_classes, device=DEVICE)
        for _ in range(n_vote):
            theta = random.uniform(0.0, 2.0 * math.pi)
            c, s  = math.cos(theta), math.sin(theta)
            rz    = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float32, device=DEVICE)
            with AMP_CTX():
                prob_sum += model(pts @ rz.T).softmax(-1)
        correct += (prob_sum.argmax(1) == labels).sum().item()
        total   += pts.shape[0]
    return correct / total


def prepare_teacher(ds_name, num_classes, train_l, val_l):
    if not USE_KD:
        return None, 0.0
    teacher = make_teacher(num_classes)
    print(f"\n[Teacher] params={sum(p.numel() for p in teacher.parameters()):,}")
    latest    = os.path.join(CKPT_DIR, f"teacher_{ds_name}_latest.pt")
    best_path = os.path.join(CKPT_DIR, f"teacher_{ds_name}_best.pth")
    opt = torch.optim.AdamW(teacher.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = make_scheduler(opt, max(1, TEACHER_EPOCHS), min(WARMUP_EP, TEACHER_EPOCHS))
    start_ep, best_teacher, _ = load_ckpt(latest, teacher, opt, sch)
    if start_ep >= TEACHER_EPOCHS and os.path.isfile(best_path):
        raw = teacher._orig_mod if hasattr(teacher, "_orig_mod") else teacher
        raw.load_state_dict(torch.load(best_path, map_location=DEVICE))
        teacher.eval()
        print(f"  [Teacher] using cached best val={best_teacher:.4f}")
        return teacher, best_teacher
    for ep in range(start_ep, TEACHER_EPOCHS):
        t0 = time.time()
        _, tr_acc = train_teacher_epoch(teacher, train_l, opt)
        sch.step()
        if (ep + 1) % 5 == 0 or ep + 1 == TEACHER_EPOCHS:
            val_acc = eval_teacher(teacher, val_l, max(1, min(N_VOTE, 3)))
            is_best = val_acc > best_teacher
            if is_best:
                best_teacher = val_acc
                save_best(best_path, teacher)
            star = "*" if is_best else " "
            print(f"  [Teacher] Ep {ep+1:3d}/{TEACHER_EPOCHS} tr={tr_acc:.4f} val={val_acc:.4f} {star} lr={opt.param_groups[0]['lr']:.5f} {time.time()-t0:.0f}s")
        save_ckpt(latest, teacher, opt, sch, ep + 1, best_teacher, [])
    raw = teacher._orig_mod if hasattr(teacher, "_orig_mod") else teacher
    if os.path.isfile(best_path):
        raw.load_state_dict(torch.load(best_path, map_location=DEVICE))
    teacher.eval()
    return teacher, best_teacher


@torch.no_grad()
def teacher_forward(teacher, pts):
    if teacher is None:
        return None
    with AMP_CTX():
        return teacher(pts)


def train_spm_epoch(model, loader, opt, teacher=None):
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        tl = teacher_forward(teacher, pts)
        with AMP_CTX():
            logits = model(pts)
            loss   = kd_ce(logits, labels, tl)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        b = pts.shape[0]
        total_loss += loss.item() * b
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n += b
    return total_loss / n, total_acc / n


def train_asp_epoch(model, loader, opt, epoch, teacher=None):
    model.train()
    model.set_gumbel_tau(gumbel_tau(epoch))
    opt.zero_grad(set_to_none=True)
    total_loss = total_acc = n = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        tl = teacher_forward(teacher, pts)
        with AMP_CTX():
            logits, logits_all = model.forward_active_train(pts)
            loss = active_loss(logits, logits_all, labels, model, tl)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        b = pts.shape[0]
        total_loss += loss.item() * b
        total_acc  += (logits.argmax(1) == labels).sum().item()
        n += b
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_spm(model, loader, n_vote=1):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        prob_sum = torch.zeros(pts.shape[0], model.num_classes, device=DEVICE)
        for _ in range(n_vote):
            theta = random.uniform(0.0, 2.0 * math.pi)
            c, s  = math.cos(theta), math.sin(theta)
            rz    = torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float32, device=DEVICE)
            with AMP_CTX():
                prob_sum += model(pts @ rz.T).softmax(-1)
        correct += (prob_sum.argmax(1) == labels).sum().item()
        total   += pts.shape[0]
    return correct / total


@torch.no_grad()
def eval_asp(model, loader):
    model.eval()
    correct = total = slices = 0
    for pts, labels in loader:
        pts, labels = pts.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with AMP_CTX():
            logits, used = model.forward_active_infer(pts, EXIT_THR)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += pts.shape[0]
        slices  += used * pts.shape[0]
    return correct / total, slices / total


# ── Dataset config / download ─────────────────────────────────────────────────
SLUGS = {
    "ModelNet10": "balraj98/modelnet10-princeton-3d-object-dataset",
    "ModelNet40": "balraj98/modelnet40-princeton-3d-object-dataset",
}

def dataset_config():
    cfg = {}
    for name in DATASET_NAMES:
        if name not in SLUGS:
            raise ValueError(f"Unsupported dataset {name!r}; choose ModelNet10 and/or ModelNet40.")
        classes = 10 if "10" in name else 40
        print(f"\n[Download] {name} ...")
        root = _download(name, SLUGS[name])
        cfg[name] = {"root": root, "classes": classes}
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_smoke():
    """Build both student variants and validate their output contracts."""
    num_classes = 10
    points = torch.randn(2, NUM_POINTS, 3, device=DEVICE)

    spm = make_spm(num_classes).eval()
    spm_logits = spm(points)
    assert spm_logits.shape == (2, num_classes)

    asp = make_asp(num_classes).eval()
    asp_logits, trajectory = asp.forward_active_train(points)
    assert asp_logits.shape == (2, num_classes)
    assert len(trajectory) == ASP_STEPS

    infer_logits, chunks = asp.forward_active_infer(points, threshold=1.1)
    assert infer_logits.shape == (2, num_classes)
    assert chunks == ASP_STEPS
    print(
        "Smoke test passed: "
        f"SPM {tuple(spm_logits.shape)}, ASP {tuple(asp_logits.shape)}, "
        f"{chunks}/{ASP_STEPS} chunks."
    )


def main():
    if args.smoke:
        run_smoke()
        return

    results   = {}
    histories = {}

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )

    for ds_name, ds_cfg in dataset_config().items():
        nc = ds_cfg["classes"]
        print("\n" + "=" * 76)
        print(f"Dataset: {ds_name}  classes={nc}")
        print("Backbone: official-like SPM group encoder + Mamba-lite mixer")
        print("=" * 76)

        train_ds = ModelNetDataset(ds_cfg["root"], NUM_POINTS, "train")
        val_ds   = ModelNetDataset(ds_cfg["root"], NUM_POINTS, "test")
        train_l  = DataLoader(train_ds, BATCH, shuffle=True,  drop_last=True,  **loader_kwargs)
        val_l    = DataLoader(val_ds,   BATCH, shuffle=False, drop_last=False, **loader_kwargs)

        teacher, teacher_best = prepare_teacher(ds_name, nc, train_l, val_l)
        if teacher is not None:
            print(f"[KD] Distilling from teacher (best val={teacher_best*100:.2f}%).")
        else:
            print("[KD] Disabled.")

        # ── SPM training ─────────────────────────────────────────────────────
        spm = make_spm(nc)
        print(f"\n[SPM] params={sum(p.numel() for p in spm.parameters()):,}")
        spm_opt    = torch.optim.AdamW(spm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        spm_sch    = make_scheduler(spm_opt, EPOCHS, WARMUP_EP)
        spm_latest = os.path.join(CKPT_DIR, f"spm_{ds_name}_latest.pt")
        spm_best_p = os.path.join(CKPT_DIR, f"spm_{ds_name}_best.pth")
        start_ep, best_spm, spm_hist = load_ckpt(spm_latest, spm, spm_opt, spm_sch)

        for ep in range(start_ep, EPOCHS):
            t0 = time.time()
            _, tr_acc = train_spm_epoch(spm, train_l, spm_opt, teacher)
            spm_sch.step()
            val_acc = None
            if (ep + 1) % 5 == 0 or ep + 1 == EPOCHS:
                val_acc = eval_spm(spm, val_l, N_VOTE)
                is_best = val_acc > best_spm
                if is_best:
                    best_spm = val_acc
                    save_best(spm_best_p, spm)
                star = "*" if is_best else " "
                print(f"  [SPM] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} val={val_acc:.4f} {star} lr={spm_opt.param_groups[0]['lr']:.5f} {time.time()-t0:.0f}s")
            spm_hist.append({"ep": ep + 1, "tr": tr_acc, "val": val_acc})
            save_ckpt(spm_latest, spm, spm_opt, spm_sch, ep + 1, best_spm, spm_hist)

        # ── ASP training ──────────────────────────────────────────────────────
        asp = make_asp(nc)
        print(f"\n[ASP] params={sum(p.numel() for p in asp.parameters()):,}")
        asp_opt    = torch.optim.AdamW(asp.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        asp_sch    = make_scheduler(asp_opt, EPOCHS, WARMUP_EP)
        asp_latest = os.path.join(CKPT_DIR, f"asp_{ds_name}_latest.pt")
        asp_best_p = os.path.join(CKPT_DIR, f"asp_{ds_name}_best.pth")
        start_ep, best_asp, asp_hist = load_ckpt(asp_latest, asp, asp_opt, asp_sch)
        best_asp_sl = ASP_STEPS

        for ep in range(start_ep, EPOCHS):
            t0 = time.time()
            _, tr_acc = train_asp_epoch(asp, train_l, asp_opt, ep, teacher)
            asp_sch.step()
            val_acc = val_sl = None
            if (ep + 1) % 5 == 0 or ep + 1 == EPOCHS:
                val_acc, val_sl = eval_asp(asp, val_l)
                is_best = val_acc > best_asp
                if is_best:
                    best_asp, best_asp_sl = val_acc, val_sl
                    save_best(asp_best_p, asp)
                star = "*" if is_best else " "
                print(f"  [ASP] Ep {ep+1:3d}/{EPOCHS} tr={tr_acc:.4f} val={val_acc:.4f} {star} slices={val_sl:.2f}/{ASP_STEPS} tau={float(asp.gumbel_tau):.3f} lr={asp_opt.param_groups[0]['lr']:.5f} {time.time()-t0:.0f}s")
            asp_hist.append({"ep": ep + 1, "tr": tr_acc, "val": val_acc})
            save_ckpt(asp_latest, asp, asp_opt, asp_sch, ep + 1, best_asp, asp_hist)

        fr_mean = asp.mean_firing_rate()
        results[ds_name] = {
            "spm_best": best_spm, "asp_best": best_asp,
            "delta_pp": (best_asp - best_spm) * 100,
            "asp_avg_chunks": best_asp_sl, "firing_rate": fr_mean,
            "teacher_best": teacher_best, "kd_enabled": teacher is not None,
        }
        histories[ds_name] = {"spm": spm_hist, "asp": asp_hist}

        print(f"\nSummary {ds_name}")
        if teacher is not None:
            print(f"  Teacher  : {teacher_best*100:.2f}%")
        print(f"  SPM best : {best_spm*100:.2f}%")
        print(f"  ASP best : {best_asp*100:.2f}%")
        print(f"  Delta    : {(best_asp-best_spm)*100:+.2f} pp")
        print(f"  ASP chunks: {best_asp_sl:.2f}/{ASP_STEPS}  Firing rate: {fr_mean:.4f}")

    with open(os.path.join(CKPT_DIR, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(CKPT_DIR, "histories.json"), "w") as f:
        json.dump(histories, f, indent=2)

    print("\n" + "=" * 76)
    print("FINAL RESULTS")
    print("=" * 76)
    print(f"{'Dataset':<14} {'Teacher':>9} {'SPM':>9} {'ASP':>9} {'Delta':>9} {'Chunks':>10}")
    for ds, r in results.items():
        print(f"{ds:<14} {r['teacher_best']*100:>8.2f}% {r['spm_best']*100:>8.2f}% {r['asp_best']*100:>8.2f}% {r['delta_pp']:>+8.2f} {r['asp_avg_chunks']:>6.2f}/{ASP_STEPS}")
    print(f"\nOutputs: {CKPT_DIR}")


if __name__ == "__main__":
    main()
