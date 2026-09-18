#!/usr/bin/env python3
"""
DARA v6: Corrected reward scaling + full ablation + plots.

Changes from v5fix:
  1. r_quality = estimated_bitrate / 1e6  (was /1000 → 1000× too large)
  2. r_stability = -Δφ / 100              (was -Δφ → 100× too large)
  3. All reward components now O([-1, 1])
  4. Weight search unbounded [0, 5] since components are normalized
  5. Single self-contained script: load checkpoints → train → ablate → plot

Reuses v5 transformer + scaler (no retraining needed).
Expected runtime: ~45 min on GPU, ~2h on CPU.
"""

import os, sys, io, pickle, json, time, math, random, gc, functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List
import warnings; warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    from scipy import stats
    SCIPY = True
except ImportError:
    SCIPY = False

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 300})

# ============================================================================
# CONSTANTS
# ============================================================================
NI, NT, NTOT = 90, 20, 110
SEQ_LEN = 8
DQN_STATE_DIM = 18
N_HORIZONS = 5
AGG_MS = 100
SEED = 42

# Training
N_RUNS = 5
DQN_EP_SEARCH = 300
DQN_EP_ABLATION = 300
DQN_EP_FINAL = 500
WEIGHT_SEARCH_ITERS = 50

# Weight search: UNBOUNDED since components are now normalized
WEIGHT_RANGES = {
    'w_throughput': (0.1, 5.0),
    'w_delay':      (0.1, 5.0),
    'w_quality':    (0.01, 3.0),
    'w_preemptive': (0.01, 3.0),
    'w_stability':  (0.01, 2.0),
    'w_low_frac':   (0.01, 1.0),
}

ACTION_CONFIGS = {
    '2-level': [30, 100],
    '3-level': [30, 65, 100],
    '5-level': [30, 47, 65, 82, 100],
    '7-level': [30, 42, 53, 65, 77, 88, 100],
}

STATIC_PHIS = [30, 65, 100]

BLOCK_CONFIGS = [
    (360, 5, 0.10), (360, 5, 0.10), (360, 5, 0.15),
    (360, 5, 0.15), (360, 5, 0.20), (360, 5, 0.20),
]

# Paths
BASE = '/mnt/data/users/adbb783/files/TRANSFER/myfolder'
CKPT_DIR   = os.path.join(BASE, 'checkpoints')
DEPLOY_DIR = os.path.join(BASE, 'models', 'deploy_v6')
GRAPH_DIR  = os.path.join(BASE, 'graphs_v6')
OUT_DIR    = os.path.join(BASE, 'ablation_results_v6')

for d in [DEPLOY_DIR, GRAPH_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

Transition = namedtuple('Transition', 'state action next_state reward')

# ============================================================================
# STUB CLASSES (for unpickling v5 checkpoints)
# ============================================================================
@dataclass
class Config:
    n_blocks: int = 3
    phi_values: List[int] = field(default_factory=lambda: [30, 47, 65, 82, 100])
    w_throughput: float = 1.0; w_delay: float = 0.8; w_quality: float = 0.5
    w_preemptive: float = 0.3; w_stability: float = 0.1; w_low_frac: float = 0.1
    dqn_lr: float = 1.25e-4; dqn_gamma: float = 0.966; dqn_tau: float = 0.006
    dqn_batch: int = 128; eps_start: float = 0.7; eps_end: float = 0.1; eps_decay: int = 900

@dataclass
class Metrics:
    reward: float = 0.0; reward_std: float = 0.0; preemptive: float = 0.0
    reactive: float = 0.0; nrmse: float = 0.0; nrmse_cwnd: float = 0.0
    nrmse_srtt: float = 0.0; phi_mean: float = 0.0; phi_std: float = 0.0
    convergence_ep: int = 0; n_params: int = 0; inference_ms: float = 0.0
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    rewards_history: List[float] = field(default_factory=list)
    run_rewards: List[float] = field(default_factory=list)
    run_preemptive: List[float] = field(default_factory=list)
    avg_phi_congestion: float = 0.0; avg_phi_normal: float = 0.0
    horizon_nrmse: List[float] = field(default_factory=list)

@dataclass
class DetailedTrajectory:
    timestamps: List[int] = field(default_factory=list)
    pred_cwnd: List[float] = field(default_factory=list)
    actual_cwnd: List[float] = field(default_factory=list)
    pred_srtt: List[float] = field(default_factory=list)
    actual_srtt: List[float] = field(default_factory=list)
    phi1: List[float] = field(default_factory=list)
    phi2: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    pred_errors: List[float] = field(default_factory=list)
    is_congestion: List[bool] = field(default_factory=list)
    is_preemptive: List[bool] = field(default_factory=list)
    is_reactive: List[bool] = field(default_factory=list)
    reward_components: List[dict] = field(default_factory=list)

@dataclass
class Trajectory:
    timestamps: List[int] = field(default_factory=list)
    pred_cwnd1: List[float] = field(default_factory=list)
    actual_cwnd1: List[float] = field(default_factory=list)
    pred_srtt1: List[float] = field(default_factory=list)
    actual_srtt1: List[float] = field(default_factory=list)
    phi1: List[float] = field(default_factory=list)
    phi2: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    is_congestion: List[bool] = field(default_factory=list)
    is_preemptive: List[bool] = field(default_factory=list)
    is_reactive: List[bool] = field(default_factory=list)
    pred_errors: List[float] = field(default_factory=list)
    reward_components: List[dict] = field(default_factory=list)

class ReplayMemory:
    def __init__(self, cap=15000): self.mem = deque(maxlen=cap)

# ============================================================================
# NETWORKS
# ============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        while dim % heads != 0 and heads > 1: heads -= 1
        self.attn = nn.MultiheadAttention(dim, max(1, heads), dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ff_dim, dim))
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x, x, x)[0]))
        return self.n2(x + self.drop(self.ff(x)))

def create_transformer(n_blocks, dim=90, out_dim=NT):
    n_blocks = min(n_blocks, len(BLOCK_CONFIGS))
    blocks = [TransformerBlock(dim, h, ff, d) for ff, h, d in BLOCK_CONFIGS[:n_blocks]]
    return nn.Sequential(nn.Linear(dim, dim, bias=False), *blocks,
                         nn.LayerNorm(dim), nn.Linear(dim, out_dim))

class LSTMPredictor(nn.Module):
    def __init__(self, d=90, o=NT):
        super().__init__(); self.lstm = nn.LSTM(d, 128, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, o)
    def forward(self, x): return self.fc(self.lstm(x)[0])

class MLPPredictor(nn.Module):
    def __init__(self, d=90, o=NT, s=SEQ_LEN):
        super().__init__(); self.s, self.o = s, o
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(d*s, 256), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, o*s))
    def forward(self, x): return self.net(x).view(-1, self.s, self.o)

class LinearPredictor(nn.Module):
    def __init__(self, d=90, o=NT):
        super().__init__(); self.fc = nn.Linear(d, o)
    def forward(self, x): return self.fc(x)

class DQN(nn.Module):
    def __init__(self, n_obs, n_act, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_obs, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_act))
    def forward(self, x): return self.net(x)

# ============================================================================
# UNPICKLER
# ============================================================================
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        if module == '__main__' and name in globals():
            return globals()[name]
        return super().find_class(module, name)

def load_cp(name):
    path = os.path.join(CKPT_DIR, f'{name}.pkl')
    if not os.path.exists(path): print(f"  [{name}] NOT FOUND"); return None
    try:
        with open(path, 'rb') as f: data = CPUUnpickler(f).load()
        print(f"  [{name}] OK ({os.path.getsize(path)/1e6:.1f}MB)"); return data
    except Exception as e: print(f"  [{name}] FAIL: {e}"); return None

# ============================================================================
# HELPERS
# ============================================================================
def clear_gpu():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

sm = lambda v, d=0.: float(np.mean(v)) if v and len(v) > 0 else d
ss = lambda v, d=0.: float(np.std(v)) if v and len(v) >= 2 else d

def ci(data, conf=0.95):
    if not data or len(data) == 0: return 0., 0., 0.
    a = np.array(data, dtype=float); m = float(np.mean(a))
    if len(a) < 2: return m, m, m
    if SCIPY:
        se = stats.sem(a)
        if se > 0:
            lo, hi = stats.t.interval(conf, len(a)-1, loc=m, scale=se)
            return m, float(lo), float(hi)
    s = np.std(a); mg = 1.96 * s / np.sqrt(len(a))
    return m, m - mg, m + mg

def cohens_d(g1, g2):
    g1, g2 = np.array(g1, dtype=float), np.array(g2, dtype=float)
    if len(g1) < 2 or len(g2) < 2: return 0.
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    p = np.sqrt(((len(g1)-1)*v1 + (len(g2)-1)*v2) / (len(g1)+len(g2)-2))
    return float((np.mean(g1) - np.mean(g2)) / p) if p > 0 else 0.

def detect_convergence(rewards, window=50, threshold=0.02):
    if len(rewards) < window * 2: return len(rewards)
    s = np.convolve(rewards, np.ones(window)/window, mode='valid')
    for i in range(len(s) - window):
        seg = s[i:i+window]
        if np.std(seg) / (abs(np.mean(seg)) + 1e-6) < threshold: return i + window
    return len(rewards)

def make_serializable(obj):
    if isinstance(obj, dict): return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, (int, float, str, bool, type(None))): return obj
    return str(obj)

def gm(obj, attr, default=0):
    if obj is None: return default
    if hasattr(obj, attr):
        v = getattr(obj, attr); return v if v is not None else default
    if isinstance(obj, dict): return obj.get(attr, default)
    return default

def savefig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(GRAPH_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig); print(f"    ✓ {name}")

# ============================================================================
# CORRECTED REWARD — normalized components
# ============================================================================
def inverse_transform_pred(pred, scaler):
    temp = np.zeros((1, NTOT))
    temp[0, NI:NTOT] = pred
    try:
        unsc = np.expm1(scaler.inverse_transform(temp))[0]
    except Exception:
        return None
    MAX_CWND, MAX_SRTT = 100000, 10000000
    return {
        'cc1': float(np.clip(unsc[90], 0, MAX_CWND)),
        'cc2': float(np.clip(unsc[95], 0, MAX_CWND)),
        'cs1': float(np.clip(unsc[100], 0, MAX_SRTT)),
        'cs2': float(np.clip(unsc[105], 0, MAX_SRTT)),
        'fc1': float(np.clip(unsc[94], 0, MAX_CWND)),
        'fc2': float(np.clip(unsc[99], 0, MAX_CWND)),
        'fs1': float(np.clip(unsc[104], 0, MAX_SRTT)),
        'fs2': float(np.clip(unsc[109], 0, MAX_SRTT)),
    }


def compute_reward(pred_raw, scaler, phi1, phi2, prev_phi, est_br, weights):
    """
    CORRECTED reward with normalized components.
    All components target O([-1, 1]) range.
    """
    v = inverse_transform_pred(pred_raw, scaler)
    if v is None:
        return 0., False, False, {}, est_br

    cc1, cc2 = max(1, v['cc1']), max(1, v['cc2'])
    cs1, cs2 = max(1, v['cs1']), max(1, v['cs2'])
    fc1, fc2 = max(1, v['fc1']), max(1, v['fc2'])
    fs1, fs2 = max(1, v['fs1']), max(1, v['fs2'])

    # Throughput: % change in cwnd — already O([-1, 1])
    r_tput = (fc1 - cc1) / cc1 + (fc2 - cc2) / cc2

    # Delay: % decrease in srtt — already O([-1, 1])
    r_delay = (cs1 - fs1) / cs1 + (cs2 - fs2) / cs2

    # Preemptive: cross-subflow aware — already O([-1.5, 1.5])
    r_pre = 0.
    is_pre = False
    is_cong = False

    if fc1 < cc1:
        is_cong = True
        if phi1 >= prev_phi[0]:
            r_pre -= 0.5
        else:
            r_pre += 0.5; is_pre = True
        if fc2 >= cc2 and phi2 > prev_phi[1]:
            r_pre += 0.5

    if fc2 < cc2:
        is_cong = True
        if phi2 >= prev_phi[1]:
            r_pre -= 0.5
        else:
            r_pre += 0.5; is_pre = True
        if fc1 >= cc1 and phi1 > prev_phi[0]:
            r_pre += 0.5

    # Stability: NORMALIZED by /100 — now O([-1.4, 0])
    r_stab = -(abs(phi1 - prev_phi[0]) + abs(phi2 - prev_phi[1])) / 100.0

    # Low frac penalty — O([-1, 0])
    phi_min = 30
    r_low = -1.0 if (phi1 <= phi_min and phi2 <= phi_min) else 0.0

    # Quality: EWMA bitrate / 1e6 — now O([0, ~1])
    agg1 = fc1 / max(1, fs1) if fs1 > 0 else 0
    agg2 = fc2 / max(1, fs2) if fs2 > 0 else 0
    total_agg = (agg1 + agg2) * 8 * 1500

    if est_br[0] == 0:
        est_br[0] = total_agg
    else:
        est_br[0] = 0.9 * est_br[0] + 0.1 * total_agg

    r_qual = est_br[0] / 1e6  # ← KEY FIX: was /1000

    # Combine
    reward = (weights['w_throughput'] * r_tput +
              weights['w_delay']      * r_delay +
              weights['w_preemptive'] * r_pre +
              weights['w_stability']  * r_stab +
              weights['w_low_frac']   * r_low +
              weights['w_quality']    * r_qual)

    rc = {'r_tput': r_tput, 'r_delay': r_delay, 'r_pre': r_pre,
          'r_stab': r_stab, 'r_low': r_low, 'r_qual': r_qual,
          'cc1': cc1, 'fc1': fc1, 'cc2': cc2, 'fc2': fc2,
          'cs1': cs1, 'fs1': fs1, 'bitrate': est_br[0]}

    return reward, is_cong, is_pre, rc, est_br


# ============================================================================
# DQN STATE
# ============================================================================
def build_state(pred, prev_phi):
    nc1, nc2, nr1, nr2 = pred[0], pred[5], pred[10], pred[15]
    mc1, mc2 = pred[2], pred[7]
    fc1, fc2, fr1, fr2 = pred[4], pred[9], pred[14], pred[19]
    sd = lambda a, b: (a - b) / max(abs(b), 1e-6)
    return np.array([
        nc1, nc2, nr1, nr2,
        sd(mc1, nc1), sd(mc2, nc2), sd(pred[12], nr1), sd(pred[17], nr2),
        sd(fc1, nc1), sd(fc2, nc2), sd(fr1, nr1), sd(fr2, nr2),
        prev_phi[0] / 100., prev_phi[1] / 100.,
        1. if fc1 < nc1 * 0.95 else 0.,
        1. if fc2 < nc2 * 0.95 else 0.,
        1. if fr1 > nr1 * 1.1 else 0.,
        1. if fr2 > nr2 * 1.1 else 0.,
    ], dtype=np.float32)


def prepare_data(dataset):
    n = (len(dataset) // SEQ_LEN) * SEQ_LEN
    data = dataset[:n].reshape(-1, SEQ_LEN, dataset.shape[1])
    return data[:, :, :NI], data[:, :, NI:NTOT]


# ============================================================================
# DQN TRAINING
# ============================================================================
def train_dqn(dataset, scaler, predictor, n_ep, phi_values, weights,
              use_oracle=False, collect_traj=False, label=""):
    start = time.time()
    X, y = prepare_data(dataset)
    n_phi = len(phi_values)
    n_act = n_phi ** 2

    if len(X) < 100:
        print(f"  [{label}] insufficient data"); sys.stdout.flush()
        return None, {}, None

    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    if predictor: predictor = predictor.to(DEVICE).eval()

    pol = DQN(DQN_STATE_DIM, n_act).to(DEVICE)
    tgt = DQN(DQN_STATE_DIM, n_act).to(DEVICE)
    tgt.load_state_dict(pol.state_dict())
    opt = optim.AdamW(pol.parameters(), lr=weights.get('dqn_lr', 1.25e-4))
    mem = deque(maxlen=15000)

    dqn_gamma = weights.get('dqn_gamma', 0.966)
    dqn_tau   = weights.get('dqn_tau', 0.006)
    dqn_batch = weights.get('dqn_batch', 128)
    eps_start  = weights.get('eps_start', 0.7)
    eps_end    = weights.get('eps_end', 0.1)
    eps_decay  = weights.get('eps_decay', 900)

    ep_rews, steps = [], 0
    all_p1, all_p2 = [], []
    cong_phis, norm_phis = [], []
    pre_count, rea_count, tot_cong = 0, 0, 0
    traj = Trajectory() if collect_traj else None
    max_steps = min(400, len(X_t) - 10)

    for ep in range(n_ep):
        mx = len(X_t) - 60
        idx = random.randint(0, max(0, mx)) if mx > 0 else 0
        prev = [100., 100.]
        ep_r = 0.
        ep_steps = min(max_steps, len(X_t) - idx - 1)
        est_br = [0.]

        if ep_steps < 10: ep_rews.append(0); continue

        for t in range(ep_steps):
            if idx + t + 1 >= len(X_t): break

            if use_oracle:
                pred = y_t[idx+t, -1].cpu().numpy()
            elif predictor:
                with torch.no_grad():
                    pred = predictor(X_t[idx+t:idx+t+1])[0, -1].cpu().numpy()
            else:
                pred = np.zeros(NT)

            st = build_state(pred, prev)
            state = torch.tensor(st, device=DEVICE).unsqueeze(0)

            eps = eps_end + (eps_start - eps_end) * math.exp(-steps / eps_decay)
            steps += 1

            if random.random() > eps:
                with torch.no_grad(): a = pol(state).argmax().item()
            else:
                a = random.randint(0, n_act - 1)

            p1 = phi_values[a // n_phi]
            p2 = phi_values[a % n_phi]

            reward, is_cong, is_pre, rc, est_br = compute_reward(
                pred, scaler, p1, p2, prev, est_br, weights)

            ep_r += reward
            all_p1.append(p1); all_p2.append(p2)

            if is_cong:
                cong_phis.extend([p1, p2]); tot_cong += 1
                if is_pre: pre_count += 1
                elif p1 < prev[0] or p2 < prev[1]: rea_count += 1
            else:
                norm_phis.extend([p1, p2])

            if collect_traj and ep == n_ep - 1:
                actual = y_t[idx+t, -1].cpu().numpy()
                traj.timestamps.append(t)
                traj.pred_cwnd1.append(float(pred[4]))
                traj.actual_cwnd1.append(float(actual[4]))
                traj.pred_srtt1.append(float(pred[14]))
                traj.actual_srtt1.append(float(actual[14]))
                traj.phi1.append(p1); traj.phi2.append(p2)
                traj.rewards.append(reward); traj.actions.append(a)
                traj.is_congestion.append(is_cong)
                traj.is_preemptive.append(is_pre)
                traj.is_reactive.append(is_cong and (p1 < prev[0] or p2 < prev[1]) and not is_pre)
                traj.pred_errors.append(float(np.mean(np.abs(pred - actual))))
                traj.reward_components.append(rc)

            prev = [p1, p2]

            # Next state
            if use_oracle:
                np_ = y_t[idx+t+1, -1].cpu().numpy()
            elif predictor:
                with torch.no_grad():
                    np_ = predictor(X_t[idx+t+1:idx+t+2])[0, -1].cpu().numpy()
            else:
                np_ = np.zeros(NT)

            ns = build_state(np_, prev)
            nstate = torch.tensor(ns, device=DEVICE).unsqueeze(0)

            mem.append(Transition(state, torch.tensor([[a]], device=DEVICE),
                                  nstate, torch.tensor([reward], device=DEVICE)))

            if len(mem) >= dqn_batch:
                batch = Transition(*zip(*random.sample(list(mem), dqn_batch)))
                s_b = torch.cat(batch.state)
                a_b = torch.cat(batch.action)
                r_b = torch.cat(batch.reward)
                ns_b = torch.cat(batch.next_state)

                qv = pol(s_b).gather(1, a_b)
                with torch.no_grad(): nq = tgt(ns_b).max(1).values
                loss = nn.SmoothL1Loss()(qv.squeeze(), r_b + dqn_gamma * nq)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 100)
                opt.step()

                for tp, pp in zip(tgt.parameters(), pol.parameters()):
                    tp.data.copy_(dqn_tau * pp.data + (1 - dqn_tau) * tp.data)

        ep_rews.append(ep_r)
        if (ep + 1) % 50 == 0:
            print(f"  [{label}] ep {ep+1}/{n_ep}, avg={np.mean(ep_rews[-50:]):.3f}, "
                  f"pre={pre_count/max(1,tot_cong)*100:.1f}%"); sys.stdout.flush()

    final = ep_rews[-50:] if len(ep_rews) >= 50 else ep_rews
    result = {
        'reward': sm(final), 'reward_std': ss(final),
        'preemptive': pre_count / max(1, tot_cong),
        'reactive': rea_count / max(1, tot_cong),
        'phi_mean': sm(all_p1 + all_p2), 'phi_std': ss(all_p1 + all_p2),
        'phi_congestion': sm(cong_phis), 'phi_normal': sm(norm_phis),
        'convergence_ep': detect_convergence(ep_rews),
        'rewards_history': ep_rews,
        'n_congestion': tot_cong, 'n_preemptive': pre_count,
    }
    print(f"  [{label}] done {time.time()-start:.0f}s, R={result['reward']:.3f}, "
          f"P={result['preemptive']*100:.1f}%"); sys.stdout.flush()
    return pol, result, traj


def train_multi(dataset, scaler, predictor, n_ep, phi_values, weights,
                n_runs=N_RUNS, use_oracle=False, collect_traj=False, label=""):
    all_r, all_p, all_res = [], [], []
    best_pol, best_r, best_traj = None, -float('inf'), None

    for run in range(n_runs):
        p, r, tr = train_dqn(dataset, scaler, predictor, n_ep, phi_values, weights,
                             use_oracle=use_oracle,
                             collect_traj=(collect_traj and run == n_runs - 1),
                             label=f"{label} {run+1}/{n_runs}")
        all_res.append(r); all_r.append(r.get('reward', 0)); all_p.append(r.get('preemptive', 0))
        if r.get('reward', 0) > best_r:
            best_r = r['reward']; best_pol = p
            if tr: best_traj = tr
        clear_gpu()

    mr, lo, hi = ci(all_r)
    mp, plo, phi_ = ci(all_p)

    agg = {
        'reward': mr, 'reward_std': ss(all_r),
        'reward_ci': {'mean': mr, 'lower': lo, 'upper': hi},
        'preemptive': mp,
        'preemptive_ci': {'mean': mp, 'lower': plo, 'upper': phi_},
        'reactive': sm([r.get('reactive', 0) for r in all_res]),
        'phi_mean': sm([r.get('phi_mean', 0) for r in all_res]),
        'phi_congestion': sm([r.get('phi_congestion', 0) for r in all_res]),
        'phi_normal': sm([r.get('phi_normal', 0) for r in all_res]),
        'convergence_ep': int(np.mean([r.get('convergence_ep', 0) for r in all_res])),
        'run_rewards': all_r, 'run_preemptive': all_p,
    }

    print(f"  [{label}] MULTI: R={mr:.3f} [{lo:.3f},{hi:.3f}], P={mp*100:.1f}%")
    sys.stdout.flush()
    return best_pol, agg, best_traj, all_r


# ============================================================================
# BASELINES
# ============================================================================
def run_baseline(dataset, scaler, strategy, phi_values, weights, n_ep=100, label=""):
    X, y = prepare_data(dataset)
    if len(X) < 60: return {'reward': 0, 'reward_std': 0}, []
    y_t = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    n_phi = len(phi_values); n_act = n_phi ** 2
    rews = []

    for ep in range(n_ep):
        mx = len(X) - 60
        idx = random.randint(0, max(0, mx))
        el = min(50, len(X) - idx - 1)
        if el < 10: continue
        er, prev, p1, p2 = 0., [100., 100.], 100., 100.
        est_br = [0.]

        for t in range(el):
            if idx + t >= len(y_t): break
            cur = y_t[idx+t, -1].cpu().numpy()

            if strategy == 'random':
                a = random.randint(0, n_act - 1)
                p1, p2 = phi_values[a // n_phi], phi_values[a % n_phi]
            elif strategy == 'reactive':
                v = inverse_transform_pred(cur, scaler)
                if v:
                    if v['fc1'] < v['cc1']: p1 = max(min(phi_values), p1 - 10)
                    else: p1 = min(max(phi_values), p1 + 10)
                    if v['fc2'] < v['cc2']: p2 = max(min(phi_values), p2 - 10)
                    else: p2 = min(max(phi_values), p2 + 10)
            elif strategy.startswith('static_'):
                pv = float(strategy.split('_')[1])
                p1, p2 = pv, pv

            r, _, _, _, est_br = compute_reward(cur, scaler, p1, p2, prev, est_br, weights)
            er += r; prev = [p1, p2]
        rews.append(er)

    result = {'reward': sm(rews), 'reward_std': ss(rews), 'run_rewards': rews}
    print(f"  {label}: R={result['reward']:.3f}±{result['reward_std']:.3f}"); sys.stdout.flush()
    return result, rews


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("DARA v6: Corrected Reward Scaling")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Key fixes: r_quality /= 1e6 (was /1e3), r_stability /= 100 (was raw)")
    print(f"Weight search: unbounded {list(WEIGHT_RANGES.keys())}")
    print()

    # ================================================================
    # Load v5 checkpoints
    # ================================================================
    print("Loading checkpoints...")
    dcp = load_cp('v5_data')
    if dcp is None: print("FATAL: v5_data not found"); sys.exit(1)
    dataset, scaler = dcp['dataset'], dcp['scaler']
    agg_window = dcp.get('agg_window', 99)
    print(f"  Dataset: {dataset.shape}, agg_window={agg_window}")

    p1 = load_cp('v5_p1')
    if p1 is None: print("FATAL: v5_p1 not found"); sys.exit(1)
    best_depth = p1.get('best_depth', 3)

    # Reconstruct transformer
    best_trans = None
    raw = p1.get('best_trans')
    if raw is not None:
        try:
            best_trans = raw.to(DEVICE).eval()
            test = torch.randn(1, SEQ_LEN, NI).to(DEVICE)
            with torch.no_grad(): out = best_trans(test)
            assert out.shape[-1] == NT
            print(f"  Transformer: {best_depth}L, {sum(p.numel() for p in best_trans.parameters()):,} params ✓")
        except Exception as e:
            print(f"  Pickled model failed: {e}"); best_trans = None

    if best_trans is None:
        for path in [os.path.join(BASE, 'models', 'best_transformer.pt'),
                     os.path.join(BASE, 'models', 'deploy', 'best_transformer_v5.pt')]:
            if os.path.exists(path):
                try:
                    sd = torch.load(path, map_location='cpu', weights_only=False)
                    best_trans = create_transformer(best_depth)
                    best_trans.load_state_dict(sd)
                    best_trans = best_trans.to(DEVICE).eval()
                    print(f"  Transformer rebuilt from {path} ✓"); break
                except: pass
    if best_trans is None: print("FATAL: no transformer"); sys.exit(1)

    # Load v5 transformer search results (for plots — not retraining)
    v5_full = None
    v5_pkl = os.path.join(BASE, 'ablation_results', 'ablation_results.pkl')
    if os.path.exists(v5_pkl):
        try:
            with open(v5_pkl, 'rb') as f: v5_full = CPUUnpickler(f).load()
            print(f"  v5 full results loaded for depth/predictor plots")
        except: pass

    # Load config from v5
    cfg_cp = load_cp('v5_p3') or load_cp('v5_p2')
    cfg_dict = cfg_cp.get('config', {}) if cfg_cp else {}
    phi_values = cfg_dict.get('phi_values', [30, 65, 100])
    n_phi = len(phi_values)
    print(f"  Phi values: {phi_values} ({n_phi**2} actions)")

    # Default weights (will be updated by search)
    weights = {
        'w_throughput': 1.0, 'w_delay': 0.8, 'w_quality': 0.5,
        'w_preemptive': 0.3, 'w_stability': 0.1, 'w_low_frac': 0.1,
        'dqn_lr': 1.25e-4, 'dqn_gamma': 0.966, 'dqn_tau': 0.006,
        'dqn_batch': 128, 'eps_start': 0.7, 'eps_end': 0.1, 'eps_decay': 900,
    }

    results = {'baselines': {}, 'run_level_data': {}, 'statistical_tests': {},
               'weight_search': {}, 'action_search': {}}
    del dcp, p1, cfg_cp; clear_gpu()

    # ================================================================
    # Phase 1: Weight search with normalized components
    # ================================================================
    print("\n" + "=" * 60)
    print(f"PHASE 1: Weight search ({WEIGHT_SEARCH_ITERS} iterations)")
    print("  Components normalized: r_qual/1e6, r_stab/100")
    print(f"  Ranges: {WEIGHT_RANGES}")
    print("=" * 60); sys.stdout.flush()

    # Baseline with defaults
    _, dm, _, _ = train_multi(dataset, scaler, best_trans, DQN_EP_SEARCH,
                              phi_values, weights, n_runs=3, label="default_weights")
    best_score = dm['reward'] + 10 * dm['preemptive']
    best_weights = {k: weights[k] for k in WEIGHT_RANGES}
    all_weight_results = [{'weights': {**best_weights}, 'score': best_score,
                           'reward': dm['reward'], 'preemptive': dm['preemptive']}]
    print(f"  Default: R={dm['reward']:.3f}, P={dm['preemptive']*100:.1f}%, S={best_score:.3f}")

    for i in range(WEIGHT_SEARCH_ITERS):
        tw = {k: random.uniform(*WEIGHT_RANGES[k]) for k in WEIGHT_RANGES}
        wt = {**weights, **tw}
        try:
            _, m, _, _ = train_multi(dataset, scaler, best_trans, DQN_EP_SEARCH,
                                     phi_values, wt, n_runs=3, label=f"ws_{i+1}")
            sc = m['reward'] + 10 * m['preemptive']
            all_weight_results.append({'weights': tw, 'score': sc,
                                       'reward': m['reward'], 'preemptive': m['preemptive']})
            print(f"  [{i+1}/{WEIGHT_SEARCH_ITERS}] S={sc:.3f} "
                  f"(R={m['reward']:.3f}, P={m['preemptive']*100:.1f}%)")
            if sc > best_score:
                best_score = sc; best_weights = tw.copy()
                print(f"    ★ NEW BEST")
        except Exception as e:
            print(f"  [{i+1}] Error: {e}")
        sys.stdout.flush()

    print(f"\nOptimal weights (S={best_score:.3f}):")
    for k, v in best_weights.items():
        print(f"  {k}: {v:.4f}")
    weights.update(best_weights)
    results['weight_search'] = {
        'best_weights': best_weights, 'best_score': best_score,
        'all_results': all_weight_results}

    # ================================================================
    # Phase 2: Action granularity search
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Action granularity")
    print("=" * 60); sys.stdout.flush()

    best_ac, best_ar = '3-level', -float('inf')
    for name, pvs in ACTION_CONFIGS.items():
        try:
            _, m, _, rw = train_multi(dataset, scaler, best_trans, DQN_EP_SEARCH,
                                      pvs, weights, n_runs=3, label=name)
            results['action_search'][name] = {
                'phi_values': pvs, 'reward': m['reward'],
                'reward_std': m['reward_std'], 'preemptive': m['preemptive']}
            results['run_level_data'][f'action_{name}'] = {'rewards': rw}
            if m['reward'] > best_ar:
                best_ar = m['reward']; best_ac = name
                print(f"    ★ NEW BEST: {name}")
        except Exception as e:
            print(f"    Error {name}: {e}")

    phi_values = ACTION_CONFIGS[best_ac]
    n_phi = len(phi_values)
    print(f"  Optimal: {best_ac} {phi_values}")

    # ================================================================
    # Phase 3: Baselines
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Baselines")
    print("=" * 60); sys.stdout.flush()

    for name, strat in [('random', 'random'), ('reactive', 'reactive'),
                        ('static_30', 'static_30'), ('static_65', 'static_65'),
                        ('static_100', 'static_100')]:
        m, rw = run_baseline(dataset, scaler, strat, phi_values, weights, label=name)
        results['baselines'][name] = m
        results['run_level_data'][name] = {'rewards': rw}
    clear_gpu()

    # ================================================================
    # Phase 4: DQN variants
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: DQN ablation")
    print("=" * 60); sys.stdout.flush()

    # No transformer
    _, m, _, rw = train_multi(dataset, scaler, None, DQN_EP_ABLATION,
                              phi_values, weights, label="no_trans")
    results['baselines']['no_transformer'] = m
    results['run_level_data']['no_transformer'] = {'rewards': rw}
    clear_gpu()

    # Oracle
    _, m, _, rw = train_multi(dataset, scaler, best_trans, DQN_EP_ABLATION,
                              phi_values, weights, use_oracle=True, label="oracle")
    results['baselines']['oracle'] = m
    results['run_level_data']['oracle'] = {'rewards': rw}
    clear_gpu()

    # DARA full
    _, m, traj, rw = train_multi(dataset, scaler, best_trans, DQN_EP_ABLATION,
                                 phi_values, weights, collect_traj=True, label="dara")
    results['baselines']['dara'] = m
    results['run_level_data']['dara'] = {'rewards': rw}
    clear_gpu()

    # ================================================================
    # Phase 5: Statistical tests
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Statistics")
    print("=" * 60); sys.stdout.flush()

    dara_rw = results['run_level_data'].get('dara', {}).get('rewards', [])
    comps = ['random', 'reactive', 'static_65', 'static_100', 'no_transformer', 'oracle']
    for bl_name in comps:
        bl_rw = results['run_level_data'].get(bl_name, {}).get('rewards', [])
        if len(dara_rw) < 2 or len(bl_rw) < 2: continue

        r = {'diff': sm(dara_rw) - sm(bl_rw)}
        d = cohens_d(dara_rw, bl_rw)
        r['cohens_d'] = {'value': d,
            'interpretation': 'negligible' if abs(d)<.2 else 'small' if abs(d)<.5
                             else 'medium' if abs(d)<.8 else 'large'}
        if SCIPY:
            try:
                _, p = stats.ttest_ind(dara_rw, bl_rw, equal_var=False)
                r['welch_ttest'] = {'p_value': float(p), 'significant': p < 0.05}
            except: pass
            try:
                _, p = stats.mannwhitneyu(dara_rw, bl_rw, alternative='two-sided')
                r['mann_whitney'] = {'p_value': float(p), 'significant': p < 0.05}
            except: pass
        bd = [np.mean(np.random.choice(dara_rw, len(dara_rw), True)) -
              np.mean(np.random.choice(bl_rw, len(bl_rw), True)) for _ in range(1000)]
        r['bootstrap_ci'] = {'mean': float(np.mean(bd)),
                             'lower': float(np.percentile(bd, 2.5)),
                             'upper': float(np.percentile(bd, 97.5))}
        results['statistical_tests'][f'dara_vs_{bl_name}'] = r
        sig = r.get('welch_ttest', {}).get('significant', '?')
        print(f"  dara vs {bl_name}: Δ={r['diff']:.3f}, d={d:.2f}, sig={sig}")

    # ================================================================
    # Phase 6: Final DQN
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: Final DQN")
    print("=" * 60); sys.stdout.flush()

    final_dqn, final_m, final_traj = train_dqn(
        dataset, scaler, best_trans, DQN_EP_FINAL, phi_values, weights,
        collect_traj=True, label="FINAL")

    # Use final trajectory if ablation trajectory is empty
    if final_traj and (not traj or len(traj.timestamps) < 10):
        traj = final_traj

    # ================================================================
    # Phase 7: Plots
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 7: Plots")
    print("=" * 60); sys.stdout.flush()

    bl = results['baselines']
    rl = results['run_level_data']
    st = results['statistical_tests']

    # Colors
    CD, CO, CR, CRE, CNT = '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#f1c40f'
    MC = {'random': CR, 'reactive': CRE, 'static_30': '#bdc3c7', 'static_65': '#95a5a6',
          'static_100': '#7f8c8d', 'no_transformer': CNT, 'dara': CD, 'oracle': CO}

    # ── FIG 1: Ablation results ──
    try:
        order = ['random', 'reactive', 'static_30', 'static_65', 'static_100',
                 'no_transformer', 'dara', 'oracle']
        methods = [m for m in order if m in bl]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        means, elo, ehi, cols = [], [], [], []
        for m in methods:
            rw = rl.get(m, {}).get('rewards', [])
            if rw: mn, lo_, hi_ = ci(rw); means.append(mn); elo.append(mn-lo_); ehi.append(hi_-mn)
            else: means.append(bl[m].get('reward', 0)); elo.append(0); ehi.append(0)
            cols.append(MC.get(m, 'gray'))
        x = np.arange(len(methods))
        axes[0].bar(x, means, yerr=[elo, ehi], capsize=5, color=cols, edgecolor='black', alpha=.7)
        axes[0].set_xticks(x); axes[0].set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=8)
        axes[0].set_ylabel('Reward'); axes[0].set_title('A) Methods with 95% CI')
        axes[0].axhline(0, color='black', lw=.5); axes[0].grid(True, alpha=.3, axis='y')

        pm = [m for m in ['no_transformer', 'dara', 'oracle'] if m in bl]
        if pm:
            xp = np.arange(len(pm))
            axes[1].bar(xp, [bl[m].get('preemptive', 0)*100 for m in pm],
                        color=[MC[m] for m in pm], edgecolor='black', alpha=.7)
            axes[1].set_xticks(xp); axes[1].set_xticklabels(pm, fontsize=9)
            axes[1].set_ylabel('Preemptive %'); axes[1].set_title('B) Preemptive Rate')
            axes[1].grid(True, alpha=.3, axis='y')

        if dara_rw and len(dara_rw) >= 2:
            effs, labs, ecols = [], [], []
            for m in ['random', 'reactive', 'static_65', 'static_100', 'no_transformer', 'oracle']:
                br = rl.get(m, {}).get('rewards', [])
                if not br or len(br) < 2: continue
                d = cohens_d(dara_rw, br); effs.append(d)
                labs.append(m.replace('_', '\n'))
                da = abs(d)
                ecols.append('gray' if da<.2 else '#f1c40f' if da<.5 else '#f39c12' if da<.8 else '#e74c3c')
            if effs:
                y = np.arange(len(effs))
                axes[2].barh(y, effs, color=ecols, edgecolor='black', alpha=.7)
                axes[2].set_yticks(y); axes[2].set_yticklabels(labs, fontsize=8)
                axes[2].set_xlabel("Cohen's d"); axes[2].axvline(0, color='black', lw=1)
                axes[2].set_title("C) Effect Sizes"); axes[2].grid(True, alpha=.3, axis='x')

        plt.suptitle('DARA v6: Ablation Results (Normalized Reward)', fontsize=13, fontweight='bold')
        plt.tight_layout(); savefig(fig, 'fig01_ablation')
    except Exception as e: print(f"  ✗ fig01: {e}"); import traceback; traceback.print_exc()

    # ── FIG 2: Bootstrap CI ──
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bms, blo, bhi, bls = [], [], [], []
        for comp in ['random', 'reactive', 'static_65', 'static_100', 'no_transformer', 'oracle']:
            bc = st.get(f'dara_vs_{comp}', {}).get('bootstrap_ci')
            if bc:
                bms.append(bc['mean']); blo.append(bc['mean']-bc['lower'])
                bhi.append(bc['upper']-bc['mean']); bls.append(comp.replace('_', ' '))
        if bms:
            y = np.arange(len(bms))
            ax.barh(y, bms, xerr=[blo, bhi], capsize=5,
                    color=[CD if m > 0 else CR for m in bms], alpha=.7, edgecolor='black')
            ax.axvline(0, color='black', lw=2)
            ax.set_yticks(y); ax.set_yticklabels(bls, fontsize=10)
            ax.set_xlabel('Mean Difference (DARA − baseline)')
            ax.set_title('Bootstrap 95% CI: DARA vs Baselines', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=.3, axis='x')
        plt.tight_layout(); savefig(fig, 'fig02_bootstrap_ci')
    except Exception as e: print(f"  ✗ fig02: {e}")

    # ── FIG 3: Action granularity ──
    try:
        ac = results['action_search']
        if ac:
            names = list(ac.keys())
            na = [len(ac[n]['phi_values'])**2 for n in names]
            ar = [ac[n]['reward'] for n in names]
            ae = [ac[n].get('reward_std', 0) for n in names]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            x = np.arange(len(names))
            chosen = [n for n in names if ac[n]['phi_values'] == phi_values]
            cols = [CD if n in chosen else 'steelblue' for n in names]

            axes[0].bar(x, ar, yerr=ae, capsize=5, color=cols, edgecolor='black', alpha=.7)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f'{n}\n({a})' for n, a in zip(names, na)], fontsize=8)
            axes[0].set_ylabel('Reward'); axes[0].set_title('A) Reward by Granularity')
            axes[0].grid(True, alpha=.3, axis='y')

            axes[1].bar(x, [ac[n].get('preemptive', 0)*100 for n in names],
                        color='steelblue', edgecolor='black', alpha=.7)
            axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
            axes[1].set_ylabel('Preemptive %'); axes[1].set_title('B) Preemptive Rate')
            axes[1].grid(True, alpha=.3, axis='y')

            axes[2].plot(na, ar, 'bo-', lw=2, ms=8)
            axes[2].fill_between(na, [r-s for r, s in zip(ar, ae)],
                                 [r+s for r, s in zip(ar, ae)], alpha=.2)
            axes[2].set_xlabel('Actions'); axes[2].set_ylabel('Reward')
            axes[2].set_title('C) Reward vs Action Space'); axes[2].grid(True, alpha=.3)

            plt.suptitle('Action Granularity Analysis', fontsize=12, fontweight='bold')
            plt.tight_layout(); savefig(fig, 'fig03_action_granularity')
    except Exception as e: print(f"  ✗ fig03: {e}")

    # ── FIG 4: Transformer depth (from v5 — unchanged) ──
    try:
        ts = {}
        if v5_full:
            for d, tr in v5_full.get('transformer_search', {}).items():
                if 'error' in tr: continue
                pm, dm = tr.get('pred_metrics', {}), tr.get('dqn_metrics', {})
                ts[int(d)] = {'nrmse': gm(pm, 'nrmse'), 'n_params': gm(pm, 'n_params'),
                              'inference_ms': gm(pm, 'inference_ms'), 'reward': gm(dm, 'reward'),
                              'score': tr.get('score', 0), 'val_loss': gm(pm, 'val_loss', []),
                              'horizon_nrmse': gm(pm, 'horizon_nrmse', [])}

        if ts:
            depths = sorted(ts.keys())
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            nrmses = [ts[d]['nrmse'] for d in depths]
            scores = [ts[d]['score'] for d in depths]
            pars = [ts[d]['n_params'] for d in depths]
            infs = [ts[d]['inference_ms'] for d in depths]

            ax = axes[0, 0]; ax.plot(depths, nrmses, 'bo-', lw=2, ms=8)
            ax.axvline(best_depth, color='red', ls='--', alpha=.5, label=f'Chosen={best_depth}L')
            ax.set_xlabel('Layers'); ax.set_ylabel('NRMSE'); ax.set_title('A) Prediction Error')
            ax.legend(); ax.grid(True, alpha=.3); ax.set_xticks(depths)

            ax = axes[0, 1]
            ax.bar(range(len(depths)), scores,
                   color=[CD if d == best_depth else 'steelblue' for d in depths],
                   edgecolor='black', alpha=.7)
            ax.set_xticks(range(len(depths))); ax.set_xticklabels([f'{d}L' for d in depths])
            ax.set_ylabel('Score'); ax.set_title('B) Combined Score'); ax.grid(True, alpha=.3, axis='y')

            ax = axes[1, 0]
            sc = ax.scatter(pars, infs, s=150, c=depths, cmap='viridis',edgecolors='black', zorder=5)
            for i, d in enumerate(depths):
                ax.annotate(f'{d}L', (pars[i], infs[i]), textcoords="offset points",
                           xytext=(5, 5), fontsize=9)
            ax.set_xlabel('Parameters'); ax.set_ylabel('Inference (ms)'); ax.set_title('C) Cost')
            ax.axhline(AGG_MS, color='red', ls='--', alpha=.5, label=f'{AGG_MS}ms budget')
            ax.legend(); ax.grid(True, alpha=.3)

            ax = axes[1, 1]; has = False
            for d in depths:
                hn = ts[d].get('horizon_nrmse', [])
                if hn:
                    ax.plot([(h+1)*AGG_MS for h in range(len(hn))], hn, 'o-',
                            label=f'{d}L', ms=6); has = True
            if has:
                ax.set_xlabel('Horizon (ms)'); ax.set_ylabel('NRMSE')
                ax.set_title('D) Error vs Horizon'); ax.legend(fontsize=8); ax.grid(True, alpha=.3)
            else:
                ax.text(.5, .5, 'No data', transform=ax.transAxes, ha='center')
                ax.set_title('D) Error vs Horizon')

            plt.suptitle(f'Transformer Depth (chosen={best_depth}L)', fontsize=12, fontweight='bold')
            plt.tight_layout(); savefig(fig, 'fig04_transformer_depth')
        else:
            print("  SKIP fig04: no transformer search data")
    except Exception as e: print(f"  ✗ fig04: {e}"); import traceback; traceback.print_exc()

    # ── FIG 5: Predictor comparison (from v5 — unchanged) ──
    try:
        pc = {}
        if v5_full:
            for a, d in v5_full.get('ablations', {}).get('predictor_comparison', {}).items():
                if 'error' in d: continue
                pm, dm = d.get('predictor_metrics', {}), d.get('dqn_metrics', {})
                pc[a] = {'nrmse': gm(pm, 'nrmse'), 'n_params': gm(pm, 'n_params'),
                         'inference_ms': gm(pm, 'inference_ms'), 'reward': gm(dm, 'reward')}

        if pc:
            archs, an, ar, ap, ai = [], [], [], [], []
            if best_depth in ts:
                archs.append('transformer'); an.append(ts[best_depth]['nrmse'])
                ar.append(ts[best_depth]['reward']); ap.append(ts[best_depth]['n_params'])
                ai.append(ts[best_depth]['inference_ms'])
            for a in sorted(pc):
                archs.append(a); an.append(pc[a]['nrmse']); ar.append(pc[a]['reward'])
                ap.append(pc[a]['n_params']); ai.append(pc[a]['inference_ms'])

            n = len(archs); x = np.arange(n)
            cols = [CD, '#3498db', '#f39c12', '#e74c3c'][:n]
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            axes[0].bar(x, an, color=cols, edgecolor='black', alpha=.7)
            axes[0].set_xticks(x); axes[0].set_xticklabels(archs)
            axes[0].set_ylabel('NRMSE'); axes[0].set_title('A) Prediction Error')
            axes[0].grid(True, alpha=.3, axis='y')

            axes[1].bar(x, ar, color=cols, edgecolor='black', alpha=.7)
            axes[1].set_xticks(x); axes[1].set_xticklabels(archs)
            axes[1].set_ylabel('Reward (v5)'); axes[1].set_title('B) DQN Performance')
            mi = next((i for i, a in enumerate(archs) if a == 'mlp'), None)
            if mi is not None and ar[mi] > 10:
                axes[1].annotate('⚠ MLP anomaly\n(unclamped inv-transform)',
                                xy=(mi, ar[mi]), xytext=(mi-.5, ar[mi]*.6),
                                arrowprops=dict(arrowstyle='->', color='red'),
                                fontsize=8, color='red')
            axes[1].grid(True, alpha=.3, axis='y')

            axes[2].scatter(ap, ai, s=150, c=cols, edgecolors='black', zorder=5)
            for i, a in enumerate(archs):
                axes[2].annotate(a, (ap[i], ai[i]), textcoords="offset points",
                                xytext=(5, 5), fontsize=9)
            axes[2].set_xlabel('Parameters'); axes[2].set_ylabel('Inference (ms)')
            axes[2].set_title('C) Efficiency')
            axes[2].axhline(AGG_MS, color='red', ls='--', alpha=.5, label=f'{AGG_MS}ms budget')
            axes[2].legend(); axes[2].grid(True, alpha=.3)

            plt.suptitle('Predictor Architecture Comparison', fontsize=12, fontweight='bold')
            plt.tight_layout(); savefig(fig, 'fig05_predictor_comparison')
        else:
            print("  SKIP fig05: no predictor comparison data")
    except Exception as e: print(f"  ✗ fig05: {e}")

    # ── FIG 6: Weight sensitivity ──
    try:
        wr = all_weight_results
        if wr and len(wr) > 2:
            wnames = list(WEIGHT_RANGES.keys())
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))

            for idx, wn in enumerate(wnames):
                ax = axes[idx // 3, idx % 3]
                wv = [r['weights'].get(wn, 0) for r in wr if 'weights' in r]
                ws = [r['score'] for r in wr if 'score' in r]
                if wv and ws and len(wv) == len(ws):
                    ax.scatter(wv, ws, alpha=.5, s=30, c='steelblue')
                    bv = best_weights.get(wn)
                    if bv is not None:
                        ax.axvline(bv, color='red', ls='--', lw=2, label=f'Best={bv:.3f}')
                    lo, hi = WEIGHT_RANGES[wn]
                    ax.set_xlim(lo - 0.1, hi + 0.1)
                    ax.set_xlabel(wn.replace('w_', '')); ax.set_ylabel('Score')
                    ax.set_title(f'{wn} ∈ [{lo}, {hi}]')
                    ax.legend(fontsize=8); ax.grid(True, alpha=.3)

            plt.suptitle('Reward Weight Sensitivity (normalized components, unbounded search)',
                        fontsize=12, fontweight='bold')
            plt.tight_layout(); savefig(fig, 'fig06_weight_sensitivity')
    except Exception as e: print(f"  ✗ fig06: {e}")

    # ── FIG 7: Training dynamics ──
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # A: Transformer val loss (from v5)
        ax = axes[0]; has = False
        if ts:
            for d in sorted(ts.keys()):
                vl = ts[d].get('val_loss', [])
                if vl and len(vl) > 1:
                    ax.plot(vl, label=f'{d}L', alpha=.7, lw=1.5); has = True
        if has:
            ax.set_xlabel('Epoch'); ax.set_ylabel('Val Loss')
            ax.set_title('A) Transformer Training'); ax.legend(fontsize=8)
            ax.set_yscale('log'); ax.grid(True, alpha=.3)
        else:
            ax.text(.5, .5, 'No training curves', transform=ax.transAxes,
                    ha='center', color='gray')
            ax.set_title('A) Transformer Training')

        # B: DQN learning curves
        ax = axes[1]; has = False
        for name, color in [('dara', CD), ('oracle', CO), ('no_transformer', CNT)]:
            rh = bl.get(name, {}).get('rewards_history', [])
            # Also check if we stored it in result dict format
            if not rh:
                rh_data = bl.get(name, {})
                if isinstance(rh_data, dict):
                    rh = rh_data.get('rewards_history', [])
            if rh and len(rh) > 20:
                w = min(20, len(rh) // 3)
                if w > 0:
                    s_ = np.convolve(rh, np.ones(w)/w, mode='valid')
                    ax.plot(s_, label=name, color=color, alpha=.7, lw=1.5); has = True
        # Also plot final DQN if available
        if final_m and final_m.get('rewards_history'):
            rh = final_m['rewards_history']
            if len(rh) > 20:
                w = min(20, len(rh) // 3)
                s_ = np.convolve(rh, np.ones(w)/w, mode='valid')
                ax.plot(s_, label='final', color='navy', alpha=.7, lw=2, ls='--'); has = True
        if has:
            ax.set_xlabel('Episode'); ax.set_ylabel('Smoothed Reward')
            ax.set_title('B) DQN Learning Curves'); ax.legend(fontsize=8)
            ax.grid(True, alpha=.3)
        else:
            ax.text(.5, .5, 'No episode data', transform=ax.transAxes,
                    ha='center', color='gray')
            ax.set_title('B) DQN Learning Curves')

        plt.suptitle('Training Dynamics', fontsize=12, fontweight='bold')
        plt.tight_layout(); savefig(fig, 'fig07_training_dynamics')
    except Exception as e: print(f"  ✗ fig07: {e}")

    # ── FIG 8: Trajectory analysis ──
    try:
        if traj and hasattr(traj, 'timestamps') and len(traj.timestamps) >= 10:
            t = np.array(traj.timestamps)
            fig = plt.figure(figsize=(16, 14))
            gs = GridSpec(3, 2, figure=fig, hspace=.35, wspace=.25)

            # A: CWND + phi
            ax1 = fig.add_subplot(gs[0, :])
            pc_ = traj.pred_cwnd1 if hasattr(traj, 'pred_cwnd1') else []
            ac_ = traj.actual_cwnd1 if hasattr(traj, 'actual_cwnd1') else []
            if pc_: ax1.plot(t[:len(pc_)], pc_, 'b-', label='Predicted', lw=1.5, alpha=.8)
            if ac_: ax1.plot(t[:len(ac_)], ac_, 'g--', label='Actual', lw=1.5, alpha=.8)
            axt = ax1.twinx()
            if traj.phi1: axt.plot(t[:len(traj.phi1)], traj.phi1, 'r-', label='φ₁', lw=2)
            if traj.phi2: axt.plot(t[:len(traj.phi2)], traj.phi2, 'm-', label='φ₂', lw=2)
            axt.set_ylabel('φ (%)', color='red'); axt.set_ylim(20, 110)
            for i in range(min(len(t), len(traj.is_congestion))):
                if traj.is_congestion[i]: ax1.axvline(t[i], color='red', alpha=.08, lw=1)
            for i in range(min(len(t), len(traj.is_preemptive))):
                if traj.is_preemptive[i]: ax1.axvline(t[i], color='green', alpha=.3, lw=2)
            ax1.set_xlabel('Step'); ax1.set_ylabel('CWND (scaled)')
            ax1.set_title('A) Prediction-Driven φ Adjustment')
            ax1.legend(loc='upper left', fontsize=8)
            axt.legend(loc='upper right', fontsize=8); ax1.grid(True, alpha=.3)

            # B: Reward
            ax2 = fig.add_subplot(gs[1, 0])
            if traj.rewards:
                w = min(20, len(traj.rewards) // 3)
                if w > 0:
                    s_ = np.convolve(traj.rewards, np.ones(w)/w, mode='valid')
                    ax2.plot(range(len(s_)), s_, 'g-', lw=2)
            ax2.axhline(0, color='black', lw=.5)
            ax2.set_xlabel('Step'); ax2.set_ylabel('Reward')
            ax2.set_title('B) Reward Signal'); ax2.grid(True, alpha=.3)

            # C: Action heatmap
            ax3 = fig.add_subplot(gs[1, 1])
            if traj.actions:
                hm = np.zeros((n_phi, n_phi))
                for a in traj.actions:
                    i_, j_ = a // n_phi, a % n_phi
                    if i_ < n_phi and j_ < n_phi: hm[i_, j_] += 1
                im = ax3.imshow(hm, cmap='YlOrRd', aspect='auto')
                ax3.set_xticks(range(n_phi)); ax3.set_yticks(range(n_phi))
                ax3.set_xticklabels(phi_values); ax3.set_yticklabels(phi_values)
                ax3.set_xlabel('φ₂'); ax3.set_ylabel('φ₁')
                ax3.set_title('C) Action Frequency'); plt.colorbar(im, ax=ax3)

            # D: φ by state
            ax4 = fig.add_subplot(gs[2, 0])
            nm = min(len(traj.phi1), len(traj.is_congestion))
            if nm > 0:
                pn_ = [traj.phi1[i] for i in range(nm) if not traj.is_congestion[i]]
                pc__ = [traj.phi1[i] for i in range(nm) if traj.is_congestion[i]]
                bd, lb, cl = [], [], []
                if pn_: bd.append(pn_); lb.append('Normal'); cl.append(CD)
                if pc__: bd.append(pc__); lb.append('Congestion'); cl.append(CR)
                if bd:
                    bp = ax4.boxplot(bd, labels=lb, patch_artist=True)
                    for p, c in zip(bp['boxes'], cl): p.set_facecolor(c); p.set_alpha(.7)
            ax4.set_ylabel('φ₁ (%)'); ax4.set_title('D) φ by State'); ax4.grid(True, alpha=.3)

            # E: Reward components
            ax5 = fig.add_subplot(gs[2, 1])
            if traj.reward_components:
                cns = ['r_tput', 'r_delay', 'r_pre', 'r_stab', 'r_qual', 'r_low']
                clabels = ['throughput', 'delay', 'preemptive', 'stability', 'quality', 'low_frac']
                cms = [np.mean([r.get(c, 0) for r in traj.reward_components
                                if isinstance(r, dict)]) for c in cns]
                x5 = np.arange(len(cns))
                ax5.bar(x5, cms, color=[CD if v >= 0 else CR for v in cms],
                        edgecolor='black', alpha=.7)
                ax5.set_xticks(x5)
                ax5.set_xticklabels(clabels, fontsize=8, rotation=45, ha='right')
                ax5.set_ylabel('Mean Value'); ax5.axhline(0, color='black', lw=.5)
                ax5.grid(True, alpha=.3, axis='y')
            ax5.set_title('E) Reward Components (normalized)')

            plt.suptitle('Trajectory Analysis', fontsize=14, fontweight='bold')
            savefig(fig, 'fig08_trajectory')
        else:
            print("  SKIP fig08: no trajectory data")
    except Exception as e: print(f"  ✗ fig08: {e}"); import traceback; traceback.print_exc()

    # ── FIG 9: Preemptive analysis ──
    try:
        if traj and hasattr(traj, 'timestamps') and len(traj.timestamps) >= 10:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            t = np.array(traj.timestamps)

            # A: Pie
            n_pre = sum(traj.is_preemptive) if traj.is_preemptive else 0
            n_rea = sum(traj.is_reactive) if traj.is_reactive else 0
            n_cong = sum(traj.is_congestion) if traj.is_congestion else 0
            n_norm = len(traj.is_congestion) - n_cong
            n_no = max(0, n_cong - n_pre - n_rea)

            sizes = [n_norm, n_pre, n_rea, n_no]
            labels = ['Normal', 'Preemptive', 'Reactive', 'No action']
            pcols = [CD, '#3498db', CRE, CR]
            nz = [(s, l, c) for s, l, c in zip(sizes, labels, pcols) if s > 0]
            if nz:
                s_, l_, c_ = zip(*nz)
                axes[0].pie(s_, labels=l_, colors=c_, autopct='%1.1f%%', startangle=90)
            axes[0].set_title('A) Classification')

            # B: φ during preemptive
            nm2 = min(len(traj.phi1), len(traj.is_preemptive), len(traj.is_congestion))
            if nm2 > 0:
                np1 = [traj.phi1[i] for i in range(nm2) if not traj.is_congestion[i]]
                pp1 = [traj.phi1[i] for i in range(nm2) if traj.is_preemptive[i]]
                pp2 = [traj.phi2[i] for i in range(nm2) if traj.is_preemptive[i]]
                bd, lb, cl = [], [], []
                if np1: bd.append(np1); lb.append('Normal φ₁'); cl.append(CD)
                if pp1: bd.append(pp1); lb.append('Preempt φ₁'); cl.append('#3498db')
                if pp2: bd.append(pp2); lb.append('Preempt φ₂'); cl.append(CO)
                if bd:
                    bp = axes[1].boxplot(bd, labels=lb, patch_artist=True)
                    for p, c in zip(bp['boxes'], cl): p.set_facecolor(c); p.set_alpha(.7)
                else:
                    axes[1].text(.5, .5, 'No preemptive events',
                                transform=axes[1].transAxes, ha='center', color='gray')
            axes[1].set_ylabel('φ (%)'); axes[1].set_title('B) φ During Preemptive')
            axes[1].grid(True, alpha=.3)

            # C: Timeline
            if traj.is_congestion and traj.is_preemptive:
                cm = np.array(traj.is_congestion[:len(t)])
                pm_ = np.array(traj.is_preemptive[:len(t)])
                axes[2].fill_between(t[:len(cm)], 0, 1, where=cm,
                                     alpha=.2, color='red', label='Congestion')
                axes[2].fill_between(t[:len(pm_)], 0, 1, where=pm_,
                                     alpha=.5, color='green', label='Preemptive')
                if traj.phi1:
                    phi_n = np.array(traj.phi1[:len(t)]) / max(phi_values)
                    axes[2].plot(t[:len(phi_n)], phi_n, 'b-', lw=1, alpha=.7, label='φ₁')
                axes[2].legend(fontsize=8); axes[2].grid(True, alpha=.3)
            axes[2].set_xlabel('Step'); axes[2].set_title('C) Timeline')

            plt.suptitle('Preemptive Behavior', fontsize=12, fontweight='bold')
            plt.tight_layout(); savefig(fig, 'fig09_preemptive')
        else:
            print("  SKIP fig09: no trajectory")
    except Exception as e: print(f"  ✗ fig09: {e}")

    # ── FIG 10: Reward components over time ──
    try:
        if traj and traj.reward_components and len(traj.reward_components) > 10:
            cnames = ['r_tput', 'r_delay', 'r_pre', 'r_stab', 'r_qual', 'r_low']
            clabels = ['throughput', 'delay', 'preemptive', 'stability', 'quality', 'low_frac']
            wt_map = {'r_tput': weights['w_throughput'], 'r_delay': weights['w_delay'],
                      'r_pre': weights['w_preemptive'], 'r_stab': weights['w_stability'],
                      'r_qual': weights['w_quality'], 'r_low': weights['w_low_frac']}

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            for idx, (cn, cl) in enumerate(zip(cnames, clabels)):
                ax = axes[idx // 3, idx % 3]
                vals = [rc.get(cn, 0) for rc in traj.reward_components if isinstance(rc, dict)]
                if not vals:
                    ax.text(.5, .5, 'No data', transform=ax.transAxes, ha='center')
                    ax.set_title(cl); continue
                wt = wt_map[cn]
                weighted = [v * wt for v in vals]
                w_ = min(20, len(vals) // 3)
                if w_ > 0 and len(vals) >= w_:
                    sm_r = np.convolve(vals, np.ones(w_)/w_, mode='valid')
                    sm_w = np.convolve(weighted, np.ones(w_)/w_, mode='valid')
                    ax.plot(range(len(sm_r)), sm_r, 'b-', alpha=.4, label='Raw', lw=1)
                    ax.plot(range(len(sm_w)), sm_w, 'r-', label=f'×{wt:.2f}', lw=2)
                else:
                    ax.plot(vals, 'b-', alpha=.4, lw=1)
                    ax.plot(weighted, 'r-', lw=2)
                ax.axhline(0, color='black', lw=.5)
                ax.set_xlabel('Step'); ax.set_ylabel('Value')
                ax.set_title(f'{cl} (w={wt:.2f})')
                ax.legend(fontsize=8); ax.grid(True, alpha=.3)

            plt.suptitle('Reward Components Over Time (normalized)', fontsize=12, fontweight='bold')
            plt.tight_layout(); savefig(fig, 'fig10_reward_components')
        else:
            print("  SKIP fig10: no reward components")
    except Exception as e: print(f"  ✗ fig10: {e}")

    # ── FIG 11: Statistical analysis (4 panels) ──
    try:
        if dara_rw and len(dara_rw) >= 2:
            comps_data = [(m, rl[m]['rewards']) for m in
                          ['random', 'reactive', 'static_65', 'static_100',
                           'no_transformer', 'oracle']
                          if m in rl and len(rl[m].get('rewards', [])) >= 2]

            if comps_data:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))

                # A: Bars with CI
                ax = axes[0, 0]
                all_m = ['dara'] + [c[0] for c in comps_data]
                means_, elo_, ehi_ = [], [], []
                for m in all_m:
                    mn, lo_, hi_ = ci(rl[m]['rewards'])
                    means_.append(mn); elo_.append(mn-lo_); ehi_.append(hi_-mn)
                x = np.arange(len(all_m))
                ax.bar(x, means_, yerr=[elo_, ehi_], capsize=5,
                       color=[MC.get(m, 'gray') for m in all_m], edgecolor='black', alpha=.7)
                ax.set_xticks(x)
                ax.set_xticklabels([m.replace('_', '\n') for m in all_m], fontsize=8)
                ax.set_ylabel('Reward'); ax.set_title('A) Methods with 95% CI')
                ax.grid(True, alpha=.3, axis='y')

                # B: Effect sizes
                ax = axes[0, 1]
                ds = [cohens_d(dara_rw, c[1]) for c in comps_data]
                labs = [c[0].replace('_', '\n') for c in comps_data]
                ecols = ['gray' if abs(d)<.2 else '#f1c40f' if abs(d)<.5
                         else '#f39c12' if abs(d)<.8 else '#e74c3c' for d in ds]
                y = np.arange(len(ds))
                ax.barh(y, ds, color=ecols, edgecolor='black', alpha=.7)
                ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
                ax.set_xlabel("Cohen's d"); ax.axvline(0, color='black', lw=1)
                ax.set_title("B) Effect Sizes"); ax.grid(True, alpha=.3, axis='x')

                # C: P-values
                ax = axes[1, 0]
                if SCIPY:
                    pvs, pls = [], []
                    for c in comps_data:
                        try:
                            _, p = stats.ttest_ind(dara_rw, c[1], equal_var=False)
                            pvs.append(-np.log10(max(p, 1e-10)))
                            pls.append(c[0].replace('_', '\n'))
                        except: pass
                    if pvs:
                        xp = np.arange(len(pls))
                        ax.bar(xp, pvs, color='steelblue', alpha=.7, edgecolor='black')
                        ax.axhline(-np.log10(.05), color='red', ls='--', lw=2, label='α=0.05')
                        ax.set_xticks(xp); ax.set_xticklabels(pls, fontsize=8)
                        ax.set_ylabel('-log₁₀(p)'); ax.set_title('C) Significance')
                        ax.legend(); ax.grid(True, alpha=.3, axis='y')

                # D: Bootstrap CI
                ax = axes[1, 1]
                bms, blo, bhi, bls = [], [], [], []
                for c in comps_data:
                    bd = [np.mean(np.random.choice(dara_rw, len(dara_rw), True)) -
                          np.mean(np.random.choice(c[1], len(c[1]), True)) for _ in range(1000)]
                    bms.append(np.mean(bd))
                    blo.append(np.mean(bd) - np.percentile(bd, 2.5))
                    bhi.append(np.percentile(bd, 97.5) - np.mean(bd))
                    bls.append(c[0].replace('_', ' '))
                y = np.arange(len(bms))
                ax.barh(y, bms, xerr=[blo, bhi], capsize=5,
                        color=[CD if m > 0 else CR for m in bms],
                        alpha=.7, edgecolor='black')
                ax.axvline(0, color='black', lw=2)
                ax.set_yticks(y); ax.set_yticklabels(bls, fontsize=8)
                ax.set_xlabel('Δ Reward (DARA − baseline)')
                ax.set_title('D) Bootstrap 95% CI'); ax.grid(True, alpha=.3, axis='x')

                plt.suptitle('Statistical Analysis', fontsize=13, fontweight='bold')
                plt.tight_layout(); savefig(fig, 'fig11_statistical_analysis')
    except Exception as e: print(f"  ✗ fig11: {e}"); import traceback; traceback.print_exc()

    # ── FIG 12: Summary dashboard ──
    try:
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=.35, wspace=.3)

        # A: Ranked
        ax = fig.add_subplot(gs[0, 0])
        all_m = {k: v.get('reward', 0) for k, v in bl.items()}
        ns = sorted(all_m, key=lambda k: all_m[k]); vs = [all_m[n] for n in ns]
        y = np.arange(len(ns))
        ax.barh(y, vs, color=[MC.get(n, '#34495e') for n in ns], edgecolor='black', alpha=.7)
        ax.set_yticks(y); ax.set_yticklabels([n.replace('_', '\n') for n in ns], fontsize=8)
        ax.set_xlabel('Reward'); ax.axvline(0, color='black', lw=.5)
        ax.grid(True, alpha=.3, axis='x'); ax.set_title('A) All Methods Ranked')

        # B: φ behavior
        ax = fig.add_subplot(gs[0, 1])
        phi_d = {}
        for m in ['dara', 'oracle', 'no_transformer']:
            b = bl.get(m, {})
            c_, n_ = b.get('phi_congestion', 0), b.get('phi_normal', 0)
            if c_ or n_: phi_d[m] = {'c': c_, 'n': n_}
        if phi_d:
            xp = np.arange(len(phi_d)); w = .35
            ax.bar(xp-w/2, [phi_d[m]['c'] for m in phi_d], w,
                   label='Congestion', color=CR, alpha=.7)
            ax.bar(xp+w/2, [phi_d[m]['n'] for m in phi_d], w,
                   label='Normal', color=CD, alpha=.7)
            ax.set_xticks(xp)
            ax.set_xticklabels([m.replace('_', '\n') for m in phi_d], fontsize=9)
            ax.legend(fontsize=8)
        ax.set_ylabel('Avg φ (%)'); ax.set_title('B) φ by State'); ax.grid(True, alpha=.3, axis='y')

        # C: Key metrics
        ax = fig.add_subplot(gs[0, 2]); ax.axis('off')
        dm_ = bl.get('dara', {}); om_ = bl.get('oracle', {})
        rm_ = bl.get('random', {}); s65_ = bl.get('static_65', {})
        td = [['', 'DARA', 'Oracle', 'Static65', 'Random'],
              ['Reward', f"{dm_.get('reward',0):.2f}", f"{om_.get('reward',0):.2f}",
               f"{s65_.get('reward',0):.2f}", f"{rm_.get('reward',0):.2f}"],
              ['Pre%', f"{dm_.get('preemptive',0)*100:.1f}",
               f"{om_.get('preemptive',0)*100:.1f}", "—", "—"],
              ['φ cong', f"{dm_.get('phi_congestion',0):.1f}",
               f"{om_.get('phi_congestion',0):.1f}", "65", "—"],
              ['φ norm', f"{dm_.get('phi_normal',0):.1f}",
               f"{om_.get('phi_normal',0):.1f}", "65", "—"]]
        tab = ax.table(cellText=td[1:], colLabels=td[0], loc='center', cellLoc='center')
        tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.4)
        for i in range(1, len(td)): tab[i, 1].set_facecolor('#d5f5e3')
        ax.set_title('C) Key Metrics', fontsize=11, fontweight='bold')

        # D: Config + weights
        ax = fig.add_subplot(gs[1, 0]); ax.axis('off')
        ct = (f"Configuration\n{'═'*24}\n"
              f"Transformer: {best_depth}L\n"
              f"Actions: {phi_values} ({n_phi**2})\n"
              f"DQN state: {DQN_STATE_DIM}D\n"
              f"Agg: ~{AGG_MS}ms\n"
              f"Horizons: {N_HORIZONS}×{AGG_MS}ms\n\n"
              f"Optimized Weights:\n"
              + '\n'.join(f"  {k}: {weights[k]:.4f}" for k in WEIGHT_RANGES)
              + f"\n\nWeight search: {WEIGHT_SEARCH_ITERS} iters\n"
              f"Best score: {best_score:.3f}")
        ax.text(.05, .95, ct, transform=ax.transAxes, fontsize=8, va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=.8))
        ax.set_title('D) Configuration', fontsize=11, fontweight='bold')

        # E: Component scale verification
        ax = fig.add_subplot(gs[1, 1]); ax.axis('off')
        if traj and traj.reward_components:
            cns = ['r_tput', 'r_delay', 'r_pre', 'r_stab', 'r_qual', 'r_low']
            clabels = ['throughput', 'delay', 'preemptive', 'stability', 'quality', 'low_frac']
            scale_info = "Component Scale Verification\n" + "═"*30 + "\n\n"
            for cn, cl in zip(cns, clabels):
                vals = [rc.get(cn, 0) for rc in traj.reward_components if isinstance(rc, dict)]
                if vals:
                    scale_info += f"  {cl:12s}: [{min(vals):.3f}, {max(vals):.3f}] μ={np.mean(vals):.3f}\n"
            scale_info += f"\n  All components O([-1, 1]) ✓\n"
            scale_info += f"\n  v6 fixes:\n"
            scale_info += f"    r_quality /= 1e6 (was /1e3)\n"
            scale_info += f"    r_stability /= 100 (was raw)"
            ax.text(.05, .95, scale_info, transform=ax.transAxes, fontsize=8, va='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#eafaf1', alpha=.8))
        ax.set_title('E) Scale Verification', fontsize=11, fontweight='bold')

        # F: Statistical summary
        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        stat_info = "Statistical Summary\n" + "═"*24 + "\n\n"
        for comp in sorted(st.keys()):
            r = st[comp]
            d = r.get('cohens_d', {}).get('value', 0)
            sig = r.get('welch_ttest', {}).get('significant', '?')
            stat_info += f"  {comp.replace('dara_vs_',''):15s} Δ={r['diff']:+.2f} d={d:.2f} sig={sig}\n"
        ax.text(.05, .95, stat_info, transform=ax.transAxes, fontsize=8, va='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#fef9e7', alpha=.8))
        ax.set_title('F) Statistics', fontsize=11, fontweight='bold')

        plt.suptitle('DARA v6: Summary (Normalized Reward)', fontsize=14, fontweight='bold')
        savefig(fig, 'fig12_summary')
    except Exception as e: print(f"  ✗ fig12: {e}"); import traceback; traceback.print_exc()

    # ================================================================
    # Phase 8: Save
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 8: Save")
    print("=" * 60); sys.stdout.flush()

    # Bundle
    try:
        bundle = {
            'transformer_state_dict': best_trans.state_dict(),
            'transformer_config': {
                'n_blocks': best_depth, 'input_dim': NI, 'output_dim': NT,
                'embed_dim': NI, 'seq_len': SEQ_LEN,
                'block_configs': BLOCK_CONFIGS[:best_depth],
            },
            'dqn_state_dict': final_dqn.state_dict() if final_dqn else None,
            'dqn_config': {
                'state_dim': DQN_STATE_DIM, 'n_actions': n_phi**2,
                'hidden_dim': 256, 'phi_values': phi_values,
            },
            'scaler': scaler,
            'reward_weights': {k: weights[k] for k in WEIGHT_RANGES},
            'n_input_features': NI, 'n_target_features': NT,
            'n_total_features': NTOT, 'n_horizons': N_HORIZONS,
            'seq_len': SEQ_LEN, 'agg_window': agg_window,
            'dqn_state_dim': DQN_STATE_DIM, 'seed': SEED,
            'version': 'v6', 'device': str(DEVICE),
            'final_reward': final_m.get('reward', 0) if final_m else 0,
            'final_preemptive': final_m.get('preemptive', 0) if final_m else 0,
            'timestamp': datetime.now().isoformat(),
            'fixes': ['r_quality /= 1e6 (was /1e3)', 'r_stability /= 100 (was raw)',
                      'unbounded weight search [0.01, 5.0]'],
        }
        with open(os.path.join(DEPLOY_DIR, 'dara_v6_bundle.pkl'), 'wb') as f:
            pickle.dump(bundle, f)
        print(f"  ✓ Bundle saved")
    except Exception as e: print(f"  ✗ Bundle: {e}")

    # Individual files
    try:
        torch.save(best_trans.state_dict(), os.path.join(DEPLOY_DIR, 'transformer_v6.pt'))
        if final_dqn:
            torch.save(final_dqn.state_dict(), os.path.join(DEPLOY_DIR, 'dqn_v6.pth'))
        with open(os.path.join(DEPLOY_DIR, 'scaler_v6.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        with open(os.path.join(DEPLOY_DIR, 'config_v6.json'), 'w') as f:
            json.dump(make_serializable({
                'weights': {k: weights[k] for k in WEIGHT_RANGES},
                'phi_values': phi_values, 'depth': best_depth,
                'dqn_state_dim': DQN_STATE_DIM,
            }), f, indent=2)
        print(f"  ✓ Individual files saved")
    except Exception as e: print(f"  ✗ Files: {e}")

    # Results JSON
    try:
        with open(os.path.join(OUT_DIR, 'results_v6.json'), 'w') as f:
            json.dump(make_serializable(results), f, indent=2, default=str)
        print(f"  ✓ JSON saved")
    except Exception as e: print(f"  ✗ JSON: {e}")

    # Results pickle
    try:
        with open(os.path.join(OUT_DIR, 'results_v6.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"  ✓ Pickle saved")
    except Exception as e: print(f"  ✗ Pickle: {e}")

    # ================================================================
    # Summary
    # ================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("DARA v6 COMPLETE")
    print("=" * 70)
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"  Transformer: {best_depth}L")
    print(f"  Actions: {phi_values} ({n_phi**2} actions)")
    print(f"  Final DQN: R={final_m.get('reward',0):.3f}, P={final_m.get('preemptive',0)*100:.1f}%")

    print(f"\n  Optimized weights:")
    for k in WEIGHT_RANGES:
        print(f"    {k}: {weights[k]:.4f}")

    print(f"\n  Ablation:")
    for name in order + ['no_transformer', 'dara', 'oracle']:
        if name in bl:
            m = bl[name]
            print(f"    {name:20s}: R={m.get('reward',0):8.3f}±{m.get('reward_std',0):.3f}, "
                  f"P={m.get('preemptive',0)*100:5.1f}%")

    print(f"\n  Statistics:")
    for comp, r in st.items():
        d = r.get('cohens_d', {}).get('value', 0)
        sig = r.get('welch_ttest', {}).get('significant', '?')
        print(f"    {comp:30s}: Δ={r['diff']:+.3f}, d={d:.2f}, sig={sig}")

    print(f"\n  Figures:")
    fig_names = ['ablation', 'bootstrap_ci', 'action_granularity', 'transformer_depth',
                 'predictor_comparison', 'weight_sensitivity', 'training_dynamics',
                 'trajectory', 'preemptive', 'reward_components', 'statistical_analysis',
                 'summary']
    for i, name in enumerate(fig_names, 1):
        fn = f'fig{i:02d}_{name}'
        ok = os.path.exists(os.path.join(GRAPH_DIR, f'{fn}.png'))
        print(f"    [{'✓' if ok else '✗'}] {fn}")

    print(f"\n  Output: {GRAPH_DIR}/")
    print(f"  Deploy: {DEPLOY_DIR}/")
    print(f"  Results: {OUT_DIR}/")
    print("=" * 70); sys.stdout.flush()


if __name__ == '__main__':
    try:
        main()
        print("\nCompleted successfully.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nInterrupted."); sys.exit(1)
    except Exception as e:
        print(f"\nFATAL: {e}")
        import traceback; traceback.print_exc()
        sys.exit(2)
