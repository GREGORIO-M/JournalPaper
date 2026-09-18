#!/usr/bin/env python3
"""
DARA v6 Scheduler

Based on the real live scheduler (doc3) with the following v6 updates:
  1. Scaler    : deploy_v6/scaler_v6.pkl         (was scalerLengthOptimal.pkl)
  2. Transformer: deploy_v6/transformer_v6.pt     (was best_model_optimal.torch)
     Uses simplified v6 TransformerBlock architecture (not 7-block NAS)
     Depth loaded from config_v6.json
  3. DQN       : deploy_v6/dqn_v6.pth            (was dara_optimized_model.pth)
     18-dimensional state via build_state_v6()   (was 4-dim observation)
  4. r_quality /= 1e6                             (was /1e3 — 1000x too large)
  5. r_stability /= 100                           (was raw Δφ — 100x too large)
  6. phi_values and reward weights from config_v6.json
  7. CSV, device paths and logging unchanged from original scheduler
"""

import os, io, pickle, json, csv, time, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque, namedtuple

# ============================================================================
# PATHS
# ============================================================================
DEPLOY_DIR  = '/root/Desktop/namespace-net-emu/mp-dccp0'

SCALER_PATH      = os.path.join(DEPLOY_DIR, 'scaler_v6.pkl')
TRANSFORMER_PATH = os.path.join(DEPLOY_DIR, 'transformer_v6.pt')
DQN_PATH         = os.path.join(DEPLOY_DIR, 'dqn_v6.pth')
CONFIG_PATH      = os.path.join(DEPLOY_DIR, 'config_v6.json')

DEVICE_PATH = "/dev/mpdccp_acpf_data"
OUTPUT_FILE = "/root/Desktop/namespace-net-emu/mp-dccp0/Scripts/subflow_data.csv"
LOG_FILE    = "/root/Desktop/namespace-net-emu/mp-dccp0/Scripts/dara_predictions.csv"

# ============================================================================
# CONSTANTS (must match v6 training)
# ============================================================================
NI, NT, NTOT  = 90, 20, 110
SEQ_LEN       = 8
DQN_STATE_DIM = 18      # v6: 18-dimensional state
SEED          = 42

BLOCK_CONFIGS = [
    (360, 5, 0.10), (360, 5, 0.10), (360, 5, 0.15),
    (360, 5, 0.15), (360, 5, 0.20), (360, 5, 0.20),
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ============================================================================
# LOGGING  (unchanged from original scheduler)
# ============================================================================

def init_log():
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'pred_cwnd1', 'pred_cwnd2', 'pred_rtt1', 'pred_rtt2',
            'obs_cwnd1',  'obs_cwnd2',  'obs_rtt1',  'obs_rtt2',
            'phi1', 'phi2', 'reward'
        ])

def log_step(gamma_p1, gamma_p2, delta_p1, delta_p2,
             current_cwnd1, current_cwnd2, current_srtt1, current_srtt2,
             phi1, phi2, reward):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            gamma_p1, gamma_p2, delta_p1, delta_p2,
            current_cwnd1, current_cwnd2, current_srtt1, current_srtt2,
            phi1, phi2, reward
        ])

# ============================================================================
# V6 NETWORK DEFINITIONS
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.attn = nn.MultiheadAttention(dim, max(1, heads),
                                          dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout),    nn.Linear(ff_dim, dim)
        )
        self.n1   = nn.LayerNorm(dim)
        self.n2   = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x, x, x)[0]))
        return self.n2(x + self.drop(self.ff(x)))


def create_transformer(n_blocks, dim=90, out_dim=NT):
    """Simplified v6 transformer (replaces 7-block NAS model)."""
    n_blocks = min(n_blocks, len(BLOCK_CONFIGS))
    blocks   = [TransformerBlock(dim, h, ff, d)
                for ff, h, d in BLOCK_CONFIGS[:n_blocks]]
    return nn.Sequential(
        nn.Linear(dim, dim, bias=False),
        *blocks,
        nn.LayerNorm(dim),
        nn.Linear(dim, out_dim)
    )


class DQN(nn.Module):
    def __init__(self, n_obs, n_act, hidden=256, n_layers=2):
        super().__init__()
        layers, in_dim = [], n_obs
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    phi_values    = cfg['phi_values']
    weights       = cfg['weights']
    depth         = cfg.get('depth', 3)
    dqn_state_dim = cfg.get('dqn_state_dim', 18)
    print(f"[config] phi_values={phi_values}, depth={depth}, dqn_state_dim={dqn_state_dim}")
    print(f"[config] weights={weights}")
    return phi_values, weights, depth, dqn_state_dim

def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[scaler] loaded from {SCALER_PATH}")
    return scaler


def load_transformer(depth):
    model = create_transformer(depth)
    sd    = torch.load(TRANSFORMER_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    with torch.no_grad():
        dummy = torch.randn(1, SEQ_LEN, NI).to(DEVICE)
        out   = model(dummy)
    assert out.shape[-1] == NT, f"Unexpected transformer output shape: {out.shape}"
    print(f"[transformer] {depth}L loaded, params={sum(p.numel() for p in model.parameters()):,}")
    return model


def load_dqn(n_actions, state_dim):
    model = DQN(state_dim, n_actions)
    sd    = torch.load(DQN_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(sd)
    model = model.to(DEVICE).eval()
    print(f"[dqn] loaded from {DQN_PATH}, state_dim={state_dim}, actions={n_actions}")
    return model

# ============================================================================
# DEVICE I/O  (unchanged from original scheduler)
# ============================================================================

def get_subflow_info():
    try:
        with open(DEVICE_PATH, 'r') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {DEVICE_PATH}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def create_dataframe(subflow_data):
    if not subflow_data:
        return pd.DataFrame(columns=[
            'Timestamp', 'sock', 'name', 'cwnd', 'frac', 'in_flight',
            'srtt', 'prio', 'subflow_queue', 'meta_queue', 'mode',
            'full_bw_reached', 'losses', 'delivered'
        ])
    data = []
    for line in subflow_data:
        parts = line.strip().split()
        if len(parts) > 2 and parts[1] == "sock":
            try:
                row = {'Timestamp': float(parts[0])}
                i = 1
                while i < len(parts) - 1:
                    key, value = parts[i], parts[i + 1]
                    if key in ('cwnd', 'frac', 'in_flight', 'srtt', 'prio',
                               'subflow_queue', 'meta_queue', 'mode',
                               'full_bw_reached', 'losses', 'delivered'):
                        try:
                            row[key] = int(value)
                        except ValueError:
                            row[key] = None
                    elif key in ('name', 'sock'):
                        row[key] = value
                    i += 2
                data.append(row)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line.strip()}. Skipping. Error: {e}")
        else:
            print(f"Skipping malformed line: {line.strip()}")
    return pd.DataFrame(data)


def set_cwnd_frac(subflow_fracs):
    try:
        message = ""
        for sock_addr, new_frac in subflow_fracs.items():
            message += f"sock {sock_addr} cwnd_frac: {new_frac}\n"
        with open(DEVICE_PATH, 'w') as f:
            f.write(message)
        return True
    except FileNotFoundError:
        print(f"Device not found: {DEVICE_PATH}")
        return False
    except Exception as e:
        print(f"Error writing to device: {e}")
        return False

# ============================================================================
# DATA PIPELINE  (unchanged from original scheduler)
# ============================================================================

def convert_timestamp_to_unix(timestamp_series):
    return pd.to_datetime(timestamp_series, format="mixed").astype(np.int64) // 10**6


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    if n_in == -1:
        for i in range(1, n_out):
            cols.append(df.shift(-i))
            names += ([f'var{j+1}(t)' if i == 0 else f'var{j+1}(t+{i})'
                       for j in range(n_vars)])
    else:
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            names += ([f'var{j+1}(t)' if i == 0 else f'var{j+1}(t+{i})'
                       for j in range(n_vars)])
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def read(scaler, loc=-15000):
    """
    Read latest data from OUTPUT_FILE, apply the pre-loaded v6 scaler,
    and return scaled dataset. Mirrors the original read() logic exactly
    but uses the passed-in scaler instead of loading scalerLengthOptimal.pkl.
    """
    zeros       = 100
    pastSeconds = 5

    try:
        with open(OUTPUT_FILE, 'r') as f:
            q = deque(f, -loc)
        if not q:
            print("Error: No data read from file.")
            return None
    except FileNotFoundError:
        print(f"Error: File {OUTPUT_FILE} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    df = pd.read_csv(
        io.StringIO(''.join(q)),
        names=["Timestamp", "sock", "name", "cwnd", "frac", "in_flight",
               "srtt", "prio", "subflow_queue", "meta_queue", 'mode',
               'full_bw_reached', 'losses', 'delivered']
    )
    if df.empty:
        print("Error: DataFrame is empty after reading CSV.")
        return None

    print(f"Read {len(df)} rows from file.")

    sock_counts    = df['sock'].value_counts()
    total_rows     = len(df)
    sock_threshold = total_rows * 0.2
    valid_socks    = sock_counts[sock_counts >= sock_threshold].index
    df             = df[df['sock'].isin(valid_socks)]
    print(f"After filtering: {len(valid_socks)} sock IDs kept, {len(df)} rows remain.")

    if len(valid_socks) < 2:
        print("Error: Fewer than 2 sock IDs meet the threshold.")
        return None

    df['frac']      = np.where(df['frac'] == 0, 1e-3, df['frac'])
    df              = df.drop(['name', 'prio'], axis=1)
    df['sock']      = pd.Categorical(df['sock']).codes
    df['Timestamp'] = convert_timestamp_to_unix(df['Timestamp'])
    df['Timestamp'] -= min(df['Timestamp'])

    sock1 = df[df['sock'] == 1]
    sock2 = df[df['sock'] == 0]
    print(f"Sock1: {len(sock1)} rows, Sock2: {len(sock2)} rows.")

    if sock1.empty or sock2.empty:
        print("Error: One or both sock DataFrames are empty.")
        return None

    data_list = [
        series_to_supervised(pd.DataFrame(sock1['cwnd']),          pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['cwnd']),          pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['frac']),          pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['frac']),          pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['in_flight']),     pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['in_flight']),     pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['srtt']),          pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['srtt']),          pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['subflow_queue']), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['subflow_queue']), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['meta_queue']),    pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['losses']),        pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['losses']),        pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['delivered']),     pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['delivered']),     pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['cwnd']),  -1, pastSeconds + 1),
        series_to_supervised(pd.DataFrame(sock2['cwnd']),  -1, pastSeconds + 1),
        series_to_supervised(pd.DataFrame(sock1['srtt']),  -1, pastSeconds + 1),
        series_to_supervised(pd.DataFrame(sock2['srtt']),  -1, pastSeconds + 1),
    ]

    result = []
    for d in data_list:
        for col in d.columns:
            result.append(d[col].tolist())

    df_feat = pd.DataFrame(result).T.replace(np.nan, 0)
    print(f"DataFrame shape after series_to_supervised: {df_feat.shape}")

    dictionary = {}
    for col in df_feat.columns:
        temp = [sum(df_feat[col][i:i+zeros]) for i in range(0, len(df_feat[col]), zeros)]
        dictionary[col] = temp
    df_agg    = pd.DataFrame(dictionary)
    df_agg[0] = list(range(len(df_agg)))
    df_agg    = df_agg.replace(np.nan, 0)

    if df_agg.empty:
        print("Error: DataFrame is empty after aggregation.")
        return None

    dataset = df_agg.values
    print(f"Final dataset shape before scaling: {dataset.shape}")

    dataset_transformed = np.log1p(dataset)
    if dataset_transformed.shape[0] == 0:
        print("Error: Transformed dataset is empty.")
        return None

    return scaler.transform(dataset_transformed)


def prepare_data(dataset, input_shape, start=90, end=110):
    if dataset is None:
        return None, None
    leng = len(dataset) * end
    while leng % (input_shape * end) != 0:
        leng -= 1
    size = leng // (input_shape * end)
    rows = input_shape * size
    if dataset.shape[1] < end:
        padding_cols = end - dataset.shape[1]
        dataset = np.pad(dataset, ((0, 0), (0, padding_cols)),
                         'constant', constant_values=0)
    data = dataset[-rows:, 0:end]
    data = data.reshape(input_shape, size, end)
    X    = data[:, :, 0:start]
    y    = data[:, :, start:end]
    return X, y

# ============================================================================
# V6 STATE BUILDER  (18-dimensional, replaces 4-dim observation)
# ============================================================================

def build_state_v6(pred, prev_phi):
    """
    Build the 18-dimensional DQN state vector used in v6 training.
    pred     : numpy array (NT,) — last-step transformer output
    prev_phi : [phi1, phi2] floats
    """
    nc1, nc2 = pred[0],  pred[5]
    nr1, nr2 = pred[10], pred[15]
    mc1, mc2 = pred[2],  pred[7]
    fc1, fc2 = pred[4],  pred[9]
    fr1, fr2 = pred[14], pred[19]
    sd = lambda a, b: (a - b) / max(abs(b), 1e-6)
    return np.array([
        nc1, nc2, nr1, nr2,
        sd(mc1, nc1),      sd(mc2, nc2),
        sd(pred[12], nr1), sd(pred[17], nr2),
        sd(fc1, nc1),      sd(fc2, nc2),
        sd(fr1, nr1),      sd(fr2, nr2),
        prev_phi[0] / 100., prev_phi[1] / 100.,
        1. if fc1 < nc1 * 0.95 else 0.,
        1. if fc2 < nc2 * 0.95 else 0.,
        1. if fr1 > nr1 * 1.10 else 0.,
        1. if fr2 > nr2 * 1.10 else 0.,
    ], dtype=np.float32)

# ============================================================================
# INVERSE TRANSFORM  (for reward computation and logging)
# ============================================================================

def inverse_transform_pred(pred, scaler):
    temp = np.zeros((1, NTOT))
    temp[0, NI:NTOT] = pred
    try:
        unsc = np.expm1(scaler.inverse_transform(temp))[0]
    except Exception:
        return None
    MAX_CWND, MAX_SRTT = 100_000, 10_000_000
    return {
        'cc1': float(np.clip(unsc[90],  0, MAX_CWND)),
        'cc2': float(np.clip(unsc[95],  0, MAX_CWND)),
        'cs1': float(np.clip(unsc[100], 0, MAX_SRTT)),
        'cs2': float(np.clip(unsc[105], 0, MAX_SRTT)),
        'fc1': float(np.clip(unsc[94],  0, MAX_CWND)),
        'fc2': float(np.clip(unsc[99],  0, MAX_CWND)),
        'fs1': float(np.clip(unsc[104], 0, MAX_SRTT)),
        'fs2': float(np.clip(unsc[109], 0, MAX_SRTT)),
    }

# ============================================================================
# V6 REWARD  (corrected scaling)
# ============================================================================

def compute_reward_v6(pred, scaler, phi1, phi2, prev_phi, weights, estimated_br):
    """
    Corrected v6 reward with normalised components.
    estimated_br is a mutable list [float] for in-place EWMA update.
    """
    v = inverse_transform_pred(pred, scaler)
    if v is None:
        return 0.

    cc1, cc2 = max(1, v['cc1']), max(1, v['cc2'])
    cs1, cs2 = max(1, v['cs1']), max(1, v['cs2'])
    fc1, fc2 = max(1, v['fc1']), max(1, v['fc2'])
    fs1, fs2 = max(1, v['fs1']), max(1, v['fs2'])

    # Throughput: % change in cwnd
    r_tput  = (fc1 - cc1) / cc1 + (fc2 - cc2) / cc2

    # Delay: % decrease in srtt
    r_delay = (cs1 - fs1) / cs1 + (cs2 - fs2) / cs2

    # Preemptive
    r_pre = 0.
    for (fc, cc, phi, pv, ofc, occ, ophi, opv) in [
        (fc1, cc1, phi1, prev_phi[0], fc2, cc2, phi2, prev_phi[1]),
        (fc2, cc2, phi2, prev_phi[1], fc1, cc1, phi1, prev_phi[0]),
    ]:
        if fc < cc:
            r_pre += 0.5 if phi < pv else -0.5
            if ofc >= occ and ophi > opv:
                r_pre += 0.5

    # Stability: V6 FIX — divide by 100
    r_stab = -(abs(phi1 - prev_phi[0]) + abs(phi2 - prev_phi[1])) / 100.0

    # Low frac penalty
    r_low = -1. if (phi1 <= 30 and phi2 <= 30) else 0.

    # Quality: V6 FIX — divide by 1e6 (not 1e3)
    agg1      = fc1 / max(1, fs1)
    agg2      = fc2 / max(1, fs2)
    total_agg = (agg1 + agg2) * 8 * 1500
    if estimated_br[0] == 0:
        estimated_br[0] = total_agg
    else:
        estimated_br[0] = 0.9 * estimated_br[0] + 0.1 * total_agg
    r_qual = estimated_br[0] / 1e6          # KEY FIX

    w = weights
    return (w['w_throughput'] * r_tput  +
            w['w_delay']      * r_delay  +
            w['w_preemptive'] * r_pre    +
            w['w_stability']  * r_stab   +
            w['w_low_frac']   * r_low    +
            w['w_quality']    * r_qual)

# ============================================================================
# MAIN
# ============================================================================

def main():
    # --- Initialise logging ---
    init_log()

    # --- Load all v6 artefacts ---
    phi_values, weights, depth, dqn_state_dim = load_config()
    scaler      = load_scaler()
    transformer = load_transformer(depth)
    n_phi       = len(phi_values)
    n_actions   = n_phi ** 2
    dqn         = load_dqn(n_actions, dqn_state_dim)

    # --- State ---
    prev_phi     = [100., 100.]
    estimated_br = [0.]

    # Persistent values mirroring original env attributes
    gamma_p1 = gamma_p2 = delta_p1 = delta_p2 = 0.
    current_cwnd1 = current_cwnd2 = current_srtt1 = current_srtt2 = 0.

    print(f"\nStarting DARA v6 LIVE INFERENCE")
    print(f"  Action space (phi values): {phi_values}")
    print(f"  DQN state dim: {dqn_state_dim}")
    print(f"  Reward weights: {weights}\n")

    while True:
        # ── 1. Read fresh subflow info for device addressing ──────────────
        subflow_info = get_subflow_info()
        subflows     = create_dataframe(subflow_info)

        # ── 2. Read and scale live CSV data ───────────────────────────────
        dataset = read(scaler, loc=-15000)
        if dataset is None:
            print("Warning: Could not read dataset, retrying...")
            time.sleep(0.1)
            continue

        # ── 3. Prepare sequences and run transformer ──────────────────────
        X, y = prepare_data(dataset, SEQ_LEN, start=NI, end=NI + NT)
        if X is None:
            print("Warning: prepare_data returned None, retrying...")
            time.sleep(0.1)
            continue

        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pred_scaled = transformer(X_t)          # (n_seq, SEQ_LEN, NT)

        last_pred = pred_scaled[0, -1].cpu().numpy()   # (NT,)

        # ── 4. Unscale for logging / reward ───────────────────────────────
        v = inverse_transform_pred(last_pred, scaler)
        if v:
            gamma_p1      = v['fc1']
            gamma_p2      = v['fc2']
            delta_p1      = v['fs1']
            delta_p2      = v['fs2']
            current_cwnd1 = v['cc1']
            current_cwnd2 = v['cc2']
            current_srtt1 = v['cs1']
            current_srtt2 = v['cs2']

        print(f"gamma_p1 (future cwnd1): {gamma_p1:.1f}")
        print(f"gamma_p2 (future cwnd2): {gamma_p2:.1f}")
        print(f"delta_p1 (future srtt1): {delta_p1:.1f}")
        print(f"delta_p2 (future srtt2): {delta_p2:.1f}")
        print(f"current_cwnd1: {current_cwnd1:.1f}")
        print(f"current_cwnd2: {current_cwnd2:.1f}")
        print(f"current_srtt1: {current_srtt1:.1f}")
        print(f"current_srtt2: {current_srtt2:.1f}")

        # ── 5. Build 18-dim state and select action ───────────────────────
        state   = build_state_v6(last_pred, prev_phi)
        state_t = torch.tensor(state, dtype=torch.float32,
                               device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            best_idx = dqn(state_t).argmax().item()
        phi1_idx = best_idx // n_phi
        phi2_idx = best_idx % n_phi
        phi1     = phi_values[phi1_idx]
        phi2     = phi_values[phi2_idx]

        # ── 6. Apply action to kernel module ─────────────────────────────
        if not subflows.empty and len(subflows) >= 2:
            subflow_fracs = {
                subflows['sock'].iloc[0]: phi1,
                subflows['sock'].iloc[1]: phi2,
            }
            set_cwnd_frac(subflow_fracs)
            print(f"Set cwnd_frac: Path1={phi1}, Path2={phi2}")
        else:
            print("Warning: subflows DataFrame is empty. Cannot set cwnd_frac.")

        # ── 7. Compute v6 reward and log ──────────────────────────────────
        reward = compute_reward_v6(last_pred, scaler, phi1, phi2,
                                   prev_phi, weights, estimated_br)
        print(f"reward: {reward}\n")

        log_step(gamma_p1, gamma_p2, delta_p1, delta_p2,
                 current_cwnd1, current_cwnd2, current_srtt1, current_srtt2,
                 phi1, phi2, reward)

        # ── 8. Update state ───────────────────────────────────────────────
        prev_phi = [float(phi1), float(phi2)]

    print('Live inference complete!')


if __name__ == '__main__':
    main()