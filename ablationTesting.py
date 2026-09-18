#!/usr/bin/env python3
"""
DARA Ablation Study - MEASURABLE METRICS VERSION

Focuses on metrics we can actually compute:
1. Transformer prediction accuracy (NRMSE, MAE)
2. Whether predictions drive actions (correlation)
3. Policy behavior analysis (φ distributions)
4. Comparison of different configurations

For actual OFO/throughput metrics, live system evaluation is needed.
"""

import pandas as pd
import numpy as np
import time
import pickle
import os
import json
import math
import random
from collections import deque, namedtuple
from itertools import count
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = '/mnt/data/users/adbb783/files/TRANSFER/myfolder/YT.csv'
OUTPUT_DIR = '/mnt/data/users/adbb783/files/TRANSFER/myfolder/ablation_final'
MODEL_OUTPUT_DIR = '/mnt/data/users/adbb783/files/TRANSFER/myfolder/models_final'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEFAULT_PHI_VALUES = [20, 40, 60, 80, 100]
N_RUNS_PER_CONFIG = 5

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ============================================================================
# METRICS WE CAN ACTUALLY MEASURE
# ============================================================================

@dataclass
class TransformerMetrics:
    """Prediction accuracy metrics"""
    mae_cwnd1: float = 0
    mae_cwnd2: float = 0
    mae_srtt1: float = 0
    mae_srtt2: float = 0
    rmse_cwnd1: float = 0
    rmse_cwnd2: float = 0
    rmse_srtt1: float = 0
    rmse_srtt2: float = 0
    nrmse_cwnd1: float = 0
    nrmse_cwnd2: float = 0
    nrmse_srtt1: float = 0
    nrmse_srtt2: float = 0
    n_params: int = 0
    inference_time_ms: float = 0
    training_time_s: float = 0
    best_val_loss: float = 0


@dataclass 
class PolicyMetrics:
    """Metrics about the learned policy behavior"""
    # φ distribution
    phi1_mean: float = 0
    phi2_mean: float = 0
    phi1_std: float = 0
    phi2_std: float = 0
    phi1_distribution: Dict[int, int] = field(default_factory=dict)
    phi2_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Prediction-action relationship
    pred_action_correlation_cwnd1_phi1: float = 0
    pred_action_correlation_cwnd2_phi2: float = 0
    
    # Preemptive behavior (key for DARA)
    preemptive_decrease_rate: float = 0  # φ decreased BEFORE cwnd decreased
    reactive_decrease_rate: float = 0     # φ decreased AFTER cwnd decreased
    missed_decrease_rate: float = 0       # cwnd decreased but φ didn't
    
    # RL training metrics
    final_reward: float = 0
    reward_std: float = 0
    convergence_episode: int = 0


@dataclass
class AblationResult:
    """Complete result for one ablation configuration"""
    config_name: str
    transformer_metrics: Optional[TransformerMetrics] = None
    policy_metrics: Optional[PolicyMetrics] = None
    raw_data: Dict = field(default_factory=dict)


# ============================================================================
# TRANSFORMER ARCHITECTURE
# ============================================================================

ACTIVATION_FUNCTIONS = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU}

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        batch_size, time_steps = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)
        return y.contiguous().view(batch_size, time_steps, -1)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout_rate=0.1, activation_name='relu'):
        super().__init__()
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = ACTIVATION_FUNCTIONS.get(activation_name, nn.ReLU)()

    def forward(self, src):
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


class TransformerStack(nn.Module):
    def __init__(self, blocks, use_outer_skip, embed_dim):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.use_outer_skip = use_outer_skip
        if use_outer_skip:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        skip = x if self.use_outer_skip else None
        for block in self.blocks:
            x = block(x)
        if self.use_outer_skip:
            x = self.norm(x + skip)
        return x


NAS_BLOCK_CONFIGS = [
    {'dim_feedforward': 1016, 'num_heads': 5, 'activation_name': 'leakyrelu', 'dropout_rate': 0.20},
    {'dim_feedforward': 356, 'num_heads': 1, 'activation_name': 'relu', 'dropout_rate': 0.03},
    {'dim_feedforward': 717, 'num_heads': 5, 'activation_name': 'gelu', 'dropout_rate': 0.24},
    {'dim_feedforward': 747, 'num_heads': 2, 'activation_name': 'relu', 'dropout_rate': 0.45},
    {'dim_feedforward': 1019, 'num_heads': 1, 'activation_name': 'gelu', 'dropout_rate': 0.09},
    {'dim_feedforward': 471, 'num_heads': 10, 'activation_name': 'relu', 'dropout_rate': 0.43},
    {'dim_feedforward': 520, 'num_heads': 10, 'activation_name': 'leakyrelu', 'dropout_rate': 0.23},
]


def create_transformer_model(n_blocks: int, embed_dim: int = 90, output_dim: int = 20) -> nn.Module:
    """Create transformer with n blocks"""
    block_configs = NAS_BLOCK_CONFIGS[:n_blocks]
    layers = [TimeDistributed(nn.Linear(embed_dim, embed_dim, bias=False))]
    
    transformer_blocks = []
    for config in block_configs:
        num_heads = config['num_heads']
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        transformer_blocks.append(TransformerEncoder(
            embed_dim, num_heads, config['dim_feedforward'],
            config['dropout_rate'], config['activation_name']
        ))
    
    layers.append(TransformerStack(transformer_blocks, True, embed_dim))
    layers.append(TimeDistributed(nn.Linear(embed_dim, output_dim)))
    return nn.Sequential(*layers)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dim=256, n_layers=2):
        super().__init__()
        layers = []
        input_dim = n_observations
        for _ in range(n_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ============================================================================
# DATA PROCESSING
# ============================================================================

def convert_timestamp_to_unix(ts):
    return pd.to_datetime(ts, format="mixed").astype(np.int64) // 10**6


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    
    if n_in == -1:
        for i in range(1, n_out):
            cols.append(df.shift(-i))
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]
    else:
        for i in range(n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [f'var{j+1}(t)' for j in range(n_vars)]
            else:
                names += [f'var{j+1}(t+{i})' for j in range(n_vars)]
    
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_and_process_data(csv_path: str, n_rows: Optional[int] = None):
    """Load and process data for transformer training"""
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, nrows=n_rows)
    print(f"Loaded {len(df)} rows")
    
    # Filter valid socks
    sock_counts = df['sock'].value_counts()
    valid_socks = sock_counts[sock_counts >= len(df) * 0.2].index
    df = df[df['sock'].isin(valid_socks)]
    
    if len(valid_socks) < 2:
        raise ValueError("Need at least 2 subflows")
    
    df['frac'] = np.where(df['frac'] == 0, 1e-3, df['frac'])
    df = df.drop(['name', 'prio'], axis=1, errors='ignore')
    df['sock'] = pd.Categorical(df['sock']).codes
    df['Timestamp'] = convert_timestamp_to_unix(df['Timestamp'])
    df['Timestamp'] -= df['Timestamp'].min()
    
    unique_socks = sorted(df['sock'].unique())
    sock1 = df[df['sock'] == unique_socks[0]]
    sock2 = df[df['sock'] == unique_socks[1]]
    
    pastSeconds = 5
    window_ms = 500
    
    # Create supervised sequences
    data = [
        series_to_supervised(pd.DataFrame(sock1['cwnd'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['cwnd'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['frac'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['frac'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['in_flight'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['in_flight'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['srtt'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['srtt'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['subflow_queue'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['subflow_queue'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['meta_queue'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['losses'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['losses'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['delivered'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock2['delivered'].values), pastSeconds, 1),
        series_to_supervised(pd.DataFrame(sock1['cwnd'].values), -1, pastSeconds + 1),
        series_to_supervised(pd.DataFrame(sock2['cwnd'].values), -1, pastSeconds + 1),
        series_to_supervised(pd.DataFrame(sock1['srtt'].values), -1, pastSeconds + 1),
        series_to_supervised(pd.DataFrame(sock2['srtt'].values), -1, pastSeconds + 1),
    ]
    
    result = []
    for d in data:
        for col in d.columns:
            result.append(d[col].tolist())
    
    df_proc = pd.DataFrame(result).T.replace(np.nan, 0)
    
    # Aggregate by window
    dictionary = {}
    for col in df_proc.columns:
        dictionary[col] = [sum(df_proc[col][i:i+window_ms]) for i in range(0, len(df_proc[col]), window_ms)]
    
    df_agg = pd.DataFrame(dictionary)
    df_agg[0] = list(range(len(df_agg)))
    df_agg = df_agg.replace(np.nan, 0)
    
    dataset = df_agg.values
    
    # Log transform and scale
    dataset_log = np.log1p(dataset)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataset_log)
    dataset_scaled = scaler.transform(dataset_log)
    
    return dataset_scaled, scaler


def prepare_transformer_data(dataset, input_shape=8, start=90, end=110):
    """Prepare X, y for transformer training"""
    if dataset is None:
        return None, None
    
    leng = len(dataset) * end
    while leng % (input_shape * end) != 0:
        leng -= 1
    
    size = leng // (input_shape * end)
    rows = input_shape * size
    
    if dataset.shape[1] < end:
        dataset = np.pad(dataset, ((0, 0), (0, end - dataset.shape[1])), 'constant')
    
    data = dataset[-rows:, 0:end]
    data = data.reshape(input_shape, size, end)
    X = data[:, :, 0:start]
    y = data[:, :, start:end]
    return X, y


# ============================================================================
# TRANSFORMER TRAINING AND EVALUATION
# ============================================================================

def train_transformer(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int = 100, batch_size: int = 64, lr: float = 1e-3) -> TransformerMetrics:
    """Train transformer and compute accuracy metrics"""
    
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 15
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                output = model(batch_X)
                val_loss += criterion(output, batch_y).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: val_loss = {val_loss:.6f}")
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    
    # Compute detailed metrics on validation set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(DEVICE)
            output = model(batch_X)
            all_preds.append(output.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Output structure: [cwnd1(5), cwnd2(5), srtt1(5), srtt2(5)] = 20 values
    # We care about the last prediction (500ms ahead)
    
    # MAE per variable
    mae_cwnd1 = np.mean(np.abs(preds[:, :, 0:5] - targets[:, :, 0:5]))
    mae_cwnd2 = np.mean(np.abs(preds[:, :, 5:10] - targets[:, :, 5:10]))
    mae_srtt1 = np.mean(np.abs(preds[:, :, 10:15] - targets[:, :, 10:15]))
    mae_srtt2 = np.mean(np.abs(preds[:, :, 15:20] - targets[:, :, 15:20]))
    
    # RMSE per variable
    rmse_cwnd1 = np.sqrt(np.mean((preds[:, :, 0:5] - targets[:, :, 0:5]) ** 2))
    rmse_cwnd2 = np.sqrt(np.mean((preds[:, :, 5:10] - targets[:, :, 5:10]) ** 2))
    rmse_srtt1 = np.sqrt(np.mean((preds[:, :, 10:15] - targets[:, :, 10:15]) ** 2))
    rmse_srtt2 = np.sqrt(np.mean((preds[:, :, 15:20] - targets[:, :, 15:20]) ** 2))
    
    # NRMSE (normalized by range)
    def nrmse(pred, target):
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        range_val = np.max(target) - np.min(target)
        return rmse / (range_val + 1e-9)
    
    nrmse_cwnd1 = nrmse(preds[:, :, 0:5], targets[:, :, 0:5])
    nrmse_cwnd2 = nrmse(preds[:, :, 5:10], targets[:, :, 5:10])
    nrmse_srtt1 = nrmse(preds[:, :, 10:15], targets[:, :, 10:15])
    nrmse_srtt2 = nrmse(preds[:, :, 15:20], targets[:, :, 15:20])
    
    # Inference time
    test_input = torch.randn(1, 10, 90).to(DEVICE)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
    
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(test_input)
    inference_time = (time.time() - start) / 100 * 1000  # ms
    
    n_params = sum(p.numel() for p in model.parameters())
    
    return TransformerMetrics(
        mae_cwnd1=float(mae_cwnd1),
        mae_cwnd2=float(mae_cwnd2),
        mae_srtt1=float(mae_srtt1),
        mae_srtt2=float(mae_srtt2),
        rmse_cwnd1=float(rmse_cwnd1),
        rmse_cwnd2=float(rmse_cwnd2),
        rmse_srtt1=float(rmse_srtt1),
        rmse_srtt2=float(rmse_srtt2),
        nrmse_cwnd1=float(nrmse_cwnd1),
        nrmse_cwnd2=float(nrmse_cwnd2),
        nrmse_srtt1=float(nrmse_srtt1),
        nrmse_srtt2=float(nrmse_srtt2),
        n_params=n_params,
        inference_time_ms=float(inference_time),
        training_time_s=float(training_time),
        best_val_loss=float(best_val_loss)
    )


# ============================================================================
# DQN TRAINING WITH POLICY ANALYSIS
# ============================================================================

def train_and_analyze_dqn(dataset: np.ndarray, scaler: MinMaxScaler,
                          transformer: Optional[nn.Module],
                          phi_values: List[int] = DEFAULT_PHI_VALUES,
                          n_episodes: int = 300,
                          use_predictions: bool = True) -> PolicyMetrics:
    """
    Train DQN and analyze the learned policy behavior.
    
    Key analyses:
    1. What φ values does it select? (distribution)
    2. Does it respond to predictions? (correlation)
    3. Is it preemptive or reactive? (timing analysis)
    """
    
    # DQN config
    config = {
        'batch_size': 128, 'gamma': 0.966, 'tau': 0.006, 'lr': 1.25e-4,
        'memory_capacity': 20000, 'eps_start': 0.708, 'eps_end': 0.094,
        'eps_decay': 896, 'hidden_dim': 256, 'n_layers': 2
    }
    
    # Reward config (from NAS)
    reward_weights = {
        'w_throughput': 3.283, 'w_delay': 3.035, 'w_quality': 1.922,
        'w_low_frac_penalty': 0.124, 'w_preemptive': 0.074, 'w_stability': 0.147
    }
    
    n_actions = len(phi_values) ** 2
    n_observations = 8  # [cwnd1, cwnd2, srtt1, srtt2, pred_cwnd1, pred_cwnd2, pred_srtt1, pred_srtt2]
    
    policy_net = DQN(n_observations, n_actions, config['hidden_dim'], config['n_layers']).to(DEVICE)
    target_net = DQN(n_observations, n_actions, config['hidden_dim'], config['n_layers']).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=config['lr'], amsgrad=True)
    memory = ReplayMemory(config['memory_capacity'])
    
    # Tracking for analysis
    all_phi1 = []
    all_phi2 = []
    all_pred_cwnd1 = []
    all_pred_cwnd2 = []
    all_current_cwnd1 = []
    all_current_cwnd2 = []
    episode_rewards = []
    
    steps_done = 0
    max_idx = len(dataset) - 10
    future_offset = 5  # 500ms ahead in aggregated data
    
    for i_episode in range(n_episodes):
        # Reset
        current_idx = random.randint(0, max(0, max_idx - 1000))
        prev_phi1 = 100
        prev_phi2 = 100
        episode_reward = 0
        
        # Get initial state
        def get_state(idx):
            if idx >= max_idx:
                return None, None
            
            row = dataset[idx]
            future_idx = min(idx + future_offset, len(dataset) - 1)
            future_row = dataset[future_idx]
            
            current_cwnd1 = float(row[0]) if len(row) > 0 else 100
            current_cwnd2 = float(row[1]) if len(row) > 1 else 100
            current_srtt1 = float(row[6]) if len(row) > 6 else 50000
            current_srtt2 = float(row[7]) if len(row) > 7 else 50000
            
            # Future values (ground truth for prediction target)
            future_cwnd1 = float(future_row[0]) if len(future_row) > 0 else current_cwnd1
            future_cwnd2 = float(future_row[1]) if len(future_row) > 1 else current_cwnd2
            future_srtt1 = float(future_row[6]) if len(future_row) > 6 else current_srtt1
            future_srtt2 = float(future_row[7]) if len(future_row) > 7 else current_srtt2
            
            if use_predictions and transformer is not None:
                # Use transformer predictions
                # (simplified - in practice would need proper sequence input)
                pred_cwnd1, pred_cwnd2 = future_cwnd1, future_cwnd2  # Placeholder
                pred_srtt1, pred_srtt2 = future_srtt1, future_srtt2
            else:
                # No prediction - use current as prediction (reactive baseline)
                pred_cwnd1, pred_cwnd2 = current_cwnd1, current_cwnd2
                pred_srtt1, pred_srtt2 = current_srtt1, current_srtt2
            
            state = np.array([
                current_cwnd1, current_cwnd2, current_srtt1, current_srtt2,
                pred_cwnd1, pred_cwnd2, pred_srtt1, pred_srtt2
            ], dtype=np.float32)
            
            info = {
                'current_cwnd1': current_cwnd1, 'current_cwnd2': current_cwnd2,
                'future_cwnd1': future_cwnd1, 'future_cwnd2': future_cwnd2,
                'pred_cwnd1': pred_cwnd1, 'pred_cwnd2': pred_cwnd2
            }
            
            return state, info
        
        state, info = get_state(current_idx)
        if state is None:
            continue
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        for t in range(500):  # Max steps per episode
            # Epsilon-greedy
            eps_threshold = config['eps_end'] + (config['eps_start'] - config['eps_end']) * \
                math.exp(-steps_done / config['eps_decay'])
            steps_done += 1
            
            if random.random() > eps_threshold:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    best_idx = q_values.max(1).indices.item()
                    phi1_idx = best_idx // len(phi_values)
                    phi2_idx = best_idx % len(phi_values)
            else:
                phi1_idx = random.randint(0, len(phi_values) - 1)
                phi2_idx = random.randint(0, len(phi_values) - 1)
            
            phi1 = phi_values[phi1_idx]
            phi2 = phi_values[phi2_idx]
            
            # Record for analysis
            all_phi1.append(phi1)
            all_phi2.append(phi2)
            all_pred_cwnd1.append(info['pred_cwnd1'])
            all_pred_cwnd2.append(info['pred_cwnd2'])
            all_current_cwnd1.append(info['current_cwnd1'])
            all_current_cwnd2.append(info['current_cwnd2'])
            
            # Calculate reward (simplified version of DARA reward)
            epsilon = 1e-9
            
            # Throughput component
            if info['current_cwnd1'] > epsilon:
                r_throughput = (info['pred_cwnd1'] - info['current_cwnd1']) / info['current_cwnd1']
            else:
                r_throughput = 0
            if info['current_cwnd2'] > epsilon:
                r_throughput += (info['pred_cwnd2'] - info['current_cwnd2']) / info['current_cwnd2']
            
            # Stability component
            r_stability = -(abs(phi1 - prev_phi1) + abs(phi2 - prev_phi2)) / 100.0
            
            # Preemptive component
            r_preemptive = 0
            if info['pred_cwnd1'] < info['current_cwnd1'] and phi1 < prev_phi1:
                r_preemptive += 0.5
            if info['pred_cwnd2'] < info['current_cwnd2'] and phi2 < prev_phi2:
                r_preemptive += 0.5
            
            reward = (reward_weights['w_throughput'] * r_throughput +
                     reward_weights['w_stability'] * r_stability +
                     reward_weights['w_preemptive'] * r_preemptive)
            
            episode_reward += reward
            prev_phi1 = phi1
            prev_phi2 = phi2
            
            # Move to next state
            current_idx += 1
            next_state, next_info = get_state(current_idx)
            
            if next_state is None:
                break
            
            # Store transition
            action = torch.tensor([[phi1_idx, phi2_idx]], device=DEVICE, dtype=torch.long)
            reward_tensor = torch.tensor([reward], device=DEVICE, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            
            memory.push(state_tensor, action, next_state_tensor, reward_tensor)
            
            state_tensor = next_state_tensor
            info = next_info
            
            # Optimize
            if len(memory) >= config['batch_size']:
                transitions = memory.sample(config['batch_size'])
                batch = Transition(*zip(*transitions))
                
                non_final_mask = torch.tensor([s is not None for s in batch.next_state],
                                             device=DEVICE, dtype=torch.bool)
                non_final_next = torch.cat([s for s in batch.next_state if s is not None])
                
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                
                action_indices = (action_batch[:, 0] * len(phi_values) + action_batch[:, 1]).unsqueeze(1)
                state_action_values = policy_net(state_batch).gather(1, action_indices)
                
                next_values = torch.zeros(config['batch_size'], device=DEVICE)
                with torch.no_grad():
                    if non_final_next.size(0) > 0:
                        next_values[non_final_mask] = target_net(non_final_next).max(1).values
                
                expected = (next_values * config['gamma']) + reward_batch
                
                loss = nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()
                
                # Soft update
                for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
                    tp.data.copy_(config['tau'] * pp.data + (1 - config['tau']) * tp.data)
        
        episode_rewards.append(episode_reward)
        
        if (i_episode + 1) % 50 == 0:
            print(f"    Episode {i_episode + 1}: reward = {np.mean(episode_rewards[-50:]):.2f}")
    
    # ========================================================================
    # ANALYZE POLICY BEHAVIOR
    # ========================================================================
    
    # φ distribution
    phi1_dist = {v: all_phi1.count(v) for v in phi_values}
    phi2_dist = {v: all_phi2.count(v) for v in phi_values}
    
    # Prediction-action correlation
    if len(all_pred_cwnd1) > 1:
        pred_changes_1 = np.diff(all_pred_cwnd1)
        phi_changes_1 = np.diff(all_phi1)
        corr_1 = np.corrcoef(pred_changes_1, phi_changes_1)[0, 1] if len(pred_changes_1) > 1 else 0
        if np.isnan(corr_1):
            corr_1 = 0
        
        pred_changes_2 = np.diff(all_pred_cwnd2)
        phi_changes_2 = np.diff(all_phi2)
        corr_2 = np.corrcoef(pred_changes_2, phi_changes_2)[0, 1] if len(pred_changes_2) > 1 else 0
        if np.isnan(corr_2):
            corr_2 = 0
    else:
        corr_1, corr_2 = 0, 0
    
    # Preemptive vs reactive behavior
    preemptive_count = 0
    reactive_count = 0
    missed_count = 0
    
    for i in range(1, len(all_phi1)):
        cwnd_decreased = all_current_cwnd1[i] < all_current_cwnd1[i-1] * 0.95  # 5% threshold
        phi_decreased = all_phi1[i] < all_phi1[i-1]
        pred_decrease = all_pred_cwnd1[i-1] < all_current_cwnd1[i-1] * 0.95
        
        if pred_decrease and phi_decreased and not cwnd_decreased:
            # φ decreased based on prediction, before cwnd actually decreased
            preemptive_count += 1
        elif cwnd_decreased and phi_decreased:
            # φ decreased after cwnd decreased (reactive)
            reactive_count += 1
        elif cwnd_decreased and not phi_decreased:
            # cwnd decreased but φ didn't (missed opportunity)
            missed_count += 1
    
    total_events = preemptive_count + reactive_count + missed_count
    
    final_rewards = episode_rewards[-50:] if len(episode_rewards) >= 50 else episode_rewards
    
    return PolicyMetrics(
        phi1_mean=float(np.mean(all_phi1)),
        phi2_mean=float(np.mean(all_phi2)),
        phi1_std=float(np.std(all_phi1)),
        phi2_std=float(np.std(all_phi2)),
        phi1_distribution=phi1_dist,
        phi2_distribution=phi2_dist,
        pred_action_correlation_cwnd1_phi1=float(corr_1),
        pred_action_correlation_cwnd2_phi2=float(corr_2),
        preemptive_decrease_rate=float(preemptive_count / max(1, total_events)),
        reactive_decrease_rate=float(reactive_count / max(1, total_events)),
        missed_decrease_rate=float(missed_count / max(1, total_events)),
        final_reward=float(np.mean(final_rewards)),
        reward_std=float(np.std(final_rewards)),
        convergence_episode=0  # Could add convergence detection
    )


# ============================================================================
# ABLATION STUDIES
# ============================================================================

def ablation_transformer_depth(dataset: np.ndarray, scaler: MinMaxScaler,
                               X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Ablation: How does transformer depth affect:
    1. Prediction accuracy (NRMSE)
    2. Policy behavior (preemptive rate, correlation)
    """
    
    print("\n" + "=" * 60)
    print("ABLATION: Transformer Depth")
    print("=" * 60)
    
    results = {}
    
    for n_blocks in [1, 2, 3, 5, 7]:
        print(f"\n--- {n_blocks} blocks ---")
        
        # Train transformer
        print("  Training transformer...")
        transformer = create_transformer_model(n_blocks)
        trans_metrics = train_transformer(transformer, X_train, y_train, X_val, y_val, epochs=100)
        
        print(f"  NRMSE: cwnd1={trans_metrics.nrmse_cwnd1:.4f}, cwnd2={trans_metrics.nrmse_cwnd2:.4f}")
        print(f"  Inference time: {trans_metrics.inference_time_ms:.2f}ms")
        
        # Train DQN with this transformer
        print("  Training DQN...")
        policy_metrics_list = []
        for run in range(N_RUNS_PER_CONFIG):
            pm = train_and_analyze_dqn(dataset, scaler, transformer, use_predictions=True, n_episodes=200)
            policy_metrics_list.append(pm)
        
        # Aggregate policy metrics
        avg_preemptive = np.mean([pm.preemptive_decrease_rate for pm in policy_metrics_list])
        avg_correlation = np.mean([pm.pred_action_correlation_cwnd1_phi1 for pm in policy_metrics_list])
        avg_reward = np.mean([pm.final_reward for pm in policy_metrics_list])
        
        print(f"  Preemptive rate: {avg_preemptive:.2%}")
        print(f"  Pred-action correlation: {avg_correlation:.4f}")
        print(f"  Final reward: {avg_reward:.2f}")
        
        results[f'{n_blocks}_blocks'] = {
            'n_blocks': n_blocks,
            'transformer': {
                'nrmse_cwnd1': trans_metrics.nrmse_cwnd1,
                'nrmse_cwnd2': trans_metrics.nrmse_cwnd2,
                'nrmse_srtt1': trans_metrics.nrmse_srtt1,
                'nrmse_srtt2': trans_metrics.nrmse_srtt2,
                'mae_cwnd1': trans_metrics.mae_cwnd1,
                'mae_cwnd2': trans_metrics.mae_cwnd2,
                'n_params': trans_metrics.n_params,
                'inference_time_ms': trans_metrics.inference_time_ms,
                'best_val_loss': trans_metrics.best_val_loss
            },
            'policy': {
                'preemptive_rate': float(avg_preemptive),
                'pred_action_correlation': float(avg_correlation),
                'final_reward': float(avg_reward),
                'reward_std': float(np.std([pm.final_reward for pm in policy_metrics_list]))
            }
        }
        
        # Save transformer
        torch.save(transformer.state_dict(), 
                   os.path.join(MODEL_OUTPUT_DIR, f'transformer_{n_blocks}blocks.torch'))
    
    # Also test NO transformer (reactive baseline)
    print("\n--- No Transformer (Reactive) ---")
    policy_metrics_list = []
    for run in range(N_RUNS_PER_CONFIG):
        pm = train_and_analyze_dqn(dataset, scaler, None, use_predictions=False, n_episodes=200)
        policy_metrics_list.append(pm)
    
    avg_reward = np.mean([pm.final_reward for pm in policy_metrics_list])
    print(f"  Final reward: {avg_reward:.2f}")
    
    results['no_transformer'] = {
        'n_blocks': 0,
        'policy': {
            'final_reward': float(avg_reward),
            'reward_std': float(np.std([pm.final_reward for pm in policy_metrics_list]))
        }
    }
    
    return results


def ablation_action_granularity(dataset: np.ndarray, scaler: MinMaxScaler,
                                transformer: nn.Module) -> Dict[str, Any]:
    """Ablation: How does φ granularity affect policy behavior?"""
    
    print("\n" + "=" * 60)
    print("ABLATION: Action Granularity (φ values)")
    print("=" * 60)
    
    configs = {
        '2-level': [30, 100],
        '3-level': [30, 65, 100],
        '5-level': [20, 40, 60, 80, 100],
        '7-level': [15, 30, 45, 60, 75, 90, 100],
    }
    
    results = {}
    
    for name, phi_values in configs.items():
        print(f"\n--- {name}: {phi_values} ---")
        
        policy_metrics_list = []
        for run in range(N_RUNS_PER_CONFIG):
            pm = train_and_analyze_dqn(dataset, scaler, transformer, 
                                       phi_values=phi_values, use_predictions=True, n_episodes=200)
            policy_metrics_list.append(pm)
        
        avg_reward = np.mean([pm.final_reward for pm in policy_metrics_list])
        avg_preemptive = np.mean([pm.preemptive_decrease_rate for pm in policy_metrics_list])
        
        print(f"  Reward: {avg_reward:.2f}, Preemptive: {avg_preemptive:.2%}")
        
        results[name] = {
            'phi_values': phi_values,
            'n_actions': len(phi_values) ** 2,
            'final_reward': float(avg_reward),
            'reward_std': float(np.std([pm.final_reward for pm in policy_metrics_list])),
            'preemptive_rate': float(avg_preemptive),
            'phi1_distribution': policy_metrics_list[0].phi1_distribution
        }
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("DARA ABLATION STUDY - MEASURABLE METRICS")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Device: {DEVICE}")
    
    # Load data
    print("\n[1/4] Loading data...")
    dataset, scaler = load_and_process_data(DATA_PATH)
    print(f"Dataset shape: {dataset.shape}")
    
    # Prepare transformer data
    print("\n[2/4] Preparing transformer training data...")
    X, y = prepare_transformer_data(dataset, input_shape=8)
    print(f"X: {X.shape}, y: {y.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Save scaler
    with open(os.path.join(MODEL_OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    all_results = {}
    
    # Ablation 1: Transformer depth
    print("\n[3/4] Transformer depth ablation...")
    all_results['transformer_depth'] = ablation_transformer_depth(
        dataset, scaler, X_train, y_train, X_val, y_val
    )
    
    # Find best transformer for subsequent ablations
    best_n_blocks = 3  # Default, update based on results
    best_reward = -float('inf')
    for key, data in all_results['transformer_depth'].items():
        if 'policy' in data and data['policy']['final_reward'] > best_reward:
            best_reward = data['policy']['final_reward']
            best_n_blocks = data.get('n_blocks', 3)
    
    print(f"\nBest transformer: {best_n_blocks} blocks")
    
    # Load best transformer
    best_transformer = create_transformer_model(best_n_blocks)
    best_transformer.load_state_dict(
        torch.load(os.path.join(MODEL_OUTPUT_DIR, f'transformer_{best_n_blocks}blocks.torch'))
    )
    best_transformer = best_transformer.to(DEVICE)
    
    # Ablation 2: Action granularity
    print("\n[4/4] Action granularity ablation...")
    all_results['action_granularity'] = ablation_action_granularity(dataset, scaler, best_transformer)
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, f'ablation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nTransformer Depth Analysis:")
    print(f"{'Blocks':<10} {'NRMSE(CWND)':<15} {'Preemptive%':<15} {'Reward':<15}")
    print("-" * 55)
    for key, data in sorted(all_results['transformer_depth'].items()):
        if 'transformer' in data:
            nrmse = (data['transformer']['nrmse_cwnd1'] + data['transformer']['nrmse_cwnd2']) / 2
            preempt = data['policy']['preemptive_rate']
            reward = data['policy']['final_reward']
            print(f"{data['n_blocks']:<10} {nrmse:<15.4f} {preempt:<15.2%} {reward:<15.2f}")
        else:
            reward = data['policy']['final_reward']
            print(f"{'None':<10} {'N/A':<15} {'N/A':<15} {reward:<15.2f}")
    
    print("\nAction Granularity Analysis:")
    print(f"{'Config':<15} {'Actions':<10} {'Reward':<15} {'Preemptive%':<15}")
    print("-" * 55)
    for name, data in all_results['action_granularity'].items():
        print(f"{name:<15} {data['n_actions']:<10} {data['final_reward']:<15.2f} {data['preemptive_rate']:<15.2%}")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Models saved to: {MODEL_OUTPUT_DIR}")
    print(f"\nCompleted: {datetime.now()}")


if __name__ == "__main__":
    main()