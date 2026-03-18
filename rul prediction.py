#!/usr/bin/env python
# coding: utf-8

# In[10]:


import time 
# 1. 导入库
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# 定义列名
index_names = ["Engine", "Cycle"]
setting_names = ["Setting 1 - c1", "Setting 2 - c2", "Setting 3 - c3"]
sensor_names = [
    "(Fan Inlet Temperature) (◦R) - s1",
    "(LPC Outlet Temperature) (◦R) - s2",
    "(HPC Outlet Temperature) (◦R) - s3",
    "(LPT Outlet Temperature) (◦R) - s4",
    "(Fan Inlet Pressure) (psia) - s5",
    "(Bypass-Duct Pressure) (psia) - s6",
    "(HPC Outlet Pressure) (psia) - s7",
    "(Physical Fan Speed) (rpm) - s8",
    "(Physical Core Speed) (rpm) - s9",
    "(Engine Pressure Ratio(P50/P2)) - s10",
    "(HPC Outlet Static Pressure) (psia) - s11",
    "(Ratio of Fuel Flow to Ps30) (pps/psia) - s12",
    "(Corrected Fan Speed) (rpm) - s13",
    "(Corrected Core Speed) (rpm) - s14",
    "(Bypass Ratio) - s15",
    "(Burner Fuel-Air Ratio) - s16",
    "(Bleed Enthalpy) - s17",
    "(Required Fan Speed) (rpm) - s18",
    "(Required Fan Conversion Speed) (rpm) - s19",
    "(High-Pressure Turbines Cool Air Flow) - s20",
    "(Low-Pressure Turbines Cool Air Flow) - s21",
]

col_names = index_names + setting_names + sensor_names
rul_col = ["RUL"]

def load_data(file_path: str, col_names: list) -> pd.DataFrame:
    data = pd.read_csv(file_path, sep="\s+", header=None, names=col_names)
    print(f"Data loaded successfully from {file_path}")
    return data

def load_multiple_data(file_paths: list, col_names: str) -> pd.DataFrame:
    dataframes = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, sep="\s+", header=None, names=col_names)
        dataframes.append(data)
        print(f"Data loaded successfully from {file_path}")
    return pd.concat(dataframes, ignore_index=True)

# 数据目录和文件名
DATA_DIR = ".\CMAPSSData"
TRAIN_FILES = ["train_FD003.txt"]
TEST_FILE = "test_FD003.txt"
RUL_FILE = "RUL_FD003.txt"

# 完整路径
TRAIN_PATHS = [os.path.join(DATA_DIR, file) for file in TRAIN_FILES]
TEST_PATH = os.path.join(DATA_DIR, TEST_FILE)
RUL_PATH = os.path.join(DATA_DIR, RUL_FILE)

# 加载数据
df_train = load_multiple_data(TRAIN_PATHS, col_names)
df_test = load_data(TEST_PATH, col_names)
df_rul = load_data(RUL_PATH, rul_col)

# 添加 Engine 列
df_rul["Engine"] = df_rul.index + 1

def add_remaining_useful_life(df: pd.DataFrame, cycles_after_last_cycle: pd.Series = None) -> pd.DataFrame:
    if cycles_after_last_cycle is not None:
        max_cycle = df.groupby("Engine")["Cycle"].transform("max")
        df = df.merge(cycles_after_last_cycle, on="Engine", how="left")
        df["RUL"] = df["RUL"] + (max_cycle - df["Cycle"])
    else:
        max_cycle = df.groupby("Engine")["Cycle"].transform("max")
        df["RUL"] = max_cycle - df["Cycle"]
    return df

# 添加 RUL
df_train = add_remaining_useful_life(df_train)
df_test = add_remaining_useful_life(df_test, df_rul)

# 限制 RUL 的上限
RUL_UPPER_BOUND = 125
df_train["RUL"] = df_train["RUL"].clip(upper=RUL_UPPER_BOUND)
df_test["RUL"] = df_test["RUL"].clip(upper=RUL_UPPER_BOUND)

# 移除不需要的列
removing_columns = ["s16", "s5", "s1", "s19", "s18", "s6",'s15','c1','c2','c3']
df_train = df_train.loc[:, ~df_train.columns.str.endswith(tuple(removing_columns))]
df_test = df_test.loc[:, ~df_test.columns.str.endswith(tuple(removing_columns))]

# 指数平滑去噪
def exponential_smoothing(data, alpha=0.3):
    smoothed_data = data.copy()
    for i in range(1, len(data)):
        smoothed_data.iloc[i] = alpha * data.iloc[i] + (1 - alpha) * smoothed_data.iloc[i - 1]
    return smoothed_data

# 对每个传感器的数据进行指数平滑
for sensor in df_train.columns.difference(["Engine", "Cycle", "RUL"]):
    df_train[sensor] = df_train.groupby("Engine")[sensor].transform(exponential_smoothing)
    df_test[sensor] = df_test.groupby("Engine")[sensor].transform(exponential_smoothing)

# 归一化
scaler = MinMaxScaler()
df_train[df_train.columns.difference(["Engine", "Cycle", "RUL"])] = scaler.fit_transform(
    df_train[df_train.columns.difference(["Engine", "Cycle", "RUL"])]
)
df_test[df_test.columns.difference(["Engine", "Cycle", "RUL"])] = scaler.transform(
    df_test[df_test.columns.difference(["Engine", "Cycle", "RUL"])]
)

# 时间窗口采样
WINDOW_SIZE = 60
STEP_SIZE = 1  # 步长 m

def time_window_data(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # 返回 Cycle 列
    data = []
    rules = []
    cycles = []  # 保存 Cycle 列
    for engine in df["Engine"].unique():
        df_engine = df[df["Engine"] == engine].copy()
        for i in range(window_size, df_engine.shape[0] + 1, step_size):
            window_data = df_engine.iloc[i - window_size : i].drop(["Engine", "Cycle", "RUL"], axis=1).values
            data.append(window_data)
            rules.append(df_engine.iloc[i - 1]["RUL"])
            cycles.append(df_engine.iloc[i - 1]["Cycle"])  # 保存当前时间窗口的 Cycle 值
    return np.array(data), np.array(rules), np.array(cycles)  # 返回 Cycle 列

# 创建时间窗口数据
X_train, y_train, cycles_train = time_window_data(df_train)
X_test, y_test, cycles_test = time_window_data(df_test)

# 划分验证集
X_train, X_val, y_train, y_val, cycles_train, cycles_val = train_test_split(
    X_train, y_train, cycles_train, test_size=0.2, random_state=42
)


# In[11]:


import torch
import torch.nn as nn
# 3. 模型定义
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=2.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:3], layers_hidden[1:4]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        batch_size, seq_len, input_size = x.size()
        x = x.view(-1, input_size)

        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        x = x.view(batch_size, seq_len, -1)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

# R-drop相关定义
def kl_loss(p, q):
    return torch.mean(torch.sum(F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none'), dim=-1))

class KANBiLSTMTimeSeriesModel(nn.Module):
    def __init__(self, kan_layers, lstm_hidden_size, output_size, num_lstm_layers=2, dropout=0.3):
        super().__init__()
        self.kan = KAN(kan_layers)
        self.kan_projection = nn.Linear(kan_layers[-1], lstm_hidden_size)
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden_size * 2)
        
        # 新增：跳跃连接的线性变换（如果维度不匹配）
        self.skip_proj = nn.Linear(kan_layers[-1], lstm_hidden_size * 2)  # 双向LSTM输出维度是 hidden_size * 2
        
        self.fc = nn.Linear(lstm_hidden_size * 2, output_size)
        self.time_embed = nn.Embedding(450, lstm_hidden_size)

    def forward(self, x, update_grid=False):
        batch_size, seq_len, input_size = x.size()
        
        # KAN 特征提取
        x_kan = self.kan(x, update_grid=update_grid)  # [batch, seq_len, kan_output]
        
        # LSTM 路径
        x_kan_proj = self.kan_projection(x_kan)  # [batch, seq_len, lstm_hidden_size]
        time_pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        time_embedding = self.time_embed(time_pos)
        x_lstm_input = x_kan_proj + time_embedding
        lstm_out, _ = self.lstm(x_lstm_input)  # [batch, seq_len, lstm_hidden_size * 2]
        lstm_out_last = lstm_out[:, -1, :]  # 取最后一个时间步
        
        # 跳跃连接：KAN 输出 -> 变换维度后与 LSTM 输出相加
        x_skip = self.skip_proj(x_kan[:, -1, :])  # 取最后一个时间步并投影
        combined = lstm_out_last + x_skip  # 残差连接
        
        # LayerNorm + 输出
        output = self.fc(self.lstm_norm(combined))
        return output
    


# In[12]:


# 4. 训练和评估
class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        mse_loss = (output - target) ** 2
        if self.weights is not None:
            mse_loss = mse_loss * self.weights
        return mse_loss.mean()

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate_model(model, X_val, y_val, device='cpu'):
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        
        predictions = model(X_val)
        predictions = predictions.cpu().numpy()
        y_val = y_val.cpu().numpy()
        
        mse = mean_squared_error(y_val, predictions)
        rmse = np.sqrt(mse)
        
        print(f"Validation RMSE: {rmse:.4f}")

def train_model_with_weighted_mse(model, X_train, y_train, criterion, optimizer, X_val, y_val, alpha=0.3, batch_size=32, epochs=50, device='cpu'):
    model.to(device)
    model.train()
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=False)
    
    for epoch in range(epochs):
        total_loss = 0
        total_kl_loss = 0
        for batch_x, batch_y in train_loader:
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            output = model(batch_x)
            
            ce_loss = criterion(output, batch_y)
            kl_loss_val = kl_loss(output, batch_y)
            
            weighted_loss = ce_loss + alpha * kl_loss_val
            
            weighted_loss.backward()
            optimizer.step()
            
            total_loss += weighted_loss.item()
            total_kl_loss += kl_loss_val.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    evaluate_model(model, X_val, y_val, device=device)
    return model


# In[13]:


# 计算时间代码
# 5. 主程序
if __name__ == "__main__":
    set_seed(52)
    from sklearn.metrics import mean_squared_error
    import math
    import time # 确保 time 被导入

    device = 'cpu' # 强制使用 CPU，因为您报告的是 CPU 性能
    batch_size=32
    
    # ----------------------------------------------------
    # 1. 初始化模型
    # ----------------------------------------------------
    model = KANBiLSTMTimeSeriesModel(
        kan_layers=[14, 26, 50],
        lstm_hidden_size=256,
        output_size=1,
        num_lstm_layers=1,
        dropout=0.4,
        
    )
    print(model)
    
    # ----------------------------------------------------
    # 2. 计算总训练时间 (Total Training Time)
    # ----------------------------------------------------
    
    # 训练模型
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    weighted_mse_criterion = WeightedMSELoss(weights=None)
    
    start_time = time.time() # 记录训练开始时间
    
    trained_model = train_model_with_weighted_mse(
        model, X_train, y_train, weighted_mse_criterion, optimizer, X_val, y_val, 
        batch_size=batch_size, epochs=50, device=device
    )
    
    end_time = time.time() # 记录训练结束时间
    total_training_time_seconds = end_time - start_time
    
    # 转换为小时和分钟
    hours = int(total_training_time_seconds // 3600)
    minutes = int((total_training_time_seconds % 3600) // 60)
    seconds = int(total_training_time_seconds % 60)
    
    print("-" * 50)
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s")
    print("-" * 50)

    # ----------------------------------------------------
    # 3. 计算平均推理时间 (Average Inference Time per Sample)
    # ----------------------------------------------------
    
    # 确保模型在 CPU 上进行评估
    trained_model.to('cpu').eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to('cpu')
    
    # 预热模型（忽略前几次运行，消除启动开销）
    with torch.no_grad():
        for _ in range(5):
            _ = trained_model(X_test_tensor[:batch_size])

    # 计时推理
    inference_start_time = time.time()
    num_samples = 0
    
    # 使用 DataLoader 分批次进行推理，模拟真实场景
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_x in test_loader:
            _ = trained_model(batch_x[0])
            num_samples += batch_x[0].size(0)

    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time
    
    average_inference_time_ms = (total_inference_time / num_samples) * 1000
    
    print(f"Total Inference Samples: {num_samples}")
    print(f"Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"Average Inference Time per Sample: {average_inference_time_ms:.4f} ms")
    print("-" * 50)
    
    


# In[4]:


# 5. 主程序
if __name__ == "__main__":
    set_seed(52)
    from sklearn.metrics import mean_squared_error
    import math
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size=32
    
    # 初始化模型
    model = KANBiLSTMTimeSeriesModel(
        kan_layers=[14, 26, 50],
        lstm_hidden_size=256,
        output_size=1,
        num_lstm_layers=1,
        dropout=0.4,
        
    )
    
    print(model)


# In[4]:


# 5. 主程序
if __name__ == "__main__":
    set_seed(52)
    from sklearn.metrics import mean_squared_error
    import math
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size=32
    
    # 初始化模型
    model = KANBiLSTMTimeSeriesModel(
        kan_layers=[14, 26, 50],
        lstm_hidden_size=256,
        output_size=1,
        num_lstm_layers=1,
        dropout=0.4,
        
    )
    
    # 训练模型
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    weighted_mse_criterion = WeightedMSELoss(weights=None)
    train_model_with_weighted_mse(
        model, X_train, y_train, weighted_mse_criterion, optimizer, X_val, y_val, 
        batch_size=batch_size, epochs=20, device=device
    )


# In[5]:


import torch 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def predict_engine_rul(model, engine_data, device=device):
    model.eval()
    with torch.no_grad():
        engine_tensor = torch.tensor(engine_data, dtype=torch.float32).unsqueeze(0).to(device)
        predictions = model(engine_tensor).cpu().numpy().flatten()
    return predictions

def plot_engine_rul_comparison(engine_id, df, model, window_size=60, device=device):
    engine_data = df[df["Engine"] == engine_id].copy().sort_values("Cycle")
    if len(engine_data) < window_size:
        raise ValueError(f"Engine {engine_id} has only {len(engine_data)} cycles, less than window_size={window_size}")
    
    feature_cols = [col for col in engine_data.columns if col not in ["Engine", "Cycle", "RUL"]]
    X_engine = engine_data[feature_cols].values
    true_rul = engine_data["RUL"].values
    
    predicted_rul = []
    for i in range(window_size, len(X_engine) + 1):
        window_data = X_engine[i - window_size:i]
        rul_pred = predict_engine_rul(model, window_data, device=device)
        predicted_rul.append(rul_pred[0])
    
    cycles = engine_data["Cycle"].values[window_size - 1:]
    true_rul_aligned = true_rul[window_size - 1:]
    
    plt.figure(figsize=(8, 6))
    plt.plot(cycles, true_rul_aligned, color='green', label="True RUL")
    plt.plot(cycles, predicted_rul, color='SkyBlue', linestyle='--', label="Predicted RUL")
    #plt.title(f" ")
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(False)
    plt.show()

# 示例调用
plot_engine_rul_comparison(engine_id=99, df=df_test, model=model, window_size=60)


# In[8]:


# 修改后的绘图函数，交换颜色并去掉预测值连线
def plot_rul_with_error_bands(true_rul_values, final_rul_predictions):
    # 计算误差带基于真实RUL值
    error_band_10_lower = np.array(true_rul_values) - 10
    error_band_10_upper = np.array(true_rul_values) + 10
    error_band_20_lower = np.array(true_rul_values) - 20
    error_band_20_upper = np.array(true_rul_values) + 20
    error_band_30_lower = np.array(true_rul_values) - 30
    error_band_30_upper = np.array(true_rul_values) + 30

    # 按实际RUL升序排列引擎单元
    sorted_indices = np.argsort(true_rul_values)
    sorted_true_rul = np.array(true_rul_values)[sorted_indices]
    sorted_pred_rul = np.array(final_rul_predictions)[sorted_indices]
    
    plt.figure(figsize=(12, 8))

    # 绘制误差带，交换颜色
    plt.fill_between(np.arange(len(sorted_true_rul)), error_band_10_lower[sorted_indices], error_band_10_upper[sorted_indices], color='yellow', alpha=0.5, label="Error band ±10")
    plt.fill_between(np.arange(len(sorted_true_rul)), error_band_20_lower[sorted_indices], error_band_20_upper[sorted_indices], color='lightcoral', alpha=0.5, label="Error band ±20")
    plt.fill_between(np.arange(len(sorted_true_rul)), error_band_30_lower[sorted_indices], error_band_30_upper[sorted_indices], color='lightblue', alpha=0.5, label="Error band ±30")
    
    # 绘制实际RUL，使用黑色圆点
    plt.plot(np.arange(len(sorted_true_rul)), sorted_true_rul, label="True RUL", color='black', linestyle='-', marker='o', markersize=4)
    
    # 绘制预测RUL，使用橙色散点而不是连线
    plt.scatter(np.arange(len(sorted_true_rul)), sorted_pred_rul, label="Predicted RUL", color='orange', marker='x', s=60)

    # 添加图例并将其放置在左上角
    plt.xlabel("Test Engine Units (Sorted by Actual RUL)")
    plt.ylabel("RUL (Cycles)")
    plt.title(" ")

    # 图例左上角
    plt.legend(loc='upper left', shadow=True, ncol=1)

    plt.grid(False)  # 关闭网格
    plt.tight_layout(pad=4.0)  # 调整图表布局，确保图例不会与图形重叠
    plt.show()

# 主流程
if __name__ == "__main__":
    engine_ids, true_rul_values, final_rul_predictions = predict_final_rul(
        model, 
        df_test, 
        window_size=60, 
        device=device
    )

    # 绘制误差带图6
    plot_rul_with_error_bands(true_rul_values, final_rul_predictions)

    rmse, score = calculate_rmse_and_score(true_rul_values, final_rul_predictions)
    print(f"RMSE: {rmse}, Score: {score}")


# In[7]:


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 确保 device 已定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义 predict_engine_rul
def predict_engine_rul(model, engine_data, device=device):
    model.eval()
    with torch.no_grad():
        engine_tensor = torch.tensor(engine_data, dtype=torch.float32).unsqueeze(0).to(device)
        predictions = model(engine_tensor).cpu().numpy().flatten()
    return predictions

# 2. 定义 predict_final_rul
def predict_final_rul(model, df_test, window_size=60, device=device): 
    model.eval() 
    final_rul_predictions = [] 
    true_rul_values = []  # 用于存储真实RUL 
 
    engine_ids = df_test["Engine"].unique() 
 
    for engine_id in engine_ids: 
        engine_data = df_test[df_test["Engine"] == engine_id].copy().sort_values("Cycle") 
 
        if len(engine_data) < window_size: 
            continue
 
        feature_cols = [col for col in engine_data.columns if col not in ["Engine", "Cycle", "RUL"]] 
        X_engine = engine_data[feature_cols].values 
        true_rul = engine_data["RUL"].values 
        predicted_rul = [] 
 
        for i in range(window_size, len(X_engine) + 1): 
            window_data = X_engine[i - window_size:i] 
            rul_pred = predict_engine_rul(model, window_data, device=device) 
            predicted_rul.append(rul_pred[0]) 
 
        true_rul_aligned = true_rul[window_size - 1:] 
        predicted_rul_aligned = predicted_rul 
         
        if len(predicted_rul_aligned) == len(true_rul_aligned):
            final_rul_predictions.append(predicted_rul_aligned[-1])  
            true_rul_values.append(true_rul_aligned[-1])  
 
    return engine_ids, true_rul_values, final_rul_predictions 

# 3. 修改后的绘图函数（蓝色+橙色，去掉网格）
def plot_final_rul_comparison(true_rul_values, final_rul_predictions):
    plt.figure(figsize=(8, 6))
    indices = np.arange(len(true_rul_values))
    plt.plot(indices, true_rul_values, label="True RUL", color='green', linestyle='-', marker='o', markersize=1)
    plt.plot(indices, final_rul_predictions, label="Predicted RUL", color='SkyBlue', linestyle='--', marker='x', markersize=4)
    #plt.title("True vs Predicted Final RUL for 100 Engines")
    plt.xlabel("Engine Index")
    plt.ylabel("RUL")
    plt.legend()
    plt.xticks(rotation=90)
    plt.grid(False)  # 关闭网格
    plt.show()

# 4. 定义 calculate_rmse_and_score
def calculate_rmse_and_score(true_values, predictions):
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    score = 0
    for i in range(len(predictions)):
        diff = predictions[i] - true_values[i]
        if diff < 0:
            score += (np.exp(-diff / 13) - 1)
        else:
            score += (np.exp(diff / 10) - 1)
    score = np.mean(score)
    return rmse, score

# 5. 主流程
if __name__ == "__main__":
    engine_ids, true_rul_values, final_rul_predictions = predict_final_rul(
        model, 
        df_test, 
        window_size=60, 
        device=device
    )

    plot_final_rul_comparison(true_rul_values, final_rul_predictions)

    rmse, score = calculate_rmse_and_score(true_rul_values, final_rul_predictions)
    print(f"RMSE: {rmse}, Score: {score}")


# In[1]:


'''20窗口 RMSE: 12.807870194441552, Score: 339.1809551577538
30窗口 RMSE: 11.668448760159382, Score: 262.2277500206507
40窗口 RMSE：11.37 score：249.02
50窗口 RMSE: 12.233657944707343, Score: 309.89928080502415
60窗口 RMSE: 10.262558936446668, Score: 162.40454179513227'''


# In[ ]:




