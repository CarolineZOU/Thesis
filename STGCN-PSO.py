import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 禁用无关警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / (denominator + 1e-6)
    return 100 * np.mean(diff)

# ----------------------
# 数据预处理模块
# ----------------------
def load_and_preprocess(filepath):
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        required_cols = ['date', 'T_AQI', 'B_AQI', 'S_AQI']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        if df['date'].isnull().any():
            raise ValueError("日期格式不正确")

        start_date = '20200101'
        end_date = '20231231'
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        df = df.loc[mask]

        exclude_suffixes = ['AQI']
        feature_cols = []
        for prefix in ['T_', 'B_', 'S_']:
            feature_cols += [
                col for col in df.columns
                if col.startswith(prefix) and not any(col.endswith(suffix) for suffix in exclude_suffixes)
            ]

        for prefix in ['T_', 'B_', 'S_']:
            cols = [f"{prefix}AQI"] + [col for col in feature_cols if col.startswith(prefix)]
            df[cols] = df[cols].ffill().bfill()

        if df.isnull().any().any():
            raise ValueError("数据仍包含缺失值")

        return df, feature_cols
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise

# ----------------------
# 图结构构建模块
# ----------------------
def build_city_graph():
    distance_matrix = np.array([[0, 120, 320], [120, 0, 280], [320, 280, 0]])
    gdp_ratio = np.array([[1.0, 0.43, 0.20], [2.33, 1.0, 0.47], [5.0, 2.15, 1.0]])
    adj_matrix = 0.7 * (1 / (distance_matrix + 1e-6)) + 0.3 * gdp_ratio
    adj_matrix /= adj_matrix.max()
    adj_matrix[adj_matrix < 0.4] = 0
    return torch.tensor(adj_matrix, dtype=torch.float)

# ----------------------
# STGCN模型定义（修正维度问题）
# ----------------------
class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.temporal_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight):
        # 空间特征提取
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index, edge_weight))
        x = self.dropout(x)

        # 时间维度处理
        x = x.permute(1, 0)  # [nodes, features] -> [features, nodes]
        x = self.temporal_conv(x.unsqueeze(0))  # [1, features, nodes]
        x = x.squeeze(0).permute(1, 0)  # [nodes, features]

        # 最终预测
        return self.fc(x).squeeze()  # 输出形状 [nodes]


def particle_swarm_optimization(fitness_func, param_ranges, num_particles=5, max_iter=3):
    particles = []
    for _ in range(num_particles):
        hidden_dim = random.randint(*param_ranges['hidden_dim'])
        dropout = random.uniform(*param_ranges['dropout_rate'])
        lr = random.uniform(*param_ranges['lr'])
        particles.append({
            'position': [hidden_dim, dropout, lr],
            'velocity': [0] * 3,
            'best_pos': [hidden_dim, dropout, lr],
            'best_fitness': -float('inf')
        })

    global_best = {'position': None, 'fitness': -float('inf')}

    for _ in range(max_iter):
        for p in particles:
            fitness = fitness_func(p['position'])
            if fitness > p['best_fitness']:
                p['best_fitness'] = fitness
                p['best_pos'] = p['position'].copy()
            if fitness > global_best['fitness']:
                global_best = {'position': p['position'].copy(), 'fitness': fitness}

        for p in particles:
            for i in range(3):
                w = 0.5
                c1 = 1.5 * random.random()
                c2 = 1.5 * random.random()
                p['velocity'][i] = w * p['velocity'][i] + c1 * (p['best_pos'][i] - p['position'][i]) + c2 * (
                            global_best['position'][i] - p['position'][i])

                if i == 0:
                    new_pos = int(p['position'][i] + p['velocity'][i])
                    new_pos = max(param_ranges['hidden_dim'][0], min(new_pos, param_ranges['hidden_dim'][1]))
                else:
                    new_pos = p['position'][i] + p['velocity'][i]
                    new_pos = max(param_ranges[['dropout_rate', 'lr'][i - 1]][0],
                                  min(new_pos, param_ranges[['dropout_rate', 'lr'][i - 1]][1]))
                p['position'][i] = new_pos

    return global_best['position'], global_best['fitness']

# ----------------------
# 主执行流程
# ----------------------
if __name__ == "__main__":
    # 配置参数
    # 配置参数
    DATA_PATH = "./data/jingjinji.xlsx"
    SAVE_DIR = "STGCN_PSO_models"
    OUTPUT_DIR = "STGCN_PSO_output"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设备检测增强
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    print(f"当前使用设备: {DEVICE}")

    # 数据加载
    df, feature_cols = load_and_preprocess(DATA_PATH)
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # 图结构构建
    adj_matrix = build_city_graph()
    edge_index = adj_matrix.nonzero().t().contiguous().to(DEVICE)
    edge_weight = adj_matrix[adj_matrix.nonzero().t()[0], adj_matrix.nonzero().t()[1]].to(DEVICE)

    cities = ['T', 'B', 'S']
    city_names = {'T': '天津', 'B': '北京', 'S': '石家庄'}
    test_metrics_list = []

    for city in cities:
        target_col = f'{city}_AQI'
        city_train = train_df.iloc[:int(0.8*len(train_df))]
        city_valid = train_df.iloc[int(0.9*len(train_df)):]

        # 数据标准化
        scaler = StandardScaler()
        train_features = scaler.fit_transform(city_train[feature_cols])
        valid_features = scaler.transform(city_valid[feature_cols])

        # 转换为张量
        x_train = torch.tensor(train_features, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(city_train[target_col].values, dtype=torch.float32).to(DEVICE)
        x_valid = torch.tensor(valid_features, dtype=torch.float32).to(DEVICE)
        y_valid = torch.tensor(city_valid[target_col].values, dtype=torch.float32).to(DEVICE)

        # 定义适应度函数
        def fitness_function(params):
            model = STGCN(
                in_channels=x_train.shape[1],
                hidden_channels=int(params[0]),
                out_channels=1
            ).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=float(params[2]))
            criterion = nn.MSELoss()

            # 快速训练
            model.train()
            for _ in range(50):
                optimizer.zero_grad()
                outputs = model(x_train, edge_index, edge_weight)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # 验证集评估
            with torch.no_grad():
                pred = model(x_valid, edge_index, edge_weight)
                return -criterion(pred, y_valid).item()

        # PSO优化
        best_params, _ = particle_swarm_optimization(
            fitness_function,
            {'hidden_dim': (64, 256), 'dropout_rate': (0.1, 0.5), 'lr': (0.001, 0.1)},
            num_particles=5, max_iter=3
        )

        # 最终模型训练
        final_model = STGCN(
            in_channels=x_train.shape[1],
            hidden_channels=int(best_params[0]),
            out_channels=1
        ).to(DEVICE)
        optimizer = optim.Adam(final_model.parameters(), lr=best_params[2])
        criterion = nn.MSELoss()

        # 训练循环
        for epoch in range(300):
            final_model.train()
            optimizer.zero_grad()
            outputs = final_model(x_train, edge_index, edge_weight)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # 保存模型
        torch.save(final_model.state_dict(), f"{SAVE_DIR}/best_model_{city}.pth")

        # 测试集预测
        final_model.eval()
        with torch.no_grad():
            test_features = scaler.transform(test_df[feature_cols])
            test_x = torch.tensor(test_features, dtype=torch.float32).to(DEVICE)
            test_pred = final_model(test_x, edge_index, edge_weight).cpu().numpy()

        test_df[f'{city}_预测值'] = test_pred

        # 评估指标
        test_metrics = {
            '城市': city_names[city],
            'MSE': mean_squared_error(test_df[target_col], test_pred),
            'RMSE': np.sqrt(mean_squared_error(test_df[target_col], test_pred)),
            'MAE': mean_absolute_error(test_df[target_col], test_pred),
            'R2': r2_score(test_df[target_col], test_pred),
            'MAPE': np.mean(np.abs((test_df[target_col] - test_pred) / (test_df[target_col] + 1e-6))) * 100,
            'SMAPE': smape(test_df[target_col], test_pred)
        }
        test_metrics_list.append(test_metrics)

    # 保存结果
    test_results_df = pd.DataFrame(test_metrics_list)
    test_results_df.to_csv(f"{OUTPUT_DIR}/测试集评估结果.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/预测结果.csv", index=False)

    VIS_DIR = os.path.join(OUTPUT_DIR, "visualization")
    os.makedirs(VIS_DIR, exist_ok=True)

    for city in cities:
        city_name = city_names[city]
        target_col = f'{city}_AQI'
        pred_col = f'{city}_预测值'

        plt.figure(figsize=(15, 6))
        plt.plot(test_df['date'], test_df[target_col], label=f'真实AQI', linewidth=2, color='#1f77b4')
        plt.plot(test_df['date'], test_df[pred_col], '--', label=f'预测AQI', linewidth=2, color='#ff7f0e', alpha=0.8)

        plt.title(f'{city_name} AQI预测对比', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('AQI指数', fontsize=12)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        y_min = max(0, min(test_df[target_col].min(), test_df[pred_col].min())) * 0.95
        y_max = max(test_df[target_col].max(), test_df[pred_col].max()) * 1.05
        plt.ylim(y_min, y_max)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VIS_DIR}/{city_name}_AQI预测对比曲线图.png", dpi=300, bbox_inches='tight')
        plt.close()

    for city in cities:
        city_name = city_names[city]
        target_col = f'{city}_AQI'
        pred_col = f'{city}_预测值'

        residuals = test_df[target_col] - test_df[pred_col]

        plt.figure(figsize=(15, 6))
        plt.scatter(test_df['date'], residuals, alpha=0.6, color='#2ca02c', label='残差分布')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='零误差线')

        plt.title(f'{city_name} AQI预测残差', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('残差 (真实值 - 预测值)', fontsize=12)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VIS_DIR}/{city_name}_AQI残差图.png", dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(18, 9))
    colors = {'天津': '#1f77b4', '北京': '#ff7f0e', '石家庄': '#2ca02c'}

    for city in cities:
        city_name = city_names[city]
        target_col = f'{city}_AQI'
        pred_col = f'{city}_预测值'

        plt.plot(test_df['date'], test_df[target_col], color=colors[city_name], linewidth=1.5, linestyle='-', label=f'{city_name}真实值')
        plt.plot(test_df['date'], test_df[pred_col], color=colors[city_name], linewidth=1.5, linestyle='--', label=f'{city_name}预测值')

    plt.title('京津冀三城市AQI预测对比', fontsize=16)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('AQI指数', fontsize=14)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=3, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/三城市AQI预测综合对比图.png", dpi=300, bbox_inches='tight')
    plt.close()

    latest_dates = test_df['date'].sort_values(ascending=False).head(30).sort_values()
    comparison_df = test_df[test_df['date'].isin(latest_dates)].copy()

    table_data = []
    for _, row in comparison_df.iterrows():
        record = {'日期': row['date'].strftime('%Y-%m-%d')}
        for city in cities:
            city_name = city_names[city]
            target_col = f'{city}_AQI'
            pred_col = f'{city}_预测值'
            record[f'{city_name}真实值'] = round(row[target_col], 1)
            record[f'{city_name}预测值'] = round(row[pred_col], 1)
            record[f'{city_name}残差'] = round(row[target_col] - row[pred_col], 1)
        table_data.append(record)

    comparison_table = pd.DataFrame(table_data)
    comparison_table.to_excel(f"{OUTPUT_DIR}/真实预测对比表格.xlsx", index=False)

    plt.figure(figsize=(18, 12))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    table = pd.plotting.table(ax, comparison_table.round(1).head(10), loc='center', cellLoc='center', colWidths=[0.1] * len(comparison_table.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.savefig(f"{VIS_DIR}/对比表格示例.png", dpi=300, bbox_inches='tight')
    plt.close()