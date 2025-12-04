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
import seaborn as sns
import random
from datetime import datetime
import os
import shap

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
        required_cols = ['date', 'T_PM2.5', 'B_PM2.5', 'S_PM2.5']
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

        exclude_suffixes = ['PM2.5']
        feature_cols = []
        for prefix in ['T_', 'B_', 'S_']:
            feature_cols += [
                col for col in df.columns
                if col.startswith(prefix) and not any(col.endswith(suffix) for suffix in exclude_suffixes)
            ]

        for prefix in ['T_', 'B_', 'S_']:
            cols = [f"{prefix}PM2.5"] + [col for col in feature_cols if col.startswith(prefix)]
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
# 增强版混合优化算法 (PSO-WOA)
# ----------------------
# 修改混合优化算法中的全局最优初始化部分
def hybrid_pso_woa(fitness_func, param_ranges, num_agents=10, max_iter=5):
    # 初始化种群
    population = []
    for _ in range(num_agents):
        individual = [
            random.randint(*param_ranges['hidden_dim']),
            random.uniform(*param_ranges['dropout_rate']),
            random.uniform(*param_ranges['lr'])
        ]
        fitness = fitness_func(individual)
        population.append({
            'position': np.array(individual),
            'velocity': np.zeros(3),
            'best_pos': np.array(individual),
            'best_fitness': fitness  # 统一使用best_fitness作为键名
        })

    # 初始化全局最优（保持键名一致性）
    global_best = {
        'position': None,
        'best_fitness': -float('inf')
    }

    # 找到初始最优解
    best_individual = max(population, key=lambda x: x['best_fitness'])
    global_best['position'] = best_individual['position'].copy()
    global_best['best_fitness'] = best_individual['best_fitness']

    for iter in range(max_iter):
        a = 2 - iter * (2 / max_iter)
        a2 = -1 + iter * (-1 / max_iter)

        for i in range(num_agents):
            # PSO更新规则
            if i < num_agents // 2:
                w = 0.5
                c1 = 1.5 * random.random()
                c2 = 1.5 * random.random()
                population[i]['velocity'] = w * population[i]['velocity'] + \
                                            c1 * random.random() * (
                                                        population[i]['best_pos'] - population[i]['position']) + \
                                            c2 * random.random() * (global_best['position'] - population[i]['position'])
                new_pos = population[i]['position'] + population[i]['velocity']
            # WOA更新规则
            else:
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2

                if np.abs(A) < 1:
                    D = np.abs(C * global_best['position'] - population[i]['position'])
                    new_pos = global_best['position'] - A * D
                else:
                    rand_agent = population[random.randint(0, num_agents - 1)]
                    D = np.abs(C * rand_agent['position'] - population[i]['position'])
                    new_pos = rand_agent['position'] - A * D

                # 螺旋更新
                l = np.random.uniform(-1, 1)
                D_prime = np.abs(global_best['position'] - population[i]['position'])
                new_pos_spiral = D_prime * np.exp(a2 * l) * np.cos(2 * np.pi * l) + global_best['position']
                if random.random() < 0.5:
                    new_pos = new_pos_spiral

            # 边界处理
            new_pos[0] = np.clip(new_pos[0], *param_ranges['hidden_dim'])
            new_pos[1] = np.clip(new_pos[1], *param_ranges['dropout_rate'])
            new_pos[2] = np.clip(new_pos[2], *param_ranges['lr'])
            new_pos[0] = int(new_pos[0])

            # 评估新位置
            new_fitness = fitness_func(new_pos)

            # 更新个体最优
            if new_fitness > population[i]['best_fitness']:
                population[i]['best_pos'] = new_pos.copy()
                population[i]['best_fitness'] = new_fitness

                # 更新全局最优
                if new_fitness > global_best['best_fitness']:
                    global_best['position'] = new_pos.copy()
                    global_best['best_fitness'] = new_fitness

            return global_best['position'], global_best['best_fitness']


# ----------------------
# 增强版STGCN模型定义
# ----------------------
class STGCN_WOA_PSO(nn.Module):
    """无注意力机制版本"""

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

        # 时间维度处理
        x = x.permute(1, 0)
        x = self.temporal_conv(x.unsqueeze(0))
        x = x.squeeze(0).permute(1, 0)

        return self.fc(x).squeeze()

    def get_gcn_weights(self):
        """获取GCN层的权重矩阵"""
        return self.gcn1.lin.weight.detach().cpu().numpy()

def plot_transfer_heatmap(model, feature_names, city_names, epoch, vis_dir, top_n=20):
    """改进版特征-城市权重热力图生成"""
    plt.figure(figsize=(12, 16))

    # 获取GCN第一层权重矩阵 [hidden_dim, input_dim]
    weights = model.get_gcn_weights()

    # 特征维度安全检查（比较列数）
    assert len(feature_names) == weights.shape[1], \
        f"特征名称数量({len(feature_names)})与权重维度({weights.shape[1]})不匹配"

    # 动态调整top_n值
    valid_top_n = min(top_n, weights.shape[1])
    if valid_top_n != top_n:
        print(f"警告：top_n自动调整为{valid_top_n}（最大可用特征数）")

    # 计算特征重要性（按列平均）
    importance = np.abs(weights).mean(axis=0)
    top_indices = np.argsort(importance)[-valid_top_n:]

    # 筛选权重和特征名称（选择对应的列）
    filtered_weights = weights[:, top_indices]
    filtered_features = [feature_names[i] for i in top_indices]

    # 创建城市标签（隐藏层节点，可能需要调整）
    city_labels = [f"节点{i+1}" for i in range(weights.shape[0])]

    # 绘制热力图
    ax = sns.heatmap(
        filtered_weights,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        annot_kws={"size": 8},
        linewidths=0.5,
        xticklabels=filtered_features,
        yticklabels=city_labels
    )

    # 优化坐标轴显示
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlabel("输入特征", fontsize=12, labelpad=10)
    plt.ylabel("隐藏层节点", fontsize=12, labelpad=10)
    plt.title(f"特征-节点权重分布 (第{epoch}轮训练)", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(f"{vis_dir}/transfer_heatmap_epoch{epoch}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------
# 主执行流程
# ----------------------
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = "./data/jingjinji.xlsx"
    SAVE_DIR = "STGCN_PSO_WOA_PM2.5_models"
    OUTPUT_DIR = "STGCN_PSO_WOA_PM2.5_output"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设备检测
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {DEVICE}")

    # 数据加载与预处理
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

    # 模型选择
    use_attention = False  # 控制是否使用注意力版本
    ModelClass = STGCN_WOA_PSO

    for city in cities:
        target_col = f'{city}_PM2.5'
        city_train = train_df.iloc[:int(0.85 * len(train_df))]
        city_valid = train_df.iloc[int(0.9 * len(train_df)):]

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
            model = ModelClass(
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
                outputs = model(x_train, edge_index, edge_weight)[0] if use_attention else model(x_train, edge_index,
                                                                                                 edge_weight)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # 验证集评估
            with torch.no_grad():
                pred = model(x_valid, edge_index, edge_weight)[0] if use_attention else model(x_valid, edge_index,
                                                                                              edge_weight)
                return -criterion(pred, y_valid).item()


        # 混合优化
        best_params, _ = hybrid_pso_woa(
            fitness_function,
            {'hidden_dim': (64, 256), 'dropout_rate': (0.1, 0.5), 'lr': (0.001, 0.1)},
            num_agents=10,
            max_iter=5
        )

        # 模型训练
        final_model = ModelClass(
            in_channels=x_train.shape[1],
            hidden_channels=int(best_params[0]),
            out_channels=1
        ).to(DEVICE)
        optimizer = optim.Adam(final_model.parameters(), lr=best_params[2])
        criterion = nn.MSELoss()

        # 训练循环
        vis_dir = os.path.join(OUTPUT_DIR, "visualization", city)
        os.makedirs(vis_dir, exist_ok=True)

        for epoch in range(300):
            final_model.train()
            optimizer.zero_grad()
            outputs = final_model(x_train, edge_index, edge_weight)[0] if use_attention else final_model(x_train,
                                                                                                         edge_index,
                                                                                                         edge_weight)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()


            def generate_feature_names(feature_cols):
                """根据特征列名动态生成中文特征名称"""
                city_mapping = {'T': '天津', 'B': '北京', 'S': '石家庄'}
                pollutant_mapping = {
                    'PM2.5': 'PM2.5浓度', 'PM10': 'PM10浓度',
                    'SO2': '二氧化硫', 'NO2': '二氧化氮',
                    'CO': '一氧化碳', 'O3': '臭氧',
                    'tem': '温度', 'pre': '气压',
                    'hum': '湿度', 'speed': '风速',
                    'dewpoint': '露点', 'precipitation': '降水量'
                }

                feature_names = []
                for col in feature_cols:
                    # 解析原始列名（示例：'T_PM2.5' -> ('天津', 'PM2.5浓度')）
                    parts = col.split('_')
                    if len(parts) >= 2:
                        city_code = parts[0]
                        pollutant_code = '_'.join(parts[1:])  # 处理复合名称如'dewpoint'
                        city_name = city_mapping.get(city_code, city_code)
                        pollutant_name = pollutant_mapping.get(pollutant_code, pollutant_code)
                        feature_names.append(f"{city_name}_{pollutant_name}")
                    else:
                        feature_names.append(col)
                return feature_names


            # 在数据加载后调用
            df, feature_cols = load_and_preprocess(DATA_PATH)
            feature_names = generate_feature_names(feature_cols)  # 根据实际特征列补充完整

            # 确保特征名称数量匹配
            assert len(feature_names) == len(feature_cols), \
                f"特征名称数量({len(feature_names)})与数据特征数({len(feature_cols)})不匹配"

            # 在训练循环中修改调用方式
            if epoch % 50 == 0:
                # 在主流程中添加特征名称生成逻辑

                # 替换原有的硬编码feature_names
                plot_transfer_heatmap(
                    model=final_model,
                    feature_names=feature_names,  # 使用动态生成的名称
                    city_names={'T': '天津', 'B': '北京', 'S': '石家庄'},
                    epoch=epoch,
                    vis_dir=vis_dir,
                    top_n=15
                )

        # 保存模型
        torch.save(final_model.state_dict(), f"{SAVE_DIR}/best_model_{city}.pth")

        # 测试集预测
        final_model.eval()
        with torch.no_grad():
            test_features = scaler.transform(test_df[feature_cols])
            test_x = torch.tensor(test_features, dtype=torch.float32).to(DEVICE)
            test_pred = final_model(test_x, edge_index, edge_weight)[0].cpu().numpy() if use_attention else final_model(
                test_x, edge_index, edge_weight).cpu().numpy()

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
        target_col = f'{city}_PM2.5'
        pred_col = f'{city}_预测值'

        plt.figure(figsize=(15, 6))
        plt.plot(test_df['date'], test_df[target_col], label=f'真实PM2.5', linewidth=2, color='#1f77b4')
        plt.plot(test_df['date'], test_df[pred_col], '--', label=f'预测PM2.5', linewidth=2, color='#ff7f0e', alpha=0.8)

        plt.title(f'{city_name} PM2.5预测对比', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('PM2.5', fontsize=12)

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
        plt.savefig(f"{VIS_DIR}/{city_name}_PM2.5预测对比曲线图.png", dpi=300, bbox_inches='tight')
        plt.close()

    for city in cities:
        city_name = city_names[city]
        target_col = f'{city}_PM2.5'
        pred_col = f'{city}_预测值'

        residuals = test_df[target_col] - test_df[pred_col]

        plt.figure(figsize=(15, 6))
        plt.scatter(test_df['date'], residuals, alpha=0.6, color='#2ca02c', label='残差分布')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='零误差线')

        plt.title(f'{city_name} PM2.5预测残差', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('残差 (真实值 - 预测值)', fontsize=12)

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VIS_DIR}/{city_name}_PM2.5残差图.png", dpi=300, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(18, 9))
    colors = {'天津': '#1f77b4', '北京': '#ff7f0e', '石家庄': '#2ca02c'}

    for city in cities:
        city_name = city_names[city]
        target_col = f'{city}_PM2.5'
        pred_col = f'{city}_预测值'

        plt.plot(test_df['date'], test_df[target_col], color=colors[city_name], linewidth=1.5, linestyle='-',
                 label=f'{city_name}真实值')
        plt.plot(test_df['date'], test_df[pred_col], color=colors[city_name], linewidth=1.5, linestyle='--',
                 label=f'{city_name}预测值')

    plt.title('京津冀三城市PM2.5预测对比', fontsize=16)
    plt.xlabel('日期', fontsize=14)
    plt.ylabel('PM2.5指数', fontsize=14)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=3, fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/三城市PM2.5预测综合对比图.png", dpi=300, bbox_inches='tight')
    plt.close()

    latest_dates = test_df['date'].sort_values(ascending=False).head(30).sort_values()
    comparison_df = test_df[test_df['date'].isin(latest_dates)].copy()

    table_data = []
    for _, row in comparison_df.iterrows():
        record = {'日期': row['date'].strftime('%Y-%m-%d')}
        for city in cities:
            city_name = city_names[city]
            target_col = f'{city}_PM2.5'
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
    table = pd.plotting.table(ax, comparison_table.round(1).head(10), loc='center', cellLoc='center',
                              colWidths=[0.1] * len(comparison_table.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.savefig(f"{VIS_DIR}/对比表格示例.png", dpi=300, bbox_inches='tight')
    plt.close()