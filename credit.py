import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
import json
# 参数网格配置
param_grid = {
    "model_params": {
        "hidden_layers": [[1024, 512, 256, 128],[512, 256, 128, 64]],  # 保持不变
        "dropout_rate": [0.3],  # 修改为列表
    },
    "training_params": {
        "epochs": [2000],
        "batch_sizes": [3000, 5000, 10000],
        "learning_rates": [0.01,0.001],
        "pos_weights": [3.0, 5.0],
        "thresholds": [round(i, 2) for i in np.arange(0.7, 0.85, 0.02)],
        "target_losses": [0.001,0.0001],
        "convergence_windows": [400],
        "min_lrs": [1e-6]
    },
    "early_stopping": {
        "patience_values": [300],
        "min_deltas": [1e-6]
    }
}

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.5):
        super(FFNN, self).__init__()
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 添加残差连接
        self.layers = nn.ModuleList(layers)
        self.shortcuts = nn.ModuleList([
            nn.Linear(input_dim, hidden_layers[i]) 
            for i in range(len(hidden_layers))
        ])
        
        self.final = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        x = self.batch_norm_input(x)
        original_x = x
        
        for i in range(0, len(self.layers), 4):
            identity = original_x
            x = self.layers[i](x)
            x = self.layers[i+1](x)
            x = self.layers[i+2](x)
            x = self.layers[i+3](x)
            x = x + self.shortcuts[i//4](identity)  # 残差连接
        
        return self.final(x)

def create_data_generator(X, y, batch_size, device):
    n = X.shape[0]
    while True:
        idxs = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_idxs = idxs[i:min(i + batch_size, n)]
            X_batch = torch.tensor(X.iloc[batch_idxs].values, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y.iloc[batch_idxs].values.squeeze(), dtype=torch.long).to(device)
            yield X_batch, y_batch

def create_data_loader(X, y, batch_size):
    # 预先将数据转换为CUDA张量
    X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y.values, dtype=torch.float32, device=device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=False,  # 由于数据已经在GPU上，关闭pin_memory
        num_workers=0      # 由于数据已经在GPU上，不需要workers
    )

class ModelCheckpoint:
    def __init__(self, model, path='best_model.pth', mode='max'):
        self.model = model
        self.path = path
        self.best_f1 = 0
        self.best_recall = 0
        self.best_metrics = None
        self.mode = mode
    
    def __call__(self, metrics):
        f1 = metrics.get('f1', 0)
        recall = metrics.get('recall', 0)
        
        if f1 > self.best_f1:
            print(f"\n发现更好的F1分数: {f1:.4f} > {self.best_f1:.4f}")
            self.best_f1 = f1
            self.best_recall = recall
            self.best_metrics = metrics
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'metrics': metrics,
                'f1': f1,
                'recall': recall
            }, self.path)
            return True
        return False

def find_optimal_threshold(model, valid_loader, device, min_precision=0.7):
    """
    在验证集上寻找最优决策阈值
    min_precision: 可接受的最小精确率
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    # 收集所有预测值和真实标签
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # 尝试不同的阈值
    thresholds = np.arange(0.4, 0.95, 0.05)
    best_threshold = 0.5
    best_recall = 0
    best_metrics = None
    
    print("\n寻找最优阈值...")
    for threshold in thresholds:
        preds = (all_outputs > threshold).astype(float)
        precision = precision_score(all_labels, preds)
        recall = recall_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)
        
        # 只有当精确率满足最低要求时才考虑
        if precision >= min_precision:
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                print(f"发现更好的阈值: {threshold:.2f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1: {f1:.4f}")
    
    return best_threshold, best_metrics

def validate_epoch(model, valid_loader, criterion, device, threshold):
    """验证函数使用固定阈值"""
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    preds = (all_outputs > threshold).astype(float)
    
    metrics = {
        'precision': precision_score(all_labels, preds),
        'recall': recall_score(all_labels, preds),
        'f1': f1_score(all_labels, preds),
        'threshold': threshold
    }
    
    return running_loss / len(valid_loader), metrics

def plot_threshold_metrics(model, valid_loader, device, save_path='threshold_metrics.png'):
    """
    绘制阈值与召回率、精确率的关系图
    """
    model.eval()
    all_outputs = []
    all_labels = []
    
    # 收集所有预测值和真实标签
    with torch.no_grad():
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # 生成更密集阈值列
    thresholds = np.linspace(0.01, 0.99, 100)
    recalls = []
    precisions = []
    f1_scores = []
    
    # 计算每个阈值的指标
    for threshold in thresholds:
        preds = (all_outputs > threshold).astype(float)
        recalls.append(recall_score(all_labels, preds))
        precisions.append(precision_score(all_labels, preds))
        f1_scores.append(f1_score(all_labels, preds))
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制主要指标
    plt.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
    plt.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
    plt.plot(thresholds, f1_scores, 'g--', label='F1 Score', alpha=0.7)
    
    # 找到F1最高点
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    # 添加最佳F1点的标记
    plt.plot(best_threshold, f1_scores[best_f1_idx], 'go', 
            label=f'Best F1: {f1_scores[best_f1_idx]:.3f} at {best_threshold:.3f}')
    
    # 添加网格图例
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 设置标签和标题
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Decision Threshold')
    
    # 添加阈值-召回的详细信息
    text_info = (f'Threshold: {best_threshold:.3f}\n'
                f'Recall: {recalls[best_f1_idx]:.3f}\n'
                f'Precision: {precisions[best_f1_idx]:.3f}\n'
                f'F1: {f1_scores[best_f1_idx]:.3f}')
    
    plt.text(1.02, 0.2, text_info, transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return best_threshold

def get_scheduler(optimizer, config):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50,
        min_lr=config['training_params']['min_lr']
    )

def train_model(model, x_train, y_train, x_test, y_test, config, device):
    train_loader = create_data_loader(x_train, y_train, config['training_params']['batch_size'])
    valid_loader = create_data_loader(x_test, y_test, config['training_params']['batch_size'])
    
    pos_weight = torch.tensor([config['training_params']['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training_params']['learning_rate'],
        weight_decay=0.01
    )
    
    scheduler = get_scheduler(optimizer, config)
    early_stopping = EarlyStopping(**config['early_stopping'])
    
    history = {
        'train_losses': [], 'valid_losses': [],
        'learning_rates': [], 'precisions': [], 'recalls': []
    }
    
    checkpoint = ModelCheckpoint(model, path='best_model.pth')
    window_size = config['training_params']['convergence_window']
    
    # 使用配置中定的阈值
    threshold = config['training_params']['threshold']
    
    print(f"\n开始训练，总轮数：{config['training_params']['epochs']}")
    for epoch in range(config['training_params']['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, metrics = validate_epoch(model, valid_loader, criterion, device, threshold)
        
        if epoch % 10 == 0:  # 每10轮打印一次详细信息
            print(f"\nEpoch {epoch}/{config['training_params']['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1: {2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-8):.4f}")
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 在调用 scheduler.step() 之前记录旧的学习率
        old_lr = current_lr
        
        scheduler.step(valid_loss)
        
        # 如果学习率发生变化，打通知
        if optimizer.param_groups[0]['lr'] != old_lr:
            print(f'Learning rate decreased to {optimizer.param_groups[0]["lr"]}')
        
        # 更新历史记录
        history['train_losses'].append(train_loss)
        history['valid_losses'].append(valid_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['precisions'].append(metrics['precision'])
        history['recalls'].append(metrics['recall'])
        
        # 检查收敛
        if check_convergence(history['valid_losses'], window_size):
            conv_metrics = get_convergence_metrics(history, window_size)
            print("\n模型已收敛，最近50轮平均指标：")
            print(f"Precision: {conv_metrics['precision']:.4f}")
            print(f"Recall: {conv_metrics['recall']:.4f}")
            
            # 如果收敛后的指标更好，更新最佳模型
            if conv_metrics['recall'] > checkpoint.best_recall:
                checkpoint(conv_metrics)
        
        if early_stopping(valid_loss):
            break
    
    return history

def evaluate_model(model, x_test, y_test, config, device):
    model.eval()
    test_loader = create_data_loader(x_test, y_test, config['training_params']['batch_size'])
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("预测值分布：", np.unique(all_preds, return_counts=True))
    print("真实值分布：", np.unique(all_labels, return_counts=True))
    
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def plot_training_history(history):
    """绘制训练历史图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失下降图
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['valid_losses'], label='Valid Loss')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 学习率变化图
    ax2.plot(history['learning_rates'])
    ax2.set_title('Learning Rate Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    # 精确率和召回率变化图
    ax3.plot(history['precisions'], label='Precision')
    ax3.plot(history['recalls'], label='Recall')
    ax3.set_title('Metrics Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # F1分数变化图
    f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                 for p, r in zip(history['precisions'], history['recalls'])]
    ax4.plot(f1_scores)
    ax4.set_title('F1 Score Over Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

class EarlyStopping:
    def __init__(self, patience=5000, min_delta=1e-8):  # 增加patience和降低min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # 移除进度条以减少开销
    for inputs, labels in train_loader:
        # 数据已在GPU上，不需要再次转移
        labels = labels.view(-1, 1)
        
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True优化内存
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def preprocess_data(df, scaler=None):
    """
    预处理数据，包括特征工程和标准化
    参数:
        df: 输入数据框
        scaler: 如果提供则使用已有的scaler进行转换，否则创建新的scaler
    返回:
        处理后的数据框和scaler
    """
    df = df.copy()
    
    # 添加特征工程
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # 创建新特征的字典
    new_features = {}
    
    # 添加交互特征
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            feat_name = f"interaction_{i}_{j}"
            new_features[feat_name] = df[numeric_features[i]] * df[numeric_features[j]]
    
    # 添加多项式特征
    for feat in numeric_features:
        new_features[f"{feat}_squared"] = df[feat] ** 2
    
    # 一次性添加所有新特征
    new_df = pd.concat([df] + [pd.Series(v, name=k) for k, v in new_features.items()], axis=1)
    
    # 标准化所有数值特征
    numeric_features = new_df.select_dtypes(include=['float64', 'int64']).columns
    if scaler is None:
        scaler = StandardScaler()
        new_df[numeric_features] = scaler.fit_transform(new_df[numeric_features])
    else:
        new_df[numeric_features] = scaler.transform(new_df[numeric_features])
    
    # 处理异常值
    for col in numeric_features:
        new_df[col] = new_df[col].clip(-3, 3)  # 限制在3个标准差内
    
    return new_df, scaler

def load_best_model(model, path='best_recall_model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n加载最佳模型:")
    print(f"Best Recall: {checkpoint['best_recall']:.4f}")
    print(f"对应的指标:")
    print(f"Precision: {checkpoint['metrics']['precision']:.4f}")
    print(f"F1: {checkpoint['metrics']['f1']:.4f}")
    return model

def save_experiment_results(config, metrics, timestamp, base_dir='experiments'):
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 生成参数配置标识
    param_id = (f"hl{len(config['model_params']['hidden_layers'])}_"
               f"dr{config['model_params']['dropout_rate']}_"
               f"lr{config['training_params']['learning_rate']}_"
               f"bs{config['training_params']['batch_size']}_"
               f"pw{config['training_params']['pos_weight']}_"
               f"th{config['training_params']['threshold']}")
    
    # 保存训练历史图
    plot_training_history(metrics['history']).savefig(
        os.path.join(experiment_dir, f"{param_id}_training_history.png"))
    plt.close()
    
    # 保存阈值分析图
    plot_threshold_metrics(metrics['model'], metrics['valid_loader'], device,
                         os.path.join(experiment_dir, f"{param_id}_threshold_metrics.png"))
    
    # 准备结果数据
    results = {
        'timestamp': timestamp,
        'param_id': param_id,
        'hidden_layers': str(config['model_params']['hidden_layers']),
        'dropout_rate': config['model_params']['dropout_rate'],
        'learning_rate': config['training_params']['learning_rate'],
        'batch_size': config['training_params']['batch_size'],
        'pos_weight': config['training_params']['pos_weight'],
        'threshold': config['training_params']['threshold'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'training_time': metrics['training_time'],
        'early_stopped': metrics.get('early_stopped', False),
        'convergence_epoch': metrics.get('convergence_epoch', None)
    }
    # 保存并排序结果
    csv_path = os.path.join(base_dir, 'experiments_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    else:
        df = pd.DataFrame([results])
    
    # 按召回率降序排序
    df = df.sort_values('recall', ascending=False)
    df.to_csv(csv_path, index=False)
    
    return results

def save_error_log(config, error_msg, timestamp, base_dir='experiments'):
    """记录训练错误信息"""
    os.makedirs(base_dir, exist_ok=True)
    error_log_path = os.path.join(base_dir, 'training_errors.csv')
    
    error_data = {
        'timestamp': timestamp,
        'hidden_layers': str(config['model_params']['hidden_layers']),
        'dropout_rate': config['model_params']['dropout_rate'],
        'batch_size': config['training_params']['batch_size'],
        'learning_rate': config['training_params']['learning_rate'],
        'pos_weight': config['training_params']['pos_weight'],
        'threshold': config['training_params']['threshold'],
        'error_message': str(error_msg),
        'has_error': 1
    }
    
    # 保存错误日志
    if os.path.exists(error_log_path):
        df = pd.read_csv(error_log_path)
        df = pd.concat([df, pd.DataFrame([error_data])], ignore_index=True)
    else:
        df = pd.DataFrame([error_data])
    
    df.to_csv(error_log_path, index=False)

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据
    dataset = load_dataset("jbrazzy/credit-card")
    df = dataset['train'].to_pandas()
    
    # 分割特征和标签
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].to_frame()
    
    # 创建训练集和测试集
    train_size = int(0.8 * len(df))
    indices = torch.randperm(len(df))
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    x_train = X.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    
    # 数据预处理
    x_train, scaler = preprocess_data(x_train)
    x_test, _ = preprocess_data(x_test, scaler=scaler)
    
    print("开始参数网格搜索...")
    
    for threshold in param_grid['training_params']['thresholds']:
        for hidden_layers in param_grid['model_params']['hidden_layers']:
            for dropout_rate in param_grid['model_params']['dropout_rate']:
                for batch_size in param_grid['training_params']['batch_sizes']:
                    for lr in param_grid['training_params']['learning_rates']:
                        for pos_weight in param_grid['training_params']['pos_weights']:
                            config = {
                                "model_params": {
                                    "hidden_layers": hidden_layers,
                                    "dropout_rate": dropout_rate
                                },
                                "training_params": {
                                    "epochs": param_grid['training_params']['epochs'][0],
                                    "batch_size": batch_size,
                                    "learning_rate": lr,
                                    "pos_weight": pos_weight,
                                    "threshold": threshold,
                                    "target_loss": param_grid['training_params']['target_losses'][0],
                                    "convergence_window": param_grid['training_params']['convergence_windows'][0],
                                    "min_lr": param_grid['training_params']['min_lrs'][0]
                                },
                                "early_stopping": {
                                    "patience": param_grid['early_stopping']['patience_values'][0],
                                    "min_delta": param_grid['early_stopping']['min_deltas'][0]
                                }
                            }
                            
                            print(f"\n开始训练新的参数组合：")
                            print(f"决策阈值: {threshold}")
                            print(f"隐藏层: {hidden_layers}")
                            print(f"Dropout率: {dropout_rate}")
                            print(f"批量大小: {batch_size}")
                            print(f"学习率: {lr}")
                            print(f"正样本权重: {pos_weight}")
                            
                            try:
                                model = train_and_evaluate(x_train, y_train, x_test, y_test, config)
                            except Exception as e:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_error_log(config, e, timestamp)
                                print(f"\n当前参数组合训练失败，错误信息: {str(e)}")
                                print("已记录错误信息，继续下一组参数训练...")
                                continue

    print("\n所有参数组合训练完成！")

def check_convergence(metrics_history, window_size=50):
    """检查最近window_size轮的指标是否稳定"""
    if len(metrics_history) < window_size:
        return False
        
    recent = metrics_history[-window_size:]
    mean_value = np.mean(recent)
    std_value = np.std(recent)
    
    # 如果标准差小于均值的1%，认为已收敛
    return std_value < (mean_value * 0.01)

def get_convergence_metrics(history, window_size=50):
    """获取收敛后的平均指标"""
    if len(history['valid_losses']) < window_size:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'valid_loss': float('inf')
        }
        
    recent_metrics = {
        'precision': np.mean(history['precisions'][-window_size:]),
        'recall': np.mean(history['recalls'][-window_size:]),
        'valid_loss': np.mean(history['valid_losses'][-window_size:])
    }
    return recent_metrics

def grid_search(param_grid, x_train, y_train, x_test, y_test, device):
    base_dir = 'experiments'
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        for threshold in param_grid['training_params']['thresholds']:
            config = {
                "model_params": param_grid["model_params"].copy(),
                "training_params": {
                    "threshold": threshold,
                    # ... 其他参数 ...
                }
            }
            
            start_time = time.time()
            model = FFNN(input_dim=x_train.shape[1],
                        hidden_layers=config['model_params']['hidden_layers'],
                        dropout_rate=config['model_params']['dropout_rate']).to(device)
            
            try:
                history = train_model(model, x_train, y_train, x_test, y_test, config, device)
                training_time = time.time() - start_time
                
                metrics = {
                    'model': model,
                    'history': history,
                    'valid_loader': create_data_loader(x_test, y_test, config['training_params']['batch_size']),
                    'precision': history['precisions'][-1],
                    'recall': history['recalls'][-1],
                    'f1': 2 * (history['precisions'][-1] * history['recalls'][-1]) / 
                         (history['precisions'][-1] + history['recalls'][-1]),
                    'training_time': training_time
                }
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_experiment_results(config, metrics, timestamp)
                
            except KeyboardInterrupt:
                print("\n训练被手动中断，保存当前结果...")
                metrics['early_stopped'] = True
                save_experiment_results(config, metrics, timestamp)
                if input("\n是否继续训练下一组参数？(y/n): ").lower() != 'y':
                    break
                    
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return

def train_and_evaluate(x_train, y_train, x_test, y_test, config):
    """
    训练并评估模型
    """
    start_time = time.time()
    
    # 初始化模型
    model = FFNN(
        input_dim=x_train.shape[1],
        hidden_layers=config['model_params']['hidden_layers'],
        dropout_rate=config['model_params']['dropout_rate']
    ).to(device)
    
    # 训练模型
    history = train_model(model, x_train, y_train, x_test, y_test, config, device)
    
    # 评估模型
    metrics = evaluate_model(model, x_test, y_test, config, device)
    print("\n最终评估结果：")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # 创建验证数据加载器
    valid_loader = create_data_loader(x_test, y_test, config['training_params']['batch_size'])
    
    # 保存训练结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_metrics = {
        'model': model,
        'history': history,
        'valid_loader': valid_loader,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'training_time': time.time() - start_time,
        'early_stopped': history.get('early_stopped', False),
        'convergence_epoch': history.get('convergence_epoch', None),
        'final_train_loss': history['train_losses'][-1],
        'final_valid_loss': history['valid_losses'][-1],
        'best_precision': max(history['precisions']),
        'best_recall': max(history['recalls'])
    }
    
    save_experiment_results(config, experiment_metrics, timestamp)
    
    return model

if __name__ == "__main__":
    main()