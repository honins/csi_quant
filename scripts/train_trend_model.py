import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import os

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_module import DataModule
from src.utils.config_loader import load_config

def train_trend_model():
    print("🚀 开始训练趋势识别模型 (Trend Recognition Model)...")
    
    config = load_config()
    dm = DataModule(config)
    
    # 1. 获取数据 (过去5年)
    print("📅 加载历史数据...")
    data = dm.get_history_data('2020-01-01', '2026-01-01')
    data = dm.preprocess_data(data)
    
    # 2. 生成 V2 特征
    print("🔧 生成多周期特征 (Weekly + ATR + ADX)...")
    data = dm.get_features_v2(data)
    
    # 3. 定义目标 (Target)
    # Bull (1): 未来20天涨幅 > 3%
    # Bear (-1): 未来20天跌幅 > 3%
    # Chop (0): 震荡
    future_days = 20
    threshold = 0.03
    
    data['future_ret'] = data['close'].shift(-future_days) / data['close'] - 1
    
    conditions = [
        data['future_ret'] > threshold,
        data['future_ret'] < -threshold
    ]
    choices = [1, -1]
    data['target'] = np.select(conditions, choices, default=0)
    
    # 4. 准备训练集
    # 移除最后20天没有标签的数据，以及前面的NaN
    features = ['adx', 'plus_di', 'minus_di', 'natr', 'obv_slope', 
                'w_trend', 'rsi', 'macd', 'dist_ma20', 'volume_ratio']
                
    # 确保特征列存在
    available_features = [f for f in features if f in data.columns]
    if len(available_features) < len(features):
        print(f"⚠️ 警告: 部分特征缺失: {set(features) - set(available_features)}")
    
    clean_data = data.dropna(subset=available_features + ['target', 'future_ret'])
    
    X = clean_data[available_features]
    y = clean_data['target']
    
    print(f"📊 样本总数: {len(X)}")
    print(f"   Bull: {sum(y==1)} ({sum(y==1)/len(y):.1%})")
    print(f"   Bear: {sum(y==-1)} ({sum(y==-1)/len(y):.1%})")
    print(f"   Chop: {sum(y==0)} ({sum(y==0)/len(y):.1%})")
    
    # 按时间切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. 训练模型
    print("🤖 训练 Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # 6. 评估
    print("\n📈 模型评估报告 (Test Set):")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Bear', 'Chop', 'Bull']))
    
    # 特征重要性
    importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    print("\n🌟 特征重要性:")
    print(importances)

if __name__ == "__main__":
    train_trend_model()
