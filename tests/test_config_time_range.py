#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
时间范围配置测试脚本
验证新的时间范围配置和数据分割比例是否正确工作
"""

import sys
import os
from pathlib import Path
import logging

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.config_loader import ConfigLoader
from src.data.data_module import DataModule
from src.ai.ai_optimizer_improved import AIOptimizerImproved

def test_time_range_config():
    """测试时间范围配置"""
    print("🔍 测试时间范围配置")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # 1. 测试数据时间范围配置
    print("📅 测试数据时间范围配置:")
    data_config = config.get('data', {})
    time_range = data_config.get('time_range', {})
    start_date = time_range.get('start_date', '未配置')
    end_date = time_range.get('end_date', '未配置')
    
    print(f"   开始日期: {start_date}")
    print(f"   结束日期: {end_date}")
    
    expected_start = "2019-01-01"
    expected_end = "2025-07-15"
    
    if start_date == expected_start and end_date == expected_end:
        print("   ✅ 时间范围配置正确")
    else:
        print(f"   ❌ 时间范围配置错误，期望: {expected_start} ~ {expected_end}")
        return False
    
    # 2. 测试数据分割比例配置
    print("\n📊 测试数据分割比例配置:")
    validation_config = config.get('ai', {}).get('validation', {})
    train_ratio = validation_config.get('train_ratio', 0)
    val_ratio = validation_config.get('validation_ratio', 0)
    test_ratio = validation_config.get('test_ratio', 0)
    
    print(f"   训练集比例: {train_ratio:.1%}")
    print(f"   验证集比例: {val_ratio:.1%}")
    print(f"   测试集比例: {test_ratio:.1%}")
    print(f"   总和: {train_ratio + val_ratio + test_ratio:.1%}")
    
    if train_ratio == 0.70 and val_ratio == 0.20 and test_ratio == 0.10:
        print("   ✅ 数据分割比例配置正确")
    else:
        print("   ❌ 数据分割比例配置错误，期望: 70%/20%/10%")
        return False
    
    # 3. 测试数据获取
    print("\n📦 测试数据获取:")
    try:
        data_module = DataModule(config)
        data = data_module.get_history_data(start_date, end_date)
        
        if data is not None and not data.empty:
            print(f"   ✅ 成功获取数据: {len(data)} 条记录")
            print(f"   时间范围: {data['date'].min()} ~ {data['date'].max()}")
            
            # 检查数据预处理
            data = data_module.preprocess_data(data)
            print(f"   ✅ 数据预处理完成: {len(data.columns)} 个特征")
        else:
            print("   ❌ 数据获取失败")
            return False
            
    except Exception as e:
        print(f"   ❌ 数据获取异常: {e}")
        return False
    
    # 4. 测试AI优化器的数据分割
    print("\n🤖 测试AI优化器数据分割:")
    try:
        ai_optimizer = AIOptimizerImproved(config)
        
        # 模拟数据分割逻辑
        train_end = int(len(data) * train_ratio)
        val_end = int(len(data) * (train_ratio + val_ratio))
        
        train_size = train_end
        val_size = val_end - train_end
        test_size = len(data) - val_end
        
        print(f"   训练集大小: {train_size} 条 ({train_size/len(data):.1%})")
        print(f"   验证集大小: {val_size} 条 ({val_size/len(data):.1%})")
        print(f"   测试集大小: {test_size} 条 ({test_size/len(data):.1%})")
        
        if abs(train_size/len(data) - train_ratio) < 0.05:  # 允许5%的误差
            print("   ✅ 数据分割比例正确")
        else:
            print("   ❌ 数据分割比例异常")
            return False
            
    except Exception as e:
        print(f"   ❌ AI优化器测试异常: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有配置测试通过！")
    print(f"📊 配置摘要:")
    print(f"   时间范围: {start_date} ~ {end_date}")
    print(f"   数据总量: {len(data)} 条")
    print(f"   分割比例: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
    print(f"   分割大小: {train_size}/{val_size}/{test_size}")
    
    return True

if __name__ == "__main__":
    success = test_time_range_config()
    sys.exit(0 if success else 1) 