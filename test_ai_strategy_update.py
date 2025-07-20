#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI训练是否会更新策略参数
"""

import os
import json
import yaml
import shutil
from datetime import datetime
import time
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import ConfigLoader
from src.ai.ai_optimizer_improved import AIOptimizerImproved
from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule

def backup_config():
    """备份当前配置"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"config/backups/test_ai_update_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # 备份策略配置文件
    if os.path.exists("config/strategy.yaml"):
        shutil.copy2("config/strategy.yaml", f"{backup_dir}/strategy_backup.yaml")
        print(f"✅ 已备份策略配置到: {backup_dir}/strategy_backup.yaml")
    
    # 备份系统配置文件
    if os.path.exists("config/system.yaml"):
        shutil.copy2("config/system.yaml", f"{backup_dir}/system_backup.yaml")
        print(f"✅ 已备份系统配置到: {backup_dir}/system_backup.yaml")
    
    return backup_dir

def load_config(config_file):
    """加载配置文件"""
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

def save_config(config, config_file):
    """保存配置文件"""
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def get_config_hash(config):
    """获取配置的哈希值用于比较"""
    return json.dumps(config, sort_keys=True, ensure_ascii=False)

def compare_configs(before_config, after_config, config_name):
    """比较配置变化"""
    print(f"\n📊 {config_name} 配置变化分析:")
    print("=" * 50)
    
    if before_config is None and after_config is None:
        print("❌ 配置文件不存在")
        return False
    
    if before_config is None:
        print("✅ 新增配置文件")
        return True
    
    if after_config is None:
        print("❌ 配置文件被删除")
        return False
    
    before_hash = get_config_hash(before_config)
    after_hash = get_config_hash(after_config)
    
    if before_hash == after_hash:
        print("➖ 配置无变化")
        return False
    else:
        print("✅ 配置已更新")
        
        # 详细比较变化
        print("\n🔍 详细变化:")
        compare_dict_changes(before_config, after_config, "")
        return True

def compare_dict_changes(before, after, path=""):
    """递归比较字典变化"""
    if not isinstance(before, dict) or not isinstance(after, dict):
        if before != after:
            print(f"  {path}: {before} → {after}")
        return
    
    all_keys = set(before.keys()) | set(after.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key not in before:
            print(f"  ➕ {current_path}: 新增 = {after[key]}")
        elif key not in after:
            print(f"  ➖ {current_path}: 删除 = {before[key]}")
        elif before[key] != after[key]:
            if isinstance(before[key], dict) and isinstance(after[key], dict):
                compare_dict_changes(before[key], after[key], current_path)
            else:
                print(f"  🔄 {current_path}: {before[key]} → {after[key]}")

def check_model_files():
    """检查模型文件变化"""
    print("\n🤖 模型文件检查:")
    print("=" * 30)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ models目录不存在")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if not model_files:
        print("❌ 没有找到模型文件")
        return
    
    # 按修改时间排序
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    print(f"📁 找到 {len(model_files)} 个模型文件:")
    for i, model_file in enumerate(model_files[:5]):  # 只显示最新的5个
        file_path = os.path.join(models_dir, model_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        size = os.path.getsize(file_path) / 1024  # KB
        marker = "🆕" if i == 0 else "  "
        print(f"{marker} {model_file} ({size:.1f}KB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

def run_ai_training():
    """直接运行AI训练"""
    print("\n🚀 开始AI训练...")
    print("=" * 30)
    
    try:
        # 直接调用AI优化器
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        data_module = DataModule(config)
        strategy_module = StrategyModule(config)
        ai_optimizer = AIOptimizerImproved(config)
        
        # 获取数据
        data_config = config.get('data', {})
        time_range = data_config.get('time_range', {})
        start_date = time_range.get('start_date', '2019-01-01')
        end_date = time_range.get('end_date', '2025-07-15')
        
        data = data_module.get_history_data(start_date, end_date)
        
        if data is None or data.empty:
            print("❌ 无法获取数据")
            return False
        
        # 运行完整AI优化（包括策略参数优化）
        start_time = time.time()
        result = ai_optimizer.run_complete_optimization(data, strategy_module)
        training_time = time.time() - start_time
        
        print(f"⏱️  训练耗时: {training_time:.1f}秒")
        
        if result.get('success', False):
            print("✅ AI完整优化成功完成")
            return True
        else:
            print(f"❌ AI优化失败: {result.get('errors', [])}")
            return False
            
    except Exception as e:
        print(f"❌ AI训练异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 AI策略更新测试")
    print("=" * 50)
    
    # 1. 备份当前配置
    backup_dir = backup_config()
    
    # 2. 记录训练前的配置
    print("\n📋 记录训练前配置...")
    before_strategy = load_config("config/strategy.yaml")
    before_system = load_config("config/system.yaml")
    
    print(f"✅ 策略配置: {'已加载' if before_strategy else '不存在'}")
    print(f"✅ 系统配置: {'已加载' if before_system else '不存在'}")
    
    # 3. 运行AI训练
    training_success = run_ai_training()
    
    # 4. 记录训练后的配置
    print("\n📋 记录训练后配置...")
    after_strategy = load_config("config/strategy.yaml")
    after_system = load_config("config/system.yaml")
    
    # 5. 比较配置变化
    strategy_changed = compare_configs(before_strategy, after_strategy, "策略")
    system_changed = compare_configs(before_system, after_system, "系统")
    
    # 6. 检查模型文件
    check_model_files()
    
    # 7. 总结
    print("\n📊 测试总结:")
    print("=" * 30)
    print(f"AI训练成功: {'✅ 是' if training_success else '❌ 否'}")
    print(f"策略配置更新: {'✅ 是' if strategy_changed else '❌ 否'}")
    print(f"系统配置更新: {'✅ 是' if system_changed else '❌ 否'}")
    
    if strategy_changed or system_changed:
        print("\n🎉 AI训练确实会更新策略参数！")
    else:
        print("\n💡 AI训练不会更新策略参数，只训练模型")
    
    # 8. 恢复配置（可选）
    print(f"\n💾 原始配置已备份到: {backup_dir}")
    print("如需恢复，请手动复制备份文件")

if __name__ == "__main__":
    main() 