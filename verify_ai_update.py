#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 python run.py ai 命令是否会更新策略参数和模型
"""

import os
import sys
import yaml
import shutil
import subprocess
import time
from datetime import datetime

def backup_files():
    """备份重要文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"config/backups/verify_ai_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # 备份配置文件
    if os.path.exists("config/strategy.yaml"):
        shutil.copy2("config/strategy.yaml", f"{backup_dir}/strategy_backup.yaml")
        print(f"✅ 已备份策略配置")
    
    if os.path.exists("config/system.yaml"):
        shutil.copy2("config/system.yaml", f"{backup_dir}/system_backup.yaml")
        print(f"✅ 已备份系统配置")
    
    # 备份最新模型文件
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            # 按修改时间排序，备份最新的模型
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            latest_model = model_files[0]
            shutil.copy2(os.path.join(models_dir, latest_model), f"{backup_dir}/model_backup.pkl")
            print(f"✅ 已备份最新模型: {latest_model}")
    
    return backup_dir

def get_file_info(file_path):
    """获取文件信息"""
    if os.path.exists(file_path):
        stat = os.stat(file_path)
        return {
            'exists': True,
            'size': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime),
            'content': None
        }
    return {'exists': False}

def load_yaml_content(file_path):
    """加载YAML文件内容"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

def compare_configs(before, after, name):
    """比较配置变化"""
    print(f"\n📊 {name} 配置变化分析:")
    print("=" * 50)
    
    if not before and not after:
        print("❌ 配置文件不存在")
        return False
    
    if not before:
        print("✅ 新增配置文件")
        return True
    
    if not after:
        print("❌ 配置文件被删除")
        return False
    
    if before == after:
        print("➖ 配置无变化")
        return False
    else:
        print("✅ 配置已更新")
        
        # 详细比较变化
        print("\n🔍 详细变化:")
        for key in set(before.keys()) | set(after.keys()):
            if key not in before:
                print(f"  ➕ {key}: 新增 = {after[key]}")
            elif key not in after:
                print(f"  ➖ {key}: 删除 = {before[key]}")
            elif before[key] != after[key]:
                if isinstance(before[key], dict) and isinstance(after[key], dict):
                    print(f"  🔄 {key}: 字典内容变化")
                    for sub_key in set(before[key].keys()) | set(after[key].keys()):
                        if sub_key not in before[key]:
                            print(f"    ➕ {sub_key}: 新增 = {after[key][sub_key]}")
                        elif sub_key not in after[key]:
                            print(f"    ➖ {sub_key}: 删除 = {before[key][sub_key]}")
                        elif before[key][sub_key] != after[key][sub_key]:
                            print(f"    🔄 {sub_key}: {before[key][sub_key]} → {after[key][sub_key]}")
                else:
                    print(f"  🔄 {key}: {before[key]} → {after[key]}")
        
        return True

def check_model_files():
    """检查模型文件变化"""
    print("\n🤖 模型文件检查:")
    print("=" * 30)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ models目录不存在")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    if not model_files:
        print("❌ 没有找到模型文件")
        return False
    
    # 按修改时间排序
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    print(f"📁 找到 {len(model_files)} 个模型文件:")
    for i, model_file in enumerate(model_files[:5]):  # 只显示最新的5个
        file_path = os.path.join(models_dir, model_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        size = os.path.getsize(file_path) / 1024  # KB
        marker = "🆕" if i == 0 else "  "
        print(f"{marker} {model_file} ({size:.1f}KB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return True

def run_ai_command():
    """运行AI命令"""
    print("\n🚀 运行 python run.py ai 命令...")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # 运行命令，设置超时时间为5分钟
        result = subprocess.run(
            ["python", "run.py", "ai"],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        duration = time.time() - start_time
        print(f"⏱️  命令执行耗时: {duration:.1f}秒")
        
        if result.returncode == 0:
            print("✅ 命令执行成功")
            return True
        else:
            print(f"❌ 命令执行失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 命令执行异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 验证 python run.py ai 命令更新功能")
    print("=" * 60)
    
    # 1. 备份文件
    backup_dir = backup_files()
    
    # 2. 记录执行前的状态
    print("\n📋 记录执行前状态...")
    
    # 配置文件信息
    before_strategy = load_yaml_content("config/strategy.yaml")
    before_system = load_yaml_content("config/system.yaml")
    
    # 模型文件信息
    models_dir = "models"
    before_model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            before_model_files = model_files[:3]  # 记录前3个最新模型
    
    print(f"✅ 策略配置: {'已加载' if before_strategy else '不存在'}")
    print(f"✅ 系统配置: {'已加载' if before_system else '不存在'}")
    print(f"✅ 模型文件: {len(before_model_files)} 个最新模型")
    
    # 3. 运行AI命令
    command_success = run_ai_command()
    
    # 4. 记录执行后的状态
    print("\n📋 记录执行后状态...")
    
    after_strategy = load_yaml_content("config/strategy.yaml")
    after_system = load_yaml_content("config/system.yaml")
    
    # 检查模型文件变化
    after_model_files = []
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            after_model_files = model_files[:3]
    
    # 5. 比较变化
    strategy_changed = compare_configs(before_strategy, after_strategy, "策略")
    system_changed = compare_configs(before_system, after_system, "系统")
    
    # 检查模型文件变化
    model_changed = False
    if before_model_files != after_model_files:
        model_changed = True
        print("\n🆕 模型文件变化:")
        print("=" * 30)
        if after_model_files and after_model_files[0] not in before_model_files:
            print(f"✅ 新增模型文件: {after_model_files[0]}")
        else:
            print("✅ 模型文件已更新")
    
    check_model_files()
    
    # 6. 总结
    print("\n📊 验证结果总结:")
    print("=" * 40)
    print(f"命令执行: {'✅ 成功' if command_success else '❌ 失败'}")
    print(f"策略配置更新: {'✅ 是' if strategy_changed else '❌ 否'}")
    print(f"系统配置更新: {'✅ 是' if system_changed else '❌ 否'}")
    print(f"模型文件更新: {'✅ 是' if model_changed else '❌ 否'}")
    
    if command_success and (strategy_changed or system_changed or model_changed):
        print("\n🎉 python run.py ai 确实会更新策略参数和模型！")
    elif command_success:
        print("\n💡 python run.py ai 只训练模型，不更新策略参数")
    else:
        print("\n❌ 命令执行失败，无法验证更新功能")
    
    # 7. 恢复选项
    print(f"\n💾 原始文件已备份到: {backup_dir}")
    response = input("是否恢复原始文件？(y/N): ").strip().lower()
    if response == 'y':
        print("🔄 恢复原始文件...")
        
        # 恢复策略配置
        backup_strategy = f"{backup_dir}/strategy_backup.yaml"
        if os.path.exists(backup_strategy):
            shutil.copy2(backup_strategy, "config/strategy.yaml")
            print("✅ 策略配置已恢复")
        
        # 恢复系统配置
        backup_system = f"{backup_dir}/system_backup.yaml"
        if os.path.exists(backup_system):
            shutil.copy2(backup_system, "config/system.yaml")
            print("✅ 系统配置已恢复")
        
        # 恢复模型文件
        backup_model = f"{backup_dir}/model_backup.pkl"
        if os.path.exists(backup_model):
            # 找到最新的模型文件并替换
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                if model_files:
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                    latest_model = model_files[0]
                    shutil.copy2(backup_model, os.path.join(models_dir, latest_model))
                    print(f"✅ 模型文件已恢复: {latest_model}")

if __name__ == "__main__":
    main() 