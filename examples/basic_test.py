#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础测试模块
用于验证系统的基本功能是否正常
"""

import os
import sys
from pathlib import Path

# 将项目根目录添加到Python路径，确保可以 `import src.*`
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def test_imports():
    """测试基本模块导入"""
    print("🔧 测试模块导入...")
    
    try:
        from src.utils.config_loader import load_config
        print("✅ 配置加载器导入成功")
        
        from src.data.data_module import DataModule
        print("✅ 数据模块导入成功")
        
        from src.strategy.strategy_module import StrategyModule
        print("✅ 策略模块导入成功")
        
        from src.ai.ai_optimizer_improved import AIOptimizerImproved
        print("✅ AI优化器导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_config_loading():
    """测试配置文件加载"""
    print("\n📁 测试配置文件加载...")
    
    try:
        from src.utils.config_loader import load_config
        
        config = load_config()
        if config:
            print("✅ 配置文件加载成功")
            print(f"   - 配置项数量: {len(config)}")
            
            # 验证关键配置项
            key_sections = ['ai', 'data', 'strategy', 'backtest']
            missing_sections = []
            
            for section in key_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"⚠️ 缺少配置部分: {missing_sections}")
            else:
                print("✅ 关键配置部分都存在")
            
            return True
        else:
            print("❌ 配置文件加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 配置加载测试失败: {e}")
        return False


def test_data_access():
    """测试数据访问"""
    print("\n📊 测试数据访问...")
    
    try:
        from src.utils.config_loader import load_config
        from src.data.data_module import DataModule
        
        config = load_config()
        if not config:
            print("❌ 无法加载配置文件")
            return False
        
        data_module = DataModule(config)
        print("✅ 数据模块初始化成功")
        
        # 检查数据文件是否存在
        data_path = config.get('data', {}).get('data_file_path', '')
        if data_path and os.path.exists(data_path):
            print(f"✅ 数据文件存在: {data_path}")
            return True
        else:
            print(f"⚠️ 数据文件不存在: {data_path}")
            print("💡 建议运行数据获取脚本下载数据")
            return True  # 不阻塞测试
            
    except Exception as e:
        print(f"❌ 数据访问测试失败: {e}")
        return False


def test_strategy_initialization():
    """测试策略模块初始化"""
    print("\n🎯 测试策略模块初始化...")
    
    try:
        from src.utils.config_loader import load_config
        from src.strategy.strategy_module import StrategyModule
        
        config = load_config()
        if not config:
            print("❌ 无法加载配置文件")
            return False
        
        strategy_module = StrategyModule(config)
        print("✅ 策略模块初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 策略模块测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("🧪 系统基础功能测试")
    print("="*60)
    
    all_passed = True
    
    # 运行各项测试
    tests = [
        ("模块导入", test_imports),
        ("配置加载", test_config_loading),
        ("数据访问", test_data_access),
        ("策略初始化", test_strategy_initialization)
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}测试:")
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有基础测试通过！")
        print("✅ 系统基础功能正常，可以继续使用其他功能")
    else:
        print("⚠️ 部分测试未通过")
        print("💡 建议检查配置文件和依赖包安装情况")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)