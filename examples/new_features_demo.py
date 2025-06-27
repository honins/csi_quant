#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新功能演示脚本
展示环境变量配置、性能监控、虚拟环境检测等新增功能
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

def demo_environment_check():
    """演示环境检测功能"""
    print("="*60)
    print("🔍 环境检测演示")
    print("="*60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"📍 Python版本: {python_version}")
    
    # 检查虚拟环境
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if in_venv:
        venv_path = os.environ.get('VIRTUAL_ENV', '当前虚拟环境')
        print(f"✅ 虚拟环境: {os.path.basename(venv_path)}")
    else:
        print("⚠️  警告: 未检测到虚拟环境")
    
    # 检查关键依赖
    dependencies = [
        ('pandas', 'pd'),
        ('numpy', 'np'), 
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('yaml', 'pyyaml'),
        ('scipy', 'scipy')
    ]
    
    print("\n📦 依赖包检查:")
    for pkg_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {pkg_name}")
        except ImportError:
            print(f"   ❌ {pkg_name} (未安装)")
    
    print()

def demo_config_path():
    """演示配置文件路径功能"""
    print("="*60)
    print("🔧 配置文件路径演示")
    print("="*60)
    
    # 检查环境变量
    config_env = os.environ.get('CSI_CONFIG_PATH')
    if config_env:
        print(f"🔧 环境变量配置: {config_env}")
        if os.path.exists(config_env):
            print("   ✅ 文件存在")
        else:
            print("   ❌ 文件不存在")
    else:
        print("💡 未设置环境变量 CSI_CONFIG_PATH")
    
    # 检查默认配置文件
    config_paths = [
        project_root / 'config' / 'config_improved.yaml',
        project_root / 'config' / 'config.yaml'
    ]
    
    print("\n📁 配置文件查找顺序:")
    for i, path in enumerate(config_paths, 1):
        exists = "✅" if path.exists() else "❌"
        print(f"   {i}. {exists} {path.name}")
    
    print()

def demo_performance_timer():
    """演示性能监控功能"""
    print("="*60)
    print("⏱️  性能监控演示")
    print("="*60)
    
    # 导入计时器类
    sys.path.insert(0, str(project_root))
    from run import PerformanceTimer
    
    timer = PerformanceTimer()
    
    # 演示短任务
    print("📊 演示短任务计时:")
    timer.start("短任务演示")
    time.sleep(2.5)  # 模拟任务执行
    duration = timer.stop()
    print(f"   返回值: {duration:.2f}秒")
    
    print()
    
    # 演示长任务
    print("📊 演示长任务计时:")
    timer.start("长任务演示")
    time.sleep(65)  # 模拟长任务执行
    timer.stop()
    
    print()

def demo_config_loading():
    """演示安全配置加载"""
    print("="*60)
    print("📄 配置文件加载演示")
    print("="*60)
    
    # 导入配置加载函数
    sys.path.insert(0, str(project_root))
    from run import load_config_safely, get_config_path
    
    # 获取配置路径
    config_path = get_config_path()
    print(f"📁 选择的配置文件: {config_path}")
    
    # 加载配置
    print("\n🔄 正在加载配置...")
    config = load_config_safely()
    
    if config:
        print("✅ 配置加载成功")
        
        # 显示一些关键配置
        if 'strategy' in config:
            strategy = config['strategy']
            print(f"   📊 策略配置:")
            print(f"      涨幅阈值: {strategy.get('rise_threshold', 'N/A')}")
            print(f"      最大天数: {strategy.get('max_days', 'N/A')}")
        
        if 'ai' in config:
            ai_config = config['ai']
            print(f"   🤖 AI配置:")
            print(f"      启用状态: {ai_config.get('enable', 'N/A')}")
            print(f"      模型类型: {ai_config.get('model_type', 'N/A')}")
    else:
        print("❌ 配置加载失败")
    
    print()

def demo_environment_variables():
    """演示环境变量设置"""
    print("="*60)
    print("🌍 环境变量设置演示")
    print("="*60)
    
    print("💡 如何设置自定义配置文件路径:")
    print()
    
    # Windows示例
    print("🪟 Windows:")
    print("   set CSI_CONFIG_PATH=C:\\path\\to\\your\\config.yaml")
    print("   python run.py ai")
    print()
    
    # Linux/Mac示例
    print("🐧 Linux/Mac:")
    print("   export CSI_CONFIG_PATH=/path/to/your/config.yaml")
    print("   python run.py ai")
    print()
    
    # 临时设置示例
    print("⚡ 临时设置（单次使用）:")
    print("   Windows: set CSI_CONFIG_PATH=config\\custom.yaml && python run.py ai")
    print("   Linux/Mac: CSI_CONFIG_PATH=config/custom.yaml python run.py ai")
    print()
    
    # 当前环境变量状态
    print("📋 当前环境变量状态:")
    env_vars = ['CSI_CONFIG_PATH', 'VIRTUAL_ENV', 'PATH']
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            if var == 'PATH':
                print(f"   {var}: [包含{len(value.split(os.pathsep))}个路径]")
            else:
                print(f"   {var}: {value}")
        else:
            print(f"   {var}: (未设置)")
    
    print()

def main():
    """主演示函数"""
    print("🚀 新功能演示脚本")
    print("展示环境变量配置、性能监控、虚拟环境检测等功能")
    print()
    
    try:
        # 1. 环境检测演示
        demo_environment_check()
        
        # 2. 配置文件路径演示
        demo_config_path()
        
        # 3. 配置文件加载演示
        demo_config_loading()
        
        # 4. 环境变量设置演示
        demo_environment_variables()
        
        # 5. 性能监控演示（较长，可选）
        user_input = input("是否演示性能监控功能？(需要约70秒) [y/N]: ").strip().lower()
        if user_input in ['y', 'yes']:
            demo_performance_timer()
        else:
            print("⏭️  跳过性能监控演示")
        
        print("="*60)
        print("✅ 所有演示完成！")
        print("💡 现在您可以使用以下新功能：")
        print("   - 环境变量配置: CSI_CONFIG_PATH")
        print("   - 性能监控: 自动显示执行时间")
        print("   - 虚拟环境检测: 自动提醒")
        print("   - 增强错误处理: 更友好的错误信息")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⛔ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 