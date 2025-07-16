#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取接口
提供统一的数据获取服务，返回标准JSON格式响应
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def check_virtual_environment():
    """检查是否在虚拟环境中运行"""
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if not in_venv:
        print("⚠️  警告: 您似乎没有在虚拟环境中运行")
        print("💡 建议: 激活虚拟环境以避免依赖冲突")
        print("   Windows: venv\\Scripts\\activate")
        print("   Linux/Mac: source venv/bin/activate")
        return False
    else:
        venv_path = os.environ.get('VIRTUAL_ENV', '当前虚拟环境')
        print(f"✅ 运行在虚拟环境: {os.path.basename(venv_path)}")
        return True

def load_config_safely(custom_config_files=None):
    """安全加载配置文件"""
    try:
        from src.utils.config_loader import load_config
        
        # 默认配置文件列表（按优先级排序）
        default_config_files = [
            'system.yaml',           # 系统基础配置
            'strategy.yaml',         # 策略优化配置
            'config.yaml'            # 兼容性配置（如果存在）
        ]
        
        # 检查环境变量配置
        env_config_path = os.environ.get('CSI_CONFIG_PATH')
        if env_config_path:
            if os.path.isabs(env_config_path):
                default_config_files.append(env_config_path)
            else:
                default_config_files.append(env_config_path)
            print(f"🔧 使用环境变量指定的额外配置文件: {env_config_path}")
        
        # 如果指定了自定义配置文件，直接使用
        if custom_config_files:
            if isinstance(custom_config_files, str):
                custom_config_files = [custom_config_files]
            config_files = custom_config_files
        else:
            config_files = default_config_files
        
        print(f"📁 使用多配置文件加载: {', '.join([os.path.basename(f) for f in config_files[:2]])}...")
        return load_config(config_files=config_files)
    except ImportError as e:
        print(f"❌ 无法导入配置加载模块: {e}")
        print("💡 请确保已激活虚拟环境并安装了所有依赖包")
        return None
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        print("💡 提示: 可以通过环境变量 CSI_CONFIG_PATH 指定额外配置文件")
        return None

def main():
    """
    数据获取主函数
    
    返回:
    dict: 标准JSON格式响应
    {
        "code": int,      # 状态码，200成功，500失败
        "msg": str,       # 响应消息
        "data": dict      # 响应数据
    }
    """
    print("="*80)
    print("📊 数据获取系统")
    print("="*80)
    
    # 检查虚拟环境
    if not check_virtual_environment():
        response = {
            "code": 500,
            "msg": "请在虚拟环境中运行数据获取",
            "data": {
                "error": "虚拟环境检查失败",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        return response
    
    # 加载配置文件
    config = load_config_safely()
    if not config:
        response = {
            "code": 500,
            "msg": "配置文件加载失败",
            "data": {
                "error": "无法加载配置文件",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        return response
    
    print(f"📋 数据获取配置:")
    data_config = config.get('data', {})
    print(f"   📂 数据源: {data_config.get('data_source', 'akshare')}")
    print(f"   📈 指数代码: {data_config.get('index_code', 'SHSE.000905')}")
    print(f"   📅 频率: {data_config.get('frequency', '1d')}")
    print(f"   💾 缓存: {'启用' if data_config.get('cache_enabled', True) else '禁用'}")
    print("="*80)
    
    try:
        # 导入数据获取模块
        from src.data.fetch_latest_data import main as fetch_main
        
        print("🚀 开始获取最新数据...")
        result = fetch_main()
        
        # 检查返回结果格式
        if isinstance(result, dict) and 'code' in result:
            # 如果返回的已经是JSON格式，直接返回
            print(f"\n✅ 数据获取完成")
            print(f"📊 状态码: {result.get('code', 'unknown')}")
            print(f"💬 消息: {result.get('msg', 'unknown')}")
            
            # 显示详细信息
            data = result.get('data', {})
            results = data.get('results', {})
            if results:
                print(f"\n📈 数据获取结果:")
                for symbol, info in results.items():
                    status = "✅ 成功" if info.get('success', False) else "❌ 失败"
                    print(f"   {symbol}: {status}")
                    print(f"      总记录数: {info.get('total_records', 0)}")
                    print(f"      最新日期: {info.get('latest_date', '未知')}")
                    print(f"      最早日期: {info.get('earliest_date', '未知')}")
            
            return result
        
        elif result is True or result is False:
            # 如果返回布尔值，转换为标准JSON格式
            response = {
                "code": 200 if result else 500,
                "msg": "数据获取完成" if result else "数据获取失败",
                "data": {
                    "success": result,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            return response
        
        else:
            # 其他情况，创建标准响应
            response = {
                "code": 200,
                "msg": "数据获取完成",
                "data": {
                    "result": str(result) if result is not None else "unknown",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            return response
            
    except ImportError as e:
        error_msg = f"无法导入数据获取模块: {e}"
        print(f"❌ {error_msg}")
        print("💡 请检查是否已安装所有依赖包: pip install -r requirements.txt")
        
        response = {
            "code": 500,
            "msg": "数据获取模块导入失败",
            "data": {
                "error": error_msg,
                "suggestion": "请检查依赖包安装情况",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        return response
        
    except Exception as e:
        error_msg = f"数据获取执行失败: {e}"
        print(f"❌ {error_msg}")
        
        response = {
            "code": 500,
            "msg": "数据获取执行失败",
            "data": {
                "error": error_msg,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        return response

if __name__ == "__main__":
    result = main()
    print(f"\n最终响应: {result}") 