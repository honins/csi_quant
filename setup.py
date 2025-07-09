#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目初始化脚本
用于初始化项目环境，包括虚拟环境、依赖安装、数据获取等
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class ProjectSetup:
    """项目初始化类"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.is_windows = platform.system().lower() == "windows"
        
    def print_header(self, title: str):
        """打印标题"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)
        
    def print_step(self, step: str):
        """打印步骤"""
        print(f"\n📋 {step}")
        print("-" * 40)
        
    def run_command(self, command: str, check: bool = True) -> bool:
        """运行命令"""
        try:
            print(f"执行命令: {command}")
            
            # 设置环境变量以处理编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONUTF8'] = '1'
            
            # 在Windows上设置代码页
            if self.is_windows:
                env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
            
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True, encoding='utf-8',
                                  env=env)
            if result.stdout:
                print(f"输出: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败: {e}")
            if e.stderr:
                print(f"错误信息: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ 命令执行异常: {e}")
            return False
            
    def check_python_version(self) -> bool:
        """检查Python版本"""
        self.print_step("检查Python版本")
        
        version = sys.version_info
        print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python版本过低，需要Python 3.8或更高版本")
            return False
            
        print("✅ Python版本符合要求")
        return True
        
    def create_virtual_environment(self) -> bool:
        """创建虚拟环境"""
        self.print_step("创建虚拟环境")
        
        if self.venv_path.exists():
            print(f"虚拟环境已存在: {self.venv_path}")
            response = input("是否重新创建虚拟环境？(y/N): ").strip().lower()
            if response == 'y':
                print("删除现有虚拟环境...")
                try:
                    # 尝试删除虚拟环境
                    shutil.rmtree(self.venv_path)
                    print("现有虚拟环境删除成功")
                except PermissionError:
                    print("❌ 无法删除虚拟环境，可能正在被使用")
                    print("请先退出虚拟环境，然后重新运行setup脚本")
                    print("退出虚拟环境命令: deactivate")
                    return False
                except Exception as e:
                    print(f"❌ 删除虚拟环境失败: {e}")
                    return False
            else:
                print("使用现有虚拟环境")
                return True
                
        print("创建虚拟环境...")
        command = f"python -m venv {self.venv_path}"
        return self.run_command(command)
        
    def get_activate_command(self) -> str:
        """获取激活虚拟环境的命令"""
        if self.is_windows:
            return str(self.venv_path / "Scripts" / "activate.bat")
        else:
            return f"source {self.venv_path / 'bin' / 'activate'}"
            
    def install_dependencies(self) -> bool:
        """安装依赖包"""
        self.print_step("安装依赖包")
        
        # 检查requirements文件
        requirements_simple = self.project_root / "requirements_simple.txt"
        
        if not self.requirements_file.exists():
            print("❌ requirements.txt文件不存在")
            return False
            
        # 构建安装命令
        if self.is_windows:
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.venv_path / "bin" / "pip"
            
        # 尝试多种安装方式
        commands = [
            # 方式1: 使用简化版本（无中文注释）
            f'"{pip_path}" install -r "{requirements_simple}"' if requirements_simple.exists() else None,
            # 方式2: 直接安装原版
            f'"{pip_path}" install -r "{self.requirements_file}"',
            # 方式3: 使用国内镜像源
            f'"{pip_path}" install -r "{self.requirements_file}" -i https://pypi.tuna.tsinghua.edu.cn/simple/',
            # 方式4: 设置环境变量后安装
            f'set PYTHONIOENCODING=utf-8 && "{pip_path}" install -r "{self.requirements_file}"',
            # 方式5: 逐个安装主要依赖
            f'"{pip_path}" install numpy pandas matplotlib seaborn scikit-learn scipy PyYAML requests akshare'
        ]
        
        # 过滤掉None值
        commands = [cmd for cmd in commands if cmd is not None]
        
        for i, command in enumerate(commands, 1):
            print(f"尝试安装方式 {i}...")
            if self.run_command(command, check=False):
                print(f"✅ 依赖安装成功 (方式 {i})")
                return True
            else:
                print(f"❌ 安装方式 {i} 失败")
                
        print("❌ 所有安装方式都失败了")
        print("\n请手动安装依赖:")
        print("1. 激活虚拟环境:")
        if self.is_windows:
            print(f"   {self.venv_path}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_path}/bin/activate")
        print("2. 安装依赖:")
        if requirements_simple.exists():
            print("   pip install -r requirements_simple.txt")
        else:
            print("   pip install -r requirements.txt")
        print("3. 或者使用国内镜像源:")
        print("   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/")
        
        return False
        
    def fetch_latest_data(self) -> bool:
        """获取最新交易数据"""
        self.print_step("获取最新交易数据")
        
        # 构建运行数据获取脚本的命令
        if self.is_windows:
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"
            
        data_script = self.project_root / "src" / "data" / "fetch_latest_data.py"
        
        if not data_script.exists():
            print("❌ 数据获取脚本不存在")
            return False
            
        command = f'"{python_path}" "{data_script}"'
        return self.run_command(command)
        
    def create_directories(self) -> bool:
        """创建必要的目录"""
        self.print_step("创建项目目录")
        
        directories = [
            "data",
            "logs", 
            "results",
            "models",
            "cache",
            "docs"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                print(f"创建目录: {directory}")
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                print(f"目录已存在: {directory}")
                
        return True
        
    def check_config_file(self) -> bool:
        """检查配置文件"""
        self.print_step("检查配置文件")
        
        config_file = self.project_root / "config" / "config.yaml"
        
        if not config_file.exists():
            print("❌ 配置文件不存在: config/config.yaml")
            return False
            
        print("✅ 配置文件存在")
        return True
        
        
    def print_summary(self, results: dict):
        """打印初始化总结"""
        self.print_header("初始化完成")
        
        print("📊 初始化结果:")
        for step, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {step}: {status}")
            
        success_count = sum(results.values())
        total_count = len(results)
        
        print(f"\n总体结果: {success_count}/{total_count} 步骤成功")
        
        if success_count == total_count:
            print("\n🎉 项目初始化完成！")
            print("\n下一步操作:")
            print("1. 激活虚拟环境:")
            if self.is_windows:
                print(f"   {self.venv_path}\\Scripts\\activate")
            else:
                print(f"   source {self.venv_path}/bin/activate")
            print("2. 运行项目:")
            print("   python run.py")
            print("3. 获取最新数据:")
            print("   python src/data/fetch_latest_data.py")
        else:
            print("\n⚠️ 部分步骤失败，请检查错误信息并手动完成")
            
    def setup(self) -> bool:
        """执行完整的初始化流程"""
        self.print_header("项目初始化")
        
        results = {}
        
        # 1. 检查Python版本
        results["Python版本检查"] = self.check_python_version()
        
        # 2. 创建虚拟环境
        if results["Python版本检查"]:
            results["虚拟环境创建"] = self.create_virtual_environment()
        else:
            results["虚拟环境创建"] = False
            
        # 3. 安装依赖
        if results.get("虚拟环境创建", False):
            results["依赖安装"] = self.install_dependencies()
        else:
            results["依赖安装"] = False
            
        # 4. 创建目录
        results["目录创建"] = self.create_directories()
        
        # 5. 检查配置文件
        results["配置文件检查"] = self.check_config_file()
        
        # 6. 获取最新数据
        if results.get("依赖安装", False):
            results["数据获取"] = self.fetch_latest_data()
        else:
            results["数据获取"] = False
            
            
        # 打印总结
        self.print_summary(results)
        
        return all(results.values())

def main():
    """主函数"""
    setup = ProjectSetup()
    success = setup.setup()
    
    if success:
        print("\n🎉 项目初始化成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 项目初始化过程中遇到问题，请检查上述错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main() 