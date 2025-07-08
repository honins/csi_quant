#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
重置参数脚本
清除过拟合的模型和参数，重新开始训练
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)

def reset_models():
    """重置模型文件"""
    logger = setup_logging()
    logger.info("🔄 开始重置模型文件...")
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        logger.info("✅ models目录不存在，无需清理")
        return
    
    # 备份当前模型
    backup_dir = project_root / "models" / "backup"
    backup_dir.mkdir(exist_ok=True)
    
    # 移动现有模型到备份目录
    model_files = list(models_dir.glob("*.pkl"))
    model_files.extend(list(models_dir.glob("*.txt")))
    
    if model_files:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)
        
        for file_path in model_files:
            if file_path.is_file():
                shutil.move(str(file_path), str(backup_subdir / file_path.name))
                logger.info(f"📦 备份模型文件: {file_path.name}")
        
        logger.info(f"✅ 模型文件已备份到: {backup_subdir}")
    else:
        logger.info("✅ 没有找到需要备份的模型文件")

def reset_optimization_history():
    """重置优化历史"""
    logger = setup_logging()
    logger.info("🔄 开始重置优化历史...")
    
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    
    # 备份优化历史
    history_files = []
    if results_dir.exists():
        history_dir = results_dir / "history"
        if history_dir.exists():
            history_files = list(history_dir.rglob("*.json"))
    
    if history_files:
        backup_dir = project_root / "results" / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"history_backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)
        
        for file_path in history_files:
            if file_path.is_file():
                # 保持目录结构
                relative_path = file_path.relative_to(results_dir)
                backup_path = backup_subdir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(backup_path))
                logger.info(f"📦 备份历史文件: {relative_path}")
        
        logger.info(f"✅ 优化历史已备份到: {backup_subdir}")
    else:
        logger.info("✅ 没有找到需要备份的优化历史文件")

def reset_config_backup():
    """重置配置文件备份"""
    logger = setup_logging()
    logger.info("🔄 开始重置配置文件备份...")
    
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    backups_dir = config_dir / "backups"
    
    if backups_dir.exists():
        # 创建新的备份目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_backup_dir = backups_dir / f"pre_reset_{timestamp}"
        new_backup_dir.mkdir(exist_ok=True)
        
        # 移动现有备份
        backup_files = list(backups_dir.glob("*.yaml"))
        for file_path in backup_files:
            if file_path.is_file():
                shutil.move(str(file_path), str(new_backup_dir / file_path.name))
                logger.info(f"📦 备份配置文件: {file_path.name}")
        
        logger.info(f"✅ 配置文件备份已整理到: {new_backup_dir}")
    else:
        logger.info("✅ 没有找到配置文件备份")

def create_fresh_start_marker():
    """创建新开始标记"""
    logger = setup_logging()
    logger.info("🔄 创建新开始标记...")
    
    project_root = Path(__file__).parent.parent
    marker_file = project_root / "FRESH_START.txt"
    
    with open(marker_file, 'w', encoding='utf-8') as f:
        f.write(f"重置时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("重置原因: 过拟合检测，降低模型复杂度\n")
        f.write("新配置:\n")
        f.write("- RandomForest: n_estimators=100, max_depth=8\n")
        f.write("- 数据分割: 60%训练/25%验证/15%测试\n")
        f.write("- 早停: patience=20, min_delta=0.005\n")
        f.write("- 过拟合检测: 启用严格检测\n")
    
    logger.info(f"✅ 新开始标记已创建: {marker_file}")

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("🚀 开始系统重置...")
    logger.info("=" * 60)
    
    try:
        # 1. 重置模型文件
        reset_models()
        
        # 2. 重置优化历史
        reset_optimization_history()
        
        # 3. 重置配置文件备份
        reset_config_backup()
        
        # 4. 创建新开始标记
        create_fresh_start_marker()
        
        logger.info("=" * 60)
        logger.info("✅ 系统重置完成！")
        logger.info("📝 建议下一步操作:")
        logger.info("   1. 运行: python run.py ai -m optimize")
        logger.info("   2. 使用新的防过拟合配置重新训练")
        logger.info("   3. 观察过拟合检测结果")
        
    except Exception as e:
        logger.error(f"❌ 重置过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main() 