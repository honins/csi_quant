#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化参数保存器模块
专门负责保存和更新 optimized_params.yaml 文件

功能：
- 保存优化后的参数
- 更新参数时间戳
- 保持文件结构和注释
- 参数验证
"""

import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .common import get_project_root


class OptimizedParamsSaver:
    """
    优化参数保存器
    
    专门负责管理 optimized_params.yaml 文件的读写
    """
    
    def __init__(self):
        self.project_root = get_project_root()
        self.config_path = self.project_root / 'config' / 'optimized_params.yaml'
        self.logger = logging.getLogger(__name__)
    
    def load_optimized_params(self) -> Dict[str, Any]:
        """
        加载优化参数配置
        
        返回:
            Dict[str, Any]: 优化参数配置
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                self.logger.warning(f"优化参数配置文件不存在: {self.config_path}")
                return self._get_default_structure()
        except Exception as e:
            # 尝试使用不安全加载读取（仅用于清理本地文件中历史遗留的numpy标记）
            self.logger.error(f"加载优化参数配置失败: {e}，尝试自动清理 numpy 标记…")
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    unsafe_data = yaml.load(f, Loader=yaml.UnsafeLoader)
                cleaned = self._to_native(unsafe_data or {})
                # 立刻覆写为纯原生类型，移除 numpy 标签
                with open(self.config_path, 'w', encoding='utf-8') as wf:
                    yaml.dump(cleaned, wf, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
                self.logger.info("已清理 optimized_params.yaml 中的 numpy 标记并重写为原生类型")
                return cleaned
            except Exception as e2:
                self.logger.error(f"自动清理失败: {e2}")
                return self._get_default_structure()
    
    def save_optimized_params(self, params: Dict[str, Any], 
                            optimization_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存优化参数
        
        参数:
            params: 优化后的参数字典
            optimization_info: 优化信息（可选）
        
        返回:
            bool: 保存是否成功
        """
        try:
            # 加载当前配置结构
            current_config = self.load_optimized_params()
            
            # 更新元数据
            current_config['metadata'] = current_config.get('metadata', {})
            current_config['metadata']['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_config['metadata']['total_params'] = len(params)
            
            if optimization_info:
                current_config['metadata'].update(optimization_info)
            
            # 更新参数
            self._update_params_by_category(current_config, params)
            
            # 转换为Python原生类型，避免写出numpy对象标记
            current_config = self._to_native(current_config)
            
            # 保存到文件
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False, indent=2)
            
            self.logger.info(f"优化参数已保存到: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存优化参数失败: {e}")
            return False
    
    def _update_params_by_category(self, config: Dict[str, Any], params: Dict[str, Any]):
        """
        按类别更新参数
        
        参数:
            config: 配置字典
            params: 新参数字典
        """
        # 基于param_config.py的参数分类映射
        param_categories = {
            # 核心策略参数（固定参数）
            'strategy_params': ['rise_threshold', 'max_days'],
            
            # 注意：final_threshold应属于confidence_weights
            
            # 置信度权重参数（核心决策参数 + 基础权重参数）
            'confidence_weights': [
                # 核心决策参数
                'final_threshold',
                'rsi_oversold_threshold', 'rsi_low_threshold',
                # 基础权重参数
                'ma_all_below', 'dynamic_confidence_adjustment',
                'market_sentiment_weight', 'trend_strength_weight'
            ],
            
            # 技术指标参数
            'technical_indicators': [
                'bb_near_threshold', 'recent_decline', 'macd_negative',
                'price_decline_threshold', 'rsi_oversold_threshold', 'rsi_low_threshold'
            ],
            
            # 成交量参数（成交量逻辑参数）
            'volume_params': [
                'volume_panic_threshold', 'volume_panic_bonus',
                'volume_surge_bonus', 'volume_shrink_penalty'
            ],
            
            # 市场参数
            'market_params': [
                'market_sentiment_weight', 'price_momentum_weight',
                'trend_strength_weight', 'dynamic_confidence_adjustment'
            ],
            
            # 策略级别参数
            'strategy_level_params': [
                'volume_weight', 'price_momentum_weight', 'bb_near_threshold',
                'volume_panic_threshold', 'volume_surge_threshold', 'volume_shrink_threshold'
            ]
        }
        
        # 构建参数到类别的映射，便于交叉清理
        param_to_category = {}
        for cat, names in param_categories.items():
            for n in names:
                param_to_category[n] = cat
        
        # 确保各类别存在
        for category in param_categories.keys():
            if category not in config:
                config[category] = {}
        
        # 先清理归属错误的参数，避免参数在多个类别中重复
        for category, cat_dict in list(config.items()):
            if category not in param_categories:
                # 非受管类别，跳过但不清理
                continue
            # 仅清理我们受管的类别中属于其他类别的已知参数
            for key in list(cat_dict.keys()):
                if key in param_to_category and param_to_category[key] != category:
                    del cat_dict[key]
                    self.logger.debug(f"移除错误类别 {category} 中的参数 {key}，应属于 {param_to_category[key]}")
        
        # 按类别更新参数（仅写入我们识别的参数）
        for category, param_names in param_categories.items():
            for param_name in param_names:
                if param_name in params:
                    config[category][param_name] = params[param_name]
                    self.logger.debug(f"更新参数 {param_name} 到类别 {category}: {params[param_name]}")
    
    def _get_default_structure(self) -> Dict[str, Any]:
        """
        获取默认配置结构
        
        返回:
            Dict[str, Any]: 默认配置结构
        """
        return {
            'metadata': {
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'optimization_version': 'v1.0',
                'total_params': 0,
                'description': '策略优化参数配置文件'
            },
            'strategy_params': {},
            'confidence_weights': {},
            'technical_indicators': {},
            'volume_params': {},
            'market_params': {},
            'strategy_level_params': {}
        }
    
    def get_all_optimized_params(self) -> Dict[str, Any]:
        """
        获取所有优化参数的扁平化字典
        
        返回:
            Dict[str, Any]: 扁平化的参数字典
        """
        config = self.load_optimized_params()
        all_params = {}
        
        # 合并所有类别的参数
        categories = ['strategy_params', 'confidence_weights', 'technical_indicators',
                     'volume_params', 'market_params', 'strategy_level_params']
        
        for category in categories:
            if category in config:
                all_params.update(config[category])
        
        return all_params
    
    def backup_current_params(self) -> bool:
        """
        备份当前参数文件
        
        返回:
            bool: 备份是否成功
        """
        try:
            if self.config_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.config_path.parent / f"optimized_params_backup_{timestamp}.yaml"
                
                import shutil
                shutil.copy2(self.config_path, backup_path)
                
                self.logger.info(f"参数文件已备份到: {backup_path}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"备份参数文件失败: {e}")
            return False

    def clean_file(self) -> bool:
        """主动清理 optimized_params.yaml 中的 numpy 对象标记并覆写为原生类型"""
        try:
            if not self.config_path.exists():
                self.logger.warning(f"文件不存在，无需清理: {self.config_path}")
                return False
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=yaml.UnsafeLoader)
            cleaned = self._to_native(data or {})
            with open(self.config_path, 'w', encoding='utf-8') as wf:
                yaml.dump(cleaned, wf, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
            self.logger.info("✅ 已清理 optimized_params.yaml，去除了 numpy 标记")
            return True
        except Exception as e:
            self.logger.error(f"❌ 清理 optimized_params.yaml 失败: {e}")
            return False

    def _to_native(self, obj: Any) -> Any:
        """递归将 numpy/自定义类型转换为 Python 原生类型，确保 YAML 安全保存"""
        try:
            import numpy as np  # 局部导入，避免在无 numpy 环境下报错
        except Exception:
            np = None
        
        if isinstance(obj, dict):
            return {k: self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_native(x) for x in obj)
        
        if np is not None:
            if isinstance(obj, (getattr(np, 'floating', ()), getattr(np, 'integer', ()) )):
                try:
                    return obj.item()
                except Exception:
                    return float(obj) if hasattr(obj, '__float__') else int(obj)
            if hasattr(np, 'bool_') and isinstance(obj, np.bool_):
                return bool(obj)
            if hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
                return obj.tolist()
        
        return obj


# 便捷函数
def save_optimized_params(params: Dict[str, Any], 
                         optimization_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    保存优化参数的便捷函数
    
    参数:
        params: 优化后的参数字典
        optimization_info: 优化信息（可选）
    
    返回:
        bool: 保存是否成功
    """
    saver = OptimizedParamsSaver()
    return saver.save_optimized_params(params, optimization_info)


def load_optimized_params() -> Dict[str, Any]:
    """
    加载优化参数的便捷函数
    
    返回:
        Dict[str, Any]: 优化参数配置
    """
    saver = OptimizedParamsSaver()
    return saver.load_optimized_params()


def clean_optimized_params_file() -> bool:
    """清理 optimized_params.yaml 中的 numpy 对象并覆写为原生数值"""
    saver = OptimizedParamsSaver()
    return saver.clean_file()


__all__ = ['OptimizedParamsSaver', 'save_optimized_params', 'load_optimized_params', 'clean_optimized_params_file']