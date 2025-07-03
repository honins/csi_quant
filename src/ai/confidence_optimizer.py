#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的置信度优化器
解决现有平滑算法变动过小的问题，提供简单而有效的解决方案
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

class SimpleConfidenceOptimizer:
    """
    简化的置信度优化器
    
    核心理念：
    1. 保持模型原始输出的主导地位
    2. 仅在必要时进行轻微调整
    3. 使用自适应的平滑强度
    4. 避免过度平滑导致的信息损失
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 简化的配置参数
        confidence_config = config.get('ai', {}).get('confidence_optimization', {})
        
        # 核心参数
        self.enabled = confidence_config.get('enabled', True)
        self.mode = confidence_config.get('mode', 'adaptive')  # adaptive | conservative | aggressive
        
        # 自适应平滑参数
        self.base_smoothing = confidence_config.get('base_smoothing', 0.2)  # 基础平滑强度
        self.large_change_threshold = confidence_config.get('large_change_threshold', 0.4)  # 大变化阈值
        self.noise_filter_threshold = confidence_config.get('noise_filter_threshold', 0.05)  # 噪音过滤阈值
        
        # 历史记录
        self.history_path = os.path.join('models', 'confidence_simple_history.json')
        self.confidence_history = self._load_confidence_history()
        
        self.logger.info(f"简化置信度优化器初始化完成，模式: {self.mode}")
    
    def optimize_confidence(self, 
                           raw_confidence: float, 
                           date: str, 
                           market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        优化置信度（简化算法）
        
        参数:
        raw_confidence: 模型原始输出
        date: 预测日期
        market_data: 市场数据
        
        返回:
        dict: 优化结果
        """
        if not self.enabled:
            return {
                'raw_confidence': raw_confidence,
                'final_confidence': raw_confidence,
                'adjustment_applied': False,
                'reason': '置信度优化已禁用'
            }
        
        try:
            # 确保原始置信度在有效范围内
            raw_confidence = max(0.0, min(1.0, raw_confidence))
            
            # 获取历史置信度
            last_confidence = self._get_last_confidence()
            
            if last_confidence is None:
                # 第一次预测，直接使用原始值
                result = {
                    'raw_confidence': raw_confidence,
                    'final_confidence': raw_confidence,
                    'adjustment_applied': False,
                    'reason': '首次预测，无需调整'
                }
            else:
                # 计算变化幅度
                change = raw_confidence - last_confidence
                change_magnitude = abs(change)
                
                # 决定是否需要调整
                if self.mode == 'aggressive':
                    # 激进模式：很少调整，保持模型原始输出
                    final_confidence = raw_confidence
                    adjustment_applied = False
                    reason = "激进模式：保持原始输出"
                    
                elif self.mode == 'conservative':
                    # 保守模式：较多平滑
                    final_confidence = self._apply_conservative_smoothing(raw_confidence, last_confidence)
                    adjustment_applied = abs(final_confidence - raw_confidence) > 0.001
                    reason = "保守模式：应用平滑"
                    
                else:  # adaptive 自适应模式
                    final_confidence, adjustment_applied, reason = self._apply_adaptive_optimization(
                        raw_confidence, last_confidence, change_magnitude, market_data
                    )
                
                result = {
                    'raw_confidence': raw_confidence,
                    'final_confidence': final_confidence,
                    'adjustment_applied': adjustment_applied,
                    'change_from_last': change,
                    'change_magnitude': change_magnitude,
                    'reason': reason
                }
            
            # 保存历史记录
            self._save_confidence_record(date, result)
            
            # 输出日志
            self._log_confidence_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"置信度优化失败: {e}")
            return {
                'raw_confidence': raw_confidence,
                'final_confidence': raw_confidence,
                'adjustment_applied': False,
                'error': str(e)
            }
    
    def _apply_adaptive_optimization(self, 
                                   raw_confidence: float, 
                                   last_confidence: float, 
                                   change_magnitude: float,
                                   market_data: pd.DataFrame = None) -> tuple:
        """
        应用自适应优化（核心算法）
        
        参数:
        raw_confidence: 原始置信度
        last_confidence: 上次置信度
        change_magnitude: 变化幅度
        market_data: 市场数据
        
        返回:
        tuple: (最终置信度, 是否调整, 原因)
        """
        
        # 1. 小变化：直接使用原始值（避免过度平滑）
        if change_magnitude <= self.noise_filter_threshold:
            return raw_confidence, False, f"小变化({change_magnitude:.3f})，保持原始值"
        
        # 2. 检查是否为市场异常情况
        if self._is_market_abnormal(market_data):
            return raw_confidence, False, "市场异常，保持原始置信度"
        
        # 3. 大变化处理：根据变化幅度确定平滑强度
        if change_magnitude >= self.large_change_threshold:
            # 大变化：轻微平滑（保留大部分信息）
            smoothing_factor = self.base_smoothing * 0.5  # 降低平滑强度
            reason = f"大变化({change_magnitude:.3f})，轻微平滑"
        else:
            # 中等变化：正常平滑
            smoothing_factor = self.base_smoothing
            reason = f"中等变化({change_magnitude:.3f})，正常平滑"
        
        # 4. 应用加权平均（主要保留原始信息）
        final_confidence = (1 - smoothing_factor) * raw_confidence + smoothing_factor * last_confidence
        
        # 5. 确保在有效范围内
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        adjustment_applied = abs(final_confidence - raw_confidence) > 0.001
        
        return final_confidence, adjustment_applied, reason
    
    def _apply_conservative_smoothing(self, raw_confidence: float, last_confidence: float) -> float:
        """应用保守平滑"""
        # 使用固定的平滑系数
        smoothing_factor = self.base_smoothing * 1.5  # 增加平滑强度
        final_confidence = (1 - smoothing_factor) * raw_confidence + smoothing_factor * last_confidence
        return max(0.0, min(1.0, final_confidence))
    
    def _is_market_abnormal(self, market_data: pd.DataFrame = None) -> bool:
        """
        检查市场是否异常（简化版）
        
        参数:
        market_data: 市场数据
        
        返回:
        bool: 是否异常
        """
        if market_data is None or len(market_data) < 10:
            return False
        
        try:
            # 检查价格剧烈变化
            if 'close' in market_data.columns and len(market_data) >= 2:
                latest_price = market_data['close'].iloc[-1]
                prev_price = market_data['close'].iloc[-2]
                price_change = abs(latest_price - prev_price) / prev_price
                
                if price_change > 0.08:  # 8%以上价格变化认为异常
                    return True
            
            # 检查成交量异常
            if 'volume' in market_data.columns and len(market_data) >= 10:
                recent_volume = market_data['volume'].iloc[-1]
                avg_volume = market_data['volume'].tail(10).mean()
                
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    if volume_ratio > 2.0 or volume_ratio < 0.3:  # 成交量异常
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _get_last_confidence(self) -> Optional[float]:
        """获取最近的置信度"""
        if self.confidence_history:
            return self.confidence_history[-1].get('final_confidence')
        return None
    
    def _load_confidence_history(self) -> List[Dict]:
        """加载置信度历史"""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"加载置信度历史失败: {e}")
        return []
    
    def _save_confidence_record(self, date: str, result: Dict[str, Any]):
        """保存置信度记录"""
        try:
            record = {
                'date': str(date),
                'raw_confidence': result['raw_confidence'],
                'final_confidence': result['final_confidence'],
                'adjustment_applied': result['adjustment_applied'],
                'reason': result.get('reason', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            self.confidence_history.append(record)
            
            # 保留最近30条记录
            if len(self.confidence_history) > 30:
                self.confidence_history = self.confidence_history[-30:]
            
            # 保存到文件
            os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump(self.confidence_history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存置信度记录失败: {e}")
    
    def _log_confidence_result(self, result: Dict[str, Any]):
        """记录置信度结果"""
        raw = result['raw_confidence']
        final = result['final_confidence']
        adjusted = result['adjustment_applied']
        reason = result.get('reason', '')
        
        if adjusted:
            change = final - raw
            self.logger.info(f"置信度优化: {raw:.4f} → {final:.4f} "
                           f"(调整: {change:+.4f}) - {reason}")
        else:
            self.logger.info(f"置信度优化: {raw:.4f} (无调整) - {reason}")
    
    def get_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        获取优化统计信息
        
        参数:
        days: 统计天数
        
        返回:
        dict: 统计结果
        """
        try:
            recent_records = self.confidence_history[-days:]
            
            if not recent_records:
                return {'error': '没有历史记录'}
            
            adjustment_count = sum(1 for r in recent_records if r.get('adjustment_applied', False))
            
            return {
                'total_predictions': len(recent_records),
                'adjustments_made': adjustment_count,
                'adjustment_rate': adjustment_count / len(recent_records),
                'mode': self.mode,
                'base_smoothing': self.base_smoothing
            }
            
        except Exception as e:
            self.logger.error(f"统计计算失败: {e}")
            return {'error': str(e)}

# =============================================================================
# 配置示例和使用说明
# =============================================================================
"""
配置示例（在config.yaml中替换现有的confidence_smoothing）:

ai:
  confidence_optimization:
    enabled: true
    mode: "adaptive"  # adaptive(推荐) | conservative | aggressive
    
    # 核心参数
    base_smoothing: 0.2              # 基础平滑强度 (0.1-0.4)
    large_change_threshold: 0.4      # 大变化阈值 (0.3-0.6)
    noise_filter_threshold: 0.05     # 噪音过滤阈值 (0.02-0.1)

算法特点：

1. 自适应调整：
   - 小变化(≤5%)：直接使用原始值
   - 中等变化(5%-40%)：轻微平滑
   - 大变化(≥40%)：保留原始信息，仅轻微调整

2. 市场感知：
   - 价格剧变时(≥8%)：保持原始置信度
   - 成交量异常时：保持原始置信度

3. 信息保留：
   - 主要权重给原始置信度(80%+)
   - 避免过度平滑导致的信息损失

解决的问题：
- ✅ 避免变动过小的问题
- ✅ 保留模型主要信息
- ✅ 简化系统复杂性
- ✅ 提供灵活的模式选择
""" 