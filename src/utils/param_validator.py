#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
参数验证工具
检查所有参数是否正确更新和配置
"""

import yaml
import os
from typing import Dict, List, Any
from src.utils.param_config import (
    FIXED_PARAMS, 
    CONFIDENCE_WEIGHT_PARAMS, 
    STRATEGY_LEVEL_PARAMS,
    get_all_optimizable_params,
    get_param_category,
    OPTIMIZABLE_PARAMS,
    OTHER_PARAMS
)

class ParamValidator:
    """参数验证器"""
    
    def __init__(self, config_path: str = 'config/strategy.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def validate_all_params(self) -> Dict[str, Any]:
        """验证所有参数"""
        results = {
            'fixed_params': self._validate_fixed_params(),
            'confidence_weight_params': self._validate_confidence_weight_params(),
            'strategy_level_params': self._validate_strategy_level_params(),
            'optimization_ranges': self._validate_optimization_ranges(),
            'summary': {}
        }
        
        # 生成摘要
        results['summary'] = self._generate_summary(results)
        return results
    
    def _validate_fixed_params(self) -> Dict[str, Any]:
        """验证固定参数"""
        results = {
            'valid': True,
            'missing': [],
            'found': [],
            'values': {}
        }
        
        # 检查多个可能的位置
        strategy_section = self.config.get('strategy', {})
        top_level = self.config
        confidence_weights = self.config.get('confidence_weights', {})
        strategy_confidence_weights = strategy_section.get('confidence_weights', {})
        
        # 合并confidence_weights
        all_confidence_weights = {**confidence_weights, **strategy_confidence_weights}
        
        for param in FIXED_PARAMS:
            # 检查strategy部分
            if param in strategy_section:
                results['found'].append(param)
                results['values'][param] = strategy_section[param]
            # 检查顶层
            elif param in top_level:
                results['found'].append(param)
                results['values'][param] = top_level[param]
            # 检查confidence_weights部分
            elif param in all_confidence_weights:
                results['found'].append(param)
                results['values'][param] = all_confidence_weights[param]
            else:
                results['missing'].append(param)
                results['valid'] = False
        
        return results
    
    def _validate_confidence_weight_params(self) -> Dict[str, Any]:
        """验证confidence_weights参数"""
        # 检查两种可能的位置：strategy.confidence_weights 和 顶层的confidence_weights
        strategy_confidence_weights = self.config.get('strategy', {}).get('confidence_weights', {})
        top_level_confidence_weights = self.config.get('confidence_weights', {})
        
        # 合并两个位置的参数
        confidence_weights = {**top_level_confidence_weights, **strategy_confidence_weights}
        
        results = {
            'valid': True,
            'missing': [],
            'found': [],
            'values': {}
        }
        
        for param in CONFIDENCE_WEIGHT_PARAMS:
            if param in confidence_weights:
                results['found'].append(param)
                results['values'][param] = confidence_weights[param]
            else:
                results['missing'].append(param)
                results['valid'] = False
        
        return results
    
    def _validate_strategy_level_params(self) -> Dict[str, Any]:
        """验证strategy级别参数"""
        strategy_config = self.config.get('strategy', {})
        results = {
            'valid': True,
            'missing': [],
            'found': [],
            'values': {}
        }
        
        for param in STRATEGY_LEVEL_PARAMS:
            if param in strategy_config:
                results['found'].append(param)
                results['values'][param] = strategy_config[param]
            else:
                results['missing'].append(param)
                results['valid'] = False
        
        return results
    
    def _validate_optimization_ranges(self) -> Dict[str, Any]:
        """验证优化范围配置"""
        optimization_ranges = self.config.get('optimization_ranges', {})
        optimizable_params = get_all_optimizable_params()  # 15个有效参数
        
        results = {
            'valid': True,
            'missing': [],
            'found': [],
            'ranges': {}
        }
        
        for param in optimizable_params:
            if param in optimization_ranges:
                results['found'].append(param)
                results['ranges'][param] = optimization_ranges[param]
            else:
                results['missing'].append(param)
                results['valid'] = False
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证摘要"""
        summary = {
            'total_fixed': len(FIXED_PARAMS),
            'total_optimizable': len(OPTIMIZABLE_PARAMS),
            'total_confidence_weights': len(CONFIDENCE_WEIGHT_PARAMS),
            'total_strategy_level': len(STRATEGY_LEVEL_PARAMS),
            'total_other': len(OTHER_PARAMS),
            'total_all': len(FIXED_PARAMS) + len(CONFIDENCE_WEIGHT_PARAMS) + len(STRATEGY_LEVEL_PARAMS) + len(OTHER_PARAMS),
            'fixed_valid': results['fixed_params']['valid'],
            'confidence_weights_valid': results['confidence_weight_params']['valid'],
            'strategy_level_valid': results['strategy_level_params']['valid'],
            'optimization_ranges_valid': results['optimization_ranges']['valid'],
            'overall_valid': all([
                results['fixed_params']['valid'],
                results['confidence_weight_params']['valid'],
                results['strategy_level_params']['valid'],
                results['optimization_ranges']['valid']
            ])
        }
        
        return summary
    
    def print_validation_report(self):
        """打印验证报告"""
        results = self.validate_all_params()
        
        print("=" * 80)
        print("📊 参数验证报告")
        print("=" * 80)
        
        # 固定参数验证
        print(f"🔒 固定参数验证: {'✅ 通过' if results['fixed_params']['valid'] else '❌ 失败'}")
        if results['fixed_params']['found']:
            print(f"   找到: {', '.join(results['fixed_params']['found'])}")
        if results['fixed_params']['missing']:
            print(f"   缺失: {', '.join(results['fixed_params']['missing'])}")
        
        # confidence_weights参数验证
        print(f"🎯 confidence_weights参数验证: {'✅ 通过' if results['confidence_weight_params']['valid'] else '❌ 失败'}")
        if results['confidence_weight_params']['found']:
            print(f"   找到: {len(results['confidence_weight_params']['found'])}/{len(CONFIDENCE_WEIGHT_PARAMS)} 个（包含 final_threshold）")
        if results['confidence_weight_params']['missing']:
            print(f"   缺失: {', '.join(results['confidence_weight_params']['missing'])}")
        
        # strategy级别参数验证
        print(f"📊 strategy级别参数验证: {'✅ 通过' if results['strategy_level_params']['valid'] else '❌ 失败'}")
        if results['strategy_level_params']['found']:
            print(f"   找到: {len(results['strategy_level_params']['found'])}/{len(STRATEGY_LEVEL_PARAMS)} 个")
        if results['strategy_level_params']['missing']:
            print(f"   缺失: {', '.join(results['strategy_level_params']['missing'])}")
        
        # 优化范围验证
        print(f"🔧 优化范围验证: {'✅ 通过' if results['optimization_ranges']['valid'] else '❌ 失败'}")
        if results['optimization_ranges']['found']:
            print(f"   找到: {len(results['optimization_ranges']['found'])}/{len(get_all_optimizable_params())} 个（15个有效参数）")
        if results['optimization_ranges']['missing']:
            print(f"   缺失: {', '.join(results['optimization_ranges']['missing'])}")
        
        # 总体验证
        print(f"\n🎯 总体验证: {'✅ 通过' if results['summary']['overall_valid'] else '❌ 失败'}")
        print(f"   固定参数: {results['summary']['total_fixed']} 个")
        print(f"   可优化参数: {results['summary']['total_optimizable']} 个（15个有效参数）")
        print(f"   其他参数: {results['summary']['total_other']} 个（不参与优化）")
        print(f"   所有参数总数: {results['summary']['total_all']} 个")
        
        print("=" * 80)
        
        return results['summary']['overall_valid']

def main():
    """主函数"""
    validator = ParamValidator()
    validator.print_validation_report()

if __name__ == "__main__":
    main()