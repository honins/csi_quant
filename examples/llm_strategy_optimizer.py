#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM驱动的策略优化脚本
自动调整策略因子参数，自动优化策略。
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import random

# 假设项目根目录在sys.path中，或者手动添加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_module import DataModule
from src.strategy.strategy_module import StrategyModule
from src.ai.ai_optimizer import AIOptimizer
from src.utils.utils import load_config

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

class LLMStrategyOptimizer:
    def __init__(self, config: dict):
        self.logger = logging.getLogger("LLMStrategyOptimizer")
        self.config = config
        self.data_module = DataModule(config)
        self.strategy_module = StrategyModule(config)
        self.ai_optimizer = AIOptimizer(config)
        self.best_score = -1.0
        self.best_params = {}

    def _simulate_llm_suggestion(self, current_params: dict, feedback: str = "") -> dict:
        """
        模拟LLM根据当前参数和回测反馈生成新的策略参数。
        在实际应用中，这里会调用LLM API。
        """
        self.logger.info(f"模拟LLM生成新参数，当前参数: {current_params}, 反馈: {feedback}")
        
        # 从配置文件获取参数范围
        optimization_config = self.config.get('optimization', {})
        param_ranges = optimization_config.get('param_ranges', {})
        rise_threshold_range = param_ranges.get('rise_threshold', {})
        max_days_range = param_ranges.get('max_days', {})
        
        # 获取rise_threshold的范围
        min_threshold = rise_threshold_range.get('min', 0.03)
        max_threshold = rise_threshold_range.get('max', 0.08)
        
        # 获取max_days的范围
        min_days = max_days_range.get('min', 15)
        max_days = max_days_range.get('max', 30)
        
        # 模拟LLM调整参数
        new_rise_threshold = round(random.uniform(min_threshold, max_threshold), 4)
        new_max_days = random.randint(min_days, max_days)
        
        # 模拟LLM根据反馈进行微调
        if "提高" in feedback and "涨幅" in feedback:
            new_rise_threshold = min(new_rise_threshold * 1.1, max_threshold) # 稍微提高涨幅阈值
        elif "降低" in feedback and "天数" in feedback:
            new_max_days = max(int(new_max_days * 0.9), min_days) # 稍微降低天数

        return {
            "rise_threshold": new_rise_threshold,
            "max_days": new_max_days
        }

    def optimize_strategy(self, num_iterations: int = 10):
        self.logger.info(f"开始LLM驱动的策略优化，迭代次数: {num_iterations}")
        try:
            # 获取所有历史数据用于优化
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d") # 过去5年数据
            
            self.logger.info(f"获取历史数据从 {start_date} 到 {end_date} 用于策略优化")
            historical_data = self.data_module.get_history_data(start_date, end_date)
            historical_data = self.data_module.preprocess_data(historical_data)

            if historical_data.empty:
                self.logger.error("历史数据为空，无法进行策略优化。")
                return False

            current_params = self.strategy_module.get_params()
            self.best_params = current_params.copy()
            self.best_score = -1.0
            
            feedback = ""

            for i in range(num_iterations):
                self.logger.info(f"--- 优化迭代 {i+1}/{num_iterations} ---")
                
                # LLM生成新参数
                new_params = self._simulate_llm_suggestion(current_params, feedback)
                self.strategy_module.update_params(new_params)
                current_params = new_params.copy()
                self.logger.info(f"LLM建议参数: {new_params}")

                # 运行回测
                self.logger.info("运行回测...")
                backtest_results = self.strategy_module.backtest(historical_data)
                
                # 评估策略
                evaluation_results = self.strategy_module.evaluate_strategy(backtest_results)
                current_score = evaluation_results["score"]
                self.logger.info(f"当前策略得分: {current_score:.4f}")

                # 更新最佳参数
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_params = current_params.copy()
                    self.logger.info(f"发现新最佳策略，得分: {self.best_score:.4f}, 参数: {self.best_params}")
                    feedback = "表现良好，尝试进一步优化。"
                else:
                    feedback = "表现不佳，尝试调整方向。"
                
                # 可视化回测结果 (可选)
                # self.strategy_module.visualize_backtest(backtest_results, save_path=f"./results/llm_opt_iter_{i+1}.png")

            self.logger.info("\n--- 策略优化完成 ---")
            self.logger.info(f"最佳策略得分: {self.best_score:.4f}")
            self.logger.info(f"最佳策略参数: {self.best_params}")
            return True
        except Exception as e:
            self.logger.error(f"LLM策略优化过程中发生异常: {e}")
            return False

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("Main")

    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        config = load_config(config_path=config_path)
        
        optimizer = LLMStrategyOptimizer(config)
        optimizer.optimize_strategy(num_iterations=20) # 运行20次迭代

    except Exception as e:
        logger.error(f"LLM策略优化脚本运行失败: {e}")


