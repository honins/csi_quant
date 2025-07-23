#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
参数优化器模块
专门负责策略参数的优化，从原AI优化器中分离出来

功能：
- 策略参数搜索和优化
- 网格搜索和随机搜索
- 真正的贝叶斯优化（使用scikit-optimize）
- 参数范围管理
- 评分函数计算
- 优化结果保存
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from itertools import product

# 贝叶斯优化相关导入
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

from ..utils.base_module import AIModule
from ..utils.common import (
    PerformanceMonitor, DataValidator, MathUtils,
    safe_execute, error_context, FileManager
)


class ParameterOptimizer(AIModule):
    """
    参数优化器
    
    专门负责策略参数的搜索和优化
    """
    
    
# 模块导出
__all__ = ['ParameterOptimizer']