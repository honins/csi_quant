#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测模块包
包含预测相关的工具函数和类
"""

from .prediction_utils import setup_logging, predict_and_validate

__all__ = ['setup_logging', 'predict_and_validate'] 