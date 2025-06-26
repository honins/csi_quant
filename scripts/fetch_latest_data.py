#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
获取最新数据脚本 - 入口文件
运行此脚本获取000852和000905的最新数据
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.fetch_latest_data import main

if __name__ == "__main__":
    main() 