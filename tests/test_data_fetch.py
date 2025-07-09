#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据获取功能测试脚本
"""

import os
import sys
import pandas as pd
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_data_files():
    """测试数据文件是否存在且格式正确"""
    print("开始测试数据文件...")
    
    data_dir = "data"
    expected_files = [
        "SHSE.000852_1d.csv",
        "SHSE.000905_1d.csv"
    ]
    
    results = {}
    
    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filename}")
            results[filename] = False
            continue
            
        try:
            # 读取CSV文件
            df = pd.read_csv(filepath)
            
            # 检查必要的列
            required_columns = ['index', 'open', 'high', 'low', 'close', 'volume', 'amount', 'date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ {filename} 缺少列: {missing_columns}")
                results[filename] = False
                continue
                
            # 检查数据行数
            if len(df) == 0:
                print(f"❌ {filename} 数据为空")
                results[filename] = False
                continue
                
            # 检查最新数据日期
            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max()
            today = datetime.now().date()
            
            print(f"✅ {filename}:")
            print(f"   - 数据行数: {len(df)}")
            print(f"   - 最新日期: {latest_date.strftime('%Y-%m-%d')}")
            print(f"   - 数据范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {latest_date.strftime('%Y-%m-%d')}")
            
            # 检查数据是否是最新的（允许1天的延迟）
            if (today - latest_date.date()).days <= 1:
                print(f"   - 数据状态: 最新 ✅")
            else:
                print(f"   - 数据状态: 可能过期 ⚠️")
                
            results[filename] = True
            
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
            results[filename] = False
    
    return results

def test_data_quality():
    """测试数据质量"""
    print("\n开始测试数据质量...")
    
    data_dir = "data"
    files = ["SHSE.000852_1d.csv", "SHSE.000905_1d.csv"]
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            continue
            
        try:
            df = pd.read_csv(filepath)
            
            # 检查数据类型
            print(f"\n📊 {filename} 数据质量检查:")
            
            # 检查价格数据
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    null_count = df[col].isnull().sum()
                    print(f"   - {col}: 范围 [{min_val:.2f}, {max_val:.2f}], 空值: {null_count}")
            
            # 检查成交量数据
            if 'volume' in df.columns:
                min_vol = df['volume'].min()
                max_vol = df['volume'].max()
                null_count = df['volume'].isnull().sum()
                print(f"   - volume: 范围 [{min_vol:.0f}, {max_vol:.0f}], 空值: {null_count}")
            
            # 检查日期连续性
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df_sorted = df.sort_values('date')
                date_diff = df_sorted['date'].diff().dt.days
                avg_gap = date_diff.mean()
                max_gap = date_diff.max()
                print(f"   - 日期间隔: 平均 {avg_gap:.1f} 天, 最大 {max_gap} 天")
                
        except Exception as e:
            print(f"❌ 数据质量检查失败 {filename}: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("数据获取功能测试")
    print("=" * 50)
    
    # 测试数据文件
    results = test_data_files()
    
    # 测试数据质量
    test_data_quality()
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"文件检查: {success_count}/{total_count} 通过")
    
    if success_count == total_count:
        print("✅ 所有数据文件检查通过！")
        return {
            "code": 200,
            "msg": "数据获取功能测试通过",
            "data": {
                "files_checked": total_count,
                "files_passed": success_count,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    else:
        print("❌ 部分数据文件检查失败")
        return {
            "code": 500,
            "msg": "数据获取功能测试失败",
            "data": {
                "files_checked": total_count,
                "files_passed": success_count,
                "results": results,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

if __name__ == "__main__":
    result = main()
    print(f"\n测试结果: {result}") 