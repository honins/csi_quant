#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings


def main():
    parser = argparse.ArgumentParser(description='按置信度分箱绘制未来最大涨幅的均值曲线')
    parser.add_argument('--csv', required=True, help='输入CSV路径，需包含列：置信度、未来最大涨幅')
    parser.add_argument('--out', required=True, help='输出PNG图像路径')
    parser.add_argument('--bins', type=int, default=10, help='分箱数量（默认10）')
    parser.add_argument('--summary', required=False, help='输出分箱统计的CSV路径（可选）')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    for col in ['置信度', '未来最大涨幅']:
        if col not in df.columns:
            raise ValueError(f'缺少列: {col}')

    # 处理可能的百分号字符串
    df['未来最大涨幅'] = df['未来最大涨幅'].astype(str).str.replace('%', '', regex=False)
    df['未来最大涨幅'] = pd.to_numeric(df['未来最大涨幅'], errors='coerce')
    df['置信度'] = pd.to_numeric(df['置信度'], errors='coerce')
    df = df.dropna(subset=['置信度', '未来最大涨幅'])

    conf = df['置信度']
    y = df['未来最大涨幅']

    # 构造分箱：优先使用分位数分箱，失败则退化为等宽分箱
    bins_count = args.bins
    try:
        binser = pd.qcut(conf, q=bins_count, duplicates='drop')
    except Exception as e:
        warnings.warn(f'qcut失败，改用等宽分箱: {e}')
        binser = pd.cut(conf, bins=bins_count, include_lowest=True)

    grouped = df.groupby(binser)
    summary = grouped.agg(
        mean_conf=('置信度', 'mean'),
        min_conf=('置信度', 'min'),
        max_conf=('置信度', 'max'),
        count=('未来最大涨幅', 'size'),
        mean_rise=('未来最大涨幅', 'mean'),
        median_rise=('未来最大涨幅', 'median'),
        q25=('未来最大涨幅', lambda s: float(np.nanpercentile(s, 25))),
        q75=('未来最大涨幅', lambda s: float(np.nanpercentile(s, 75))),
    ).sort_values('mean_conf')

    # 绘图
    x = summary['mean_conf'].values
    mean_y = summary['mean_rise'].values
    q25 = summary['q25'].values
    q75 = summary['q75'].values

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(x, mean_y, marker='o', color='#1f77b4', label='分箱均值')
    ax.fill_between(x, q25, q75, color='#1f77b4', alpha=0.18, label='IQR 25%-75%')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlabel('置信度（分箱均值）')
    ax.set_ylabel('未来最大涨幅（分箱均值）')
    ax.set_title(f'置信度分箱均值曲线（{len(summary)}分位）')
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)

    # 写出分箱统计CSV（可选）
    if args.summary:
        sum_path = Path(args.summary)
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        summary_out = summary.reset_index(drop=True)
        summary_out.index = summary_out.index + 1  # bin编号从1开始
        summary_out.to_csv(sum_path, index_label='bin')

    # 控制台输出简要统计（便于快速查看）
    print('bins=', len(summary))
    print('out_png=', str(out_path))
    if args.summary:
        print('out_csv=', str(sum_path))
    print('head:')
    print(summary[['min_conf', 'max_conf', 'count', 'mean_rise']].head(5).to_string())
    print('tail:')
    print(summary[['min_conf', 'max_conf', 'count', 'mean_rise']].tail(5).to_string())


if __name__ == '__main__':
    # 尝试设置中文字体，避免中文显示为方块
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    try:
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    except Exception:
        pass
    main()