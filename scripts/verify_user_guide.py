#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证USER_GUIDE.md文档质量和完整性的脚本
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_file_exists(file_path):
    """检查文件是否存在"""
    return os.path.exists(file_path)

def check_markdown_syntax(content):
    """检查Markdown语法基本正确性"""
    issues = []
    lines = content.split('\n')
    
    # 检查标题层级
    heading_levels = []
    for i, line in enumerate(lines, 1):
        if line.startswith('#'):
            level = len(line.split()[0])
            heading_levels.append((i, level, line.strip()))
    
    # 检查标题层级是否合理（不能跳级太多）
    for i in range(1, len(heading_levels)):
        prev_level = heading_levels[i-1][1]
        curr_level = heading_levels[i][1]
        if curr_level > prev_level + 1:
            issues.append(f"第{heading_levels[i][0]}行: 标题层级跳跃过大 (从 {'#'*prev_level} 到 {'#'*curr_level})")
    
    # 检查代码块配对
    code_block_count = content.count('```')
    if code_block_count % 2 != 0:
        issues.append("代码块(```)数量不匹配，可能有未闭合的代码块")
    
    # 检查链接格式
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, content)
    for link_text, link_url in links:
        if not link_url.strip():
            issues.append(f"空链接: [{link_text}]()")
    
    return issues

def check_command_examples(content):
    """检查命令示例的完整性"""
    issues = []
    
    # 检查是否包含基本命令
    basic_commands = [
        'python run.py b',
        'python run.py ai -m full',
        'python run.py s',
        'python run.py r',
        'python run.py opt',
        'python run.py bot'
    ]
    
    for cmd in basic_commands:
        if cmd not in content:
            issues.append(f"缺少基本命令示例: {cmd}")
    
    # 检查虚拟环境说明
    venv_keywords = ['venv', '虚拟环境', 'activate']
    if not any(keyword in content for keyword in venv_keywords):
        issues.append("缺少虚拟环境相关说明")
    
    return issues

def check_section_completeness(content):
    """检查章节完整性"""
    required_sections = [
        '系统概述',
        '快速开始',
        '命令参考',
        '配置文件',
        '使用场景',
        '输出文件',
        '故障排除',
        '高级用法'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    return missing_sections

def check_links_validity(content, base_path):
    """检查内部链接的有效性"""
    issues = []
    
    # 提取所有相对路径链接
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, content)
    
    for link_text, link_url in links:
        # 跳过外部链接和锚点链接
        if link_url.startswith(('http://', 'https://', '#')):
            continue
            
        # 检查本地文件链接
        if not link_url.startswith('/'):
            full_path = os.path.join(base_path, link_url)
            if not os.path.exists(full_path):
                issues.append(f"无效链接: [{link_text}]({link_url}) - 文件不存在")
    
    return issues

def analyze_document_structure(content):
    """分析文档结构"""
    lines = content.split('\n')
    structure = {
        'total_lines': len(lines),
        'headings': [],
        'code_blocks': 0,
        'tables': 0,
        'links': 0,
        'lists': 0
    }
    
    for line in lines:
        if line.startswith('#'):
            level = len(line.split()[0])
            title = line.strip()
            structure['headings'].append((level, title))
        elif line.strip().startswith('```'):
            structure['code_blocks'] += 1
        elif '|' in line and line.strip().startswith('|'):
            structure['tables'] += 1
        elif re.search(r'\[([^\]]+)\]\(([^)]+)\)', line):
            structure['links'] += len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line))
        elif line.strip().startswith(('-', '*', '+')):
            structure['lists'] += 1
    
    structure['code_blocks'] = structure['code_blocks'] // 2  # 每个代码块有开始和结束
    
    return structure

def main():
    """主函数"""
    print("🔍 验证 USER_GUIDE.md 文档质量...")
    print("=" * 50)
    
    # 检查文件是否存在
    user_guide_path = "USER_GUIDE.md"
    if not check_file_exists(user_guide_path):
        print("❌ USER_GUIDE.md 文件不存在!")
        return
    
    # 读取文档内容
    try:
        with open(user_guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return
    
    print(f"📄 文档大小: {len(content):,} 字符")
    print()
    
    # 分析文档结构
    print("📊 文档结构分析:")
    structure = analyze_document_structure(content)
    print(f"  - 总行数: {structure['total_lines']:,}")
    print(f"  - 标题数量: {len(structure['headings'])}")
    print(f"  - 代码块数量: {structure['code_blocks']}")
    print(f"  - 表格行数: {structure['tables']}")
    print(f"  - 链接数量: {structure['links']}")
    print(f"  - 列表项数: {structure['lists']}")
    print()
    
    # 显示文档大纲
    print("📋 文档大纲:")
    for level, title in structure['headings'][:10]:  # 只显示前10个标题
        indent = "  " * (level - 1)
        print(f"{indent}- {title}")
    if len(structure['headings']) > 10:
        print(f"  ... 还有 {len(structure['headings']) - 10} 个标题")
    print()
    
    # 检查Markdown语法
    print("🔍 检查Markdown语法:")
    markdown_issues = check_markdown_syntax(content)
    if markdown_issues:
        for issue in markdown_issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Markdown语法检查通过")
    print()
    
    # 检查命令示例
    print("🔍 检查命令示例:")
    command_issues = check_command_examples(content)
    if command_issues:
        for issue in command_issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ 命令示例检查通过")
    print()
    
    # 检查章节完整性
    print("🔍 检查章节完整性:")
    missing_sections = check_section_completeness(content)
    if missing_sections:
        for section in missing_sections:
            print(f"  ❌ 缺少章节: {section}")
    else:
        print("  ✅ 章节完整性检查通过")
    print()
    
    # 检查链接有效性
    print("🔍 检查链接有效性:")
    link_issues = check_links_validity(content, os.path.dirname(user_guide_path) or ".")
    if link_issues:
        for issue in link_issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ 链接有效性检查通过")
    print()
    
    # 总结
    total_issues = len(markdown_issues) + len(command_issues) + len(missing_sections) + len(link_issues)
    print("📊 验证总结:")
    print(f"  - 发现问题数量: {total_issues}")
    
    if total_issues == 0:
        print("  🎉 文档质量验证通过！")
        return_code = 0
    else:
        print("  ⚠️  发现一些需要改进的地方")
        return_code = 1
    
    print()
    print("💡 建议:")
    print("  - 定期运行此脚本验证文档质量")
    print("  - 添加新功能时及时更新用户指南")
    print("  - 确保所有链接指向正确的文件")
    print("  - 保持命令示例与实际代码同步")
    
    return return_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 