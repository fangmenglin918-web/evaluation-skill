#!/usr/bin/env python3
"""
评分差异分析脚本

分析人工评分与模型评分之间的差异模式，识别系统性偏差和问题区域。

使用方法:
    python analyze_discrepancy.py <input_file> [--output <output_dir>]

输入文件格式 (CSV/Excel):
    必需列: human_score, model_score
    可选列: case_id, category, dimension, content, reasoning

输出:
    - 差异分布分析
    - 按类别/维度分组分析
    - 系统性偏差检测
    - 问题案例列表
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

def load_data(filepath: str):
    """加载评分数据"""
    import pandas as pd
    
    path = Path(filepath)
    if path.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    required_cols = ['human_score', 'model_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    return df


def analyze_discrepancy_distribution(df) -> dict:
    """分析差异分布"""
    import numpy as np
    
    diff = df['model_score'] - df['human_score']
    
    analysis = {
        'mean_diff': float(diff.mean()),
        'std_diff': float(diff.std()),
        'median_diff': float(diff.median()),
        'distribution': {}
    }
    
    # 差异分布
    for d in range(-4, 5):
        count = (diff == d).sum()
        pct = count / len(diff) * 100
        analysis['distribution'][f'diff_{d:+d}'] = {
            'count': int(count),
            'percentage': f'{pct:.1f}%'
        }
    
    # 偏差方向
    analysis['model_higher'] = float((diff > 0).mean() * 100)
    analysis['model_lower'] = float((diff < 0).mean() * 100)
    analysis['exact_match'] = float((diff == 0).mean() * 100)
    
    return analysis


def detect_systematic_bias(df) -> dict:
    """检测系统性偏差"""
    import numpy as np
    from scipy import stats
    
    diff = df['model_score'] - df['human_score']
    
    # t检验判断是否存在系统性偏差
    t_stat, p_value = stats.ttest_1samp(diff, 0)
    
    bias = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'direction': None,
        'recommendation': None
    }
    
    mean_diff = diff.mean()
    if bias['significant']:
        if mean_diff > 0.2:
            bias['direction'] = '模型系统性偏高'
            bias['recommendation'] = '建议提高Prompt的严格程度，强化扣分条款'
        elif mean_diff < -0.2:
            bias['direction'] = '模型系统性偏低'
            bias['recommendation'] = '建议放宽判定阈值，或增加积极评价的条款'
        else:
            bias['direction'] = '存在轻微偏差'
            bias['recommendation'] = '可以进行微调优化'
    else:
        bias['direction'] = '无显著系统性偏差'
        bias['recommendation'] = '整体校准良好，关注特定类型的问题'
    
    return bias


def analyze_by_score_level(df) -> dict:
    """按分数档位分析"""
    import numpy as np
    
    analysis = {}
    
    for score in sorted(df['human_score'].unique()):
        subset = df[df['human_score'] == score]
        if len(subset) == 0:
            continue
        
        model_scores = subset['model_score']
        exact_match = (model_scores == score).mean() * 100
        
        analysis[f'score_{score}'] = {
            'count': len(subset),
            'exact_agreement': f'{exact_match:.1f}%',
            'model_mean': float(model_scores.mean()),
            'model_distribution': dict(model_scores.value_counts().sort_index())
        }
    
    return analysis


def analyze_by_category(df, category_col: str) -> dict:
    """按类别分析"""
    if category_col not in df.columns:
        return {}
    
    import numpy as np
    
    analysis = {}
    
    for category in df[category_col].unique():
        subset = df[df[category_col] == category]
        if len(subset) < 5:  # 样本太少跳过
            continue
        
        diff = subset['model_score'] - subset['human_score']
        exact = (diff == 0).mean() * 100
        adjacent = (abs(diff) <= 1).mean() * 100
        
        analysis[str(category)] = {
            'count': len(subset),
            'exact_agreement': f'{exact:.1f}%',
            'adjacent_agreement': f'{adjacent:.1f}%',
            'mean_diff': float(diff.mean()),
            'bias_direction': '偏高' if diff.mean() > 0.2 else ('偏低' if diff.mean() < -0.2 else '正常')
        }
    
    return analysis


def find_problem_cases(df, threshold: int = 2) -> list:
    """找出差异较大的问题案例"""
    import numpy as np
    
    df = df.copy()
    df['diff'] = df['model_score'] - df['human_score']
    df['abs_diff'] = abs(df['diff'])
    
    problems = df[df['abs_diff'] >= threshold].sort_values('abs_diff', ascending=False)
    
    result = []
    for _, row in problems.head(20).iterrows():  # 最多返回20个
        case = {
            'human_score': int(row['human_score']),
            'model_score': int(row['model_score']),
            'diff': int(row['diff']),
        }
        
        # 添加可选字段
        if 'case_id' in row:
            case['case_id'] = str(row['case_id'])
        if 'category' in row:
            case['category'] = str(row['category'])
        if 'dimension' in row:
            case['dimension'] = str(row['dimension'])
        if 'content' in row and not (hasattr(row['content'], '__len__') and len(str(row['content'])) > 200):
            case['content'] = str(row['content'])[:200]
        
        result.append(case)
    
    return result


def generate_recommendations(discrepancy_dist: dict, bias: dict, score_analysis: dict) -> list:
    """生成优化建议"""
    recommendations = []
    
    # 基于系统性偏差的建议
    if bias['significant']:
        recommendations.append({
            'type': '系统性偏差',
            'issue': bias['direction'],
            'suggestion': bias['recommendation']
        })
    
    # 基于分数档位分析的建议
    for score_key, data in score_analysis.items():
        score = score_key.replace('score_', '')
        agreement = float(data['exact_agreement'].replace('%', ''))
        if agreement < 50:
            recommendations.append({
                'type': '特定档位问题',
                'issue': f'{score}分档位一致率过低 ({data["exact_agreement"]})',
                'suggestion': f'检查{score}分的判定标准是否清晰，增加锚点示例'
            })
    
    # 基于差异分布的建议
    if discrepancy_dist['model_higher'] > 30:
        recommendations.append({
            'type': '分数分布偏移',
            'issue': f'模型评分偏高的比例过大 ({discrepancy_dist["model_higher"]:.1f}%)',
            'suggestion': '强化Prompt中的扣分条款，增加负面案例'
        })
    
    if discrepancy_dist['model_lower'] > 30:
        recommendations.append({
            'type': '分数分布偏移',
            'issue': f'模型评分偏低的比例过大 ({discrepancy_dist["model_lower"]:.1f}%)',
            'suggestion': '检查Prompt是否过于严格，放宽部分判定条件'
        })
    
    return recommendations


def print_report(analysis: dict):
    """打印分析报告"""
    print("\n" + "="*60)
    print("评分差异分析报告")
    print("="*60)
    
    # 差异分布
    print("\n【差异分布】")
    dist = analysis['discrepancy_distribution']
    print(f"  平均差异(模型-人工): {dist['mean_diff']:+.2f}")
    print(f"  差异标准差: {dist['std_diff']:.2f}")
    print(f"  模型偏高: {dist['model_higher']:.1f}%")
    print(f"  模型偏低: {dist['model_lower']:.1f}%")
    print(f"  完全一致: {dist['exact_match']:.1f}%")
    
    # 系统性偏差
    print("\n【系统性偏差检测】")
    bias = analysis['systematic_bias']
    print(f"  判定: {bias['direction']}")
    print(f"  p值: {bias['p_value']:.4f} ({'显著' if bias['significant'] else '不显著'})")
    print(f"  建议: {bias['recommendation']}")
    
    # 按分数档位
    print("\n【按分数档位分析】")
    for score_key, data in analysis['score_level_analysis'].items():
        score = score_key.replace('score_', '')
        print(f"  {score}分: 样本{data['count']}个, 一致率{data['exact_agreement']}")
    
    # 按类别分析
    if analysis['category_analysis']:
        print("\n【按类别分析】")
        for cat, data in analysis['category_analysis'].items():
            print(f"  {cat}: 样本{data['count']}个, 一致率{data['exact_agreement']}, {data['bias_direction']}")
    
    # 问题案例
    print("\n【差异较大的案例】(前5个)")
    for i, case in enumerate(analysis['problem_cases'][:5], 1):
        print(f"  {i}. 人工{case['human_score']}分 vs 模型{case['model_score']}分 (差{case['diff']:+d})")
        if 'case_id' in case:
            print(f"     Case ID: {case['case_id']}")
    
    # 优化建议
    print("\n【优化建议】")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"  {i}. [{rec['type']}] {rec['issue']}")
        print(f"     → {rec['suggestion']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='分析评分差异')
    parser.add_argument('input_file', help='输入文件路径 (CSV/Excel)')
    parser.add_argument('--output', '-o', help='输出目录路径')
    parser.add_argument('--category', '-c', default='category', help='类别列名 (默认: category)')
    args = parser.parse_args()
    
    try:
        import pandas as pd
        import numpy as np
        from scipy import stats
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install pandas numpy scipy openpyxl")
        sys.exit(1)
    
    # 加载数据
    df = load_data(args.input_file)
    
    # 执行分析
    analysis = {
        'discrepancy_distribution': analyze_discrepancy_distribution(df),
        'systematic_bias': detect_systematic_bias(df),
        'score_level_analysis': analyze_by_score_level(df),
        'category_analysis': analyze_by_category(df, args.category),
        'problem_cases': find_problem_cases(df),
    }
    
    # 生成建议
    analysis['recommendations'] = generate_recommendations(
        analysis['discrepancy_distribution'],
        analysis['systematic_bias'],
        analysis['score_level_analysis']
    )
    
    # 打印报告
    print_report(analysis)
    
    # 保存结果
    if args.output:
        import json
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
        
        # 导出问题案例
        if analysis['problem_cases']:
            problem_df = pd.DataFrame(analysis['problem_cases'])
            problem_df.to_csv(output_dir / 'problem_cases.csv', index=False)
        
        print(f"\n结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
