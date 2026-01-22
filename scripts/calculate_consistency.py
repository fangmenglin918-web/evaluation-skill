#!/usr/bin/env python3
"""
评估一致性计算脚本

计算人工评分与模型评分之间的各类一致性指标。

使用方法:
    python calculate_consistency.py <input_file> [--output <output_file>]

输入文件格式 (CSV/Excel):
    必需列: human_score, model_score
    可选列: case_id, category, dimension

输出:
    - 完全一致率
    - 相邻一致率
    - Cohen's Kappa
    - 加权Kappa
    - Pearson相关系数
    - MAE (平均绝对误差)
"""

import argparse
import sys
from pathlib import Path

def load_data(filepath: str) -> tuple:
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
    
    return df['human_score'].values, df['model_score'].values, df


def exact_agreement(human: list, model: list) -> float:
    """计算完全一致率"""
    import numpy as np
    return np.mean(np.array(human) == np.array(model))


def adjacent_agreement(human: list, model: list, threshold: int = 1) -> float:
    """计算相邻一致率"""
    import numpy as np
    return np.mean(np.abs(np.array(human) - np.array(model)) <= threshold)


def cohens_kappa(human: list, model: list) -> float:
    """计算Cohen's Kappa系数"""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(human, model)


def weighted_kappa(human: list, model: list, weights: str = 'linear') -> float:
    """计算加权Kappa系数"""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(human, model, weights=weights)


def pearson_correlation(human: list, model: list) -> float:
    """计算Pearson相关系数"""
    import numpy as np
    return np.corrcoef(human, model)[0, 1]


def mean_absolute_error(human: list, model: list) -> float:
    """计算平均绝对误差"""
    import numpy as np
    return np.mean(np.abs(np.array(human) - np.array(model)))


def score_distribution(scores: list, name: str) -> dict:
    """计算分数分布"""
    import numpy as np
    from collections import Counter
    counter = Counter(scores)
    total = len(scores)
    dist = {f"{name}_score_{k}": f"{v/total*100:.1f}%" for k, v in sorted(counter.items())}
    return dist


def confusion_matrix_analysis(human: list, model: list) -> dict:
    """生成混淆矩阵分析"""
    import numpy as np
    from collections import defaultdict
    
    matrix = defaultdict(lambda: defaultdict(int))
    for h, m in zip(human, model):
        matrix[h][m] += 1
    
    return dict(matrix)


def calculate_all_metrics(human: list, model: list) -> dict:
    """计算所有一致性指标"""
    metrics = {
        'sample_count': len(human),
        'exact_agreement': exact_agreement(human, model),
        'adjacent_agreement': adjacent_agreement(human, model),
        'cohens_kappa': cohens_kappa(human, model),
        'weighted_kappa_linear': weighted_kappa(human, model, 'linear'),
        'weighted_kappa_quadratic': weighted_kappa(human, model, 'quadratic'),
        'pearson_correlation': pearson_correlation(human, model),
        'mae': mean_absolute_error(human, model),
    }
    
    # 添加分布信息
    metrics.update(score_distribution(human, 'human'))
    metrics.update(score_distribution(model, 'model'))
    
    return metrics


def interpret_metrics(metrics: dict) -> dict:
    """解读指标结果"""
    interpretations = {}
    
    # 完全一致率解读
    ea = metrics['exact_agreement']
    if ea >= 0.7:
        interpretations['exact_agreement'] = '良好 (≥70%)'
    elif ea >= 0.5:
        interpretations['exact_agreement'] = '可接受 (50-70%)'
    else:
        interpretations['exact_agreement'] = '需要优化 (<50%)'
    
    # 相邻一致率解读
    aa = metrics['adjacent_agreement']
    if aa >= 0.9:
        interpretations['adjacent_agreement'] = '良好 (≥90%)'
    elif aa >= 0.8:
        interpretations['adjacent_agreement'] = '可接受 (80-90%)'
    else:
        interpretations['adjacent_agreement'] = '需要优化 (<80%)'
    
    # Kappa解读
    kappa = metrics['cohens_kappa']
    if kappa >= 0.8:
        interpretations['cohens_kappa'] = '极好 (≥0.80)'
    elif kappa >= 0.6:
        interpretations['cohens_kappa'] = '较好 (0.60-0.80)'
    elif kappa >= 0.4:
        interpretations['cohens_kappa'] = '中等 (0.40-0.60)'
    else:
        interpretations['cohens_kappa'] = '较差 (<0.40)'
    
    # MAE解读
    mae = metrics['mae']
    if mae <= 0.5:
        interpretations['mae'] = '良好 (≤0.5)'
    elif mae <= 1.0:
        interpretations['mae'] = '可接受 (0.5-1.0)'
    else:
        interpretations['mae'] = '需要优化 (>1.0)'
    
    return interpretations


def print_report(metrics: dict, interpretations: dict):
    """打印分析报告"""
    print("\n" + "="*60)
    print("评估一致性分析报告")
    print("="*60)
    
    print(f"\n样本数量: {metrics['sample_count']}")
    
    print("\n【核心指标】")
    print(f"  完全一致率: {metrics['exact_agreement']*100:.1f}% - {interpretations['exact_agreement']}")
    print(f"  相邻一致率: {metrics['adjacent_agreement']*100:.1f}% - {interpretations['adjacent_agreement']}")
    print(f"  Cohen's Kappa: {metrics['cohens_kappa']:.3f} - {interpretations['cohens_kappa']}")
    print(f"  加权Kappa(线性): {metrics['weighted_kappa_linear']:.3f}")
    print(f"  加权Kappa(二次): {metrics['weighted_kappa_quadratic']:.3f}")
    print(f"  Pearson相关系数: {metrics['pearson_correlation']:.3f}")
    print(f"  MAE: {metrics['mae']:.3f} - {interpretations['mae']}")
    
    print("\n【分数分布】")
    print("  人工评分分布:")
    for k, v in metrics.items():
        if k.startswith('human_score_'):
            score = k.replace('human_score_', '')
            print(f"    {score}分: {v}")
    
    print("  模型评分分布:")
    for k, v in metrics.items():
        if k.startswith('model_score_'):
            score = k.replace('model_score_', '')
            print(f"    {score}分: {v}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='计算评估一致性指标')
    parser.add_argument('input_file', help='输入文件路径 (CSV/Excel)')
    parser.add_argument('--output', '-o', help='输出文件路径 (JSON)')
    args = parser.parse_args()
    
    try:
        # 检查依赖
        import pandas as pd
        import numpy as np
        from sklearn.metrics import cohen_kappa_score
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install pandas numpy scikit-learn openpyxl")
        sys.exit(1)
    
    # 加载数据
    human, model, df = load_data(args.input_file)
    
    # 计算指标
    metrics = calculate_all_metrics(human, model)
    interpretations = interpret_metrics(metrics)
    
    # 打印报告
    print_report(metrics, interpretations)
    
    # 保存结果
    if args.output:
        import json
        result = {
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in metrics.items()},
            'interpretations': interpretations
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {args.output}")


if __name__ == '__main__':
    main()
