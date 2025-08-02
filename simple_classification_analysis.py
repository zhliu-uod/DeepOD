#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测分类错误原因分析（无外部依赖版本）
基于您提供的图片分析分类错误的可能原因
"""

def analyze_tabular_metrics_issue():
    """
    分析tabular_metrics函数中的分类问题
    """
    
    print("=== 异常检测分类错误分析 ===\n")
    
    print("根据您提供的图片，显示了多个样本的Score和Label，")
    print("其中出现了分类错误的情况。让我分析可能的原因：\n")
    
    print("1. 阈值选择问题 (最可能的原因)")
    print("   在deepod/metrics/_anomaly_detection.py的tabular_metrics函数中：")
    print("   ```python")
    print("   ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)")
    print("   thresh = np.percentile(y_score, ratio)")
    print("   y_pred = (y_score >= thresh).astype(int)")
    print("   ```")
    print("   ")
    print("   问题：")
    print("   - 该方法假设分数最高的k%样本就是异常（k=异常样本比例）")
    print("   - 但这个假设在很多情况下不成立")
    print("   - 如果模型学习不好，正常样本可能获得更高的分数")
    print("   - 百分位数阈值可能不是最优的分类阈值")
    
    print("\n2. 模型学习问题")
    print("   - 模型可能没有正确学习到异常模式")
    print("   - 训练数据质量问题（包含异常样本或标注错误）")
    print("   - 特征不够判别性")
    print("   - 模型参数需要调优")
    
    print("\n3. 数据分布问题")
    print("   - 正常和异常样本在特征空间中重叠严重")
    print("   - 异常样本内部差异很大")
    print("   - 数据标注不一致")
    
    print("\n4. 评估方法局限性")
    print("   - 百分位数方法不适用于所有场景")
    print("   - 需要根据具体应用选择合适的阈值")
    
    return True


def suggest_solutions():
    """
    提供具体的解决方案
    """
    
    print("\n=== 解决方案建议 ===\n")
    
    print("1. 改进阈值选择方法")
    print("   推荐使用以下方法替代原始的百分位数方法：")
    print("   ")
    print("   a) 最优F1阈值：")
    print("   ```python")
    print("   from sklearn.metrics import precision_recall_curve")
    print("   precision, recall, thresholds = precision_recall_curve(y_true, y_score)")
    print("   f1_scores = 2 * precision * recall / (precision + recall + 1e-8)")
    print("   best_threshold = thresholds[np.argmax(f1_scores)]")
    print("   ```")
    print("   ")
    print("   b) 基于业务需求的阈值：")
    print("   - 如果要最小化误报：提高阈值，优先精确率")
    print("   - 如果要最小化漏报：降低阈值，优先召回率")
    print("   ")
    print("   c) 使用验证集调优阈值")
    
    print("\n2. 改进模型训练")
    print("   - 确保训练集只包含正常样本（无监督异常检测）")
    print("   - 增加训练轮数或调整学习率")
    print("   - 尝试不同的模型架构")
    print("   - 使用数据增强技术")
    
    print("\n3. 数据预处理优化")
    print("   - 检查数据标准化/归一化")
    print("   - 特征选择和特征工程")
    print("   - 处理噪声和异常值")
    print("   - 确保数据质量")
    
    print("\n4. 模型选择")
    print("   - 尝试不同的异常检测算法：")
    print("     * Isolation Forest")
    print("     * One-Class SVM") 
    print("     * Local Outlier Factor")
    print("     * AutoEncoder")
    print("     * SVDD (Support Vector Data Description)")
    print("   - 使用集成方法")
    
    print("\n5. 评估指标多样化")
    print("   - 不要仅依赖F1分数")
    print("   - 同时查看AUC-ROC和AUC-PR")
    print("   - 分析混淆矩阵")
    print("   - 考虑业务特定的评估指标")


def code_fix_examples():
    """
    提供具体的代码修复示例
    """
    
    print("\n=== 代码修复示例 ===\n")
    
    print("修复后的tabular_metrics函数：")
    print("```python")
    print("def improved_tabular_metrics(y_true, y_score, method='best_f1'):")
    print("    from sklearn import metrics")
    print("    import numpy as np")
    print("    ")
    print("    # 计算AUC（不依赖阈值）")
    print("    auc_roc = metrics.roc_auc_score(y_true, y_score)")
    print("    auc_pr = metrics.average_precision_score(y_true, y_score)")
    print("    ")
    print("    if method == 'best_f1':")
    print("        # 寻找最优F1阈值")
    print("        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)")
    print("        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)")
    print("        best_idx = np.argmax(f1_scores)")
    print("        threshold = thresholds[best_idx]")
    print("    elif method == 'percentile':")
    print("        # 原始方法")
    print("        ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)")
    print("        threshold = np.percentile(y_score, ratio)")
    print("    else:")
    print("        # 自定义阈值")
    print("        threshold = method")
    print("    ")
    print("    # 基于阈值分类")
    print("    y_pred = (y_score >= threshold).astype(int)")
    print("    ")
    print("    # 计算F1分数")
    print("    precision = metrics.precision_score(y_true, y_pred)")
    print("    recall = metrics.recall_score(y_true, y_pred)")
    print("    f1 = metrics.f1_score(y_true, y_pred)")
    print("    ")
    print("    return auc_roc, auc_pr, f1, threshold")
    print("```")
    
    print("\n使用示例：")
    print("```python")
    print("# 使用改进的方法")
    print("auc_roc, auc_pr, f1, threshold = improved_tabular_metrics(y_true, y_score, 'best_f1')")
    print("print(f'最优阈值: {threshold:.4f}, F1: {f1:.4f}')")
    print("```")


def debugging_checklist():
    """
    提供调试检查清单
    """
    
    print("\n=== 调试检查清单 ===\n")
    
    print("当遇到分类错误时，请依次检查：")
    print("")
    print("□ 1. 数据检查")
    print("   - 异常样本比例是否合理（通常1-10%）")
    print("   - 数据标注是否正确")
    print("   - 是否存在数据泄露")
    print("")
    print("□ 2. 分数分布检查") 
    print("   - 异常样本分数是否普遍高于正常样本")
    print("   - 分数分布是否有明显分离")
    print("   - 是否存在分数反转（异常分数低于正常分数）")
    print("")
    print("□ 3. 阈值检查")
    print("   - 当前使用的阈值是否合理")
    print("   - 尝试不同的阈值选择方法")
    print("   - 手动调整阈值观察效果")
    print("")
    print("□ 4. 模型检查")
    print("   - 训练是否收敛")
    print("   - 模型参数是否需要调优")
    print("   - 尝试其他算法")
    print("")
    print("□ 5. 评估指标检查")
    print("   - 不要只看F1分数")
    print("   - 查看精确率和召回率的平衡")
    print("   - 考虑AUC-ROC和AUC-PR")


if __name__ == "__main__":
    print("基于您提供的异常检测分类结果图片，")
    print("我来分析可能的分类错误原因和解决方案。\n")
    
    analyze_tabular_metrics_issue()
    suggest_solutions() 
    code_fix_examples()
    debugging_checklist()
    
    print("\n=== 总结 ===")
    print("您遇到的分类错误很可能是由于tabular_metrics函数中")
    print("使用百分位数方法选择阈值导致的。这个方法假设分数最高的")
    print("k%样本就是异常，但在实际情况中这个假设可能不成立。")
    print("")
    print("建议：")
    print("1. 使用最优F1阈值方法替代百分位数方法")
    print("2. 检查模型是否正确学习了异常模式") 
    print("3. 尝试不同的异常检测算法")
    print("4. 根据业务需求调整评估策略")
    print("")
    print("希望这些分析和建议能帮助您解决分类错误问题！")