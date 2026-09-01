# Experiment Archive
# 实验归档

This folder archives the earlier research stages from V1 to V4. These versions show how the project evolved before the final V5 Transformer baseline.

本文件夹归档 V1 至 V4 的早期研究阶段，用于展示项目在最终 V5 Transformer baseline 之前的迭代过程。

The main repository README focuses on V5. This archive is mainly used for research traceability, model comparison, and project documentation.

主 README 重点展示 V5，本目录主要用于研究追踪、模型对比和项目复盘。

Raw market data, licensed datasets, cache files, parquet files, large intermediate outputs, and full prediction files are not included.

本目录不包含原始行情数据、授权数据集、缓存文件、parquet 文件、大型中间结果和完整预测明细。

---

## Version Overview
## 版本概览

| Version | Focus | Description | 中文说明 |
|---|---|---|---|
| V1 | HMM market state recognition | Built the initial two-stage framework: HMM regime recognition plus stock return prediction. | 建立初始两阶段框架：HMM 市场状态识别 + 个股收益预测。 |
| V2 | Strict supervised prediction | Added strict lagged features, walk-forward validation, leakage checks, transaction costs, and portfolio diagnostics. | 强化严格滞后特征、walk-forward 验证、信息泄漏检查、交易成本和组合诊断。 |
| V3 | HMM state features | Tested whether HMM state probabilities add predictive value in supervised models. | 测试 HMM 状态概率在监督预测中的增量作用。 |
| V4 | HMM-informed Transformer | Explored multi-scale Transformer and HMM-informed Transformer using 5/20/60-day windows. | 尝试多尺度 Transformer 与 HMM-informed Transformer。 |

---

## Research Path
## 研究路径

```text
V1: HMM market regimes
        ↓
V2: strict supervised prediction
        ↓
V3: HMM state feature testing
        ↓
V4: HMM-informed Transformer exploration
        ↓
V5: Transformer baseline under the stricter V2 evaluation framework
```

```text
V1：HMM 市场状态识别
        ↓
V2：严格监督式个股收益预测
        ↓
V3：HMM 状态变量增量测试
        ↓
V4：HMM-informed Transformer 探索
        ↓
V5：在 V2 严格评估框架下的 Transformer baseline
```

---

## Folder Structure
## 文件夹结构

```text
experiments/
├── README.md
├── v1_hmm_market_state/
├── v2_supervised_prediction/
├── v3_hmm_state_prediction/
└── v4_hmm_informed_transformer/
```

Each version folder may contain:

每个版本文件夹可能包含：

```text
notebooks/   Jupyter notebooks / 实验代码
figures/     Result figures / 结果图表
reports/     Reports and summaries / 报告与总结
```

---

## Data Policy
## 数据说明

Only notebooks, figures, reports, and high-level summaries are included in this archive.

本归档仅保留 notebook、图表、报告和高层总结。

Raw market data, licensed datasets, cache files, parquet files, large intermediate outputs, and full prediction files are not included.

本目录不包含原始行情数据、授权数据集、缓存文件、parquet 文件、大型中间结果和完整预测明细。

Full reproduction requires access to the corresponding local A-share data panel, labels, and cache files.

完整复现实验需要具备对应的本地 A 股数据面板、标签文件和缓存文件。

---

## Notes
## 说明

This archive is designed for documentation and research review rather than direct plug-and-play execution.

本归档主要用于文档展示和研究复盘，而不是开箱即用的完整运行环境。

The latest and primary experiment is V5, which is documented in the main repository README and the main notebook.

最新且主要展示的实验是 V5，相关内容已在仓库主 README 和主 notebook 中说明。
