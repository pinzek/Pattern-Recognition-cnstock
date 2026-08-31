# 个股截面预测 V2：Test 集预测效果报告

## 一、任务说明

本版本按照老师要求，从单纯的市场状态识别转为监督预测任务。目标是使用日线数据预测个股未来1日涨跌幅和未来1日是否上涨。

预测形式为：

X_t, Z_t -> r_{t+1}

其中，X_t 表示个股和市场日线特征，Z_t 表示市场状态变量，r_{t+1} 表示未来1日收益率。

## 二、Train/Test 切分

训练集：2020-01-02 至 2024-12-31

测试集：2025-01-02 至 2026-01-30

## 三、Test 集指标

### base_no_state

- 样本数：1351492
- 交易日数：263
- Overall IC：0.0750
- Overall Rank IC：0.0406
- Daily IC Mean：0.1249
- Daily IC t值：24.8941
- Daily Rank IC Mean：0.0794
- Daily Rank IC t值：9.4485
- 方向准确率：51.968%
- AUC：0.5351
- Top组日均收益：0.326%
- Top-Bottom日均收益：0.392%
- Top-Bottom日频Sharpe：0.4038

### with_state

- 样本数：1351492
- 交易日数：263
- Overall IC：0.0638
- Overall Rank IC：0.0494
- Daily IC Mean：0.1242
- Daily IC t值：25.5822
- Daily Rank IC Mean：0.0842
- Daily Rank IC t值：11.2406
- 方向准确率：54.195%
- AUC：0.5622
- Top组日均收益：0.348%
- Top-Bottom日均收益：0.401%
- Top-Bottom日频Sharpe：0.4555

## 四、结果解释

本版本更贴近量化研究的核心问题：不是只解释市场处于什么状态，而是在 Test 集上直接评估未来涨跌幅和涨跌方向预测效果。重点关注 Daily Rank IC、方向准确率、AUC、Top组收益和Top-Bottom多空收益。

如果 with_state 模型相对 base_no_state 在 Daily Rank IC、AUC 或分组收益上有所提升，说明市场状态变量对个股收益预测具有增量信息。如果提升不明显，则说明当前状态变量对预测帮助有限，需要继续改进特征、模型或预测目标。
