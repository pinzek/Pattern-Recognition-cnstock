# Transformer 与 HMM-informed Transformer 效果对比报告

## 一、实验目的

本实验在现有A股日线数据基础上，比较普通多尺度Transformer与加入HMM状态变量后的HMM-informed Transformer在test集上的预测效果。核心目标是验证HMM多状态机制是否能够为Transformer多头注意力模型提供增量信息。

## 二、模型设置

普通Transformer使用过去5日、20日、60日三个窗口作为短、中、长期日线序列输入，通过Transformer Encoder和Multi-Head Attention融合不同时间尺度信息，预测未来1日涨跌幅和未来1日是否上涨。

HMM-informed Transformer在普通Transformer基础上，额外加入HMM识别得到的市场状态概率，使模型在预测个股未来收益时能够同时考虑当前market regime信息。

## 三、Test集效果对比

| 模型 | IC | Rank IC | Daily IC Mean | Daily Rank IC Mean | 方向准确率 | AUC | MSE | Top组日均收益 | Top-Bottom日均收益 | Top-Bottom Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transformer_no_hmm | 0.0509 | 0.0429 | 0.0925 | 0.0814 | 0.5087 | 0.5194 | 0.001000 | 0.003121 | 0.003892 | 0.3065 |
| transformer_with_hmm | 0.0867 | 0.0954 | 0.0775 | 0.0796 | 0.5248 | 0.5457 | 0.000910 | 0.002685 | 0.003036 | 0.2481 |

## 四、实验结果解读

从预测指标来看，加入HMM状态变量后，Transformer的整体预测相关性和方向判断能力有所提升。普通Transformer的IC为0.0509，Rank IC为0.0429，方向准确率为50.87%，AUC为0.5194；加入HMM状态变量后，IC提升到0.0867，Rank IC提升到0.0954，方向准确率提升到52.48%，AUC提升到0.5457，同时MSE从0.001000下降到0.000910。这说明HMM识别出的市场状态变量确实为Transformer提供了一定的market regime信息，对未来涨跌幅预测和涨跌方向判断有一定增量帮助。

但是，从每日截面排序和组合收益指标来看，加入HMM后效果并不是全面提升。普通Transformer的Daily Rank IC Mean为0.0814，Top组日均收益为0.3121%，Top-Bottom日均收益为0.3892%；加入HMM后，Daily Rank IC Mean小幅下降到0.0796，Top组日均收益下降到0.2685%，Top-Bottom日均收益下降到0.3036%。这说明HMM状态变量虽然提升了整体预测相关性和方向判断能力，但在每日截面排序和组合收益转化上还不够稳定。

## 五、初步结论

综合来看，HMM多状态机制与Transformer多头注意力机制的结合是有价值的。HMM状态变量能够帮助模型理解不同市场regime，并在IC、Rank IC、方向准确率、AUC和MSE等预测指标上带来改善。这说明HMM状态信息确实包含一定的增量预测信息。

不过，当前方法只是将HMM状态概率简单拼接到Transformer输入特征中，收益端指标并没有同步提升，说明这种增量信息还没有被稳定地转化为组合收益。下一步可以继续优化模型结构，例如将HMM状态作为单独的regime embedding，或者作为attention bias融入多头注意力机制，而不是仅作为普通数值特征输入。同时，也可以增加训练样本量、训练轮数，并在更多test区间上验证结果稳定性。

因此，本实验的结论可以概括为：HMM-informed Transformer在预测层面相对普通Transformer有一定提升，但在组合收益层面还不够稳定。后续需要进一步改进HMM状态信息和多头注意力机制的结合方式，使其不仅提升预测相关性，也能更稳定地提升排序能力和策略收益。
