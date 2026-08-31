# Transformer Baseline Summary

## Experiment

- Model: Transformer Encoder sequence regressor
- Label horizon: `1d` (`label_1d_raw`)
- Sequence length: 20 trading observations per stock
- Features: return_5d, return_10d, return_20d, turnover_20d_mean, log_neg_market_value
- Walk-forward: 504 training days / 5 embargo days / monthly test
- Scaler and winsorization bounds fitted inside each training window only
- Training rows per window: capped at 120000 sequences
- Device: mps

## Test results

| Metric | Value |
|---|---:|
| Valid evaluated rows | 4,440,862 |
| Test days | 931 |
| Rank IC mean | 0.038227 |
| Rank IC std | 0.142991 |
| ICIR | 0.267340 |
| IC positive ratio | 60.47% |
| Direction accuracy | 50.21% |
| MSE | 0.001129 |
| Top annualized return | 32.30% |
| Bottom annualized return | -3.16% |
| Top-Bottom gross annualized return | 35.46% |
| Top-Bottom net annualized return | 29.30% |
| Net annualized compound return | 32.39% |
| Net annualized volatility | 15.68% |
| Sharpe | 1.869117 |
| Max drawdown | -23.90% |
| Mean top turnover | 25.29% |
| Mean bottom turnover | 23.45% |

## Output files

The full prediction parquet files, window audit parquet files, and large intermediate cache files are not included in this repository.

Included repository outputs:

- Notebook: `notebooks/v5_transformer_baseline.ipynb`
- Summary: `reports/transformer_1d_summary.md`
- Group return figure: `figures/transformer_1d_group_returns.png`
- Loss curves: `figures/transformer_1d_loss_curves.png`
- Long-short NAV figure: `figures/transformer_1d_nav.png`
- Daily Rank IC figure: `figures/transformer_1d_rank_ic.png`

Excluded local/internal outputs:

- Full prediction parquet files
- Window audit parquet files
- Raw market data
- Large cache files

## Interpretation note

This is a sequence-model baseline. It should be compared against `ridge_baseline_v2`, `mlp_baseline`, and other V2 outputs under the same evaluator before making any conclusion. A useful Transformer result should improve not only validation loss, but also out-of-sample Rank IC, net spread return, Sharpe, turnover, and drawdown under the shared evaluation contract.
