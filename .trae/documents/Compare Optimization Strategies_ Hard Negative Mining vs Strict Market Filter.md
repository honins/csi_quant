# Plan: Implement "Strict Market Filter" Strategy

## 1. Save Current Version (Hard Negative Mining)
- Create a new branch `feat/hard_negative_mining` to preserve the current successful optimization (Hard Negative Mining + Environment Awareness).
- Commit all current changes to this branch.

## 2. Implement New Strategy (Strict Market Filter)
- Create a new branch `feat/strict_market_filter` based on the current state.
- **Revert Hard Negative Mining**:
    - Modify `src/ai/ai_optimizer_improved.py`: Remove the logic that forces labels to 0 for the "2025-03 to 2025-05" period.
- **Implement Strict Filtering**:
    - Modify `src/prediction/prediction_utils.py`:
    - Instead of just adjusting thresholds for `Sideways` and `Bear` markets, **strictly abandon** trading in these environments.
    - Logic: If `trend_regime` is not `'bull'`, force `is_predicted_low_point = False` and skip the trade.

## 3. Retrain and Verify
- **Retrain AI**: Run `python run.py ai` to train a new model *without* the hard negative bias (pure data + strict filtering strategy).
- **Backtest**: Run a 1-year rolling backtest (`2025-01-09` to `2026-01-09`) to evaluate the new strategy.
- **Comparison**: Compare the results (Accuracy, Precision, Recall) with the "Hard Negative Mining" version.

## 4. Expected Outcome
- This strategy tests whether "avoiding uncertain markets" is more effective than "learning to identify traps".
- We expect **higher Precision** (fewer false positives in bad markets) but potentially **lower Recall** (missing opportunities in early trend reversals).
