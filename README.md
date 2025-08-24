# Classification-Model-for-Trading
Adaptive trading framework using ML-driven market regime classification to toggle between rule-based strategies in Backtrader. Includes feature engineering, regime prediction, dynamic strategy switching, and trade-level analytics. Built for modularity, extensibility, and robust performance benchmarking.

This is a modular trading framework that combines machine learning-based market regime classification with dynamic strategy execution using Backtrader. The project leverages engineered features like seasonal correlation, MACD momentum, volatility spikes, and price deviation to train a Random Forest classifier that labels each market day as either rallying or sideways.
Based on the predicted regime, the system toggles between two rule-based strategies:
- Strategy A: RSI-driven mean reversion for sideways markets
- Strategy B: EMA-based trend following for rallying conditions
The framework includes:
- Feature engineering pipeline for regime detection
- ML model training and prediction integration
- Strategy switching logic within Backtrader
- Trade-level performance tracking and win ratio analytics
- Transition logs for regime shifts
